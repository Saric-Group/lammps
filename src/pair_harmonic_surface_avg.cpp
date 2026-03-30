/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Felix Wodaczek (ISTA)
------------------------------------------------------------------------- */

#include "pair_harmonic_surface_avg.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "error.h"
#include "fix_property_atom.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairHarmonicSurfaceAvg::PairHarmonicSurfaceAvg(LAMMPS *lmp) : Pair(lmp), k(nullptr), r_zero(nullptr), cut(nullptr), cut_tang(nullptr),\
                                                             id_fix_store_nnvecs(nullptr), fix_store_nnvecs(nullptr),
                                                             idx_nnvec_contributors(-1), idx_avg_nvecs(-1),
                                                             nnvec_contributors_atom(nullptr), avg_nvecs_atom(nullptr)
{
  born_matrix_enable = 1;
  writedata = 1;
  comm_forward = 4;
  cfstyle = ATOMVECS;
}

/* ---------------------------------------------------------------------- */

PairHarmonicSurfaceAvg::~PairHarmonicSurfaceAvg()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r_zero);
    memory->destroy(cut);
    memory->destroy(cut_tang);
    memory->destroy(cutsq);
    memory->destroy(normal_factor);

    if (id_fix_store_nnvecs && modify->get_fix_by_id(id_fix_store_nnvecs)) {
      modify->delete_fix(id_fix_store_nnvecs);
      delete[] id_fix_store_nnvecs;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype, isurf, iinteract;
  double fxtmp, fytmp, fztmp;
  double delx, dely, delz, rsq, factor_lj;
  double normx, normy, normz, normr, rotation[3][3], normal_dist, tang_dist;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double costheta_max;

  ev_init(eflag, vflag);

  // grow local vectors and arrays if necessary
  // removed since all handled via fix store/atom now
  // if (atom->nmax > nmax) grow_local();

  // reset pointers to atom properties in case they were reallocated
  // that being nnvec_contributors_atom and avg_nvecs_atom
  find_atom_properties();

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int ntotal = atom->nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR, "Pair style harmonic/surface/avg requires atom style ellipsoid");

  calculate_mean_normal_vectors();

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    fxtmp = fytmp = fztmp = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];

      // extract normal vector for surface atom
      if (jtype == surface_type) {
        isurf = j;
        iinteract = i;
      } else if (itype == surface_type) {
        isurf = i;
        iinteract = j;
      } else {
        error->all(FLERR, "Pair between type %d and %d does not contain given surface type %d.",
                   itype, jtype, surface_type);
      }

      // extract average surface normal at interactor atom
      normx = avg_nvecs_atom[iinteract][0];
      normy = avg_nvecs_atom[iinteract][1];
      normz = avg_nvecs_atom[iinteract][2];

      delx = x[iinteract][0] - x[isurf][0];
      dely = x[iinteract][1] - x[isurf][1];
      delz = x[iinteract][2] - x[isurf][2];
      rsq = delx * delx + dely * dely + delz * delz;

      normr = sqrt(normx * normx + normy * normy + normz * normz);
      if (normr == 0.0 || nnvec_contributors_atom[iinteract] == 0) continue;
      normal_dist = std::abs(delx * normx + dely * normy + delz * normz) / normr;
      tang_dist = sqrt(rsq - normal_dist * normal_dist);
      if (rsq >= cutsq[itype][jtype] || tang_dist > cut_tang[itype][jtype]) continue;

      double r = sqrt(rsq);

      const double align = std::abs(delx * normx + dely * normy + delz * normz) / r;    // [0, 1] alignment of delta with surface normal.
      double delta = r_zero[itype][jtype] - (r * align);
      const double prefactor = factor_lj * delta * k[itype][jtype]; // * theta_fact;
      const double fpair = 2.0 * prefactor / (float) nnvec_contributors_atom[iinteract]; //  / r;

      if (itype == surface_type) {
        // fpair is negative when larger than r_zero
        // therefore change sign here to move surface atoms inwards
        // therefore along normal, this should point inwards
        fxtmp -= fpair * normx;
        fytmp -= fpair * normy;
        fztmp -= fpair * normz;
        if (newton_pair || j < nlocal) {
          f[j][0] += fpair * normx;
          f[j][1] += fpair * normy;
          f[j][2] += fpair * normz;
        }
      } else if (jtype == surface_type) {
        fxtmp += fpair * normx;
        fytmp += fpair * normy;
        fztmp += fpair * normz;
        if (newton_pair || j < nlocal) {
          f[j][0] -= fpair * normx;
          f[j][1] -= fpair * normy;
          f[j][2] -= fpair * normz;
        }
      } else {
        error->all(FLERR, "Pair between type %d and %d does not contain given surface type %d.",
                   itype, jtype, surface_type);
      }

      if (evflag) {
        const double philj = prefactor * delta;
        ev_tally(i, j, nlocal, newton_pair, philj, 0.0, fpair, delx, dely, delz);
      }
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(k, n, n, "pair:k");
  memory->create(r_zero, n, n, "pair:r_zero");
  memory->create(cut, n, n, "pair:cut");
  memory->create(cut_tang, n, n, "pair:cut_tang");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(normal_factor, n, n, "pair:normal_factor");

  // nmax = atom->nmax;
  // memory->create(nnvec_contributors, atom->nmax, "pair_harmonic_surface_avg:nnvec_contributors");
  // memory->create(avg_nvecs, atom->nmax, 3, "pair_harmonic_surface_avg:avg_nvecs");

  // register number of contributors and mean-field normal vector for debugging
  setup_custom_atom_properties();
}

/* ----------------------------------------------------------------------
   custom atom properties for use in output/computations later on
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::setup_custom_atom_properties()
{
  id_fix_store_nnvecs = "harmonic_surface_avg_props_internal";

  // Keep per-atom state in an internal fix so values persist and migrate
  Fix* myfix = nullptr;
  if (!modify->get_fix_by_id(id_fix_store_nnvecs)) {
    myfix = modify->add_fix(std::string(id_fix_store_nnvecs) +
                    " all property/atom i_nnvec_contributors d2_avg_nvecs 3 ghost yes");
  }
  myfix = modify->get_fix_by_id(id_fix_store_nnvecs);
  if (!myfix) {
    error->all(FLERR, "Failed to create fix to store custom atom properties for pair style harmonic/surface/avg");
  }
  // this is not strictly necessary, but could allow for easier access to fix if needed
  fix_store_nnvecs = dynamic_cast<FixPropertyAtom *>(myfix);

  int flag = -1;
  int cols = -1;

  idx_nnvec_contributors = atom->find_custom("nnvec_contributors", flag, cols);
  if (idx_nnvec_contributors < 0) {
    error->all(FLERR, "Failed to create custom atom property nnvec_contributors via internal fix property/atom");
  }
  if (flag != 0 || cols != 0) {
    error->all(FLERR, "Custom atom property nnvec_contributors must be an integer scalar (i_nnvec_contributors)");
  }

  idx_avg_nvecs = atom->find_custom("avg_nvecs", flag, cols);
  if (idx_avg_nvecs < 0) {
    error->all(FLERR, "Failed to create custom atom property avg_nvecs via internal fix property/atom");
  }
  if (flag != 1 || cols != 3) {
    error->all(FLERR, "Custom atom property avg_nvecs must be a 3-column double array (d2_avg_nvecs)");
  }

  nnvec_contributors_atom = atom->ivector[idx_nnvec_contributors];
  avg_nvecs_atom = atom->darray[idx_avg_nvecs];
}

void PairHarmonicSurfaceAvg::find_atom_properties()
{
  int flag = -1;
  int cols = -1;
  idx_nnvec_contributors = atom->find_custom("nnvec_contributors", flag, cols);
  idx_avg_nvecs = atom->find_custom("avg_nvecs", flag, cols);

  if (idx_nnvec_contributors < 0 || idx_avg_nvecs < 0) {
    error->all(FLERR, "Custom atom properties nnvec_contributors and avg_nvecs must be allocated before pair style harmonic/surface/avg can be allocated");
  }

  nnvec_contributors_atom = atom->ivector[idx_nnvec_contributors];
  avg_nvecs_atom = atom->darray[idx_avg_nvecs];
}

/* ----------------------------------------------------------------------
   mean average vector calculation and propagator
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::calculate_mean_normal_vectors()
{ 
  int i, j, ii, jj, inum, jnum, itype, jtype, isurf, iinteract;
  double normx, normy, normz, normr, rotation[3][3], normal_dist, tang_dist;
  double delx, dely, delz, rsq, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;
  
  double **x = atom->x;
  int *type = atom->type;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // zero out normal vector contributors and average normal vectors for each atom
  for (ii = 0; ii < atom->nlocal; ii++) {
    nnvec_contributors_atom[ii] = 0;
    avg_nvecs_atom[ii][0] = 0.0;
    avg_nvecs_atom[ii][1] = 0.0;
    avg_nvecs_atom[ii][2] = 0.0;
  }

  // fill normal vectors with averages of nearest interaction partners
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];

      // determine normal vector at surface atom
      if (jtype == surface_type) {
        isurf = j;
        iinteract = i;
      } else if (itype == surface_type) {
        isurf = i;
        iinteract = j;
      } else {
        continue; // not interacting with surface, skip quietly
        error->all(FLERR, "Pair between type %d and %d does not contain given surface type %d.",
                   itype, jtype, surface_type);
      }

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq[itype][jtype]) continue;

      if (!atom->ellipsoid_flag) error->all(FLERR, "Atom with index %d and type %d is not an ellipsoid, cannot obtain normal for pair style harmonic/surface/avg", isurf, surface_type);
      // taken from pair_ylz.cpp
      // does this mean longest axis has to be x?
      // or are the ellipsoid axis sorted by length?
      double *iquat = avec->bonus[atom->ellipsoid[isurf]].quat;
      MathExtra::quat_to_mat_trans(iquat, rotation);
      // YlZ ellipsoids point outward, so make them point inward here
      normx = normal_factor[itype][jtype] * rotation[0][0];
      normy = normal_factor[itype][jtype] * rotation[0][1];
      normz = normal_factor[itype][jtype] * rotation[0][2];

      normr = sqrt(normx * normx + normy * normy + normz * normz);
      normx /= normr;
      normy /= normr;
      normz /= normr;

      avg_nvecs_atom[iinteract][0] += normx;
      avg_nvecs_atom[iinteract][1] += normy;
      avg_nvecs_atom[iinteract][2] += normz;
      nnvec_contributors_atom[iinteract]++;
    }
  }

  for (int ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (nnvec_contributors_atom[i] > 0) {
      avg_nvecs_atom[i][0] /= nnvec_contributors_atom[i];
      avg_nvecs_atom[i][1] /= nnvec_contributors_atom[i];
      avg_nvecs_atom[i][2] /= nnvec_contributors_atom[i];
    }
  }

  // all owned atoms now know their average surface normal
  // forward comm average surface normal to ghosts of other procs
  cfstyle = ATOMVECS; // switch to communicating class variables for forward comm, we need the average normal vector and number of contributors for each atom in the force calculation
  comm->forward_comm(this);

  // Re-count contributors using distance projected along the averaged local surface normal.
  // This multiplicity is used to split the force among neighbors in the force loop below.
  for (ii = 0; ii < atom->nlocal; ii++) nnvec_contributors_atom[ii] = 0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj] & NEIGHMASK;
      jtype = type[j];


      if (jtype == surface_type) {
        isurf = j;
        iinteract = i;
      } else if (itype == surface_type) {
        isurf = i;
        iinteract = j;
      } else {
        continue;
      }

      normx = avg_nvecs_atom[iinteract][0];
      normy = avg_nvecs_atom[iinteract][1];
      normz = avg_nvecs_atom[iinteract][2];
      normr = sqrt(normx * normx + normy * normy + normz * normz);
      if (normr == 0.0) continue;

      delx = x[iinteract][0] - x[isurf][0];
      dely = x[iinteract][1] - x[isurf][1];
      delz = x[iinteract][2] - x[isurf][2];
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq[itype][jtype]) continue;

      normal_dist = std::abs(delx * normx + dely * normy + delz * normz) / normr;
      tang_dist = sqrt(rsq - normal_dist * normal_dist);
      if (tang_dist <= cut_tang[itype][jtype]) nnvec_contributors_atom[iinteract]++;
    }
  }

  // forward new number of neighbours again to use in force calculation
  cfstyle = ATOMVECS; // switch to communicating class variables for forward comm, we need the average normal vector and number of contributors for each atom in the force calculation
  comm->forward_comm(this);
}


/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  surface_type = utils::numeric(FLERR, arg[0], false, lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::coeff(int narg, char **arg)
{
  if (!(narg == 5 || narg == 6 || narg == 7))
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double k_one = utils::numeric(FLERR, arg[2], false, lmp);
  double r_zero_one = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[4], false, lmp);
  double cut_tang_one = cut_one;
  int normal_factor_one = -1;
  if (narg == 6) {
    cut_tang_one = utils::numeric(FLERR, arg[5], false, lmp);
  } else if (narg == 7) {
    cut_tang_one = utils::numeric(FLERR, arg[5], false, lmp);
    normal_factor_one = utils::inumeric(FLERR, arg[6], false, lmp);
  }

  if (!(normal_factor_one == 1 || normal_factor_one == -1))
    error->all(FLERR, "Multiplier for surface normal has to be either -1 or 1, found %d.",
               normal_factor_one);

  // int surf_type = utils::inumeric(FLERR, arg[5], false, lmp); // TODO: projection, read an extra variable here for which atom type to extract normal from

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      k[i][j] = k_one;
      r_zero[i][j] = r_zero_one;
      cut[i][j] = cut_one;
      cut_tang[i][j] = cut_tang_one;
      normal_factor[i][j] = normal_factor_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairHarmonicSurfaceAvg::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
    cut_tang[i][j] = mix_distance(cut_tang[i][i], cut_tang[j][j]);
    k[i][j] = mix_energy(k[i][i], k[j][j], cut[i][i], cut[j][j]);
  }
  k[j][i] = k[i][j];
  r_zero[j][i] = r_zero[i][j];
  cut[j][i] = cut[i][j];
  cut_tang[j][i] = cut_tang[i][j];
  normal_factor[j][i] = normal_factor[i][j];
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&k[i][j], sizeof(double), 1, fp);
        fwrite(&r_zero[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&cut_tang[i][j], sizeof(double), 1, fp);
        fwrite(&normal_factor[i][j], sizeof(int), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &k[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &r_zero[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut_tang[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &normal_factor[i][j], sizeof(int), 1, fp, nullptr, error);
        }
        MPI_Bcast(&k[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&r_zero[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut_tang[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&normal_factor[i][j], 1, MPI_INT, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::write_restart_settings(FILE *fp)
{
  fwrite(&surface_type, sizeof(int), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &surface_type, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&surface_type, 1, MPI_INT, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g %g %d\n", i, k[i][i], r_zero[i][i], cut[i][i], cut_tang[i][i],
            normal_factor[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %d\n", i, j, k[i][j], r_zero[i][j], cut[i][j], cut_tang[i][j],
              normal_factor[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairHarmonicSurfaceAvg::single(int i, int j, int itype, int jtype, double rsq,
                                      double /*factor_coul*/, double factor_lj, double &fforce)
{
  if (rsq >= cutsq[itype][jtype]) {
    fforce = 0.0;
    return 0.0;
  }
  error->all(FLERR, "Single not implemented for pair style harmonic/surface/avg");
  const double delx = atom->x[i][0] - atom->x[j][0];
  const double dely = atom->x[i][1] - atom->x[j][1];
  const double delz = atom->x[i][2] - atom->x[j][2];

  const double r = sqrt(delx * delx + delz * delz);    // TODO: projection
  const double delta = r_zero[itype][jtype] - r;
  const double philj = factor_lj * delta * delta * k[itype][jtype];
  fforce = 2.0 * philj / (r * delta);
  return philj;
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSurfaceAvg::born_matrix(int i, int j, int itype, int jtype, double rsq,
                                         double /*factor_coul*/, double factor_lj, double &dupair,
                                         double &du2pair)
{
  error->all(FLERR, "Born matrix not implemented for pair style harmonic/surface/avg");
  const double delx = atom->x[i][0] - atom->x[j][0];
  const double dely = atom->x[i][1] - atom->x[j][1];
  const double delz = atom->x[i][2] - atom->x[j][2];
  double r = sqrt(delx * delx + delz * delz);    // TODO: projection
  double dr = r - r_zero[itype][jtype];

  double du = 0;
  double du2 = 2 * k[itype][jtype];
  if (r > 0) du = du2 * dr;

  dupair = factor_lj * du;
  du2pair = factor_lj * du2;
}

/* ---------------------------------------------------------------------- */

void *PairHarmonicSurfaceAvg::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "k") == 0) return (void *) k;
  if (strcmp(str, "r_zero") == 0) return (void *) r_zero;
  if (strcmp(str, "cut") == 0) return (void *) cut;
  if (strcmp(str, "cut_tang") == 0) return (void *) cut_tang;
  if (strcmp(str, "normal_factor") == 0) return (void *) normal_factor;
  return nullptr;
}

/* ----------------------------------------------------------------------
  grow local vectors and arrays if necessary
  keep them all atom->nmax in length even if ghost storage not needed
  removed since all handled via fix store/atom now, see setup_custom_atom_properties and find_atom_properties
------------------------------------------------------------------------- */
/*
void PairHarmonicSurfaceAvg::grow_local()
{
  if (allocated) {
    memory->destroy(nnvec_contributors);
    memory->destroy(avg_nvecs);
  }

  int nmax = atom->nmax;
  memory->grow(nnvec_contributors, nmax, "pair_harmonic_surface_avg:nnvec_contributors");
  memory->grow(avg_nvecs, nmax, 3, "pair_harmonic_surface_avg:avg_nvecs");

  // update pointers in case they were reallocated
  find_atom_properties();
}  
*/

int PairHarmonicSurfaceAvg::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                              int * /*pbc*/)
{
  int i, j, m = 0;

  if (cfstyle == ATOMVECS) {
    // Refresh atom property pointers in case atom list was reallocated
    find_atom_properties();
    
    for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = ubuf(nnvec_contributors_atom[j]).d;
        buf[m++] = avg_nvecs_atom[j][0];
        buf[m++] = avg_nvecs_atom[j][1];
        buf[m++] = avg_nvecs_atom[j][2];
    }
  }
  
  return m;
}

void PairHarmonicSurfaceAvg::unpack_forward_comm(int n, int first, double *buf)
{
  int i, k, m, last;
  m = 0;
  last = first + n;

  if (cfstyle == ATOMVECS) {
    // Refresh atom property pointers in case atom list was reallocated
    find_atom_properties();

    for (i = first; i < last; i++) {
      nnvec_contributors_atom[i] = (int) ubuf(buf[m++]).i;
      avg_nvecs_atom[i][0] = buf[m++];
      avg_nvecs_atom[i][1] = buf[m++];
      avg_nvecs_atom[i][2] = buf[m++];
    }
  }
}