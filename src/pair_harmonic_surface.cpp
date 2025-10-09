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

#include "pair_harmonic_surface.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairHarmonicSurface::PairHarmonicSurface(LAMMPS *lmp) : Pair(lmp), k(nullptr), r_zero(nullptr), cut(nullptr)
{
  born_matrix_enable = 1;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairHarmonicSurface::~PairHarmonicSurface()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r_zero);
    memory->destroy(cut);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSurface::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype, isurf;
  double xtmp, ytmp, ztmp, fxtmp, fytmp, fztmp;
  double delx, dely, delz, rsq, factor_lj;
  double normx, normy, normz, normr, rotation[3][3];
  int *ilist, *jlist, *numneigh, **firstneigh;

  double costheta_max;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR, "Pair style harmonic/surface requires atom style ellipsoid");

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    fxtmp = fytmp = fztmp = 0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      // determine normal vector at surface atom
      if (jtype == surface_type) {
        isurf = j;
      } else if (itype == surface_type) {
        isurf = i;
      } else {
        error->all(FLERR, "Pair between type %d and %d does not contain given surface type %d.", itype, jtype, surface_type);
      }
      // simply set normal vectors as pointing radially inward this way:
      // normx = - x[isurf][0];
      // normy = 0;
      // normz = - x[isurf][2];

      if (!atom->ellipsoid_flag) error->all(FLERR, "Atom with index %d and type %d is not an ellipsoid, cannot obtain normal for pair style harmonic/surface", isurf, surface_type);
      // taken from pair_ylz.cpp
      // does this mean longest axis has to be x?
      // or are the ellipsoid axis sorted by length?
      double* iquat = avec->bonus[atom->ellipsoid[isurf]].quat;
      MathExtra::quat_to_mat_trans(iquat, rotation);
      // YlZ ellipsoids point outward, so make them point inward here
      normx = -rotation[0][0];
      normy = -rotation[0][1];
      normz = -rotation[0][2];

      normr = sqrt(normx * normx + normy * normy + normz * normz);
      normx /= normr;
      normy /= normr;
      normz /= normr;

      if (rsq < cutsq[itype][jtype]) {
        const double r = sqrt(rsq);
        costheta_max = r_zero[itype][jtype] / cut[itype][jtype];
        const double align = std::abs(delx * normx + dely * normy + delz * normz) / r; // [0, 1] alignment of delta with surface normal. TODO: properly calculate normals
        double theta_fact = (align - costheta_max) / (1.0 - costheta_max); // linear decay of force magnitude when going away from ideal alignment
        theta_fact = theta_fact > 0 ? theta_fact : 0;
        double delta = r_zero[itype][jtype] - r;
        const double prefactor = factor_lj * delta * k[itype][jtype] * theta_fact;
        const double fpair = 2.0 * prefactor / r;
        
        if (itype == surface_type) {
          // fpair is negative when larger than r_zero
          // therefore change sign here to move surface atoms inwards
          // therefore along normal, this should point inwards
          fxtmp -= align * fpair * normx;
          fytmp -= align * fpair * normy;
          fztmp -= align * fpair * normz;
          if (newton_pair || j < nlocal) {
              f[j][0] += align * fpair * normx;
              f[j][1] += align * fpair * normy;
              f[j][2] += align * fpair * normz;
          }
        } else if (jtype == surface_type) {
          fxtmp += align * fpair * normx;
          fytmp += align * fpair * normy;
          fztmp += align * fpair * normz;
          if (newton_pair || j < nlocal) {
              f[j][0] -= align * fpair * normx;
              f[j][1] -= align * fpair * normy;
              f[j][2] -= align * fpair * normz;
          }
        } else {
          error->all(FLERR, "Pair between type %d and %d does not contain given surface type %d.", itype, jtype, surface_type);
        }

        if (evflag) {
          const double philj = prefactor * delta;
          ev_tally(i, j, nlocal, newton_pair, philj, 0.0, fpair, delx, dely, delz);
        }
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

void PairHarmonicSurface::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(k, n, n, "pair:k");
  memory->create(r_zero, n, n, "pair:r_zero");
  memory->create(cut, n, n, "pair:cut");
  memory->create(cutsq, n, n, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairHarmonicSurface::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  surface_type = utils::numeric(FLERR, arg[0], false, lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairHarmonicSurface::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double k_one = utils::numeric(FLERR, arg[2], false, lmp);
  double r_zero_one = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[4], false, lmp);
  // int surf_type = utils::inumeric(FLERR, arg[5], false, lmp); // TODO: projection, read an extra variable here for which atom type to extract normal from

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      k[i][j] = k_one;
      r_zero[i][j] = r_zero_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairHarmonicSurface::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
    k[i][j] = mix_energy(k[i][i], k[j][j], cut[i][i], cut[j][j]);
  }
  k[j][i] = k[i][j];
  r_zero[j][i] = r_zero[i][j];
  cut[j][i] = cut[i][j];
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHarmonicSurface::write_restart(FILE *fp)
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
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHarmonicSurface::read_restart(FILE *fp)
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
        }
        MPI_Bcast(&k[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&r_zero[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHarmonicSurface::write_restart_settings(FILE *fp)
{
  fwrite(&surface_type, sizeof(int), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHarmonicSurface::read_restart_settings(FILE *fp)
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

void PairHarmonicSurface::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g\n", i, k[i][i], r_zero[i][i], cut[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairHarmonicSurface::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) fprintf(fp, "%d %d %g %g %g\n", i, j, k[i][j], r_zero[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairHarmonicSurface::single(int i, int j, int itype, int jtype, double rsq,
                               double /*factor_coul*/, double factor_lj, double &fforce)
{
  if (rsq >= cutsq[itype][jtype]) {
    fforce = 0.0;
    return 0.0;
  }
  error->all(FLERR, "Using single on accident");
  const double delx = atom->x[i][0] - atom->x[j][0];
  const double dely = atom->x[i][1] - atom->x[j][1];
  const double delz = atom->x[i][2] - atom->x[j][2];

  const double r = sqrt(delx * delx + delz * delz); // TODO: projection
  const double delta = r_zero[itype][jtype] - r;
  const double philj = factor_lj * delta * delta * k[itype][jtype];
  fforce = 2.0 * philj / (r * delta);
  return philj;
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSurface::born_matrix(int i, int j, int itype, int jtype, double rsq,
                            double /*factor_coul*/, double factor_lj, double &dupair,
                            double &du2pair)
{
  const double delx = atom->x[i][0] - atom->x[j][0];
  const double dely = atom->x[i][1] - atom->x[j][1];
  const double delz = atom->x[i][2] - atom->x[j][2];
  double r = sqrt(delx * delx + delz * delz); // TODO: projection
  double dr = r - r_zero[itype][jtype];

  double du = 0;
  double du2 = 2 * k[itype][jtype];
  if (r > 0) du = du2 * dr;

  dupair = factor_lj * du;
  du2pair = factor_lj * du2;
}

/* ---------------------------------------------------------------------- */

void *PairHarmonicSurface::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "k") == 0) return (void *) k;
  if (strcmp(str, "r_zero") == 0) return (void *) r_zero;
  if (strcmp(str, "cut") == 0) return (void *) cut;
  return nullptr;
}
