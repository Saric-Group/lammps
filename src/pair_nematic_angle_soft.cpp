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
   Pair style nematic/angle/soft: soft interaction between the dipoles of
   two beads with preferred crossing angles +-theta0. Based on the 2D style of
   Andraz Gnidovec (branch feature/update-treadmilling/dipole), generalised
   to 3D with a surface normal.

   E = (1 + cos(pi r / rc)) U(theta)
   U = Amin + (A - Amin) (1 - exp(-2 kappa sin^2(theta - theta0)))
                         (1 - exp(-2 kappa sin^2(theta + theta0)))

   theta is the signed angle from mu_i to mu_j, measured in the tangent
   plane of the surface: n is the normalised sum of the surface normals of the
   two beads, mu_k' = mu_k - (mu_k.n) n the projections, c = cos(theta) =
   mu_i'.mu_j' / (|mu_i'| |mu_j'|), s = sin(theta) = n.(mu_i' x mu_j') / (...).
   Pairs whose normals cancel or with a dipole (almost) along n are skipped.

   The torques are the exact derivatives of E with respect to the dipole
   orientations. The force is the radial derivative of the cosine envelope
   only: the normal is taken as constant, so the force is not conservative
   where the normal changes with the bead positions (like harmonic/surface).
------------------------------------------------------------------------- */

#include "pair_nematic_angle_soft.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNematicSoft::PairNematicSoft(LAMMPS *lmp) :
    Pair(lmp), cut_global(0.0), intermol_flag(0), normal_fixed(0), name_normal("avg_nvecs"), Aamp(nullptr), Amin(nullptr), kappa(nullptr),
    theta0(nullptr), cut(nullptr), c0(nullptr), s0(nullptr)
{
  single_enable = 0;
  orientation_flag = 1;
  restartinfo = 0;
  normal_vec[0] = normal_vec[1] = normal_vec[2] = 0.0;
}

/* ---------------------------------------------------------------------- */

PairNematicSoft::~PairNematicSoft()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(Aamp);
    memory->destroy(Amin);
    memory->destroy(kappa);
    memory->destroy(theta0);
    memory->destroy(cut);
    memory->destroy(c0);
    memory->destroy(s0);
  }
}

/* ---------------------------------------------------------------------- */

void PairNematicSoft::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(Aamp, n + 1, n + 1, "pair:Aamp");
  memory->create(Amin, n + 1, n + 1, "pair:Amin");
  memory->create(kappa, n + 1, n + 1, "pair:kappa");
  memory->create(theta0, n + 1, n + 1, "pair:theta0");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(c0, n + 1, n + 1, "pair:c0");
  memory->create(s0, n + 1, n + 1, "pair:s0");
}

/* ----------------------------------------------------------------------
   pair_style nematic/angle/soft cutoff [intermol] [normal d2_name | normal fixed nx ny nz]
------------------------------------------------------------------------- */

void PairNematicSoft::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style nematic/angle/soft command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global <= 0.0) error->all(FLERR, "Pair style nematic/angle/soft: cutoff must be positive");

  intermol_flag = 0;
  normal_fixed = 0;
  name_normal = "avg_nvecs";
  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "intermol") == 0) {
      intermol_flag = 1;
      iarg++;
    } else if (strcmp(arg[iarg], "normal") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style nematic/angle/soft normal", error);
      if (strcmp(arg[iarg + 1], "fixed") == 0) {
        if (iarg + 5 > narg)
          utils::missing_cmd_args(FLERR, "pair_style nematic/angle/soft normal fixed", error);
        for (int k = 0; k < 3; k++) normal_vec[k] = utils::numeric(FLERR, arg[iarg + 2 + k], false, lmp);
        const double len = sqrt(normal_vec[0] * normal_vec[0] + normal_vec[1] * normal_vec[1] +
                                normal_vec[2] * normal_vec[2]);
        if (len == 0.0) error->all(FLERR, "Pair style nematic/angle/soft: fixed normal is zero");
        for (int k = 0; k < 3; k++) normal_vec[k] /= len;
        normal_fixed = 1;
        iarg += 5;
      } else {
        normal_fixed = 0;
        name_normal = arg[iarg + 1];
        if (utils::strmatch(name_normal, "^d2_")) name_normal = name_normal.substr(3);
        iarg += 2;
      }
    } else {
      error->all(FLERR, "Unknown keyword {} for pair_style nematic/angle/soft", arg[iarg]);
    }
  }
  if (allocated)
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
}

/* ----------------------------------------------------------------------
   pair_coeff I J A Amin kappa theta0 [cut]
------------------------------------------------------------------------- */

void PairNematicSoft::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  const double A_one = utils::numeric(FLERR, arg[2], false, lmp);
  const double Amin_one = utils::numeric(FLERR, arg[3], false, lmp);
  const double kappa_one = utils::numeric(FLERR, arg[4], false, lmp);
  const double theta0_one = utils::numeric(FLERR, arg[5], false, lmp);
  double cut_one = cut_global;
  if (narg == 7) cut_one = utils::numeric(FLERR, arg[6], false, lmp);
  if (cut_one <= 0.0) error->all(FLERR, "Invalid cutoff specified for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++)
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      Aamp[i][j] = A_one;
      Amin[i][j] = Amin_one;
      kappa[i][j] = kappa_one;
      theta0[i][j] = theta0_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairNematicSoft::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Pair style nematic/angle/soft requires atom attributes mu and torque");
  if (intermol_flag && !atom->molecule_flag)
    error->all(FLERR, "Pair style nematic/angle/soft intermol requires molecule IDs");
  if (!normal_fixed) {
    int flag, cols, ghost;
    const int index = atom->find_custom_ghost(name_normal.c_str(), flag, cols, ghost);
    if (index < 0 || flag != 1 || cols != 3 || !ghost)
      error->all(FLERR,
                 "Pair style nematic/angle/soft needs a 3-column per-atom array d2_{} with ghost "
                 "values as surface normal (e.g. from pair harmonic/surface/avg) or 'normal fixed'",
                 name_normal);
  }
  neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

double PairNematicSoft::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "Pair style nematic/angle/soft: no mixing, set pair_coeff {} {}", i, j);
  Aamp[j][i] = Aamp[i][j];
  Amin[j][i] = Amin[i][j];
  kappa[j][i] = kappa[i][j];
  theta0[j][i] = theta0[i][j];
  cut[j][i] = cut[i][j];
  c0[i][j] = c0[j][i] = cos(theta0[i][j]);
  s0[i][j] = s0[j][i] = sin(theta0[i][j]);
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   @FelixWodaczek/dipole3d name of the per-atom normal array (without d2_) for fix bond/react
   energy_check, nullptr with 'normal fixed'
------------------------------------------------------------------------- */

void *PairNematicSoft::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "orientation_normal_name") == 0 && !normal_fixed) return (void *) &name_normal;
  return nullptr;
}

/* ----------------------------------------------------------------------
   @FelixWodaczek/dipole3d energy of a pair for fix bond/react energy_check: same energy as compute()
   (atoms without dipole, pairs with cancelling normals or a dipole along n give 0)
   n_i, n_j are the surface normals of the atoms (ignored with 'normal fixed')
   the molecule gate of 'intermol' is up to the caller
------------------------------------------------------------------------- */

double PairNematicSoft::single_orientation(int itype, int jtype, double rsq, int special_order,
                                           const double *mu_i, const double *mu_j, const double *n_i,
                                           const double *n_j)
{
  constexpr double LMIN = 1.0e-2;
  if (!mu_i || !mu_j || mu_i[3] <= 0.0 || mu_j[3] <= 0.0) return 0.0;
  if (rsq >= cutsq[itype][jtype] || rsq == 0.0) return 0.0;

  double n[3];
  if (normal_fixed) {
    for (int k = 0; k < 3; k++) n[k] = normal_vec[k];
  } else {
    if (!n_i || !n_j) return 0.0;
    for (int k = 0; k < 3; k++) n[k] = n_i[k] + n_j[k];
    const double nlen = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (nlen < 1.0e-8) return 0.0;
    for (int k = 0; k < 3; k++) n[k] /= nlen;
  }

  const double mui_n = mu_i[0] * n[0] + mu_i[1] * n[1] + mu_i[2] * n[2];
  const double muj_n = mu_j[0] * n[0] + mu_j[1] * n[1] + mu_j[2] * n[2];
  double pi[3], pj[3];
  for (int k = 0; k < 3; k++) {
    pi[k] = mu_i[k] - mui_n * n[k];
    pj[k] = mu_j[k] - muj_n * n[k];
  }
  const double li = sqrt(pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2]);
  const double lj = sqrt(pj[0] * pj[0] + pj[1] * pj[1] + pj[2] * pj[2]);
  if (li < LMIN || lj < LMIN) return 0.0;
  for (int k = 0; k < 3; k++) {
    pi[k] /= li;
    pj[k] /= lj;
  }

  const double rc = cut[itype][jtype];
  const double Aij = Aamp[itype][jtype];
  const double Amin_ij = Amin[itype][jtype];
  const double kij = kappa[itype][jtype];
  const double c = pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2];
  const double s = n[0] * (pi[1] * pj[2] - pi[2] * pj[1]) + n[1] * (pi[2] * pj[0] - pi[0] * pj[2]) +
      n[2] * (pi[0] * pj[1] - pi[1] * pj[0]);
  const double sm = s * c0[itype][jtype] - c * s0[itype][jtype];
  const double sp = s * c0[itype][jtype] + c * s0[itype][jtype];
  const double U = Amin_ij + (Aij - Amin_ij) * (1.0 - exp(-2.0 * kij * sm * sm)) *
      (1.0 - exp(-2.0 * kij * sp * sp));
  return force->special_lj[special_order] * (1.0 + cos(M_PI * sqrt(rsq) / rc)) * U;
}

/* ---------------------------------------------------------------------- */

void PairNematicSoft::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  tagint *molecule = atom->molecule;
  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;
  double *special_lj = force->special_lj;

  double **normal = nullptr;
  if (!normal_fixed) {
    int flag, cols;
    normal = atom->darray[atom->find_custom(name_normal.c_str(), flag, cols)];
  }
  constexpr double LMIN = 1.0e-2;    // smallest |mu'| (unit mu) with a defined angle

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (mu[i][3] <= 0.0) continue;
    const int itype = type[i];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      if (mu[j][3] <= 0.0) continue;
      if (intermol_flag && molecule[i] == molecule[j]) continue;
      const int jtype = type[j];

      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq[itype][jtype] || rsq == 0.0) continue;

      // common surface normal of the pair
      double n[3];
      if (normal_fixed) {
        n[0] = normal_vec[0];
        n[1] = normal_vec[1];
        n[2] = normal_vec[2];
      } else {
        n[0] = normal[i][0] + normal[j][0];
        n[1] = normal[i][1] + normal[j][1];
        n[2] = normal[i][2] + normal[j][2];
        const double nlen = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nlen < 1.0e-8) continue;
        n[0] /= nlen;
        n[1] /= nlen;
        n[2] /= nlen;
      }

      // dipoles projected on the tangent plane (dipoles are unit vectors)
      const double mui_n = mu[i][0] * n[0] + mu[i][1] * n[1] + mu[i][2] * n[2];
      const double muj_n = mu[j][0] * n[0] + mu[j][1] * n[1] + mu[j][2] * n[2];
      double pi[3], pj[3];
      for (int k = 0; k < 3; k++) {
        pi[k] = mu[i][k] - mui_n * n[k];
        pj[k] = mu[j][k] - muj_n * n[k];
      }
      const double li = sqrt(pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2]);
      const double lj = sqrt(pj[0] * pj[0] + pj[1] * pj[1] + pj[2] * pj[2]);
      if (li < LMIN || lj < LMIN) continue;
      for (int k = 0; k < 3; k++) {
        pi[k] /= li;
        pj[k] /= lj;
      }

      const double r = sqrt(rsq);
      const double rc = cut[itype][jtype];
      const double Aij = Aamp[itype][jtype];
      const double Amin_ij = Amin[itype][jtype];
      const double kij = kappa[itype][jtype];
      const double c0ij = c0[itype][jtype];
      const double s0ij = s0[itype][jtype];

      const double c = pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2];
      const double s = n[0] * (pi[1] * pj[2] - pi[2] * pj[1]) +
                       n[1] * (pi[2] * pj[0] - pi[0] * pj[2]) +
                       n[2] * (pi[0] * pj[1] - pi[1] * pj[0]);
      const double sm = s * c0ij - c * s0ij;    // sin(theta - theta0)
      const double sp = s * c0ij + c * s0ij;    // sin(theta + theta0)
      const double em = exp(-2.0 * kij * sm * sm);
      const double ep = exp(-2.0 * kij * sp * sp);
      const double U = Amin_ij + (Aij - Amin_ij) * (1.0 - em) * (1.0 - ep);

      const double xarg = M_PI * r / rc;
      const double Sr = 1.0 + cos(xarg);
      const double dSdr = -(M_PI / rc) * sin(xarg);

      // radial force from the cosine envelope only
      const double fpair = -factor_lj * dSdr * U / r;
      const double fx = fpair * delx, fy = fpair * dely, fz = fpair * delz;

      // dU/dtheta, theta = angle from mu_i' to mu_j' about n
      const double cm = c * c0ij + s * s0ij;    // cos(theta - theta0)
      const double cp = c * c0ij - s * s0ij;    // cos(theta + theta0)
      const double dfm = 4.0 * kij * em * sm * cm;
      const double dfp = 4.0 * kij * ep * sp * cp;
      const double tau = factor_lj * Sr * (Aij - Amin_ij) * (dfm * (1.0 - ep) + (1.0 - em) * dfp);

      // E depends on theta_j - theta_i. Torque on a dipole = mu x (-dE/dmu) with
      // dtheta_k/dmu_k = (n x p_k) / l_k: tau [n - (mu_k.n)/l_k p_k] for i, minus that for j
      double ti[3], tj[3];
      for (int k = 0; k < 3; k++) {
        ti[k] = tau * (n[k] - mui_n / li * pi[k]);
        tj[k] = -tau * (n[k] - muj_n / lj * pj[k]);
      }

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      torque[i][0] += ti[0];
      torque[i][1] += ti[1];
      torque[i][2] += ti[2];
      if (newton_pair || j < nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;
        torque[j][0] += tj[0];
        torque[j][1] += tj[1];
        torque[j][2] += tj[2];
      }

      if (evflag) {
        const double evdwl = eflag ? factor_lj * Sr * U : 0.0;
        ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
