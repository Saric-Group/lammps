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
   two beads with preferred crossing angles +-theta0. Based on the style of
   Andraz Gnidovec (branch feature/update-treadmilling/dipole).

   E = (1 + cos(pi r / rc)) U(theta)
   U = Amin + (A - Amin) (1 - exp(-2 kappa sin^2(theta - theta0)))
                         (1 - exp(-2 kappa sin^2(theta + theta0)))

   This version assumes unit dipoles in the xy plane, theta is the signed
   angle from mu_i to mu_j, the torque has only a z component.
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
    Pair(lmp), cut_global(0.0), intermol_flag(0), Aamp(nullptr), Amin(nullptr), kappa(nullptr),
    theta0(nullptr), cut(nullptr), c0(nullptr), s0(nullptr)
{
  single_enable = 0;
  restartinfo = 0;
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
   pair_style nematic/angle/soft cutoff [intermol]
------------------------------------------------------------------------- */

void PairNematicSoft::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2) error->all(FLERR, "Illegal pair_style nematic/angle/soft command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global <= 0.0) error->all(FLERR, "Pair style nematic/angle/soft: cutoff must be positive");
  intermol_flag = 0;
  if (narg == 2) {
    if (strcmp(arg[1], "intermol") == 0)
      intermol_flag = 1;
    else
      error->all(FLERR, "Unknown keyword {} for pair_style nematic/angle/soft", arg[1]);
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

      const double r = sqrt(rsq);
      const double rc = cut[itype][jtype];
      const double Aij = Aamp[itype][jtype];
      const double Amin_ij = Amin[itype][jtype];
      const double kij = kappa[itype][jtype];
      const double c0ij = c0[itype][jtype];
      const double s0ij = s0[itype][jtype];

      // unit dipoles in the xy plane
      const double c = mu[i][0] * mu[j][0] + mu[i][1] * mu[j][1];
      const double s = mu[i][0] * mu[j][1] - mu[i][1] * mu[j][0];
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

      // dU/dtheta with theta = angle from mu_i to mu_j (c = cos, s = sin of theta)
      const double cm = c * c0ij + s * s0ij;    // cos(theta - theta0)
      const double cp = c * c0ij - s * s0ij;    // cos(theta + theta0)
      const double dfm = 4.0 * kij * em * sm * cm;
      const double dfp = 4.0 * kij * ep * sp * cp;
      const double dUdtheta = (Aij - Amin_ij) * (dfm * (1.0 - ep) + (1.0 - em) * dfp);
      // E depends on theta_j - theta_i: torque_i = +dE/dtheta, torque_j = -dE/dtheta
      const double tau = factor_lj * Sr * dUdtheta;

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      torque[i][2] += tau;
      if (newton_pair || j < nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;
        torque[j][2] -= tau;
      }

      if (evflag) {
        const double evdwl = eflag ? factor_lj * Sr * U : 0.0;
        ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
