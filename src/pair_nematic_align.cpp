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
   Pair style nematic/align: alignment of the dipole mu of a bead with the
   line to its neighbours. Based on the nematic/align of Andraz Gnidovec
   (branch feature/update-treadmilling/dipole), rewritten for this fork.

   E = -eps (1 - r/rc)^2 [(mu_i.u)^2 + (mu_j.u)^2],  u = (x_i - x_j)/r

   With max_dist N only beads of the same molecule ID whose per-atom
   chain indices (i_chain_index, from fix chain/index or fix bond/react
   ... chain_index) differ by at most N interact, i.e. the dipoles are
   aligned with the backbone of their own filament.

   Forces and torques are the exact derivatives of E with respect to the
   positions and the orientations of the dipoles.
------------------------------------------------------------------------- */

#include "pair_nematic_align.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNematicAlign::PairNematicAlign(LAMMPS *lmp) :
    Pair(lmp), cut_global(0.0), max_dist(0), epsilon(nullptr), cut(nullptr), no_radial(nullptr)
{
  single_enable = 0;
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

PairNematicAlign::~PairNematicAlign()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(epsilon);
    memory->destroy(cut);
    memory->destroy(no_radial);
  }
}

/* ---------------------------------------------------------------------- */

void PairNematicAlign::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(no_radial, n + 1, n + 1, "pair:no_radial");
}

/* ----------------------------------------------------------------------
   pair_style nematic/align cutoff [max_dist N]
------------------------------------------------------------------------- */

void PairNematicAlign::settings(int narg, char **arg)
{
  if (narg != 1 && narg != 3) error->all(FLERR, "Illegal pair_style nematic/align command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global <= 0.0) error->all(FLERR, "Pair style nematic/align: cutoff must be positive");
  max_dist = 0;
  if (narg == 3) {
    if (strcmp(arg[1], "max_dist") != 0)
      error->all(FLERR, "Unknown keyword {} for pair_style nematic/align", arg[1]);
    max_dist = utils::inumeric(FLERR, arg[2], false, lmp);
    if (max_dist < 1) error->all(FLERR, "Pair style nematic/align: max_dist must be at least 1");
  }

  if (allocated)
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
}

/* ----------------------------------------------------------------------
   pair_coeff I J epsilon [cut] [no_radial]
------------------------------------------------------------------------- */

void PairNematicAlign::coeff(int narg, char **arg)
{
  int no_radial_one = 0;
  if (narg > 3 && strcmp(arg[narg - 1], "no_radial") == 0) {
    no_radial_one = 1;
    narg--;
  }
  if (narg < 3 || narg > 4) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  const double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double cut_one = cut_global;
  if (narg == 4) cut_one = utils::numeric(FLERR, arg[3], false, lmp);
  if (cut_one <= 0.0) error->all(FLERR, "Invalid cutoff specified for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++)
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      cut[i][j] = cut_one;
      no_radial[i][j] = no_radial_one;
      setflag[i][j] = 1;
      count++;
    }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairNematicAlign::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Pair style nematic/align requires atom attributes mu and torque");
  if (max_dist) {
    if (!atom->molecule_flag)
      error->all(FLERR, "Pair style nematic/align max_dist requires molecule IDs");
    int flag, cols, ghost;
    const int index = atom->find_custom_ghost("chain_index", flag, cols, ghost);
    if (index < 0 || flag != 0 || cols != 0 || !ghost)
      error->all(FLERR,
                 "Pair style nematic/align max_dist needs the per-atom property i_chain_index "
                 "with ghost values (fix chain/index or fix bond/react ... chain_index)");
  }
  neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

double PairNematicAlign::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "Pair style nematic/align: no mixing, set pair_coeff {} {}", i, j);
  epsilon[j][i] = epsilon[i][j];
  cut[j][i] = cut[i][j];
  no_radial[j][i] = no_radial[i][j];
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairNematicAlign::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;
  double *special_lj = force->special_lj;

  tagint *molecule = atom->molecule;
  int *chain_index = nullptr;
  if (max_dist) {
    int flag, cols;
    chain_index = atom->ivector[atom->find_custom("chain_index", flag, cols)];
  }

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
      const int jtype = type[j];
      if (max_dist) {
        // same filament and at most max_dist monomers apart along the chain
        if (molecule[i] != molecule[j]) continue;
        const int ci = chain_index[i], cj = chain_index[j];
        if (ci < 0 || cj < 0 || abs(ci - cj) > max_dist) continue;
      }

      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq[itype][jtype] || rsq == 0.0) continue;

      const double r = sqrt(rsq);
      const double rinv = 1.0 / r;
      const double rc = cut[itype][jtype];
      const double eps = factor_lj * epsilon[itype][jtype];

      const double ux = delx * rinv, uy = dely * rinv, uz = delz * rinv;
      const double mui_u = mu[i][0] * ux + mu[i][1] * uy + mu[i][2] * uz;
      const double muj_u = mu[j][0] * ux + mu[j][1] * uy + mu[j][2] * uz;
      const double S = mui_u * mui_u + muj_u * muj_u;

      const double one_minus = 1.0 - r / rc;
      const double g = -eps * one_minus * one_minus;    // E = g S
      const double dgdr = 2.0 * eps * one_minus / rc;

      // tangential part: -g dS/dx_i, with dS/dx_i = 2/r [(mu_i.u) a_i + (mu_j.u) a_j],
      // a_k = mu_k - (mu_k.u) u
      const double tp = -2.0 * g * rinv;
      double fx = tp * (mui_u * (mu[i][0] - mui_u * ux) + muj_u * (mu[j][0] - muj_u * ux));
      double fy = tp * (mui_u * (mu[i][1] - mui_u * uy) + muj_u * (mu[j][1] - muj_u * uy));
      double fz = tp * (mui_u * (mu[i][2] - mui_u * uz) + muj_u * (mu[j][2] - muj_u * uz));
      // radial part: -dg/dr S u
      if (!no_radial[itype][jtype]) {
        fx -= dgdr * S * ux;
        fy -= dgdr * S * uy;
        fz -= dgdr * S * uz;
      }

      // torque = mu x (-dE/dmu) = -2 g (mu.u) (mu x u)
      const double tc = -2.0 * g;
      torque[i][0] += tc * mui_u * (mu[i][1] * uz - mu[i][2] * uy);
      torque[i][1] += tc * mui_u * (mu[i][2] * ux - mu[i][0] * uz);
      torque[i][2] += tc * mui_u * (mu[i][0] * uy - mu[i][1] * ux);

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      if (newton_pair || j < nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;
        torque[j][0] += tc * muj_u * (mu[j][1] * uz - mu[j][2] * uy);
        torque[j][1] += tc * muj_u * (mu[j][2] * ux - mu[j][0] * uz);
        torque[j][2] += tc * muj_u * (mu[j][0] * uy - mu[j][1] * ux);
      }

      if (evflag) {
        const double evdwl = eflag ? g * S : 0.0;
        ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
