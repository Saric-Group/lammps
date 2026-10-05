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
   Attraction between filament beads that only acts if their filament
   directions (from fix filament/direction) are aligned:

     E = g(c) U(r),  c = t_i . t_j

   U is the attractive cosine/squared well (-eps for r < sigma, then
   -eps cos^2(pi (r - sigma) / (2 (cut - sigma))) up to cut) and g a
   smoothstep switch that is 0 at c = cos_off and 1 at c = cos_on.
   cos_on > cos_off selects parallel, cos_on < cos_off anti-parallel pairs.

   The force is the central force -g dU/dr. The dependence of g on the
   positions of the bonded neighbours that define t is neglected, so the
   style is not conservative.
------------------------------------------------------------------------- */

#include "pair_align_cosine_squared.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

/* ---------------------------------------------------------------------- */

PairAlignCosineSquared::PairAlignCosineSquared(LAMMPS *lmp) : Pair(lmp), id_fix(nullptr), fix_direction(nullptr)
{
  writedata = 0;
  restartinfo = 0;
  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairAlignCosineSquared::~PairAlignCosineSquared()
{
  delete[] id_fix;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(cut);
    memory->destroy(cos_on);
    memory->destroy(cos_off);
  }
}

/* ---------------------------------------------------------------------- */

void PairAlignCosineSquared::allocate()
{
  allocated = 1;
  const int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(epsilon, n, n, "pair:epsilon");
  memory->create(sigma, n, n, "pair:sigma");
  memory->create(cut, n, n, "pair:cut");
  memory->create(cos_on, n, n, "pair:cos_on");
  memory->create(cos_off, n, n, "pair:cos_off");
}

/* ----------------------------------------------------------------------
   pair_style align/cosine/squared fix-ID cutoff
------------------------------------------------------------------------- */

void PairAlignCosineSquared::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Pair style align/cosine/squared needs a fix ID and a cutoff");
  delete[] id_fix;
  id_fix = utils::strdup(arg[0]);
  cut_global = utils::numeric(FLERR, arg[1], false, lmp);

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   pair_coeff I J eps sigma [cut [cos_on cos_off]]
------------------------------------------------------------------------- */

void PairAlignCosineSquared::coeff(int narg, char **arg)
{
  if (narg != 4 && narg != 5 && narg != 7)
    error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  const double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  const double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = cut_global;
  double cos_on_one = 0.9;
  double cos_off_one = 0.7;
  if (narg >= 5) cut_one = utils::numeric(FLERR, arg[4], false, lmp);
  if (narg == 7) {
    cos_on_one = utils::numeric(FLERR, arg[5], false, lmp);
    cos_off_one = utils::numeric(FLERR, arg[6], false, lmp);
  }
  if (cut_one <= sigma_one)
    error->all(FLERR, "Pair style align/cosine/squared needs cutoff > sigma");
  if (cos_on_one == cos_off_one)
    error->all(FLERR, "Pair style align/cosine/squared needs cos_on != cos_off");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      cos_on[i][j] = cos_on_one;
      cos_off[i][j] = cos_off_one;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ---------------------------------------------------------------------- */

void PairAlignCosineSquared::init_style()
{
  fix_direction = modify->get_fix_by_id(id_fix);
  if (!fix_direction)
    error->all(FLERR, "Pair style align/cosine/squared: fix ID {} does not exist", id_fix);
  if (strcmp(fix_direction->style, "filament/direction") != 0)
    error->all(FLERR, "Pair style align/cosine/squared: fix {} is not of style filament/direction",
               id_fix);
  int dim;
  auto nevery = (int *) fix_direction->extract("nevery", dim);
  if (!nevery || *nevery != 1)
    error->all(FLERR,
               "Pair style align/cosine/squared needs fix filament/direction with Nevery = 1, "
               "otherwise forces use outdated directions");

  neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

double PairAlignCosineSquared::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "Pair style align/cosine/squared: no mixing, set pair_coeff {} {}", i, j);

  epsilon[j][i] = epsilon[i][j];
  sigma[j][i] = sigma[i][j];
  cut[j][i] = cut[i][j];
  cos_on[j][i] = cos_on[i][j];
  cos_off[j][i] = cos_off[i][j];
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairAlignCosineSquared::get_directions(double **&dir, int *&nbond)
{
  int dim;
  dir = (double **) fix_direction->extract("dir", dim);
  nbond = (int *) fix_direction->extract("nbond", dim);
  if (!dir || !nbond) error->all(FLERR, "Pair style align/cosine/squared: no directions from fix {}", id_fix);
}

/* ----------------------------------------------------------------------
   smoothstep switch on the cosine of the angle between the directions
------------------------------------------------------------------------- */

double PairAlignCosineSquared::gate(double c, int itype, int jtype)
{
  double t = (c - cos_off[itype][jtype]) / (cos_on[itype][jtype] - cos_off[itype][jtype]);
  if (t <= 0.0) return 0.0;
  if (t >= 1.0) return 1.0;
  return t * t * (3.0 - 2.0 * t);
}

/* ----------------------------------------------------------------------
   attractive cosine/squared well U(r) and dU/dr
------------------------------------------------------------------------- */

void PairAlignCosineSquared::well(double r, int itype, int jtype, double &u, double &dudr)
{
  const double eps = epsilon[itype][jtype];
  const double sig = sigma[itype][jtype];
  if (r <= sig) {
    u = -eps;
    dudr = 0.0;
    return;
  }
  const double scale = 0.5 * MY_PI / (cut[itype][jtype] - sig);
  const double arg = scale * (r - sig);
  const double cosarg = cos(arg);
  u = -eps * cosarg * cosarg;
  dudr = eps * sin(2.0 * arg) * scale;
}

/* ---------------------------------------------------------------------- */

void PairAlignCosineSquared::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **dir;
  int *nbond;
  get_directions(dir, nbond);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;
  double *special_lj = force->special_lj;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (nbond[i] == 0) continue;
    const int itype = type[i];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      if (nbond[j] == 0) continue;
      const int jtype = type[j];

      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq[itype][jtype]) continue;

      const double c = dir[i][0] * dir[j][0] + dir[i][1] * dir[j][1] + dir[i][2] * dir[j][2];
      const double g = gate(c, itype, jtype);
      if (g == 0.0) continue;

      const double r = sqrt(rsq);
      double u, dudr;
      well(r, itype, jtype, u, dudr);
      const double fpair = -factor_lj * g * dudr / r;

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delx * fpair;
        f[j][1] -= dely * fpair;
        f[j][2] -= delz * fpair;
      }

      const double evdwl = eflag ? factor_lj * g * u : 0.0;
      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

double PairAlignCosineSquared::single(int i, int j, int itype, int jtype, double rsq,
                                      double /*factor_coul*/, double factor_lj, double &fforce)
{
  fforce = 0.0;
  double **dir;
  int *nbond;
  get_directions(dir, nbond);
  if (nbond[i] == 0 || nbond[j] == 0) return 0.0;

  const double c = dir[i][0] * dir[j][0] + dir[i][1] * dir[j][1] + dir[i][2] * dir[j][2];
  const double g = gate(c, itype, jtype);
  if (g == 0.0) return 0.0;

  const double r = sqrt(rsq);
  double u, dudr;
  well(r, itype, jtype, u, dudr);
  fforce = -factor_lj * g * dudr / r;
  return factor_lj * g * u;
}
