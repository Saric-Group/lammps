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
   Harmonic spring between filament beads that meet on a given side:

     E = K (r - r0)^2  for r < cut

   with the left vectors of fix filament/direction ... side. "left" acts
   only if j is on the left of i and i on the left of j (a left-left
   contact, (x_j - x_i).left_i > 0 and (x_i - x_j).left_j > 0), "right"
   only for right-right contacts. Anti-parallel neighbours meet left-left
   or right-right, parallel ones never do.

   nth N: only beads whose chain index (i_chain_index) is a multiple of N
   interact, both beads of a pair have to be. Use fix chain/index ...
   modulo N (or a multiple of N), so that relabelling keeps the indices
   modulo N and the springs stay on the same monomers.
   "any" has no side condition (needs no side keyword on the fix).

   Optional keywords after nth and cut (as treadmill_hoomd's SideHarmonic):
     parallel c      dir_i . dir_j >= c
     antiparallel c  dir_i . dir_j <= c (with both, either one suffices)
     lateral c       |cos| of the pair axis with both directions <= c,
                     skips neighbours along a filament
     region w        both beads in the band |y| < w/2 (box centred on
                     y = 0, y wrapped into the box)

   The force is the central harmonic force. The sides, directions and chain
   index switch it on and off, so energy is not conserved when they change.
------------------------------------------------------------------------- */

#include "pair_harmonic_side.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairHarmonicSide::PairHarmonicSide(LAMMPS *lmp) :
    Pair(lmp), id_fix(nullptr), fix_direction(nullptr), need_chain_index(0), need_left(0)
{
  writedata = 0;
  restartinfo = 0;
  single_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairHarmonicSide::~PairHarmonicSide()
{
  delete[] id_fix;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(cut);
    memory->destroy(side);
    memory->destroy(nth);
    memory->destroy(parallel_cos);
    memory->destroy(antiparallel_cos);
    memory->destroy(check_alignment);
    memory->destroy(lateral_cos);
    memory->destroy(region_half);
  }
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSide::allocate()
{
  allocated = 1;
  const int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(k, n, n, "pair:k");
  memory->create(r0, n, n, "pair:r0");
  memory->create(cut, n, n, "pair:cut");
  memory->create(side, n, n, "pair:side");
  memory->create(nth, n, n, "pair:nth");
  memory->create(parallel_cos, n, n, "pair:parallel_cos");
  memory->create(antiparallel_cos, n, n, "pair:antiparallel_cos");
  memory->create(check_alignment, n, n, "pair:check_alignment");
  memory->create(lateral_cos, n, n, "pair:lateral_cos");
  memory->create(region_half, n, n, "pair:region_half");
}

/* ----------------------------------------------------------------------
   pair_style harmonic/side fix-ID cutoff
------------------------------------------------------------------------- */

void PairHarmonicSide::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Pair style harmonic/side needs a fix ID and a cutoff");
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
   pair_coeff I J K r0 left|right|any [nth [cut]] [parallel c] [antiparallel c]
              [lateral c] [region w]
------------------------------------------------------------------------- */

void PairHarmonicSide::coeff(int narg, char **arg)
{
  if (narg < 5)
    error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  const double k_one = utils::numeric(FLERR, arg[2], false, lmp);
  const double r0_one = utils::numeric(FLERR, arg[3], false, lmp);
  int side_one;
  if (strcmp(arg[4], "left") == 0)
    side_one = LEFT;
  else if (strcmp(arg[4], "right") == 0)
    side_one = RIGHT;
  else if (strcmp(arg[4], "any") == 0)
    side_one = ANY;
  else
    error->all(FLERR, "Pair style harmonic/side: side must be left, right or any, not {}", arg[4]);

  // optional positional nth and cut, then keywords
  int iarg = 5;
  int nth_one = 1;
  if (iarg < narg && utils::is_integer(arg[iarg]))
    nth_one = utils::inumeric(FLERR, arg[iarg++], false, lmp);
  if (nth_one < 1) error->all(FLERR, "Pair style harmonic/side: nth must be >= 1");
  double cut_one = cut_global;
  if (iarg < narg && utils::is_double(arg[iarg]))
    cut_one = utils::numeric(FLERR, arg[iarg++], false, lmp);
  if (cut_one <= 0.0) error->all(FLERR, "Pair style harmonic/side: cutoff must be > 0");

  double parallel_one = 2.0, antiparallel_one = -2.0, lateral_one = 1.0, region_one = -1.0;
  int alignment_one = 0;
  while (iarg < narg) {
    if (iarg + 1 >= narg)
      error->all(FLERR, "Pair style harmonic/side: keyword {} needs a value", arg[iarg]);
    const double value = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
    if (strcmp(arg[iarg], "parallel") == 0) {
      if (value < -1.0 || value > 1.0)
        error->all(FLERR, "Pair style harmonic/side: parallel must be between -1 and 1");
      parallel_one = value;
      alignment_one = 1;
    } else if (strcmp(arg[iarg], "antiparallel") == 0) {
      if (value < -1.0 || value > 1.0)
        error->all(FLERR, "Pair style harmonic/side: antiparallel must be between -1 and 1");
      antiparallel_one = value;
      alignment_one = 1;
    } else if (strcmp(arg[iarg], "lateral") == 0) {
      if (value < 0.0 || value > 1.0)
        error->all(FLERR, "Pair style harmonic/side: lateral must be between 0 and 1");
      lateral_one = value;
    } else if (strcmp(arg[iarg], "region") == 0) {
      if (value <= 0.0) error->all(FLERR, "Pair style harmonic/side: region width must be > 0");
      region_one = 0.5 * value;
    } else
      error->all(FLERR, "Pair style harmonic/side: unknown keyword {}", arg[iarg]);
    iarg += 2;
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      k[i][j] = k_one;
      r0[i][j] = r0_one;
      cut[i][j] = cut_one;
      side[i][j] = side_one;
      nth[i][j] = nth_one;
      parallel_cos[i][j] = parallel_one;
      antiparallel_cos[i][j] = antiparallel_one;
      check_alignment[i][j] = alignment_one;
      lateral_cos[i][j] = lateral_one;
      region_half[i][j] = region_one;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSide::init_style()
{
  fix_direction = modify->get_fix_by_id(id_fix);
  if (!fix_direction) error->all(FLERR, "Pair style harmonic/side: fix ID {} does not exist", id_fix);
  if (strcmp(fix_direction->style, "filament/direction") != 0)
    error->all(FLERR, "Pair style harmonic/side: fix {} is not of style filament/direction", id_fix);
  int dim;
  auto nevery = (int *) fix_direction->extract("nevery", dim);
  if (!nevery || *nevery != 1)
    error->all(FLERR,
               "Pair style harmonic/side needs fix filament/direction with Nevery = 1, "
               "otherwise forces use outdated sides");
  need_left = 0;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      if (setflag[i][j] && side[i][j] != ANY) need_left = 1;
  if (need_left && !fix_direction->extract("left", dim))
    error->all(FLERR,
               "Pair style harmonic/side left or right needs fix filament/direction {} with the "
               "side keyword",
               id_fix);

  need_chain_index = 0;
  int nth_max = 1;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      if (setflag[i][j] && nth[i][j] > 1) {
        need_chain_index = 1;
        nth_max = MAX(nth_max, nth[i][j]);
      }
  if (need_chain_index) {
    int flag, cols, ghost;
    const int index = atom->find_custom_ghost("chain_index", flag, cols, ghost);
    if (index < 0 || flag != 0 || cols != 0 || !ghost)
      error->all(FLERR,
                 "Pair style harmonic/side with nth > 1 needs the per-atom property i_chain_index "
                 "with ghost values (fix chain/index or fix bond/react ... chain_index)");
    // relabelling must keep the indices modulo every nth
    for (auto *ifix : modify->get_fix_by_style("^chain/index$")) {
      auto *modulo = (int *) ifix->extract("modulo", dim);
      for (int i = 1; i <= atom->ntypes; i++)
        for (int j = i; j <= atom->ntypes; j++)
          if (modulo && setflag[i][j] && *modulo % nth[i][j] != 0 && comm->me == 0)
            error->warning(FLERR,
                           "Pair style harmonic/side nth {} of types {} {} does not divide the "
                           "modulo {} of fix chain/index {}: relabelling moves the springs to other "
                           "monomers",
                           nth[i][j], i, j, *modulo, ifix->id);
    }
  }

  neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

double PairHarmonicSide::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "Pair style harmonic/side: no mixing, set pair_coeff {} {}", i, j);

  k[j][i] = k[i][j];
  r0[j][i] = r0[i][j];
  cut[j][i] = cut[i][j];
  side[j][i] = side[i][j];
  nth[j][i] = nth[i][j];
  parallel_cos[j][i] = parallel_cos[i][j];
  antiparallel_cos[j][i] = antiparallel_cos[i][j];
  check_alignment[j][i] = check_alignment[i][j];
  lateral_cos[j][i] = lateral_cos[i][j];
  region_half[j][i] = region_half[i][j];
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSide::get_arrays(double **&dir, double **&left, int *&chain_index)
{
  int dim;
  dir = (double **) fix_direction->extract("dir", dim);
  if (!dir) error->all(FLERR, "Pair style harmonic/side: no directions from fix {}", id_fix);
  left = (double **) fix_direction->extract("left", dim);
  if (need_left && !left)
    error->all(FLERR, "Pair style harmonic/side: no left vectors from fix {}", id_fix);
  chain_index = nullptr;
  if (need_chain_index) {
    int flag, cols;
    chain_index = atom->ivector[atom->find_custom("chain_index", flag, cols)];
  }
}

/* ----------------------------------------------------------------------
   y wrapped into the periodic box, so ghost images get the y of their atom
------------------------------------------------------------------------- */

double PairHarmonicSide::wrapped_y(double y) const
{
  if (domain->yperiodic) {
    if (y >= domain->boxhi[1])
      y -= domain->yprd;
    else if (y < domain->boxlo[1])
      y += domain->yprd;
  }
  return y;
}

/* ----------------------------------------------------------------------
   true if the pair meets all conditions of its pair_coeff: chain indices
   multiples of nth, both beads in the band, the side, the alignment of
   the directions and the lateral condition; del = x_i - x_j
------------------------------------------------------------------------- */

bool PairHarmonicSide::active(int i, int j, int itype, int jtype, double delx, double dely,
                              double delz, double **dir, double **left, int *chain_index)
{
  const int n = nth[itype][jtype];
  if (n > 1) {
    const int ci = chain_index[i], cj = chain_index[j];
    if (ci < 0 || cj < 0 || ci % n || cj % n) return false;
  }
  const double half = region_half[itype][jtype];
  if (half > 0.0) {
    double **x = atom->x;
    if (fabs(wrapped_y(x[i][1])) >= half || fabs(wrapped_y(x[j][1])) >= half) return false;
  }
  const int s = side[itype][jtype];
  if (s != ANY) {
    // side of j seen from i and of i seen from j, 0 for beads without a left vector
    const double si = -(delx * left[i][0] + dely * left[i][1] + delz * left[i][2]);
    const double sj = delx * left[j][0] + dely * left[j][1] + delz * left[j][2];
    if (s == LEFT && !(si > 0.0 && sj > 0.0)) return false;
    if (s == RIGHT && !(si < 0.0 && sj < 0.0)) return false;
  }
  if (check_alignment[itype][jtype]) {
    const double c = dir[i][0] * dir[j][0] + dir[i][1] * dir[j][1] + dir[i][2] * dir[j][2];
    if (!(c >= parallel_cos[itype][jtype] || c <= antiparallel_cos[itype][jtype])) return false;
  }
  const double lat = lateral_cos[itype][jtype];
  if (lat < 1.0) {
    // skips pairs along a filament: |cos| of the pair axis with either direction above lat
    const double r = sqrt(delx * delx + dely * dely + delz * delz);
    if (r == 0.0) return false;
    const double ci = delx * dir[i][0] + dely * dir[i][1] + delz * dir[i][2];
    const double cj = delx * dir[j][0] + dely * dir[j][1] + delz * dir[j][2];
    if (fabs(ci) > lat * r || fabs(cj) > lat * r) return false;
  }
  return true;
}

/* ---------------------------------------------------------------------- */

void PairHarmonicSide::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **dir, **left;
  int *chain_index;
  get_arrays(dir, left, chain_index);

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
    const int itype = type[i];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      const int jtype = type[j];

      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq[itype][jtype]) continue;
      if (!active(i, j, itype, jtype, delx, dely, delz, dir, left, chain_index)) continue;

      const double r = sqrt(rsq);
      const double dr = r - r0[itype][jtype];
      const double rk = k[itype][jtype] * dr;
      const double fpair = (r > 0.0) ? -factor_lj * 2.0 * rk / r : 0.0;

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delx * fpair;
        f[j][1] -= dely * fpair;
        f[j][2] -= delz * fpair;
      }

      const double evdwl = eflag ? factor_lj * rk * dr : 0.0;
      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

double PairHarmonicSide::single(int i, int j, int itype, int jtype, double rsq,
                                double /*factor_coul*/, double factor_lj, double &fforce)
{
  fforce = 0.0;
  double **dir, **left;
  int *chain_index;
  get_arrays(dir, left, chain_index);

  double **x = atom->x;
  const double delx = x[i][0] - x[j][0];
  const double dely = x[i][1] - x[j][1];
  const double delz = x[i][2] - x[j][2];
  if (!active(i, j, itype, jtype, delx, dely, delz, dir, left, chain_index)) return 0.0;

  const double r = sqrt(rsq);
  const double dr = r - r0[itype][jtype];
  const double rk = k[itype][jtype] * dr;
  fforce = (r > 0.0) ? -factor_lj * 2.0 * rk / r : 0.0;
  return factor_lj * rk * dr;
}
