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
   fix ID group surface/normal [every N]

   Averages the surface normals of pair style harmonic/surface/avg (d2_avg_nvecs,
   i_nnvec_contributors of owned and ghost atoms) once per step in pre_force, instead of at
   every force evaluation of the pair. The pair must be given 'normals fix'; its neighbour list
   and coefficients are used. Under run_style respa the averaging runs once per (outer) step,
   at the first pre_force_respa call, so the pair's force on an inner level reuses it.
   Fixes defined after this one (filament/direction side, nematic/angle/soft) see the normals
   of the current step.
------------------------------------------------------------------------- */

#include "fix_surface_normal.h"

#include "error.h"
#include "force.h"
#include "modify.h"
#include "pair_harmonic_surface_avg.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSurfaceNormal::FixSurfaceNormal(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), nevery(1), last_step(-1), pair(nullptr)
{
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "every") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix surface/normal every", error);
      nevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (nevery < 1) error->all(FLERR, "Fix surface/normal every must be > 0");
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix surface/normal keyword {}", arg[iarg]);
  }
}

/* ---------------------------------------------------------------------- */

int FixSurfaceNormal::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceNormal::init()
{
  // nsub 0 returns the style only if there is exactly one instance
  pair = dynamic_cast<PairHarmonicSurfaceAvg *>(force->pair_match("^harmonic/surface/avg$", 0, 0));
  if (!pair) {
    if (force->pair_match("^harmonic/surface/avg$", 0, 1))
      error->all(FLERR, "Fix surface/normal supports only one instance of pair style harmonic/surface/avg");
    error->all(FLERR, "Fix surface/normal requires pair style harmonic/surface/avg");
  }
  if (!pair->external_normals)
    error->all(FLERR,
               "Fix surface/normal requires pair_style harmonic/surface/avg ... normals fix, "
               "otherwise the pair recomputes the normals at every force evaluation");

  // fixes that read the normals in their pre_force should come after this one
  for (const auto &style : {"^filament/direction$"})
    for (const auto *other : modify->get_fix_by_style(style))
      if (modify->find_fix(other->id) < modify->find_fix(id))
        error->warning(FLERR,
                       "Fix {} is defined before fix surface/normal {}: it sees the normals "
                       "of the previous step",
                       other->id, id);
}

/* ---------------------------------------------------------------------- */

void FixSurfaceNormal::refresh(int force_update)
{
  if (!force_update && (update->ntimestep % nevery)) return;
  pair->update_normals();
  last_step = update->ntimestep;
}

void FixSurfaceNormal::setup_pre_force(int /*vflag*/)
{
  refresh(1);
}

void FixSurfaceNormal::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel == 0) setup_pre_force(vflag);
}

void FixSurfaceNormal::pre_force(int /*vflag*/)
{
  refresh(0);
}

// once per step: at the first pre_force_respa call (innermost level, after reneighbouring)
void FixSurfaceNormal::pre_force_respa(int /*vflag*/, int /*ilevel*/, int /*iloop*/)
{
  if (update->ntimestep == last_step) return;
  refresh(0);
}

void FixSurfaceNormal::min_pre_force(int /*vflag*/)
{
  refresh(1);
}
