// clang-format off
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
   fix ID group director/bind d2_name eps [nth N]

   Binding of the director e (atom->mu, the binding face of a filament bead;
   see bond_style director and angle_style director/bend) to the surface
   normal n of the same atom (3-column per-atom array d2_name, e.g.
   d2_cylinder_normal or d2_avg_nvecs):

     E = eps (1 - e . n^)  per atom,  torque = eps e x n^

   on every atom of the group with a director (mu != 0) and a normal
   (|n| >= 1e-6): one binding site per bead, also at chain ends and on
   nucleated dimers. With nth N only atoms whose chain index (i_chain_index)
   is a multiple of N bind, e.g. one site per monomer with N beads per
   monomer; atoms with chain index -1 do not bind. No force (the normal is
   taken as given). The global scalar is the binding energy; fix_modify
   energy yes adds it to the potential energy. Under run_style respa the
   torque is applied on the outermost level (fix_modify respa to change).
------------------------------------------------------------------------- */

#include "fix_director_bind.h"

#include "atom.h"
#include "error.h"
#include "math_extra.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double NORMAL_SMALL = 1.0e-6;

/* ---------------------------------------------------------------------- */

FixDirectorBind::FixDirectorBind(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), eps(0.0), nth(1), ilevel_respa(0), energy_local(0.0), energy_all(0.0),
    energy_flag(0)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix director/bind", error);
  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;
  respa_level_support = 1;

  std::string name = arg[3];
  if (!utils::strmatch(name, "^d2_"))
    error->all(FLERR, "Fix director/bind needs a per-atom array d2_<name>, got {}", name);
  normal_name = name.substr(3);
  eps = utils::numeric(FLERR, arg[4], false, lmp);

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nth") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix director/bind nth", error);
      nth = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (nth < 1) error->all(FLERR, "Fix director/bind: nth must be >= 1");
      iarg += 2;
    } else error->all(FLERR, "Unknown fix director/bind keyword {}", arg[iarg]);
  }
}

/* ---------------------------------------------------------------------- */

int FixDirectorBind::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDirectorBind::init()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Fix director/bind needs atom attributes mu and torque (atom style with dipole and sphere)");
  int flag, cols;
  int index = atom->find_custom(normal_name.c_str(), flag, cols);
  if (index < 0 || flag != 1 || cols != 3)
    error->all(FLERR, "Fix director/bind needs a 3-column per-atom array d2_{}", normal_name);
  if (nth > 1) {
    index = atom->find_custom("chain_index", flag, cols);
    if (index < 0 || flag != 0 || cols != 0)
      error->all(FLERR, "Fix director/bind nth {} needs the per-atom chain index i_chain_index", nth);
  }
  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixDirectorBind::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^respa")) {
    auto *respa = dynamic_cast<Respa *>(update->integrate);
    respa->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    respa->copy_f_flevel(ilevel_respa);
  } else {
    post_force(vflag);
  }
}

/* ---------------------------------------------------------------------- */

void FixDirectorBind::post_force(int /*vflag*/)
{
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  int flag, cols;
  double **normal = atom->darray[atom->find_custom(normal_name.c_str(), flag, cols)];
  int *chain_index = nullptr;
  if (nth > 1) chain_index = atom->ivector[atom->find_custom("chain_index", flag, cols)];

  energy_local = 0.0;
  energy_flag = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (chain_index && (chain_index[i] < 0 || chain_index[i] % nth != 0)) continue;
    const double mlen = MathExtra::len3(mu[i]);
    const double nlen = MathExtra::len3(normal[i]);
    if (mlen <= 0.0 || nlen < NORMAL_SMALL) continue;
    double e[3], n[3], tq[3];
    MathExtra::scale3(1.0 / mlen, mu[i], e);
    MathExtra::scale3(1.0 / nlen, normal[i], n);
    energy_local += eps * (1.0 - MathExtra::dot3(e, n));
    MathExtra::cross3(e, n, tq);
    torque[i][0] += eps * tq[0];
    torque[i][1] += eps * tq[1];
    torque[i][2] += eps * tq[2];
  }
}

/* ---------------------------------------------------------------------- */

void FixDirectorBind::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixDirectorBind::compute_scalar()
{
  if (energy_flag == 0) {
    MPI_Allreduce(&energy_local, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    energy_flag = 1;
  }
  return energy_all;
}
