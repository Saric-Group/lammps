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
   Direction of treadmilling filaments from their bond topology, and
   per-atom counts of parallel and anti-parallel neighbours.

   The direction of a bond is the order in which it is stored: with
   newton_bond on, every bond lives on its first atom and the bond list
   holds it as (atom1, atom2), so the data and molecule files define the
   direction. bacterial_septation writes all filament bonds from the
   shrinking end (type 2) to the growing end (type 3).
------------------------------------------------------------------------- */

#include "fix_filament_direction.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cctype>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double SMALL = 1.0e-12;
static constexpr double NO_NEIGHBOUR = 2.0;    // cosmin of atoms without a counted neighbour

/* ---------------------------------------------------------------------- */

FixFilamentDirection::FixFilamentDirection(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), bondtype_flag(nullptr), type_flag(nullptr), id_props(nullptr), dir(nullptr), nbond(nullptr),
    npar(nullptr), nanti(nullptr), cosmin(nullptr), list(nullptr)
{
  if (narg < 4) utils::missing_cmd_args(FLERR, "fix filament/direction", error);
  if (!atom->molecular) error->all(FLERR, "Fix filament/direction requires a molecular system");

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Fix filament/direction Nevery must be > 0");

  const int nbondtypes = atom->nbondtypes;
  memory->create(bondtype_flag, nbondtypes + 1, "filament/direction:bondtype_flag");
  for (int i = 0; i <= nbondtypes; i++) bondtype_flag[i] = 1;
  memory->create(type_flag, atom->ntypes + 1, "filament/direction:type_flag");
  for (int i = 0; i <= atom->ntypes; i++) type_flag[i] = 1;

  orient = BOND_ORDER;
  cutoff = 1.5;
  parallel_cos = 0.8;
  antiparallel_cos = -0.8;
  lateral_cos = 1.0;
  exclude_special = 1;
  mu_flag = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "bondtypes") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction bondtypes", error);
      for (int i = 0; i <= nbondtypes; i++) bondtype_flag[i] = 0;
      // any number of types or type ranges, up to the next keyword
      iarg++;
      int nranges = 0;
      while (iarg < narg && (isdigit(arg[iarg][0]) || arg[iarg][0] == '*')) {
        int lo, hi;
        utils::bounds(FLERR, arg[iarg], 1, nbondtypes, lo, hi, error);
        for (int i = lo; i <= hi; i++) bondtype_flag[i] = 1;
        nranges++;
        iarg++;
      }
      if (nranges == 0) error->all(FLERR, "Fix filament/direction bondtypes needs at least one type");
    } else if (strcmp(arg[iarg], "types") == 0) {
      // atom types of filament beads; unlike a dynamic group always up to date for atoms
      // that fix bond/react created in the same step
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction types", error);
      for (int i = 0; i <= atom->ntypes; i++) type_flag[i] = 0;
      iarg++;
      int nranges = 0;
      while (iarg < narg && (isdigit(arg[iarg][0]) || arg[iarg][0] == '*')) {
        int lo, hi;
        utils::bounds(FLERR, arg[iarg], 1, atom->ntypes, lo, hi, error);
        for (int i = lo; i <= hi; i++) type_flag[i] = 1;
        nranges++;
        iarg++;
      }
      if (nranges == 0) error->all(FLERR, "Fix filament/direction types needs at least one type");
    } else if (strcmp(arg[iarg], "orient") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction orient", error);
      if (strcmp(arg[iarg + 1], "bond_order") == 0)
        orient = BOND_ORDER;
      else
        error->all(FLERR, "Unknown fix filament/direction orient value: {}", arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "cutoff") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction cutoff", error);
      cutoff = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (cutoff < 0.0) error->all(FLERR, "Fix filament/direction cutoff must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "parallel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction parallel", error);
      parallel_cos = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "antiparallel") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "fix filament/direction antiparallel", error);
      antiparallel_cos = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "lateral") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction lateral", error);
      lateral_cos = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (lateral_cos < 0.0 || lateral_cos > 1.0)
        error->all(FLERR, "Fix filament/direction lateral must be between 0 and 1");
      iarg += 2;
    } else if (strcmp(arg[iarg], "exclude") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction exclude", error);
      if (strcmp(arg[iarg + 1], "special") == 0)
        exclude_special = 1;
      else if (strcmp(arg[iarg + 1], "none") == 0)
        exclude_special = 0;
      else
        error->all(FLERR, "Unknown fix filament/direction exclude value: {}", arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "mu") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix filament/direction mu", error);
      mu_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix filament/direction keyword: {}", arg[iarg]);
  }

  if (parallel_cos < antiparallel_cos)
    error->all(FLERR, "Fix filament/direction needs parallel >= antiparallel");

  name_dir = std::string(id) + "_dir";
  name_nbond = std::string(id) + "_nbond";
  name_npar = std::string(id) + "_npar";
  name_nanti = std::string(id) + "_nanti";
  name_cosmin = std::string(id) + "_cosmin";

  vector_flag = 1;
  size_vector = 3;
  extvector = 0;
  dynamic_group_allow = 1;
  comm_forward = 4;
  comm_reverse = 4;
}

/* ---------------------------------------------------------------------- */

FixFilamentDirection::~FixFilamentDirection()
{
  if (id_props && modify->nfix) modify->delete_fix(id_props);
  delete[] id_props;
  memory->destroy(bondtype_flag);
  memory->destroy(type_flag);
}

/* ----------------------------------------------------------------------
   per-atom output lives in an internal fix property/atom, so it migrates
   with the atoms, goes into restart files and can be used in dumps and
   atom-style variables as d2_ID_dir[1-3], i_ID_nbond, i_ID_npar,
   i_ID_nanti and d_ID_cosmin
------------------------------------------------------------------------- */

void FixFilamentDirection::post_constructor()
{
  id_props = utils::strdup(std::string(id) + "_props_internal");
  modify->add_fix(fmt::format("{} all property/atom d2_{} 3 i_{} i_{} i_{} d_{} ghost yes", id_props,
                              name_dir, name_nbond, name_npar, name_nanti, name_cosmin));
  find_properties();
  for (int i = 0; i < atom->nlocal; i++) cosmin[i] = NO_NEIGHBOUR;
}

/* ---------------------------------------------------------------------- */

void FixFilamentDirection::find_properties()
{
  int flag, cols;
  int index = atom->find_custom(name_dir.c_str(), flag, cols);
  if (index < 0 || flag != 1 || cols != 3)
    error->all(FLERR, "Fix filament/direction: per-atom property d2_{} is missing", name_dir);
  dir = atom->darray[index];
  nbond = atom->ivector[atom->find_custom(name_nbond.c_str(), flag, cols)];
  npar = atom->ivector[atom->find_custom(name_npar.c_str(), flag, cols)];
  nanti = atom->ivector[atom->find_custom(name_nanti.c_str(), flag, cols)];
  cosmin = atom->dvector[atom->find_custom(name_cosmin.c_str(), flag, cols)];
}

/* ---------------------------------------------------------------------- */

int FixFilamentDirection::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFilamentDirection::init()
{
  if (orient == BOND_ORDER && !force->newton_bond)
    error->all(FLERR,
               "Fix filament/direction with orient bond_order requires newton_bond on: "
               "only then is a bond stored on its first atom");
  if (mu_flag && !atom->mu_flag)
    error->all(FLERR, "Fix filament/direction mu yes requires an atom style with dipoles");

  if (cutoff > 0.0) {
    double cutneigh = cutoff + neighbor->skin;
    if (cutneigh > comm->get_comm_cutoff())
      error->all(FLERR,
                 "Fix filament/direction cutoff + skin ({}) is larger than the communication "
                 "cutoff ({}), increase it with comm_modify cutoff",
                 cutneigh, comm->get_comm_cutoff());
    // the list is used until the next reneighboring, so it includes the skin like
    // a pair style list does
    auto req = neighbor->add_request(this);
    req->set_cutoff(cutneigh);
  }
}

/* ---------------------------------------------------------------------- */

void FixFilamentDirection::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixFilamentDirection::setup_pre_force(int /*vflag*/)
{
  compute();
}

// a dynamic group gets its members in FixGroup::setup(), after setup_pre_force(),
// so recompute once it is set (fixes are set up in the order they were defined)
void FixFilamentDirection::setup(int /*vflag*/)
{
  if (group->dynamic[igroup]) compute();
}

void FixFilamentDirection::pre_force(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;
  compute();
}

void FixFilamentDirection::min_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixFilamentDirection::compute()
{
  find_properties();
  compute_direction();
  if (cutoff > 0.0) compute_alignment();
}

/* ----------------------------------------------------------------------
   +1 if the bond (i0, i1) of the bond list points from i0 to i1, -1 if
   it points the other way, 0 if its direction is unknown
------------------------------------------------------------------------- */

int FixFilamentDirection::bond_sign(int /*i0*/, int /*i1*/)
{
  // orient bond_order: the bond list holds bonds as (atom1, atom2)
  return 1;
}

/* ----------------------------------------------------------------------
   direction of an atom = normalised sum of the unit vectors of its bonds,
   i.e. the bisector x(i+1) - x(i-1) inside a chain, the bond at its ends
------------------------------------------------------------------------- */

void FixFilamentDirection::compute_direction()
{
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int newton_bond = force->newton_bond;
  double **x = atom->x;
  int **bondlist = neighbor->bondlist;
  const int nbondlist = neighbor->nbondlist;

  for (int i = 0; i < nall; i++) {
    dir[i][0] = dir[i][1] = dir[i][2] = 0.0;
    nbond[i] = 0;
  }

  for (int n = 0; n < nbondlist; n++) {
    const int i0 = bondlist[n][0];
    const int i1 = bondlist[n][1];
    const int type = bondlist[n][2];
    if (type <= 0 || !bondtype_flag[type]) continue;
    if (!is_bead(i0) || !is_bead(i1)) continue;

    const int sign = bond_sign(i0, i1);
    if (sign == 0) continue;

    // bondlist partners are already the closest image
    double d[3] = {x[i1][0] - x[i0][0], x[i1][1] - x[i0][1], x[i1][2] - x[i0][2]};
    const double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (r < SMALL) continue;
    const double scale = sign / r;
    d[0] *= scale;
    d[1] *= scale;
    d[2] *= scale;

    // with newton_bond off a bond across processors is in the list of both of them,
    // so each adds only to its own atom
    if (newton_bond || i0 < nlocal) {
      dir[i0][0] += d[0];
      dir[i0][1] += d[1];
      dir[i0][2] += d[2];
      nbond[i0]++;
    }
    if (newton_bond || i1 < nlocal) {
      dir[i1][0] += d[0];
      dir[i1][1] += d[1];
      dir[i1][2] += d[2];
      nbond[i1]++;
    }
  }

  if (newton_bond) {
    commflag = COMM_DIRECTION;
    comm->reverse_comm(this, 4);
  }

  double **mu = atom->mu;
  for (int i = 0; i < nlocal; i++) {
    const double len = sqrt(dir[i][0] * dir[i][0] + dir[i][1] * dir[i][1] + dir[i][2] * dir[i][2]);
    if (len > SMALL) {
      dir[i][0] /= len;
      dir[i][1] /= len;
      dir[i][2] /= len;
    } else {
      // opposite bonds cancel (inconsistent bond directions) or no bond at all
      dir[i][0] = dir[i][1] = dir[i][2] = 0.0;
    }
    if (mu_flag && is_bead(i)) {
      mu[i][0] = dir[i][0];
      mu[i][1] = dir[i][1];
      mu[i][2] = dir[i][2];
      mu[i][3] = (len > SMALL) ? 1.0 : 0.0;
    }
  }

  commflag = COMM_DIRECTION;
  comm->forward_comm(this, 4);
}

/* ---------------------------------------------------------------------- */

bool FixFilamentDirection::is_special(int i, int j)
{
  const tagint jtag = atom->tag[j];
  const int nspecial = atom->nspecial[i][2];
  tagint *special = atom->special[i];
  for (int k = 0; k < nspecial; k++)
    if (special[k] == jtag) return true;
  return false;
}

/* ----------------------------------------------------------------------
   count parallel and anti-parallel neighbours within the cutoff
------------------------------------------------------------------------- */

void FixFilamentDirection::compute_alignment()
{
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int newton_pair = force->newton_pair;
  const double cutsq = cutoff * cutoff;
  double **x = atom->x;

  for (int i = 0; i < nall; i++) {
    npar[i] = nanti[i] = 0;
    cosmin[i] = NO_NEIGHBOUR;
  }

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (!is_bead(i) || nbond[i] == 0) continue;
    if (dir[i][0] == 0.0 && dir[i][1] == 0.0 && dir[i][2] == 0.0) continue;

    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!is_bead(j) || nbond[j] == 0) continue;
      if (dir[j][0] == 0.0 && dir[j][1] == 0.0 && dir[j][2] == 0.0) continue;

      const double delx = x[j][0] - x[i][0];
      const double dely = x[j][1] - x[i][1];
      const double delz = x[j][2] - x[i][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq || rsq < SMALL) continue;
      if (exclude_special && is_special(i, j)) continue;

      if (lateral_cos < 1.0) {
        // skip contacts along the filament axis, e.g. head to head
        const double r = sqrt(rsq);
        const double ri = fabs(delx * dir[i][0] + dely * dir[i][1] + delz * dir[i][2]) / r;
        const double rj = fabs(delx * dir[j][0] + dely * dir[j][1] + delz * dir[j][2]) / r;
        if (ri > lateral_cos || rj > lateral_cos) continue;
      }

      const double c = dir[i][0] * dir[j][0] + dir[i][1] * dir[j][1] + dir[i][2] * dir[j][2];
      const int is_par = (c >= parallel_cos);
      const int is_anti = (c <= antiparallel_cos);

      npar[i] += is_par;
      nanti[i] += is_anti;
      if (c < cosmin[i]) cosmin[i] = c;
      if (newton_pair || j < nlocal) {
        npar[j] += is_par;
        nanti[j] += is_anti;
        if (c < cosmin[j]) cosmin[j] = c;
      }
    }
  }

  if (newton_pair) {
    commflag = COMM_ALIGNMENT;
    comm->reverse_comm(this, 3);
  }
}

/* ----------------------------------------------------------------------
   global vector: number of atoms with a direction, with a parallel and
   with an anti-parallel neighbour
------------------------------------------------------------------------- */

double FixFilamentDirection::compute_vector(int n)
{
  find_properties();
  const int nlocal = atom->nlocal;
  bigint count = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!is_bead(i)) continue;
    if (n == 0)
      count += (nbond[i] > 0);
    else if (n == 1)
      count += (npar[i] > 0);
    else
      count += (nanti[i] > 0);
  }
  bigint all;
  MPI_Allreduce(&count, &all, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  return (double) all;
}

/* ---------------------------------------------------------------------- */

int FixFilamentDirection::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                            int * /*pbc*/)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    buf[m++] = dir[j][0];
    buf[m++] = dir[j][1];
    buf[m++] = dir[j][2];
    buf[m++] = ubuf(nbond[j]).d;
  }
  return m;
}

void FixFilamentDirection::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    dir[i][0] = buf[m++];
    dir[i][1] = buf[m++];
    dir[i][2] = buf[m++];
    nbond[i] = (int) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

int FixFilamentDirection::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    if (commflag == COMM_DIRECTION) {
      buf[m++] = dir[i][0];
      buf[m++] = dir[i][1];
      buf[m++] = dir[i][2];
      buf[m++] = ubuf(nbond[i]).d;
    } else {
      buf[m++] = ubuf(npar[i]).d;
      buf[m++] = ubuf(nanti[i]).d;
      buf[m++] = cosmin[i];
    }
  }
  return m;
}

void FixFilamentDirection::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    if (commflag == COMM_DIRECTION) {
      dir[j][0] += buf[m++];
      dir[j][1] += buf[m++];
      dir[j][2] += buf[m++];
      nbond[j] += (int) ubuf(buf[m++]).i;
    } else {
      npar[j] += (int) ubuf(buf[m++]).i;
      nanti[j] += (int) ubuf(buf[m++]).i;
      const double c = buf[m++];
      if (c < cosmin[j]) cosmin[j] = c;
    }
  }
}

/* ---------------------------------------------------------------------- */

void *FixFilamentDirection::extract(const char *str, int &dim)
{
  if (strcmp(str, "nevery") == 0) {
    dim = 0;
    return (void *) &nevery;
  }
  find_properties();
  if (strcmp(str, "dir") == 0) {
    dim = 2;
    return (void *) dir;
  }
  if (strcmp(str, "nbond") == 0) {
    dim = 1;
    return (void *) nbond;
  }
  return nullptr;
}
