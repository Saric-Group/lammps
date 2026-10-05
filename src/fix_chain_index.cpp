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
   Per-atom chain index of treadmilling filaments (i_chain_index):
   0 at the tail (type 2), increasing along the backbone bonds to the
   head (type 3), -1 for atoms that are not part of a well-formed chain.

   fix bond/react (keyword chain_index) and fix nucleate (chain_index yes)
   keep the index up to date when atoms are added or removed; this fix
   recomputes it from the topology every Nevery steps. It gathers the
   backbone bonds of the group on all processors and walks every chain
   from its tail, so the result does not depend on the order in which the
   bonds are stored or on newton_bond.
------------------------------------------------------------------------- */

#include "fix_chain_index.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char PROPERTY_FIX_ID[] = "CHAIN_INDEX_PROPERTY";

/* ---------------------------------------------------------------------- */

namespace {
std::vector<tagint> allgather(const std::vector<tagint> &local, MPI_Comm world, int nprocs)
{
  int n = local.size();
  std::vector<int> counts(nprocs), displs(nprocs);
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  int total = 0;
  for (int p = 0; p < nprocs; p++) {
    displs[p] = total;
    total += counts[p];
  }
  std::vector<tagint> all(total + 1);    // +1 keeps data() valid for empty lists
  std::vector<tagint> send(local);
  send.push_back(0);
  MPI_Allgatherv(send.data(), n, MPI_LMP_TAGINT, all.data(), counts.data(), displs.data(),
                 MPI_LMP_TAGINT, world);
  all.resize(total);
  return all;
}

void parse_types(LAMMPS *lmp, int narg, char **arg, int &iarg, int ntypes, int *flags,
                 const char *keyword)
{
  for (int i = 0; i <= ntypes; i++) flags[i] = 0;
  iarg++;
  int nranges = 0;
  while (iarg < narg && (isdigit(arg[iarg][0]) || arg[iarg][0] == '*')) {
    int lo, hi;
    utils::bounds(FLERR, arg[iarg], 1, ntypes, lo, hi, lmp->error);
    for (int i = lo; i <= hi; i++) flags[i] = 1;
    nranges++;
    iarg++;
  }
  if (nranges == 0) lmp->error->all(FLERR, "Fix chain/index {} needs at least one bond type", keyword);
}
}    // namespace

/* ---------------------------------------------------------------------- */

FixChainIndex::FixChainIndex(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), bondtype_flag(nullptr), attach_flag(nullptr)
{
  if (narg < 4) utils::missing_cmd_args(FLERR, "fix chain/index", error);
  if (!atom->molecular) error->all(FLERR, "Fix chain/index requires a molecular system");

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Fix chain/index Nevery must be > 0");

  const int nbondtypes = atom->nbondtypes;
  memory->create(bondtype_flag, nbondtypes + 1, "chain/index:bondtype_flag");
  memory->create(attach_flag, nbondtypes + 1, "chain/index:attach_flag");
  for (int i = 0; i <= nbondtypes; i++) {
    bondtype_flag[i] = 1;
    attach_flag[i] = 0;
  }
  tail_type = 2;
  head_type = 3;
  warnflag = 1;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "bondtypes") == 0) {
      parse_types(lmp, narg, arg, iarg, nbondtypes, bondtype_flag, "bondtypes");
    } else if (strcmp(arg[iarg], "attach") == 0) {
      parse_types(lmp, narg, arg, iarg, nbondtypes, attach_flag, "attach");
    } else if (strcmp(arg[iarg], "tail") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix chain/index tail", error);
      tail_type = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "head") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix chain/index head", error);
      head_type = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "warn") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix chain/index warn", error);
      warnflag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix chain/index keyword: {}", arg[iarg]);
  }
  for (int i = 1; i <= nbondtypes; i++)
    if (bondtype_flag[i] && attach_flag[i])
      error->all(FLERR, "Fix chain/index bond type {} is both a backbone and an attach type", i);

  for (double &value : stats) value = 0.0;
  vector_flag = 1;
  size_vector = 6;
  extvector = 0;
  global_freq = 1;    // values of the last relabelling
  dynamic_group_allow = 1;
  comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixChainIndex::~FixChainIndex()
{
  // the property fix is shared with fix bond/react and fix nucleate and stays defined
  memory->destroy(bondtype_flag);
  memory->destroy(attach_flag);
}

/* ----------------------------------------------------------------------
   per-atom integer property i_chain_index, created on first use and shared
   by fix chain/index, fix bond/react and fix nucleate. New properties start
   at -1 (not part of a chain). A user-defined fix property/atom
   i_chain_index ghost yes (e.g. to read the values with read_data) is used
   as it is.
------------------------------------------------------------------------- */

int FixChainIndex::find_or_create_property(LAMMPS *lmp)
{
  int flag, cols;
  int index = lmp->atom->find_custom("chain_index", flag, cols);
  if (index >= 0) {
    if (flag != 0 || cols != 0)
      lmp->error->all(FLERR, "Per-atom property chain_index must be an integer vector (i_chain_index)");
    return index;
  }
  Fix *property =
      lmp->modify->add_fix(fmt::format("{} all property/atom i_chain_index ghost yes", PROPERTY_FIX_ID));
  index = lmp->atom->find_custom("chain_index", flag, cols);
  // keep values restored from a restart file
  if (!property->restart_reset) {
    int *chain_index = lmp->atom->ivector[index];
    for (int i = 0; i < lmp->atom->nlocal; i++) chain_index[i] = -1;
  }
  return index;
}

/* ---------------------------------------------------------------------- */

void FixChainIndex::post_constructor()
{
  find_or_create_property(lmp);
}

/* ---------------------------------------------------------------------- */

int FixChainIndex::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixChainIndex::setup_pre_force(int /*vflag*/)
{
  relabel();
}

// a dynamic group gets its members in FixGroup::setup(), after setup_pre_force()
void FixChainIndex::setup(int /*vflag*/)
{
  if (group->dynamic[igroup]) relabel();
}

void FixChainIndex::pre_force(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;
  relabel();
}

void FixChainIndex::min_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixChainIndex::relabel()
{
  int flag, cols;
  const int index = atom->find_custom("chain_index", flag, cols);
  if (index < 0) error->all(FLERR, "Fix chain/index: per-atom property i_chain_index is missing");
  int *chain_index = atom->ivector[index];

  const int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  const int newton_bond = force->newton_bond;

  // local group atoms (tag, type, old index) and bonds (tag, tag, 0 backbone / 1 attach)
  std::vector<tagint> my_atoms, my_bonds;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    my_atoms.push_back(tag[i]);
    my_atoms.push_back(type[i]);
    my_atoms.push_back(chain_index[i]);
    for (int m = 0; m < num_bond[i]; m++) {
      const int btype = bond_type[i][m];
      if (btype <= 0) continue;
      const tagint partner = bond_atom[i][m];
      // with newton_bond off every bond is stored on both of its atoms
      if (!newton_bond && partner < tag[i]) continue;
      if (bondtype_flag[btype] || attach_flag[btype]) {
        my_bonds.push_back(tag[i]);
        my_bonds.push_back(partner);
        my_bonds.push_back(bondtype_flag[btype] ? 0 : 1);
      }
    }
  }
  const std::vector<tagint> all_atoms = allgather(my_atoms, world, comm->nprocs);
  const std::vector<tagint> all_bonds = allgather(my_bonds, world, comm->nprocs);

  // every processor walks all chains: the result is identical everywhere
  std::unordered_map<tagint, int> atom_type, old_index;
  for (size_t k = 0; k < all_atoms.size(); k += 3) {
    atom_type[all_atoms[k]] = (int) all_atoms[k + 1];
    old_index[all_atoms[k]] = (int) all_atoms[k + 2];
  }
  std::unordered_map<tagint, std::vector<tagint>> backbone;
  std::vector<std::pair<tagint, tagint>> backbone_bonds, attach_bonds;
  for (size_t k = 0; k < all_bonds.size(); k += 3) {
    const tagint a = all_bonds[k], b = all_bonds[k + 1];
    if (!atom_type.count(a) || !atom_type.count(b)) continue;    // partner not in group
    if (all_bonds[k + 2] == 0) {
      backbone[a].push_back(b);
      backbone[b].push_back(a);
      backbone_bonds.emplace_back(a, b);
    } else
      attach_bonds.emplace_back(a, b);
  }

  // sorted tails make the walk order (and the warnings) reproducible
  std::vector<tagint> tails;
  for (const auto &entry : backbone)
    if (atom_type[entry.first] == tail_type && entry.second.size() == 1) tails.push_back(entry.first);
  std::sort(tails.begin(), tails.end());

  std::unordered_map<tagint, int> label;
  std::unordered_set<tagint> visited;
  int nchains = 0, longest = 0, no_head = 0, branched = 0, unreached = 0, reordered = 0;
  std::vector<tagint> path;
  for (const tagint tail : tails) {
    if (visited.count(tail)) continue;
    path.clear();
    bool ok = true, branch = false;
    tagint previous = 0, current = tail;
    while (true) {
      if (visited.count(current)) {
        ok = false;    // ran into another chain
        break;
      }
      visited.insert(current);
      path.push_back(current);
      const auto &neighbours = backbone[current];
      if (neighbours.size() > 2) {
        ok = false;
        branch = true;
        break;
      }
      tagint next = 0;
      for (const tagint n : neighbours)
        if (n != previous) next = n;
      if (next == 0) break;
      previous = current;
      current = next;
    }
    if (ok && atom_type[path.back()] != head_type) {
      ok = false;
      no_head++;
    }
    if (branch) branched++;
    if (!ok) continue;
    for (int k = 0; k < (int) path.size(); k++) label[path[k]] = k;
    nchains++;
    longest = MAX(longest, (int) path.size());
  }
  for (const auto &entry : backbone)
    if (!visited.count(entry.first)) unreached++;

  // side beads take the index of the backbone bead they are attached to
  for (const auto &bond : attach_bonds) {
    const auto a = label.find(bond.first), b = label.find(bond.second);
    if (a != label.end() && b == label.end() && !backbone.count(bond.second))
      label[bond.second] = a->second;
    else if (b != label.end() && a == label.end() && !backbone.count(bond.first))
      label[bond.first] = b->second;
  }

  // bonds whose direction by index changed: 0 if the indices were maintained correctly
  for (const auto &bond : backbone_bonds) {
    const int old_a = old_index[bond.first], old_b = old_index[bond.second];
    const auto a = label.find(bond.first), b = label.find(bond.second);
    if (old_a < 0 || old_b < 0 || a == label.end() || b == label.end()) continue;
    if ((old_b > old_a) != (b->second > a->second)) reordered++;
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const auto it = label.find(tag[i]);
    chain_index[i] = (it == label.end()) ? -1 : it->second;
  }
  comm->forward_comm(this, 1);

  const bool anomalies_changed =
      (no_head != stats[2]) || (branched != stats[3]) || (unreached != stats[4]);
  stats[0] = nchains;
  stats[1] = longest;
  stats[2] = no_head;
  stats[3] = branched;
  stats[4] = unreached;
  stats[5] = reordered;

  if (warnflag && comm->me == 0 && anomalies_changed && (no_head || branched || unreached))
    error->warning(FLERR,
                   "Fix chain/index at step {}: {} chains without a head, {} branched chains, {} "
                   "backbone atoms not reachable from a tail (indices set to -1)",
                   update->ntimestep, no_head, branched, unreached);
}

/* ----------------------------------------------------------------------
   1 chains, 2 longest chain, 3 chains without head, 4 branched chains,
   5 backbone atoms not reachable from a tail, 6 backbone bonds whose
   direction by index changed in the last relabelling
------------------------------------------------------------------------- */

double FixChainIndex::compute_vector(int n)
{
  return stats[n];
}

/* ---------------------------------------------------------------------- */

int FixChainIndex::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int flag, cols;
  int *chain_index = atom->ivector[atom->find_custom("chain_index", flag, cols)];
  for (int i = 0; i < n; i++) buf[i] = ubuf(chain_index[list[i]]).d;
  return n;
}

void FixChainIndex::unpack_forward_comm(int n, int first, double *buf)
{
  int flag, cols;
  int *chain_index = atom->ivector[atom->find_custom("chain_index", flag, cols)];
  for (int i = 0; i < n; i++) chain_index[first + i] = (int) ubuf(buf[i]).i;
}
