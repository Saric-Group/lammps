/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#include "fix_backbone_info.h"
#include "atom.h"
#include "error.h"
#include "neighbor.h"
#include "utils.h"
#include <algorithm>
#include <queue>
#include <unordered_set>

using namespace LAMMPS_NS;

FixBackboneInfo::FixBackboneInfo(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg != 4)
    error->all(FLERR,
               "Illegal fix backbone/info command. Syntax: fix ID group-ID backbone/info max_dist");

  max_dist = utils::inumeric(FLERR, arg[3], false, lmp);
  if (max_dist < 1) error->all(FLERR, "Max bond distance for fix backbone/info must be >= 1");

  nevery = 1;
}

FixBackboneInfo::~FixBackboneInfo() {}

int FixBackboneInfo::setmask()
{
  // Tell LAMMPS to call pre_force() every timestep.
  int mask = 0;
  mask |= FixConst::PRE_FORCE;
  return mask;
}

void FixBackboneInfo::init()
{
  if (atom->map_style == 0)
    error->all(FLERR, "FixBackboneInfo requires an atom map; use 'atom_modify map yes'");
  compute_all_backbone_maps();
}

// --- MODIFIED BFS ---
// This function now populates a vector of tags instead of a map.
void FixBackboneInfo::run_bfs_from_atom(int start_idx)
{
  if (start_idx < 0 || start_idx >= atom->nlocal) return;

  tagint start_tag = atom->tag[start_idx];
  backbone_neighbors[start_tag].clear();    // Clear the vector for this atom

  std::queue<std::pair<int, int>> q;    // Pair of {atom_index, distance}
  std::unordered_set<int> visited;      // Use a hash set for efficient O(1) visited checks

  // Initialize the search
  q.push({start_idx, 0});
  visited.insert(start_idx);
  // Add the atom to its own neighbor list. This is harmless and simplifies
  // the pair style logic, which expects to find bonded partners in this list.
  backbone_neighbors[start_tag].push_back(start_tag);

  while (!q.empty()) {
    auto current = q.front();
    q.pop();
    int current_idx = current.first;
    int current_dist = current.second;

    if (current_dist >= max_dist) continue;

    for (int neighbor_idx : adj[current_idx]) {
      // Check if the neighbor has been visited
      if (visited.find(neighbor_idx) == visited.end()) {
        visited.insert(neighbor_idx);    // Mark as visited

        tagint neighbor_tag = atom->tag[neighbor_idx];
        backbone_neighbors[start_tag].push_back(neighbor_tag);    // Add neighbor tag to vector

        // Only continue the search from local atoms
        if (neighbor_idx < atom->nlocal) { q.push({neighbor_idx, current_dist + 1}); }
      }
    }
  }
}

void FixBackboneInfo::compute_all_backbone_maps()
{
  adj.assign(atom->nmax, std::vector<int>());

  // Loop over all local atoms and their bonds
  for (int i = 0; i < atom->nlocal; i++) {
    for (int j = 0; j < atom->num_bond[i]; j++) {
      int neighbor_idx = atom->map(atom->bond_atom[i][j]);

      // Only consider bonds where both atoms are "known" (local or ghost)
      if (neighbor_idx >= 0) {
        // Add the symmetric bond to our internal list
        adj[i].push_back(neighbor_idx);
        // If the neighbor is also local, add the reverse bond too
        if (neighbor_idx < atom->nlocal) { adj[neighbor_idx].push_back(i); }
      }
    }
  }

  // Run the BFS using guaranteed-symmetric adjacency list.
  backbone_neighbors.clear();
  for (int i = 0; i < atom->nlocal; ++i) { run_bfs_from_atom(i); }
}

void FixBackboneInfo::pre_force(int /*vflag*/)
{
  // We need to recompute the backbone maps if neighbor list was rebuilt
  if (neighbor->ago == 0) { compute_all_backbone_maps(); }
}
