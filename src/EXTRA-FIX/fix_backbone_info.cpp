/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#include "fix_backbone_info.h"
#include "atom.h"
#include "error.h"
#include "utils.h"
#include "neighbor.h"
#include <queue>
#include <algorithm>


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


void FixBackboneInfo::run_bfs_from_atom(int start_idx)
{
  if (start_idx < 0 || start_idx >= atom->nlocal) return;

  tagint start_tag = atom->tag[start_idx];
  backbone_neighbors[start_tag].clear();

  std::queue<std::pair<int, int>> q;
  std::map<int, int> visited;

  q.push({start_idx, 0});
  visited[start_idx] = 0;
  backbone_neighbors[start_tag][start_tag] = 0;

  while (!q.empty()) {
    auto current = q.front();
    q.pop();
    int current_idx = current.first;
    int current_dist = current.second;

    if (current_dist == max_dist) continue;

    for (int neighbor_idx : adj[current_idx]) {
      if (visited.find(neighbor_idx) == visited.end()) {
        tagint neighbor_tag = atom->tag[neighbor_idx];
        visited[neighbor_idx] = current_dist + 1;
        backbone_neighbors[start_tag][neighbor_tag] = current_dist + 1;
        if (neighbor_idx < atom->nlocal) {
          q.push({neighbor_idx, current_dist + 1});
        }
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
        if (neighbor_idx < atom->nlocal) {
          adj[neighbor_idx].push_back(i);
        }
      }
    }
  }

  // Run the BFS using guaranteed-symmetric adjacency list.
  backbone_neighbors.clear();
  for (int i = 0; i < atom->nlocal; ++i) {
    run_bfs_from_atom(i);
  }
}


void FixBackboneInfo::pre_force(int /*vflag*/)
{
  // We need to recompute the backbone maps if neighbor list was rebuilt
  if (neighbor->ago == 0) {
    compute_all_backbone_maps();
  }
}
