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
#include <vector>

using namespace LAMMPS_NS;

FixBackboneInfo::FixBackboneInfo(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg != 4)
    error->all(FLERR, "Illegal fix backbone/info command. Syntax: fix ID group-ID backbone/info max_dist");

  max_dist = utils::inumeric(FLERR, arg[3], false, lmp);
  if (max_dist < 1) error->all(FLERR, "Max bond distance for fix backbone/info must be >= 1");

  nevery = 1;
  cache_valid = false;
}

FixBackboneInfo::~FixBackboneInfo() {}

int FixBackboneInfo::setmask()
{
  return FixConst::PRE_FORCE;
}

void FixBackboneInfo::init()
{
  if (atom->map_style == 0)
    error->all(FLERR, "FixBackboneInfo requires 'atom_modify map yes'");
  
  // Mark invalid to force build on first step
  cache_valid = false;
}

void FixBackboneInfo::pre_force(int /*vflag*/)
{
  // If neighbor list was rebuilt, our index-based cache is potentially wrong.
  // We mark it invalid. The Pair style will trigger the rebuild.
  // This is because it seems that combining with bond/react, the list can get stale if recomputed here before it is used in the Pair style
  if (neighbor->ago == 0) {
    cache_valid = false;
  }
}

void FixBackboneInfo::ensure_cache()
{
  if (cache_valid) return;
  build_cache();
  cache_valid = true;
}

void FixBackboneInfo::build_cache()
{
  int nlocal = atom->nlocal;
  int nmax = atom->nmax;

  // Resize
  if ((int)backbone_cache.size() != nmax) {
      backbone_cache.resize(nmax);
      adj.resize(nmax);
      visited_flag.resize(nmax, -1);
  }
  
  // Clear
  for (int i = 0; i < nmax; i++) {
      adj[i].clear();
      backbone_cache[i].clear(); 
      visited_flag[i] = -1; 
  }

  // Build Adjacency
  // Logic: Symmetrize only for local atoms.
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;

  for (int i = 0; i < nlocal; i++) {
    for (int k = 0; k < num_bond[i]; k++) {
      tagint partner_tag = bond_atom[i][k];
      int partner_idx = atom->map(partner_tag);

      if (partner_idx >= 0 && partner_idx < nmax) {
        adj[i].push_back(partner_idx);
        
        // Only back-link if partner is local
        if (partner_idx < nlocal) {
             adj[partner_idx].push_back(i);
        }
      }
    }
  }

  // BFS
  std::queue<std::pair<int, int>> q;
  tagint *tag = atom->tag;

  for (int i = 0; i < nlocal; i++) {
    int current_visit_id = i; 

    std::queue<std::pair<int, int>> empty_q;
    std::swap(q, empty_q);

    q.push({i, 0});
    visited_flag[i] = current_visit_id;

    while (!q.empty()) {
      auto current = q.front();
      q.pop();
      int u = current.first;
      int dist = current.second;

      if (dist >= max_dist) continue;

      for (int v : adj[u]) {
        if (visited_flag[v] != current_visit_id) {
          visited_flag[v] = current_visit_id;
          
          backbone_cache[i].push_back(tag[v]);

          // Do not traverse through ghosts
          if (v < nlocal) {
             q.push({v, dist + 1});
          }
        }
      }
    }
  }
}