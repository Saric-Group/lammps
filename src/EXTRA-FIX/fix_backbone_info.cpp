#include "fix_backbone_info.h"
#include "atom.h"
#include "error.h"
#include "modify.h"
#include "utils.h"
#include <queue>
#include <algorithm>


using namespace LAMMPS_NS;

FixBackboneInfo::FixBackboneInfo(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  // From debugging, we know narg is 4 for the command:
  // "fix BI all backbone/info 2"
  if (narg != 4)
    error->all(FLERR,
               "Illegal fix backbone/info command. Syntax: fix ID group-ID backbone/info max_dist");

  // From debugging, we know the max_dist argument is at index 3.
  max_dist = utils::inumeric(FLERR, arg[3], false, lmp);
  if (max_dist < 1) error->all(FLERR, "Max bond distance for fix backbone/info must be >= 1");

  // This fix doesn't need to run on its own. It's only driven by callbacks.
  nevery = 2000000000;
}

FixBackboneInfo::~FixBackboneInfo() {}

int FixBackboneInfo::setmask()
{
  // This fix doesn't modify atoms directly, so it returns 0
  return 0;
}

// void FixBackboneInfo::post_constructor()
// {
//   if (atom->map_style == 0)
//     error->all(FLERR, "FixBackboneInfo requires an atom map; use 'atom_modify map yes'");
//   // At the very start of a run, perform a full calculation of all maps
//   compute_all_backbone_maps();
// }


void FixBackboneInfo::init()
{
  if (atom->map_style == 0)
    error->all(FLERR, "FixBackboneInfo requires an atom map; use 'atom_modify map yes'");
  // At the very start of a run, perform a full calculation of all maps
  compute_all_backbone_maps();
}


// void FixBackboneInfo::run_bfs_from_atom(int start_idx)
// {
//   if (start_idx < 0 || start_idx >= atom->nlocal) return;

//   tagint start_tag = atom->tag[start_idx];
//   backbone_neighbors[start_tag].clear();    // Use the TAG as the key

//   std::queue<std::pair<int, int>> q;
//   std::map<int, int> visited;

//   q.push({start_idx, 0});
//   visited[start_idx] = 0;
//   backbone_neighbors[start_tag][start_tag] = 0;

//   while (!q.empty()) {
//     auto current = q.front();
//     q.pop();
//     int current_idx = current.first;
//     int current_dist = current.second;

//     if (current_dist == max_dist) continue;

//     // Explore neighbors of the current LOCAL atom
//     for (int k = 0; k < atom->num_bond[current_idx]; ++k) {
//       tagint neighbor_tag = atom->bond_atom[current_idx][k];
//       int neighbor_idx = atom->map(neighbor_tag);

//       if (neighbor_idx >= 0 && visited.find(neighbor_idx) == visited.end()) {
//         visited[neighbor_idx] = current_dist + 1;
//         backbone_neighbors[start_tag][neighbor_tag] = current_dist + 1;    // Store tag -> tag
//         if (neighbor_idx < atom->nlocal) { q.push({neighbor_idx, current_dist + 1}); }
//       }
//     }
//   }
// }

// void FixBackboneInfo::compute_all_backbone_maps()
// {
//   backbone_neighbors.clear();    // Clear the main map
//   for (int i = 0; i < atom->nlocal; ++i) { run_bfs_from_atom(i); }
// }

// This BFS now uses the symmetric adjacency list "adj"
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

    // CRITICAL CHANGE: Iterate over our own symmetric list "adj"
    // instead of the potentially asymmetric atom->bond_atom.
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
  // --- START OF NEW GRAPH BUILD ---
  // 1. Create our own symmetric adjacency list (adj) to ensure the
  //    graph is traversable regardless of the global "newton bond" setting.

  adj.assign(atom->nmax, std::vector<int>()); // Resize and clear

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
  // --- END OF NEW GRAPH BUILD ---


  // 2. Now, run the BFS using our guaranteed-symmetric adjacency list.
  backbone_neighbors.clear();
  for (int i = 0; i < atom->nlocal; ++i) {
    run_bfs_from_atom(i);
  }
}

// void FixBackboneInfo::post_reaction_callback_local(tagint tag1, tagint tag2)
// {
//   std::set<tagint> tags_to_update;    // Now a set of TAGS

//   // 1. Check if tag1 exists as a key in the main map
//   if (backbone_neighbors.count(tag1)) {
//     // 2. Iterate over the inner map {neighbor_tag -> dist}
//     for (const auto &pair : backbone_neighbors[tag1]) {
//       // 3. Insert the neighbor_tag into the set
//       tags_to_update.insert(pair.first);
//     }
//   }

//   // (Same logic for tag2)
//   if (backbone_neighbors.count(tag2)) {
//     for (const auto &pair : backbone_neighbors[tag2]) { tags_to_update.insert(pair.first); }
//   }

//   tags_to_update.insert(tag1);
//   tags_to_update.insert(tag2);

//   // 4. Final safety check before running BFS
//   for (tagint tag : tags_to_update) {
//     int idx = atom->map(tag);
//     // This check ensures we only run BFS for atoms that are STILL local
//     if (idx >= 0 && idx < atom->nlocal) { run_bfs_from_atom(idx); }
//   }
// }


// THE CORRECTED, UNIFIED CALLBACK
void FixBackboneInfo::post_reaction_callback_local(tagint tag1, tagint tag2)
{
  // 1. ALWAYS rebuild the symmetric adjacency list from the CURRENT,
  //    post-reaction atom data. This is essential and correct.
  adj.assign(atom->nmax, std::vector<int>());
  for (int i = 0; i < atom->nlocal; i++) {
    for (int j = 0; j < atom->num_bond[i]; j++) {
      int neighbor_idx = atom->map(atom->bond_atom[i][j]);
      if (neighbor_idx >= 0) {
        adj[i].push_back(neighbor_idx);
        if (neighbor_idx < atom->nlocal) {
          adj[neighbor_idx].push_back(i);
        }
      }
    }
  }

  // 2. Use the "smart" logic to determine the update set. This version
  //    correctly handles the case where tag1 or tag2 are new atoms.
  std::set<tagint> tags_to_update;

  // For the first atom involved in the reaction:
  // If it's an existing atom (i.e., it's in our old map), find its old
  // neighborhood to define the full zone of influence.
  if (backbone_neighbors.count(tag1)) {
    for (const auto& pair : backbone_neighbors.at(tag1)) {
      tags_to_update.insert(pair.first);
    }
  }
  // In ALL cases (whether it's a new or old atom), the atom itself
  // and its immediate new neighbors must be updated.
  tags_to_update.insert(tag1);
  int idx1 = atom->map(tag1);
  if (idx1 >= 0) { // Add its new direct neighbors
      for (int neighbor_idx : adj[idx1]) {
          tags_to_update.insert(atom->tag[neighbor_idx]);
      }
  }


  // Do the same for the second atom.
  if (backbone_neighbors.count(tag2)) {
    for (const auto& pair : backbone_neighbors.at(tag2)) {
      tags_to_update.insert(pair.first);
    }
  }
  tags_to_update.insert(tag2);
  int idx2 = atom->map(tag2);
  if (idx2 >= 0) { // Add its new direct neighbors
      for (int neighbor_idx : adj[idx2]) {
          tags_to_update.insert(atom->tag[neighbor_idx]);
      }
  }

  // 3. Run the BFS for each affected atom using the new, correct graph.
  for (tagint tag : tags_to_update) {
    int idx = atom->map(tag);
    if (idx >= 0 && idx < atom->nlocal) { // Safety check
      run_bfs_from_atom(idx);
    }
  }
}



// void FixBackboneInfo::validate()
// {
//   if (comm->me == 0) {
//     fprintf(screen, "\n--- RUNNING FIX BACKBONE/INFO VALIDATION ---\n");
//   }

//   // 1. Store a copy of the current map (built by local callbacks)
//   std::map<tagint, std::map<tagint, int>> map_from_callbacks = backbone_neighbors;

//   // 2. Perform a full, from-scratch rebuild into the live map
//   compute_all_backbone_maps();
//   std::map<tagint, std::map<tagint, int>>& map_from_rebuild = backbone_neighbors;

//   // 3. Compare the two maps
//   bool mismatch_found = false;

//   // Check if the number of top-level keys (atoms) is the same
//   if (map_from_callbacks.size() != map_from_rebuild.size()) {
//     mismatch_found = true;
//     if (comm->me == 0) {
//       fprintf(screen, "VALIDATION FAILED: Top-level map sizes differ! (%zu vs %zu)\n",
//               map_from_callbacks.size(), map_from_rebuild.size());
//     }
//   }

//   if (!mismatch_found) {
//     // Iterate through the rebuilt map (our ground truth)
//     for (auto const& [atom_tag, rebuild_inner_map] : map_from_rebuild) {
//       // Check if the atom exists in the callback-built map
//       if (map_from_callbacks.find(atom_tag) == map_from_callbacks.end()) {
//         mismatch_found = true;
//         if (comm->me == 0) {
//           fprintf(screen, "VALIDATION FAILED: Atom tag %d in rebuild map not found in callback map.\n", atom_tag);
//         }
//         break; // Stop on first error for cleaner output
//       }

//       const auto& callback_inner_map = map_from_callbacks[atom_tag];

//       // Check if the inner maps are identical
//       if (rebuild_inner_map.size() != callback_inner_map.size() ||
//           !std::equal(rebuild_inner_map.begin(), rebuild_inner_map.end(), callback_inner_map.begin())) {
//         mismatch_found = true;
//         if (comm->me == 0) {
//           fprintf(screen, "VALIDATION FAILED: Neighbor map for atom tag %d is different.\n", atom_tag);
//           fprintf(screen, "  Rebuild map has %zu neighbors.\n", rebuild_inner_map.size());
//           fprintf(screen, "  Callback map has %zu neighbors.\n", callback_inner_map.size());
//         }
//         break;
//       }
//     }
//   }

//   // 4. Print summary
//   if (!mismatch_found) {
//     if (comm->me == 0) {
//       fprintf(screen, "VALIDATION SUCCESSFUL: Maps are identical.\n");
//     }
//   }

//   // Restore the original map (important!)
//   backbone_neighbors = map_from_callbacks;

//   if (comm->me == 0) {
//     fprintf(screen, "--- VALIDATION COMPLETE ---\n\n");
//   }
// }