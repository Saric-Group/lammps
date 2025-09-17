#ifdef FIX_CLASS
FixStyle(backbone/info, FixBackboneInfo)

#else

#ifndef LMP_FIX_BACKBONE_INFO_H
#define LMP_FIX_BACKBONE_INFO_H

#include "fix.h"
#include <vector>
#include <map>
#include <set>

namespace LAMMPS_NS {

class FixBackboneInfo : public Fix {
public:
  FixBackboneInfo(class LAMMPS *, int, char **);
  ~FixBackboneInfo() override;
  int setmask() override;
  // void post_constructor() override; 
  // void init() override {};
  void init() override;

  // The local update callback called by fix bond/react
  void post_reaction_callback_local(tagint, tagint);

  // void validate();
  
  // The public data structure that the pair style will access
  // For each atom index [i], it stores a map of {neighbor_tag -> distance_in_bonds}
  std::map<tagint, std::map<tagint, int>> backbone_neighbors;
  

private:
  int max_dist; // Max bond distance to search

  // The full rebuild, used only at the start of a run
  void compute_all_backbone_maps();

  // Helper function to run a single BFS from a starting atom index
  void run_bfs_from_atom(int start_idx);

  // The symmetric representation of the bond graph.
  // This is the key to being independent of "newton bond".
  std::vector<std::vector<int>> adj;
};

} // namespace LAMMPS_NS

#endif
#endif