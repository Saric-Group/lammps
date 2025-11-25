/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(backbone/info, FixBackboneInfo)

#else

#ifndef LMP_FIX_BACKBONE_INFO_H
#define LMP_FIX_BACKBONE_INFO_H

#include "fix.h"
#include <vector>

namespace LAMMPS_NS {

class FixBackboneInfo : public Fix {
public:
  FixBackboneInfo(class LAMMPS *, int, char **);
  ~FixBackboneInfo() override;
  int setmask() override;
  void init() override;
  void pre_force(int) override; // Called every timestep

  // Memory management required for atom deletion/sorting
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  
  std::vector<std::vector<tagint>> backbone_cache;

  // Optimized getter for the Pair Style
  inline const std::vector<tagint>* get_backbone_partners(int i) const {
    if (i >= 0 && i < (int)backbone_cache.size()) {
        return &backbone_cache[i];
    }
    return nullptr;
  }

  void ensure_cache();
  

private:
  int max_dist;
  bool cache_valid;
  
  // Internal scratch containers
  std::vector<std::vector<int>> adj; 
  std::vector<int> visited_flag;

  void build_cache();
};

} // namespace LAMMPS_NS

#endif
#endif