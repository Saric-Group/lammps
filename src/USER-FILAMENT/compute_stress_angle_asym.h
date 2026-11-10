/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(stress/angle/asym,ComputeStressAngleAsym);
// clang-format on
#else

#ifndef LMP_COMPUTE_STRESS_ANGLE_ASYM_H
#define LMP_COMPUTE_STRESS_ANGLE_ASYM_H

#include "compute.h"

namespace LAMMPS_NS {

// per-atom 2D stress of angle_style harmonic with the three angle forces split into two pairs of
// opposite forces, one along each bond; columns xx, yy, xy, yx (first index position, second force),
// not symmetrised: the antisymmetric part is the bending couple

class ComputeStressAngleAsym : public Compute {
 public:
  ComputeStressAngleAsym(class LAMMPS *, int, char **);
  ~ComputeStressAngleAsym() override;
  void init() override;
  void compute_peratom() override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

 private:
  int nmax;
  double **stress;
  double *k, *theta0;    // pointers into the angle style, not copies
  class Angle *angle;
};

}    // namespace LAMMPS_NS

#endif
#endif
