/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(nematic/align, PairNematicAlign);
#else

#ifndef LMP_PAIR_NEMATIC_ALIGN_H
#define LMP_PAIR_NEMATIC_ALIGN_H

#include "pair.h"    
#include "fix_backbone_info.h"

class FixBackboneInfo;

  

namespace LAMMPS_NS {

class PairNematicAlign : public Pair {
 public:
  PairNematicAlign(class LAMMPS *);
  ~PairNematicAlign() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  // double single(int, int, int, int, double, double, double, double &) override;

  void write_restart(FILE *fp) override;
  void read_restart(FILE *fp) override;

 protected:
  double cut_global;
  double **epsilon;
  double **cut;

  int **no_radial_flag;

  char *fix_id;
  FixBackboneInfo *fix_bi;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif