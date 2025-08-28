/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(nematic/angle/soft, PairNematicSoft);
#else

#ifndef LMP_PAIR_NEMATIC_SOFT_H
#define LMP_PAIR_NEMATIC_SOFT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairNematicSoft : public Pair {
 public:
  PairNematicSoft(class LAMMPS *);
  ~PairNematicSoft() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double single(int, int, int, int, double, double, double, double &) override;
  double single_orientation(int, int, double, double, const double *, const double *) override;

 protected:
  double cut_global;

  double **Aamp;
  double **kappa;
  double **theta0;
  double **alpha;  // geometric correction parameter
  double ** epsilon; // depth of the potential well
  double **c0, **s0, **c0s0, **cos2t0, **c_fac; // cached trig
  double **cut;

  virtual void allocate();
};

} // namespace LAMMPS_NS

#endif
#endif