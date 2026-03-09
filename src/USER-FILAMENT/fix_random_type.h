/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(random/type, FixRandomType)

#else

#ifndef LMP_FIX_RANDOM_TYPE_H
#define LMP_FIX_RANDOM_TYPE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRandomType : public Fix {
 public:
  FixRandomType(class LAMMPS *, int, char **);
  ~FixRandomType() override;
  int setmask() override;
  void end_of_step() override;

 private:
  int nevery, type_from, type_to;
  double fraction;
  class RanPark *random;
};

}

#endif
#endif