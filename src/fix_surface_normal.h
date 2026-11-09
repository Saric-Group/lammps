/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(surface/normal,FixSurfaceNormal);
// clang-format on
#else

#ifndef LMP_FIX_SURFACE_NORMAL_H
#define LMP_FIX_SURFACE_NORMAL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSurfaceNormal : public Fix {
 public:
  FixSurfaceNormal(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void setup_pre_force(int) override;
  void setup_pre_force_respa(int, int) override;
  void pre_force(int) override;
  void pre_force_respa(int, int, int) override;
  void min_pre_force(int) override;

 private:
  int nevery;
  bigint last_step;
  class PairHarmonicSurfaceAvg *pair;

  void refresh(int force_update);
};

}    // namespace LAMMPS_NS

#endif
#endif
