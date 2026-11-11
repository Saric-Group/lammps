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
FixStyle(director/bind,FixDirectorBind);
// clang-format on
#else

#ifndef LMP_FIX_DIRECTOR_BIND_H
#define LMP_FIX_DIRECTOR_BIND_H

#include "fix.h"

#include <string>

namespace LAMMPS_NS {

class FixDirectorBind : public Fix {
 public:
  FixDirectorBind(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  double compute_scalar() override;

 protected:
  std::string normal_name;    // per-atom array without d2_
  double eps;
  int nth;
  int ilevel_respa;
  double energy_local, energy_all;
  int energy_flag;
};

}    // namespace LAMMPS_NS

#endif
#endif
