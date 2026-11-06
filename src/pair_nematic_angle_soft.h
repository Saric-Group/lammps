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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nematic/angle/soft,PairNematicSoft);
// clang-format on
#else

#ifndef LMP_PAIR_NEMATIC_ANGLE_SOFT_H
#define LMP_PAIR_NEMATIC_ANGLE_SOFT_H

#include "pair.h"

#include <string>

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
  void *extract(const char *, int &) override;
  double single_orientation(int, int, double, int, const double *, const double *, const double *,
                            const double *) override;

 protected:
  double cut_global;
  int intermol_flag;
  int normal_fixed;           // 1: constant normal normal_vec, 0: per-atom array name_normal
  double normal_vec[3];
  std::string name_normal;    // 3-column per-atom array d2_name with the surface normal
  double **Aamp, **Amin, **kappa, **theta0, **cut;
  double **c0, **s0;    // cos and sin of theta0

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
