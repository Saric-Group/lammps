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
PairStyle(nematic/align,PairNematicAlign);
// clang-format on
#else

#ifndef LMP_PAIR_NEMATIC_ALIGN_H
#define LMP_PAIR_NEMATIC_ALIGN_H

#include "pair.h"

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

 protected:
  double cut_global;
  double **epsilon, **cut;
  int **no_radial;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
