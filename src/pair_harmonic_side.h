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
PairStyle(harmonic/side,PairHarmonicSide);
// clang-format on
#else

#ifndef LMP_PAIR_HARMONIC_SIDE_H
#define LMP_PAIR_HARMONIC_SIDE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairHarmonicSide : public Pair {
 public:
  PairHarmonicSide(class LAMMPS *);
  ~PairHarmonicSide() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  enum { LEFT = 1, RIGHT = -1 };

  char *id_fix;
  class Fix *fix_direction;
  double cut_global;
  double **k, **r0, **cut;
  int **side, **nth;
  int need_chain_index;    // 1 if any pair_coeff uses nth > 1

  void allocate();
  void get_arrays(double **&, int *&);
  bool active(int, int, int, int, double, double, double, double **, int *);
};

}    // namespace LAMMPS_NS

#endif
#endif
