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
  enum { LEFT = 1, RIGHT = -1, ANY = 0 };

  char *id_fix;
  class Fix *fix_direction;
  double cut_global;
  double **k, **r0, **cut;
  int **side, **nth;
  double **parallel_cos, **antiparallel_cos;    // alignment gate, 2 / -2 if unset
  int **check_alignment;                        // 1 if parallel or antiparallel is set
  double **lateral_cos;                         // 1 = off
  double **region_half;                         // half width of the band |y| < w/2, -1 = off
  int need_chain_index;                         // 1 if any pair_coeff uses nth > 1
  int need_left;                                // 1 if any pair_coeff uses left or right

  void allocate();
  void get_arrays(double **&, double **&, int *&);
  bool active(int, int, int, int, double, double, double, double **, double **, int *);
  double wrapped_y(double) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
