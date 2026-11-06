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
PairStyle(align/cosine/squared,PairAlignCosineSquared);
// clang-format on
#else

#ifndef LMP_PAIR_ALIGN_COSINE_SQUARED_H
#define LMP_PAIR_ALIGN_COSINE_SQUARED_H

#include "pair.h"

namespace LAMMPS_NS {

class PairAlignCosineSquared : public Pair {
 public:
  PairAlignCosineSquared(class LAMMPS *);
  ~PairAlignCosineSquared() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  char *id_fix;
  class Fix *fix_direction;
  double cut_global;
  double **epsilon, **sigma, **cut, **cos_on, **cos_off;

  void allocate();
  void get_directions(double **&, int *&);
  double gate(double, int, int);
  void well(double, int, int, double &, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif
