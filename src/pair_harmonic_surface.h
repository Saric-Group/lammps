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
PairStyle(harmonic/surface,PairHarmonicSurface);
// clang-format on
#else

#ifndef LMP_PAIR_HARMONIC_SURFACE_H
#define LMP_PAIR_HARMONIC_SURFACE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairHarmonicSurface : public Pair {
 public:
  PairHarmonicSurface(class LAMMPS *);
  ~PairHarmonicSurface() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void born_matrix(int, int, int, int, double, double, double, double &, double &) override;
  void *extract(const char *, int &) override;

 protected:
  double **k, **r_zero, **cut;
  int surface_type; // atom-type of particles to obtain normal from, must be ellipsoids

  class AtomVecEllipsoid *avec; // to access orientation of ellipsoids

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
