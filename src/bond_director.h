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

#ifdef BOND_CLASS
// clang-format off
BondStyle(director,BondDirector);
// clang-format on
#else

#ifndef LMP_BOND_DIRECTOR_H
#define LMP_BOND_DIRECTOR_H

#include "bond.h"

namespace LAMMPS_NS {

class BondDirector : public Bond {
 public:
  BondDirector(class LAMMPS *);
  ~BondDirector() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  void *extract(const char *, int &) override;

  // energy of one bond with r = x2 - x1 and unit directors e1, e2 (nullptr = none); fills the force on
  // atom 1 (atom 2 gets -f1) and the torques on both directors
  double bond_energy(int type, const double *r, const double *e1, const double *e2, double *f1,
                     double *tq1, double *tq2) const;

 protected:
  double *k, *r0, *k_perp, *k_tw, *tau0;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
