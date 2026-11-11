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

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(director/bend,AngleDirectorBend);
// clang-format on
#else

#ifndef LMP_ANGLE_DIRECTOR_BEND_H
#define LMP_ANGLE_DIRECTOR_BEND_H

#include "angle.h"

namespace LAMMPS_NS {

class AngleDirectorBend : public Angle {
 public:
  AngleDirectorBend(class LAMMPS *);
  ~AngleDirectorBend() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double equilibrium_angle(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, int, int, int) override;
  void *extract(const char *, int &) override;

  // energy of one angle; r1 = x2 - x1, r2 = x3 - x2; e2 unit director of atom 2 (nullptr = none).
  // Fills the forces on atoms 1 and 3 (f2 = -f1 - f3) and the torque on the director of atom 2.
  double bend(int type, const double *r1, const double *r2, const double *e2, double *f1, double *f3,
              double *tq2) const;

 protected:
  double *k_b, *delta0, *alpha0;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
