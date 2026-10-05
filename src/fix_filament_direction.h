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
FixStyle(filament/direction,FixFilamentDirection);
// clang-format on
#else

#ifndef LMP_FIX_FILAMENT_DIRECTION_H
#define LMP_FIX_FILAMENT_DIRECTION_H

#include "atom.h"
#include "fix.h"

#include <string>

namespace LAMMPS_NS {

class FixFilamentDirection : public Fix {
 public:
  FixFilamentDirection(class LAMMPS *, int, char **);
  ~FixFilamentDirection() override;

  int setmask() override;
  void post_constructor() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup_pre_force(int) override;
  void setup(int) override;
  void pre_force(int) override;
  void min_pre_force(int) override;
  double compute_vector(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void *extract(const char *, int &) override;

 protected:
  enum { BOND_ORDER };
  enum { COMM_DIRECTION, COMM_ALIGNMENT };

  int nevery;
  int *bondtype_flag;    // 1 for the bond types that define the direction
  int *type_flag;        // 1 for the atom types that are filament beads
  int orient;            // how the direction of a bond is decided
  double cutoff;         // alignment neighbour cutoff, 0 = no alignment counting
  double parallel_cos, antiparallel_cos, lateral_cos;
  int exclude_special;
  int mu_flag;

  // per-atom output, stored in an internal fix property/atom
  char *id_props;
  std::string name_dir, name_nbond, name_npar, name_nanti, name_cosmin;
  double **dir;
  int *nbond, *npar, *nanti;
  double *cosmin;

  int commflag;
  class NeighList *list;

  void find_properties();
  void compute();
  void compute_direction();
  void compute_alignment();
  int bond_sign(int, int);
  bool is_special(int, int);
  bool is_bead(int i) const { return (atom->mask[i] & groupbit) && type_flag[atom->type[i]]; }
};

}    // namespace LAMMPS_NS

#endif
#endif
