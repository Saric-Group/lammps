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
FixStyle(chain/index,FixChainIndex);
// clang-format on
#else

#ifndef LMP_FIX_CHAIN_INDEX_H
#define LMP_FIX_CHAIN_INDEX_H

#include "fix.h"

namespace LAMMPS_NS {

class FixChainIndex : public Fix {
 public:
  FixChainIndex(class LAMMPS *, int, char **);
  ~FixChainIndex() override;

  int setmask() override;
  void post_constructor() override;
  void setup_pre_force(int) override;
  void setup(int) override;
  void pre_force(int) override;
  void min_pre_force(int) override;
  double compute_vector(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  // look up or create the per-atom property i_chain_index, return its index
  static int find_or_create_property(class LAMMPS *);

 protected:
  int nevery;
  int *bondtype_flag;    // 1 for backbone bond types
  int *attach_flag;      // 1 for bond types that attach side beads to the backbone
  int tail_type, head_type;
  int warnflag;
  double stats[6];

  void relabel();
};

}    // namespace LAMMPS_NS

#endif
#endif
