/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(addlipid,FixAddLipid)

#else

#ifndef LMP_FIX_ADDLIPID_H
#define LMP_FIX_ADDLIPID_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAddLipid : public Fix {
 public:
  FixAddLipid(class LAMMPS *, int, char **);
  ~FixAddLipid();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void pre_exchange();
  void post_constructor();
  void unlimit_bond();
  void update_everything();

  //int pack_reverse_comm_size(int, int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;




  int count_global;

 private:
  int lipid_type, add_num, add_flag, added_num, dt, vel_flag, near_num_limit, stabilization_flag,test;
  double d_cut, d_break, r_detect, a_min, d_Bezier, a_near_min;
  tagint maxtag_all;
  double **xnew;
  int *masknew;
  class NeighList *list;

  // For internal nve/limit integration
  int custom_exclude_flag;
  Fix *fix1;                   // nve/limit used to relax reaction sites
  Fix *fix2;                   // properties/atom used to indicate 1) relaxing atoms
                //                                  2) to which 'react' atom belongs
  Fix *fix3;                   // property/atom used for system-wide thermostat
  Fix *fix4;                   // used to add ignorance tags


  //  Fpr creation/handling of nearest neighbour lists
  int ignore_neigh_flag;        // 0 -> all neighbours excluded, 1 -> stabilized excluded, 2-> custom number of neighbours and/or exclusion duration
  int stabilize_neigh_flag;     // if at least 1 neighbour is stabilized
  int need_neighbours_flag;     // 0 -> no neighbour calculations, 1 -> one neirest neighbours list (for statted and/or ignored), 2-> two neighbourlist (for statted and ignored individually)
  int num_nn;             // number of neirest neighbours   // for both limit/exclude (at the moment)
  int num_nn_ignore;      // yet to be implemented

  int random_ignorance_flag;    // 0 -> all pairs are considered, 1 -> pairs are exclude BEFORE acceptance of insertion (any calculation really->fast), 2 -> pairs are excluded AFTER acceptance of inserion
  double p_add;                 // probability to consider pair for adding
  class RanMars **random_test;

  int group_consideration_flag;  //0 -> no group considere additionally, 1/2 -> group considered is inclusive/exclusive
  int consider_group;     // group for inclusion/exclusion in adding procedure


  int limit_duration;      // indicates how long to limit dynamics
  int ignore_duration;    // -||- exclude flagged particles from adding procedure


  int *statted_vec;         // NOT USED ATM: should be used for more efficient comm_reverse() of ghost tags
  int *limit_vec;

  char *nve_limit_xmax;    // indicates max distance allowed to move when relaxing
  char *id_fix1;           // id of internally created fix nve/limit
  char *id_fix2;           // id of internally created fix per-atom properties
  char *id_fix3;           // id of internally created 'stabilization group' per-atom property fix
  char *id_fix4;           // id of internally created 'ignore group' per-atom property fix
  char *limit_id;          // name of 'limit group' (basically a countdown running down for limit_duration, statted with nve/limit in the meantime)
  char *statted_id;        // name of 'stabilization group' per-atom property
  char *ignored_id;         // name of 'ignore' per-atom property
  char *ignore_id;          // same as limit_tags are to statted_tags
  char *master_group;      // group containing relaxing atoms from all fix rxns
  char *exclude_group;     // group for system-wide thermostat


};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix pour requires atom attributes radius, rmass

The atom style defined does not have these attributes.

E: Invalid atom type in fix pour command

Self-explanatory.

E: Must specify a region in fix pour

Self-explanatory.

E: Fix pour region does not support a bounding box

Not all regions represent bounded volumes.  You cannot use
such a region with the fix pour command.

E: Fix pour region cannot be dynamic

Only static regions can be used with fix pour.

E: Insertion region extends outside simulation box

Self-explanatory.

E: Must use a z-axis cylinder region with fix pour

Self-explanatory.

E: Must use a block or cylinder region with fix pour

Self-explanatory.

E: Must use a block region with fix pour for 2d simulations

Self-explanatory.

E: Cannot use fix_pour unless atoms have IDs

Self-explanatory.

E: Fix pour molecule must have coordinates

The defined molecule does not specify coordinates.

E: Fix pour molecule must have atom types

The defined molecule does not specify atom types.

E: Invalid atom type in fix pour mol command

The atom types in the defined molecule are added to the value
specified in the create_atoms command, as an offset.  The final value
for each atom must be between 1 to N, where N is the number of atom
types.

E: Fix pour molecule template ID must be same as atom style template ID

When using atom_style template, you cannot pour molecules that are
not in that template.

E: Cannot use fix pour rigid and not molecule

Self-explanatory.

E: Cannot use fix pour shake and not molecule

Self-explanatory.

E: Cannot use fix pour rigid and shake

These two attributes are conflicting.

E: No fix gravity defined for fix pour

Gravity is required to use fix pour.

E: Fix pour insertion count per timestep is 0

Self-explanatory.

E: Cannot use fix pour with triclinic box

This option is not yet supported.

E: Gravity must point in -z to use with fix pour in 3d

Self-explanatory.

E: Gravity must point in -y to use with fix pour in 2d

Self-explanatory.

E: Gravity changed since fix pour was created

The gravity vector defined by fix gravity must be static.

E: Fix pour rigid fix does not exist

Self-explanatory.

E: Fix pour and fix rigid/small not using same molecule template ID

Self-explanatory.

E: Fix pour shake fix does not exist

Self-explanatory.

E: Fix pour and fix shake not using same molecule template ID

Self-explanatory.

W: Less insertions than requested

The fix pour command was unsuccessful at finding open space
for as many particles as it tried to insert.

E: Too many total atoms

See the setting for bigint in the src/lmptype.h file.

E: New atom IDs exceed maximum allowed ID

See the setting for tagint in the src/lmptype.h file.

E: Fix pour region ID does not exist

Self-explanatory.

E: Molecule template ID for fix pour does not exist

Self-explanatory.

E: Fix pour polydisperse fractions do not sum to 1.0

Self-explanatory.

E: Cannot change timestep with fix pour

This is because fix pour pre-computes the time delay for particles to
fall out of the insertion volume due to gravity.

*/
