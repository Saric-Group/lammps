/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.
   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef REGION_CLASS
// clang-format off
RegionStyle(capsule,RegCapsule);
// clang-format on
#else

#ifndef LMP_REGION_CAPSULE_H
#define LMP_REGION_CAPSULE_H

#include "region.h"

namespace LAMMPS_NS {

class RegCapsule : public Region {
  friend class FixPour;
  friend class Region2VMD;

 public:
  RegCapsule(class LAMMPS *, int, char **);
  ~RegCapsule() override;
  void init() override;
  int inside(double, double, double) override;
  int surface_interior(double *, double) override;
  int surface_exterior(double *, double) override;
  void shape_update() override;

  // Dynamic velocity handling
  void set_velocity_shape() override;
  void velocity_contact_shape(double *, double *) override;

 private:
  // Capsule geometry
  char axis;                       // main axis: 'x', 'y', 'z'
  double c1, c2;                   // lateral offsets along non-axial directions
  double radius;                    // cylinder radius
  double lo, hi;                    // cylinder axial min/max
  double lohemisphereradius, hihemisphereradius; // hemisphere radii

  // Previous values for velocity computation
  double rprev, rlo_prev, rhi_prev;
  double prev_loh, prev_hih;
  double prev_ctr1, prev_ctr2;
  double prev_cc1, prev_cc2;
  double prev_c1, prev_c2;
  double prev_center1, prev_center2;


  // Axis-aligned center for cylinder velocity
  double xcenter[3];

  // Styles and variable indices for dynamic parameters
  int c1style, c1var;
  int c2style, c2var;
  int rstyle, rvar;
  int lostyle, histyle;
  int lovar, hivar;
  int lohemisphereradiusstyle, hihemisphereradiusstyle;
  int lohemisphereradiusvar, hihemisphereradiusvar;

  // Strings for variable names
  char *c1str, *c2str, *rstr;
  char *lostr, *histr;
  char *lohemisphereradiusstr, *hihemisphereradiusstr;

  // Precompute hemisphere offsets for velocity
  double H1, H2, dH1, dH2;
  double center1, center2; // centers of hemispheres along axis
  double newlo, newhi;     // extended positions including hemispheres

  // Helper functions
  void variable_check();
};

}    // namespace LAMMPS_NS

#endif
#endif
