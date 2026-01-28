/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.
   Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
   the U.S. Government retains certain rights in this software.
   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#include "region_capsule.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "update.h"
#include "variable.h"
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr double BIG = 1.0e20;

/* ---------------------------------------------------------------------- */

RegCapsule::RegCapsule(LAMMPS *lmp, int narg, char **arg) :
    Region(lmp, narg, arg),
    c1str(nullptr), c2str(nullptr), rstr(nullptr),
    lostr(nullptr), histr(nullptr), lohemisphereradiusstr(nullptr), hihemisphereradiusstr(nullptr)
{
  c1style = c2style = rstyle = CONSTANT;
  lostyle = histyle = CONSTANT;
  lohemisphereradiusvar = hihemisphereradiusvar = CONSTANT;
  options(narg - 10, &arg[10]);

  if (strcmp(arg[2], "x") != 0 && strcmp(arg[2], "y") != 0 && strcmp(arg[2], "z") != 0)
    error->all(FLERR, "Illegal region capsule axis: {}", arg[2]);
  axis = arg[2][0];

  // Initialize prev variables to zero to prevent uninitialized access
  for (int i = 0; i < 5; ++i) prev[i] = 0.0;
  prev_loh = prev_hih = 0.0;
  prev_ctr1 = prev_ctr2 = 0.0;
  prev_cc1 = prev_cc2 = 0.0;

  // center coordinates
  if (axis == 'x') {
    if (utils::strmatch(arg[3], "^v_")) { c1str = utils::strdup(arg[3]+2); c1 = 0.0; c1style = VARIABLE; varshape = 1; }
    else { c1 = utils::numeric(FLERR,arg[3],false,lmp)*yscale; c1style = CONSTANT; }
    if (utils::strmatch(arg[4], "^v_")) { c2str = utils::strdup(arg[4]+2); c2 = 0.0; c2style = VARIABLE; varshape = 1; }
    else { c2 = utils::numeric(FLERR,arg[4],false,lmp)*zscale; c2style = CONSTANT; }
  } else if (axis == 'y') {
    if (utils::strmatch(arg[3], "^v_")) { c1str = utils::strdup(arg[3]+2); c1 = 0.0; c1style = VARIABLE; varshape = 1; }
    else { c1 = utils::numeric(FLERR,arg[3],false,lmp)*xscale; c1style = CONSTANT; }
    if (utils::strmatch(arg[4], "^v_")) { c2str = utils::strdup(arg[4]+2); c2 = 0.0; c2style = VARIABLE; varshape = 1; }
    else { c2 = utils::numeric(FLERR,arg[4],false,lmp)*zscale; c2style = CONSTANT; }
  } else {
    if (utils::strmatch(arg[3], "^v_")) { c1str = utils::strdup(arg[3]+2); c1 = 0.0; c1style = VARIABLE; varshape = 1; }
    else { c1 = utils::numeric(FLERR,arg[3],false,lmp)*xscale; c1style = CONSTANT; }
    if (utils::strmatch(arg[4], "^v_")) { c2str = utils::strdup(arg[4]+2); c2 = 0.0; c2style = VARIABLE; varshape = 1; }
    else { c2 = utils::numeric(FLERR,arg[4],false,lmp)*yscale; c2style = CONSTANT; }
  }

  // radius
  if (utils::strmatch(arg[5], "^v_")) { rstr = utils::strdup(arg[5]+2); radius = 0.0; rstyle = VARIABLE; varshape = 1; }
  else { radius = utils::numeric(FLERR,arg[5],false,lmp); if (axis=='x') radius*=yscale; else radius*=xscale; rstyle = CONSTANT; }

  // low bound lo
  if (utils::strmatch(arg[6], "^v_")) { lostr = utils::strdup(arg[6]+2); lo = 0.0; lostyle = VARIABLE; varshape = 1; } 
  else { lo = utils::numeric(FLERR,arg[6],false,lmp); if (axis=='x') lo*=yscale; else lo*=xscale; lostyle = CONSTANT; }

  // high bound hi
  if (utils::strmatch(arg[7], "^v_")) { histr = utils::strdup(arg[7]+2); hi = 0.0; histyle = VARIABLE; varshape = 1; } 
  else { hi = utils::numeric(FLERR,arg[7],false,lmp); if (axis=='x') hi*=yscale; else hi*=xscale; histyle = CONSTANT; }

   // low bound lo
  if (utils::strmatch(arg[8], "^v_")) { lohemisphereradiusstr = utils::strdup(arg[8]+2); lohemisphereradius = 0.0; lohemisphereradiusstyle = VARIABLE; varshape = 1; } 
  else { lohemisphereradius = utils::numeric(FLERR,arg[8],false,lmp); if (axis=='x') lohemisphereradius*=yscale; else lohemisphereradius*=xscale; lohemisphereradiusstyle = CONSTANT; }

  // high bound hi
  if (utils::strmatch(arg[9], "^v_")) { hihemisphereradiusstr = utils::strdup(arg[9]+2); hihemisphereradius = 0.0; hihemisphereradiusstyle = VARIABLE; varshape = 1; } 
  else { hihemisphereradius = utils::numeric(FLERR,arg[9],false,lmp); if (axis=='x') hihemisphereradius*=yscale; else hihemisphereradius*=xscale; hihemisphereradiusstyle = CONSTANT; }

  if (varshape) {
    variable_check();
    shape_update();
  }

  if (radius <= 0.0) error->all(FLERR,"Illegal radius {} in region capsule command",radius);

  // code checks for the whole box if its in/out of the region - can be sped up with the original LAMMPS setup of a bounding box
  bboxflag=0;

  cmax=3;
  contact = new Contact[cmax];
  tmax = 1;
}

/* ---------------------------------------------------------------------- */

RegCapsule::~RegCapsule()
{
  delete[] c1str;
  delete[] c2str;
  delete[] rstr;
  delete[] lostr;
  delete[] histr;
  delete[] lohemisphereradiusstr;
  delete[] hihemisphereradiusstr;
  delete[] contact;
}

/* ---------------------------------------------------------------------- */

void RegCapsule::init()
{
  Region::init();
  if (varshape) variable_check();
}

/* ---------------------------------------------------------------------- */

void RegCapsule::shape_update()
{
  if(c1style==VARIABLE) c1 = input->variable->compute_equal(c1var);
  if(c2style==VARIABLE) c2 = input->variable->compute_equal(c2var);
  if(rstyle==VARIABLE) {
    radius = input->variable->compute_equal(rvar);
    if(radius<0.0) error->one(FLERR,"Variable evaluation in region gave bad value");
  }
  if(lostyle==VARIABLE) lo = input->variable->compute_equal(lovar);
  if(histyle==VARIABLE) hi = input->variable->compute_equal(hivar);
  if(lohemisphereradiusstyle==VARIABLE) {
    lohemisphereradius = input->variable->compute_equal(lohemisphereradiusvar);
    if(lohemisphereradius<0.0) error->one(FLERR,"Variable evaluation in region gave bad value");
  }

  if(hihemisphereradiusstyle==VARIABLE) {
    hihemisphereradius = input->variable->compute_equal(hihemisphereradiusvar);
    if(hihemisphereradius<0.0) error->one(FLERR,"Variable evaluation in region gave bad value");
  }

  if(axis=='x'){ if(c1style==VARIABLE)c1*=yscale; if(c2style==VARIABLE)c2*=zscale; if(rstyle==VARIABLE)radius*=yscale; if(lostyle==VARIABLE)lo*=xscale; if(histyle==VARIABLE)hi*=xscale; }
  else if(axis=='y'){ if(c1style==VARIABLE)c1*=xscale; if(c2style==VARIABLE)c2*=zscale; if(rstyle==VARIABLE)radius*=xscale; if(lostyle==VARIABLE)lo*=yscale; if(histyle==VARIABLE)hi*=yscale; }
  else{ if(c1style==VARIABLE)c1*=xscale; if(c2style==VARIABLE)c2*=yscale; if(rstyle==VARIABLE)radius*=xscale; if(lostyle==VARIABLE)lo*=zscale; if(histyle==VARIABLE)hi*=zscale; }
}

/* ---------------------------------------------------------------------- */

void RegCapsule::variable_check()
{
  if(c1style==VARIABLE){ c1var=input->variable->find(c1str); if(c1var<0) error->all(FLERR,"Variable {} for region capsule does not exist",c1str); if(!input->variable->equalstyle(c1var)) error->all(FLERR,"Variable {} for region capsule is invalid style",c1str);}
  if(c2style==VARIABLE){ c2var=input->variable->find(c2str); if(c2var<0) error->all(FLERR,"Variable {} for region capsule does not exist",c2str); if(!input->variable->equalstyle(c2var)) error->all(FLERR,"Variable {} for region capsule is invalid style",c2str);}
  if(rstyle==VARIABLE){ rvar=input->variable->find(rstr); if(rvar<0) error->all(FLERR,"Variable {} for region capsule does not exist",rstr); if(!input->variable->equalstyle(rvar)) error->all(FLERR,"Variable {} for region capsule is invalid style",rstr);}
  if(lostyle==VARIABLE){ lovar=input->variable->find(lostr); if(lovar<0) error->all(FLERR,"Variable {} for region capsule does not exist",lostr); if(!input->variable->equalstyle(lovar)) error->all(FLERR,"Variable {} for region capsule is invalid style",lostr);}
  if(histyle==VARIABLE){ hivar=input->variable->find(histr); if(hivar<0) error->all(FLERR,"Variable {} for region capsule does not exist",histr); if(!input->variable->equalstyle(hivar)) error->all(FLERR,"Variable {} for region capsule is invalid style",histr);}
  if(lohemisphereradiusstyle==VARIABLE){ lohemisphereradiusvar=input->variable->find(lohemisphereradiusstr); if(lohemisphereradiusvar<0) error->all(FLERR,"Variable {} for region capsule does not exist",lohemisphereradiusstr); if(!input->variable->equalstyle(lohemisphereradiusvar)) error->all(FLERR,"Variable {} for region capsule is invalid style",lohemisphereradiusstr);}
  if(hihemisphereradiusstyle==VARIABLE){ hihemisphereradiusvar=input->variable->find(hihemisphereradiusstr); if(hihemisphereradiusvar<0) error->all(FLERR,"Variable {} for region capsule does not exist",hihemisphereradiusstr); if(!input->variable->equalstyle(hihemisphereradiusvar)) error->all(FLERR,"Variable {} for region capsule is invalid style",hihemisphereradiusstr);}
}

/* ---------------------------------------------------------------------- */

int RegCapsule::inside(double x, double y, double z)
{
  double del1, del2, dist;
  double dx, dy, dz;  // predeclare all axis offsets
  double H1,dH1,H2,dH2,center1,center2, newlo,newhi; //calculate position of the hemisphere centers
  const double EPS = 1e-12;

  H1 = sqrt(fmax(EPS,lohemisphereradius*lohemisphereradius - radius*radius));
  dH1 = lohemisphereradius - H1;
  center1 = lo + H1;
  newlo = lo - dH1;
  H2 = sqrt(fmax(EPS,hihemisphereradius*hihemisphereradius - radius*radius));
  dH2 = hihemisphereradius - H2;
  center2 = hi - H2;
  newhi = hi + dH2;

  if(axis=='x'){
    //cylinder
    del1 = y - c1;
    del2 = z - c2;
    dist = sqrt(del1*del1 + del2*del2);
    if(dist <= radius && x >= lo && x <= hi) return 1;

    //low hemisphere
    dx = x - center1;
    dist = sqrt(del1*del1 + del2*del2 + dx*dx);
    if(dist <= lohemisphereradius && x>=newlo && x<=lo) return 1;

    //high hemisphere
    dx = x - center2;
    dist = sqrt(del1*del1 + del2*del2 + dx*dx);
    if(dist <= hihemisphereradius && x>=hi && x<=newhi) return 1;

  } else if(axis=='y'){
    del1 = x - c1;
    del2 = z - c2;
    dist = sqrt(del1*del1 + del2*del2);
    if(dist <= radius && y >= lo && y <= hi) return 1;

    dy = y - center1;
    dist = sqrt(del1*del1 + del2*del2 + dy*dy);
    if(dist <= lohemisphereradius && y>=newlo && y<=lo) return 1;

    dy = y - center2;
    dist = sqrt(del1*del1 + del2*del2 + dy*dy);
    if(dist <= hihemisphereradius && y>= hi && y<=newhi) return 1;

  } else { // axis=='z'
    del1 = x - c1;
    del2 = y - c2;
    dist = sqrt(del1*del1 + del2*del2);
    if(dist <= radius && z >= lo && z <= hi) return 1;

    dz = z - center1;
    dist = sqrt(del1*del1 + del2*del2 + dz*dz);
    if(dist <= lohemisphereradius && z>=newlo && z<=lo) return 1;

    dz = z - center2;
    dist = sqrt(del1*del1 + del2*del2 + dz*dz);
    if(dist <= hihemisphereradius && z>= hi && z<=newhi) return 1;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   contact if 0 <= x < cutoff from one or more inner surfaces of cylinder
   can be one contact for each of 3 cylinder surfaces
   no contact if outside (possible if called from union/intersect)
   delxyz = vector from nearest point on cylinder to x
   special case: no contact with curved surf if x is on center axis
------------------------------------------------------------------------- */

int RegCapsule::surface_interior(double *x, double cutoff)
{
  double del1, del2, r, delta;
  double dx, dy, dz, dist;
  const double EPS = 1e-12;

  //-----------------------------------------------------------------------
  // Compute hemisphere geometry (same as inside())
  //-----------------------------------------------------------------------
  double H1 = sqrt(fmax(EPS, lohemisphereradius*lohemisphereradius - radius*radius));
  double dH1 = lohemisphereradius - H1;
  double center1 = lo + H1;
  double newlo = lo - dH1;

  double H2 = sqrt(fmax(EPS, hihemisphereradius*hihemisphereradius - radius*radius));
  double dH2 = hihemisphereradius - H2;
  double center2 = hi - H2;
  double newhi = hi + dH2;

  int n = 0;

  //-----------------------------------------------------------------------
  // Capsule aligned along X-axis
  //-----------------------------------------------------------------------
  if (axis == 'x') {
    del1 = x[1] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1 * del1 + del2 * del2);

    // 1. Cylinder contact (side wall)
    if (r <= radius && x[0] >= lo && x[0] <= hi) {
      delta = radius - r;
      if (delta < cutoff && r > 0.0) {
        contact[n].r = delta;
        contact[n].delx = 0.0;
        contact[n].dely = del1 * (1.0 - radius / r);
        contact[n].delz = del2 * (1.0 - radius / r);
        contact[n].radius = - 2.0 * radius;
        contact[n].iwall = 0; // curved wall
        contact[n].varflag = 1;
        n++;
      }
    }

    // 2. Low hemisphere contact (below lo)
    if (x[0] < lo && x[0] >= newlo) {
      dx = x[0] - center1;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist <= lohemisphereradius) {
        delta = lohemisphereradius - dist;
        if (delta < cutoff) {
          contact[n].r = delta;
          contact[n].delx = dx * (1.0 - lohemisphereradius / dist);
          contact[n].dely = del1 * (1.0 - lohemisphereradius / dist);
          contact[n].delz = del2 * (1.0 - lohemisphereradius / dist);
          contact[n].radius = - lohemisphereradius;
          contact[n].iwall = 0; // low end
          contact[n].varflag = 1;
          n++;
        }
      }
    }

    // 3. High hemisphere contact (above hi)
    if (x[0] > hi && x[0] <= newhi) {
      dx = x[0] - center2;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist <= hihemisphereradius) {
        delta = hihemisphereradius - dist;
        if (delta < cutoff) {
          contact[n].r = delta;
          contact[n].delx = dx * (1.0 - hihemisphereradius / dist);
          contact[n].dely = del1 * (1.0 - hihemisphereradius / dist);
          contact[n].delz = del2 * (1.0 - hihemisphereradius / dist);
          contact[n].radius = - hihemisphereradius;
          contact[n].iwall = 0; // high end
          contact[n].varflag = 1;
          n++;
        }
      }
    }

  //-----------------------------------------------------------------------
  // Capsule aligned along Y-axis
  //-----------------------------------------------------------------------
  } else if (axis == 'y') {
    del1 = x[0] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1 * del1 + del2 * del2);

    // 1. Cylinder contact (side wall)
    if (r <= radius && x[1] >= lo && x[1] <= hi) {
      delta = radius - r;
      if (delta < cutoff && r > 0.0) {
        contact[n].r = delta;
        contact[n].delx = del1 * (1.0 - radius / r);
        contact[n].dely = 0.0;
        contact[n].delz = del2 * (1.0 - radius / r);
        contact[n].radius = - 2.0 * radius;
        contact[n].iwall = 0;
        contact[n].varflag = 1;
        n++;
      }
    }

    // 2. Low hemisphere contact (below lo)
    if (x[1] < lo && x[1] >= newlo) {
      dy = x[1] - center1;
      dist = sqrt(del1*del1 + del2*del2 + dy*dy);
      if (dist <= lohemisphereradius) {
        delta = lohemisphereradius - dist;
        if (delta < cutoff) {
          contact[n].r = delta;
          contact[n].delx = del1 * (1.0 - lohemisphereradius / dist);
          contact[n].dely = dy * (1.0 - lohemisphereradius / dist);
          contact[n].delz = del2 * (1.0 - lohemisphereradius / dist);
          contact[n].radius = - lohemisphereradius;
          contact[n].iwall = 0;
          contact[n].varflag = 1;
          n++;
        }
      }
    }

    // 3. High hemisphere contact (above hi)
    if (x[1] > hi && x[1] <= newhi) {
      dy = x[1] - center2;
      dist = sqrt(del1*del1 + del2*del2 + dy*dy);
      if (dist <= hihemisphereradius) {
        delta = hihemisphereradius - dist;
        if (delta < cutoff) {
          contact[n].r = delta;
          contact[n].delx = del1 * (1.0 - hihemisphereradius / dist);
          contact[n].dely = dy * (1.0 - hihemisphereradius / dist);
          contact[n].delz = del2 * (1.0 - hihemisphereradius / dist);
          contact[n].radius = - hihemisphereradius;
          contact[n].iwall = 0;
          contact[n].varflag = 1;
          n++;
        }
      }
    }

  //-----------------------------------------------------------------------
  // Capsule aligned along Z-axis
  //-----------------------------------------------------------------------
  } else {
    del1 = x[0] - c1;
    del2 = x[1] - c2;
    r = sqrt(del1 * del1 + del2 * del2);

    // 1. Cylinder contact (side wall)
    if (r <= radius && x[2] >= lo && x[2] <= hi) {
      delta = radius - r;
      if (delta < cutoff && r > 0.0) {
        contact[n].r = delta;
        contact[n].delx = del1 * (1.0 - radius / r);
        contact[n].dely = del2 * (1.0 - radius / r);
        contact[n].delz = 0.0;
        contact[n].radius = - 2.0 * radius;
        contact[n].iwall = 0;
        contact[n].varflag = 1;
        n++;
      }
    }

    // 2. Low hemisphere contact (below lo)
    if (x[2] < lo && x[2] >= newlo) {
      dz = x[2] - center1;
      dist = sqrt(del1*del1 + del2*del2 + dz*dz);
      if (dist <= lohemisphereradius) {
        delta = lohemisphereradius - dist;
        if (delta < cutoff) {
          contact[n].r = delta;
          contact[n].delx = del1 * (1.0 - lohemisphereradius / dist);
          contact[n].dely = del2 * (1.0 - lohemisphereradius / dist);
          contact[n].delz = dz * (1.0 - lohemisphereradius / dist);
          contact[n].radius = - lohemisphereradius;
          contact[n].iwall = 0;
          contact[n].varflag = 1;
          n++;
        }
      }
    }

    // 3. High hemisphere contact (above hi)
    if (x[2] > hi && x[2] <= newhi) {
      dz = x[2] - center2;
      dist = sqrt(del1*del1 + del2*del2 + dz*dz);
      if (dist <= hihemisphereradius) {
        delta = hihemisphereradius - dist;
        if (delta < cutoff) {
          contact[n].r = delta;
          contact[n].delx = del1 * (1.0 - hihemisphereradius / dist);
          contact[n].dely = del2 * (1.0 - hihemisphereradius / dist);
          contact[n].delz = dz * (1.0 - hihemisphereradius / dist);
          contact[n].radius = - hihemisphereradius;
          contact[n].iwall = 0;
          contact[n].varflag = 1;
          n++;
        }
      }
    }
  }

  return n;
}


/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from outer surface of cylinder
   no contact if inside (possible if called from union/intersect)
   delxyz = vector from nearest point on cylinder to x
------------------------------------------------------------------------- */

int RegCapsule::surface_exterior(double *x, double cutoff)
{
  double del1, del2, r;
  double dx, dist;
  double xp = 0.0, yp = 0.0, zp = 0.0;
  double d2, d2prev;
  double crad = 0.0;
  int varflag = 0;
  const double EPS = 1e-12;

  // ----------------------------------------------------------------------
  // Precompute hemisphere geometry
  // ----------------------------------------------------------------------
  double H1 = sqrt(fmax(EPS, lohemisphereradius*lohemisphereradius - radius*radius));
  double dH1 = lohemisphereradius - H1;
  double center1 = lo + H1;
  double newlo = lo - dH1;

  double H2 = sqrt(fmax(EPS, hihemisphereradius*hihemisphereradius - radius*radius));
  double dH2 = hihemisphereradius - H2;
  double center2 = hi - H2;
  double newhi = hi + dH2;

  d2prev = BIG;

  // ======================================================================
  // Capsule aligned along X-axis
  // ======================================================================
  if (axis == 'x') {
    del1 = x[1] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1*del1 + del2*del2);

    // Quick reject
    if (r >= radius + cutoff && x[0] > lo && x[0] < hi) return 0;
    if (x[0] < newlo - cutoff || x[0] > newhi + cutoff) return 0;

    // --- 1. Cylindrical wall ---
    if (r > radius && x[0] >= lo && x[0] <= hi) {
      xp = x[0];
      yp = c1 + del1 * radius / r;
      zp = c2 + del2 * radius / r;

      crad = 2.0 * radius;
      varflag = 1;

      add_contact(0, x, xp, yp, zp);
      contact[0].radius = crad;
      contact[0].varflag = varflag;
      contact[0].iwall = 0;
      if (contact[0].r < cutoff) return 1;
    }

    // --- 2. Lower hemisphere --- maybe multiple contacts possible, take closest - still shouldnt influence the top
    if (x[0] < lo && x[0] >= newlo) {
      dx = x[0] - center1;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist > lohemisphereradius) {
        double delta = dist - lohemisphereradius;
        if (delta < cutoff) {
          contact[0].r = delta;
          double scale = 1.0 - lohemisphereradius / dist;
          contact[0].delx = dx * scale;
          contact[0].dely = del1 * scale;
          contact[0].delz = del2 * scale;
          contact[0].radius = lohemisphereradius;
          contact[0].iwall = 0;
          contact[0].varflag = 1;
          return 1;
        }
      }
    }

    // --- 3. Upper hemisphere ---
    if (x[0] > hi && x[0] <= newhi) {
      dx = x[0] - center2;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist > hihemisphereradius) {
        double delta = dist - hihemisphereradius;
        if (delta < cutoff) {
          contact[0].r = delta;
          double scale = 1.0 - hihemisphereradius / dist;
          contact[0].delx = dx * scale;
          contact[0].dely = del1 * scale;
          contact[0].delz = del2 * scale;
          contact[0].radius = hihemisphereradius;
          contact[0].iwall = 0;
          contact[0].varflag = 1;
          return 1;
        }
      }
    }

    return 0;
  }

  // ======================================================================
  // Capsule aligned along Y-axis
  // ======================================================================
  else if (axis == 'y') {
    del1 = x[0] - c1;
    del2 = x[2] - c2;
    r = sqrt(del1*del1 + del2*del2);

    if (r >= radius + cutoff && x[1] > lo && x[1] < hi) return 0;
    if (x[1] < newlo - cutoff || x[1] > newhi + cutoff) return 0;

    // Cylindrical wall
    if (r > radius && x[1] >= lo && x[1] <= hi) {
      yp = x[1];
      xp = c1 + del1 * radius / r;
      zp = c2 + del2 * radius / r;

      crad = 2.0 * radius;
      varflag = 1;

      add_contact(0, x, xp, yp, zp);
      contact[0].radius = crad;
      contact[0].varflag = varflag;
      contact[0].iwall = 0;
      if (contact[0].r < cutoff) return 1;
    }

    // Lower hemisphere
    if (x[1] < lo && x[1] >= newlo) {
      dx = x[1] - center1;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist > lohemisphereradius) {
        double delta = dist - lohemisphereradius;
        if (delta < cutoff) {
          contact[0].r = delta;
          double scale = 1.0 - lohemisphereradius / dist;
          contact[0].delx = del1 * scale;
          contact[0].dely = dx * scale;
          contact[0].delz = del2 * scale;
          contact[0].radius = lohemisphereradius;
          contact[0].iwall = 0;
          contact[0].varflag = 1;
          return 1;
        }
      }
    }

    // Upper hemisphere
    if (x[1] > hi && x[1] <= newhi) {
      dx = x[1] - center2;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist > hihemisphereradius) {
        double delta = dist - hihemisphereradius;
        if (delta < cutoff) {
          contact[0].r = delta;
          double scale = 1.0 - hihemisphereradius / dist;
          contact[0].delx = del1 * scale;
          contact[0].dely = dx * scale;
          contact[0].delz = del2 * scale;
          contact[0].radius = hihemisphereradius;
          contact[0].iwall = 0;
          contact[0].varflag = 1;
          return 1;
        }
      }
    }

    return 0;
  }

  // ======================================================================
  // Capsule aligned along Z-axis
  // ======================================================================
  else {
    del1 = x[0] - c1;
    del2 = x[1] - c2;
    r = sqrt(del1*del1 + del2*del2);

    if (r >= radius + cutoff && x[2] > lo && x[2] < hi) return 0;
    if (x[2] < newlo - cutoff || x[2] > newhi + cutoff) return 0;

    // Cylindrical wall
    if (r > radius && x[2] >= lo && x[2] <= hi) {
      zp = x[2];
      xp = c1 + del1 * radius / r;
      yp = c2 + del2 * radius / r;

      crad = 2.0 * radius;
      varflag = 1;

      add_contact(0, x, xp, yp, zp);
      contact[0].radius = crad;
      contact[0].varflag = varflag;
      contact[0].iwall = 0;
      if (contact[0].r < cutoff) return 1;
    }

    // Lower hemisphere
    if (x[2] < lo && x[2] >= newlo) {
      dx = x[2] - center1;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist > lohemisphereradius) {
        double delta = dist - lohemisphereradius;
        if (delta < cutoff) {
          contact[0].r = delta;
          double scale = 1.0 - lohemisphereradius / dist;
          contact[0].delx = del1 * scale;
          contact[0].dely = del2 * scale;
          contact[0].delz = dx * scale;
          contact[0].radius = lohemisphereradius;
          contact[0].iwall = 0;
          contact[0].varflag = 1;
          return 1;
        }
      }
    }

    // Upper hemisphere
    if (x[2] > hi && x[2] <= newhi) {
      dx = x[2] - center2;
      dist = sqrt(del1*del1 + del2*del2 + dx*dx);
      if (dist > hihemisphereradius) {
        double delta = dist - hihemisphereradius;
        if (delta < cutoff) {
          contact[0].r = delta;
          double scale = 1.0 - hihemisphereradius / dist;
          contact[0].delx = del1 * scale;
          contact[0].dely = del2 * scale;
          contact[0].delz = dx * scale;
          contact[0].radius = hihemisphereradius;
          contact[0].iwall = 0;
          contact[0].varflag = 1;
          return 1;
        }
      }
    }

    return 0;
  }
}



/* ----------------------------------------------------------------------
   Set values needed to calculate velocity due to shape changes.
   These values do not depend on the contact, so this function is
   called once per timestep by fix/wall/gran/region.

------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Update geometric parameters for velocity calculation of dynamic capsule
------------------------------------------------------------------------- */

void RegCapsule::set_velocity_shape()
{
  // 1. Compute capsule center depending on the main axis
  if (axis == 'x') {
    xcenter[0] = 0.0;
    xcenter[1] = c1;
    xcenter[2] = c2;
  } else if (axis == 'y') {
    xcenter[0] = c1;
    xcenter[1] = 0.0;
    xcenter[2] = c2;
  } else { // 'z'
    xcenter[0] = c1;
    xcenter[1] = c2;
    xcenter[2] = 0.0;
  }

  forward_transform(xcenter[0], xcenter[1], xcenter[2]);
  const double EPS = 1e-12;

  // 2. Compute hemisphere geometry (local coords)
  H1 = sqrt(fmax(EPS, lohemisphereradius*lohemisphereradius - radius*radius));
  dH1 = lohemisphereradius - H1;
  center1 = lo + H1;
  newlo = lo - dH1;

  H2 = sqrt(fmax(EPS, hihemisphereradius*hihemisphereradius - radius*radius));
  dH2 = hihemisphereradius - H2;
  center2 = hi - H2;
  newhi = hi + dH2;

  // Transform centers to simulation coords (if rotated)
  double cx1 = 0.0, cy1 = 0.0, cz1 = 0.0;
  double cx2 = 0.0, cy2 = 0.0, cz2 = 0.0;

  if (axis == 'x') {
    cx1 = center1; cy1 = c1; cz1 = c2;
    cx2 = center2; cy2 = c1; cz2 = c2;
  } else if (axis == 'y') {
    cx1 = c1; cy1 = center1; cz1 = c2;
    cx2 = c1; cy2 = center2; cz2 = c2;
  } else { // 'z'
    cx1 = c1; cy1 = c2; cz1 = center1;
    cx2 = c1; cy2 = c2; cz2 = center2;
  }

  forward_transform(cx1, cy1, cz1);
  forward_transform(cx2, cy2, cz2);

  // --------------------------------------------------------------------
  // 3. Store previous values for velocity computation
  // --------------------------------------------------------------------

  // For the first step, initialize previous values directly
  if (update->ntimestep == 0) {
    rprev         = radius;
    rlo_prev      = lohemisphereradius;
    rhi_prev      = hihemisphereradius;
    prev_center1  = center1;
    prev_center2  = center2;
    prev_c1       = c1;
    prev_c2       = c2;
  } else {
    // otherwise, compute previous from stored values
    rprev         = prev[4];       // cylinder radius from last step
    rlo_prev      = prev_loh;      // previous lower hemisphere radius
    rhi_prev      = prev_hih;      // previous upper hemisphere radius
    prev_center1  = prev_ctr1;
    prev_center2  = prev_ctr2;
    prev_c1       = prev_cc1;
    prev_c2       = prev_cc2;
  }

  // --------------------------------------------------------------------
  // 4. Update stored values for next timestep
  // --------------------------------------------------------------------
  prev[4]   = radius;               // existing slot used by LAMMPS (main radius)
  prev_loh  = lohemisphereradius;
  prev_hih  = hihemisphereradius;
  prev_ctr1 = center1;
  prev_ctr2 = center2;
  prev_cc1  = c1;
  prev_cc2  = c2;
}



/* ----------------------------------------------------------------------
   add velocity due to shape change to wall velocity
------------------------------------------------------------------------- */

void RegCapsule::velocity_contact_shape(double *vwall, double *xc)
{
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double del1, del2, dist, growth;

  if (axis == 'x') {
    del1 = xc[1] - c1;
    del2 = xc[2] - c2;
    double r = sqrt(del1*del1 + del2*del2);

    if (xc[0] >= lo && xc[0] <= hi) {
      // cylindrical section
      if (r > 0.0) {
        growth = 1.0 - rprev / radius;
        dx = 0.0;
        dy = del1 * growth;
        dz = del2 * growth;
      }
    } else if (xc[0] < lo && xc[0] >= newlo) {
      // lower hemisphere
      dist = sqrt(del1*del1 + del2*del2 + (xc[0]-center1)*(xc[0]-center1));
      if (dist > 0.0) {
        growth = 1.0 - rlo_prev / lohemisphereradius;
        dx = (xc[0] - center1) * growth;
        dy = del1 * growth;
        dz = del2 * growth;
      }
    } else if (xc[0] > hi && xc[0] <= newhi) {
      // upper hemisphere
      dist = sqrt(del1*del1 + del2*del2 + (xc[0]-center2)*(xc[0]-center2));
      if (dist > 0.0) {
        growth = 1.0 - rhi_prev / hihemisphereradius;
        dx = (xc[0] - center2) * growth;
        dy = del1 * growth;
        dz = del2 * growth;
      }
    }
  } else if (axis == 'y') {
    del1 = xc[0] - c1;
    del2 = xc[2] - c2;
    double r = sqrt(del1*del1 + del2*del2);

    if (xc[1] >= lo && xc[1] <= hi) {
      // cylinder
      if (r > 0.0) {
        growth = 1.0 - rprev / radius;
        dx = del1 * growth;
        dy = 0.0;
        dz = del2 * growth;
      }
    } else if (xc[1] < lo && xc[1] >= newlo) {
      // lower hemisphere
      dist = sqrt(del1*del1 + del2*del2 + (xc[1]-center1)*(xc[1]-center1));
      if (dist > 0.0) {
        growth = 1.0 - rlo_prev / lohemisphereradius;
        dx = del1 * growth;
        dy = (xc[1] - center1) * growth;
        dz = del2 * growth;
      }
    } else if (xc[1] > hi && xc[1] <= newhi) {
      // upper hemisphere
      dist = sqrt(del1*del1 + del2*del2 + (xc[1]-center2)*(xc[1]-center2));
      if (dist > 0.0) {
        growth = 1.0 - rhi_prev / hihemisphereradius;
        dx = del1 * growth;
        dy = (xc[1] - center2) * growth;
        dz = del2 * growth;
      }
    }
  } else { // axis == 'z'
    del1 = xc[0] - c1;
    del2 = xc[1] - c2;
    double r = sqrt(del1*del1 + del2*del2);

    if (xc[2] >= lo && xc[2] <= hi) {
      // cylinder
      if (r > 0.0) {
        growth = 1.0 - rprev / radius;
        dx = del1 * growth;
        dy = del2 * growth;
        dz = 0.0;
      }
    } else if (xc[2] < lo && xc[2] >= newlo) {
      // lower hemisphere
      dist = sqrt(del1*del1 + del2*del2 + (xc[2]-center1)*(xc[2]-center1));
      if (dist > 0.0) {
        growth = 1.0 - rlo_prev / lohemisphereradius;
        dx = del1 * growth;
        dy = del2 * growth;
        dz = (xc[2] - center1) * growth;
      }
    } else if (xc[2] > hi && xc[2] <= newhi) {
      // upper hemisphere
      dist = sqrt(del1*del1 + del2*del2 + (xc[2]-center2)*(xc[2]-center2));
      if (dist > 0.0) {
        growth = 1.0 - rhi_prev / hihemisphereradius;
        dx = del1 * growth;
        dy = del2 * growth;
        dz = (xc[2] - center2) * growth;
      }
    }
  }

  vwall[0] += dx / update->dt;
  vwall[1] += dy / update->dt;
  vwall[2] += dz / update->dt;
}
