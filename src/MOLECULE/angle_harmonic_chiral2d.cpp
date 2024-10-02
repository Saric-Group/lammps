// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "angle_harmonic_chiral2d.h"

#include <cmath>
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"


using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.001

/* ---------------------------------------------------------------------- */

AngleHarmonicChiral2D::AngleHarmonicChiral2D(LAMMPS *lmp) : Angle(lmp)
{
  k = nullptr;
  theta0 = nullptr;
}

/* ---------------------------------------------------------------------- */

AngleHarmonicChiral2D::~AngleHarmonicChiral2D()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(theta0);
  }
}

/* ---------------------------------------------------------------------- */

void AngleHarmonicChiral2D::compute(int eflag, int vflag)
{
  int i1,i2,i3,n,type;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle,f1[3],f3[3];
  double dtheta,tk;
  double rsq1,rsq2,r1,r2,c,s,a,a11,a12,a22;

  eangle = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];

    c = -delx1*delx2 - dely1*dely2;
    s = delx1*dely2 - dely1*delx2;

    tk = k[type] * (atan2(s,c) - theta0[type]);

    f1[0] = tk * (c*dely2+s*delx2) / (c*c+s*s);
    f1[1] = tk * (s*dely2-c*delx2) / (c*c+s*s);
    f3[0] = tk * (s*delx1-c*dely1) / (c*c+s*s);
    f3[1] = tk * (s*dely1+c*delx1) / (c*c+s*s);

    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] -= f1[0];
      f[i1][1] -= f1[1];
      f[i1][2] -= f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f1[0] + f3[0];
      f[i2][1] += f1[1] + f3[1];
      f[i2][2] += f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] -= f3[0];
      f[i3][1] -= f3[1];
      f[i3][2] -= f3[2];
    }
    
    if (eflag) eangle = tk*dtheta;
    if (evflag) ev_tally(i1,i2,i3,nlocal,newton_bond,eangle,f1,f3,
                         delx1,dely1,delz1,delx2,dely2,delz2);
  }
}

/* ---------------------------------------------------------------------- */

void AngleHarmonicChiral2D::allocate()
{
  allocated = 1;
  int n = atom->nangletypes;

  memory->create(k,n+1,"angle:k");
  memory->create(theta0,n+1,"angle:theta0");

  memory->create(setflag,n+1,"angle:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void AngleHarmonicChiral2D::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nangletypes,ilo,ihi,error);

  double k_one = utils::numeric(FLERR,arg[1],false,lmp);
  double theta0_one = utils::numeric(FLERR,arg[2],false,lmp);

  // convert theta0 from degrees to radians

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    theta0[i] = theta0_one/180.0 * MY_PI;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleHarmonicChiral2D::equilibrium_angle(int i)
{
  return theta0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleHarmonicChiral2D::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nangletypes,fp);
  fwrite(&theta0[1],sizeof(double),atom->nangletypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleHarmonicChiral2D::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nangletypes,fp,nullptr,error);
    utils::sfread(FLERR,&theta0[1],sizeof(double),atom->nangletypes,fp,nullptr,error);
  }
  MPI_Bcast(&k[1],atom->nangletypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&theta0[1],atom->nangletypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleHarmonicChiral2D::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp,"%d %g %g\n",i,k[i],theta0[i]/MY_PI*180.0);
}

/* ---------------------------------------------------------------------- */

double AngleHarmonicChiral2D::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1,dely1,delz1);
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2,dely2,delz2);
  double r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

  double c = delx1*delx2 + dely1*dely2;
  double s = delx1*dely2 - dely1*delx2;
  c /= r1*r2;
  s /= r1*r2;

  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  if (s > 1.0) s = 1.0;
  if (s < -1.0) s = -1.0;

  //if (s < SMALL) s = SMALL;

  double dtheta = atan2(s,-c) - theta0[type];
  double tk = k[type] * dtheta;
  return tk*dtheta;
}
