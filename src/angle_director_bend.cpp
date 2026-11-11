// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   angle_style director/bend
   angle_coeff N K_b delta0 alpha0

   Bend of an intrinsically curved filament whose beads carry a unit
   director e (atom->mu, the side of the membrane-binding face; rotated by
   fix nve/sphere update dipole). For the angle 1-2-3 with unit bond vectors
   t1 = (x2-x1)/|..|, t2 = (x3-x2)/|..|, bend b = t2 - t1, mean tangent
   t = (t1+t2)/|..| and q = director of atom 2 perpendicular to t (unit):

     E = K_b |b - delta0 (cos(alpha0) q + sin(alpha0) t x q)|^2

   delta0 = |b| at the minimum (= 2 sin(d/2), d = 180 deg - theta), alpha0 in
   degrees (0: the filament curves towards its director). A single smooth
   well in the bond vectors for any delta0, also near straight. The other
   terms of the model: perpendicularity and twist of the directors in
   bond_style director, binding of the director to a surface normal in fix
   director/bind. Atom 2 without director: E = K_b |b|^2. Forces on the
   beads, torque -e x dE/de on the director of atom 2. Needs atom attributes
   mu and torque (atom style dipole + sphere).
------------------------------------------------------------------------- */

#include "angle_director_bend.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using MathConst::DEG2RAD;
using MathConst::RAD2DEG;
using MathConst::MY_PI;
using namespace MathExtra;

static constexpr double SMALL = 1.0e-8;

/* ---------------------------------------------------------------------- */

// adjoint of u = v/|v|: g_v += (g_u - (g_u.u) u)/|v|
static inline void unit_adjoint(const double *g_u, const double *u, double len, double *g_v)
{
  const double gu = dot3(g_u, u);
  for (int k = 0; k < 3; k++) g_v[k] += (g_u[k] - gu * u[k]) / len;
}

// adjoint of p = v - (v.t) t: g_v += g_p - (g_p.t) t, g_t += -((v.t) g_p + (g_p.t) v)
static inline void proj_adjoint(const double *g_p, const double *v, const double *t, double *g_v,
                                double *g_t)
{
  const double vt = dot3(v, t), pt = dot3(g_p, t);
  for (int k = 0; k < 3; k++) {
    g_v[k] += g_p[k] - pt * t[k];
    g_t[k] -= vt * g_p[k] + pt * v[k];
  }
}

/* ---------------------------------------------------------------------- */

AngleDirectorBend::AngleDirectorBend(LAMMPS *_lmp) :
    Angle(_lmp), k_b(nullptr), delta0(nullptr), alpha0(nullptr)
{
  born_matrix_enable = 0;
}

/* ---------------------------------------------------------------------- */

AngleDirectorBend::~AngleDirectorBend()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k_b);
    memory->destroy(delta0);
    memory->destroy(alpha0);
  }
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Angle style director/bend needs atom attributes mu and torque "
               "(atom style with dipole and sphere)");
}

/* ---------------------------------------------------------------------- */

double AngleDirectorBend::bend(int type, const double *r1, const double *r2, const double *e2,
                               double *f1, double *f3, double *tq2) const
{
  for (int d = 0; d < 3; d++) f1[d] = f3[d] = tq2[d] = 0.0;
  const double len1 = len3(r1), len2 = len3(r2);
  if (len1 == 0.0 || len2 == 0.0) return 0.0;

  double t1[3], t2[3], b[3], s[3];
  scale3(1.0 / len1, r1, t1);
  scale3(1.0 / len2, r2, t2);
  sub3(t2, t1, b);
  add3(t1, t2, s);
  const double slen = len3(s);

  double energy = 0.0;
  double g_t1[3] = {0.0, 0.0, 0.0}, g_t2[3] = {0.0, 0.0, 0.0};
  double g_e2[3] = {0.0, 0.0, 0.0};
  double g_b[3] = {0.0, 0.0, 0.0}, g_ts[3] = {0.0, 0.0, 0.0};

  double ts[3] = {0.0, 0.0, 0.0}, p2[3] = {0.0, 0.0, 0.0};
  double plen = 0.0;
  if (e2 && slen >= SMALL) {
    scale3(1.0 / slen, s, ts);
    const double cp = dot3(e2, ts);
    for (int d = 0; d < 3; d++) p2[d] = e2[d] - cp * ts[d];
    plen = len3(p2);
  }

  if (plen < SMALL) {
    // no director (or director along the backbone, or folded back): prefer straight
    energy = k_b[type] * lensq3(b);
    scale3(2.0 * k_b[type], b, g_b);
  } else {
    // intrinsic bend towards the face
    double q[3], a[3], target[3], diff[3];
    scale3(1.0 / plen, p2, q);
    cross3(ts, q, a);
    const double ca = delta0[type] * cos(alpha0[type]), sa = delta0[type] * sin(alpha0[type]);
    for (int d = 0; d < 3; d++) {
      target[d] = ca * q[d] + sa * a[d];
      diff[d] = b[d] - target[d];
    }
    energy = k_b[type] * lensq3(diff);
    scale3(2.0 * k_b[type], diff, g_b);
    double g_q[3], g_a[3], tmp[3];
    scale3(-ca, g_b, g_q);    // g_target = -g_b
    scale3(-sa, g_b, g_a);
    // a = ts x q
    cross3(q, g_a, tmp);
    add3(g_ts, tmp, g_ts);
    cross3(g_a, ts, tmp);
    add3(g_q, tmp, g_q);
    double g_p2[3] = {0.0, 0.0, 0.0};
    unit_adjoint(g_q, q, plen, g_p2);
    proj_adjoint(g_p2, e2, ts, g_e2, g_ts);

    // ts = s/|s|, s = t1 + t2
    double g_s[3] = {0.0, 0.0, 0.0};
    unit_adjoint(g_ts, ts, slen, g_s);
    add3(g_t1, g_s, g_t1);
    add3(g_t2, g_s, g_t2);
  }

  // b = t2 - t1
  sub3(g_t1, g_b, g_t1);
  add3(g_t2, g_b, g_t2);

  // t = r/|r|; r1 = x2 - x1 -> F1 = +dE/dr1, r2 = x3 - x2 -> F3 = -dE/dr2
  double g_r1[3] = {0.0, 0.0, 0.0}, g_r2[3] = {0.0, 0.0, 0.0};
  unit_adjoint(g_t1, t1, len1, g_r1);
  unit_adjoint(g_t2, t2, len2, g_r2);
  for (int d = 0; d < 3; d++) {
    f1[d] = g_r1[d];
    f3[d] = -g_r2[d];
  }

  // torque on the director: -e x dE/de
  if (e2) cross3(g_e2, e2, tq2);
  return energy;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::compute(int eflag, int vflag)
{
  double r1[3], r2[3], f1[3], f3[3], tq2[3], e2[3];
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int **anglelist = neighbor->anglelist;
  const int nanglelist = neighbor->nanglelist;
  const int nlocal = atom->nlocal;
  const int newton_bond = force->newton_bond;

  for (int n = 0; n < nanglelist; n++) {
    const int i1 = anglelist[n][0], i2 = anglelist[n][1], i3 = anglelist[n][2];
    const int type = anglelist[n][3];

    // the topology lists hold the closest images
    for (int d = 0; d < 3; d++) {
      r1[d] = x[i2][d] - x[i1][d];
      r2[d] = x[i3][d] - x[i2][d];
    }
    const double mlen = len3(mu[i2]);
    const double *ep = nullptr;
    if (mlen > 0.0) {
      scale3(1.0 / mlen, mu[i2], e2);
      ep = e2;
    }
    const double eangle = bend(type, r1, r2, ep, f1, f3, tq2);

    if (newton_bond || i1 < nlocal)
      for (int d = 0; d < 3; d++) f[i1][d] += f1[d];
    if (newton_bond || i2 < nlocal)
      for (int d = 0; d < 3; d++) {
        f[i2][d] -= f1[d] + f3[d];
        torque[i2][d] += tq2[d];
      }
    if (newton_bond || i3 < nlocal)
      for (int d = 0; d < 3; d++) f[i3][d] += f3[d];

    // ev_tally takes del1 = x1 - x2 = -r1, del2 = x3 - x2 = r2
    if (evflag)
      ev_tally(i1, i2, i3, nlocal, newton_bond, eangle, f1, f3, -r1[0], -r1[1], -r1[2], r2[0],
               r2[1], r2[2]);
  }
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::allocate()
{
  allocated = 1;
  const int np1 = atom->nangletypes + 1;
  memory->create(k_b, np1, "angle:k_b");
  memory->create(delta0, np1, "angle:delta0");
  memory->create(alpha0, np1, "angle:alpha0");
  memory->create(setflag, np1, "angle:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR, "Incorrect args for angle coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nangletypes, ilo, ihi, error);
  const double k_one = utils::numeric(FLERR, arg[1], false, lmp);
  const double delta_one = utils::numeric(FLERR, arg[2], false, lmp);
  const double alpha_one = utils::numeric(FLERR, arg[3], false, lmp);
  if (delta_one < 0.0 || delta_one > 2.0)
    error->all(FLERR, "Angle style director/bend: delta0 must be in [0, 2] (|b| = 2 sin(d/2))");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k_b[i] = k_one;
    delta0[i] = delta_one;
    alpha0[i] = alpha_one * DEG2RAD;
    setflag[i] = 1;
    count++;
  }
  if (count == 0) error->all(FLERR, "Incorrect args for angle coefficients" + utils::errorurl(21));
}

/* ---------------------------------------------------------------------- */

double AngleDirectorBend::equilibrium_angle(int i)
{
  return MY_PI - 2.0 * asin(0.5 * delta0[i]);
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::write_restart(FILE *fp)
{
  const int n = atom->nangletypes;
  for (double *p : {k_b, delta0, alpha0}) fwrite(&p[1], sizeof(double), n, fp);
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::read_restart(FILE *fp)
{
  allocate();
  const int n = atom->nangletypes;
  for (double *p : {k_b, delta0, alpha0}) {
    if (comm->me == 0) utils::sfread(FLERR, &p[1], sizeof(double), n, fp, nullptr, error);
    MPI_Bcast(&p[1], n, MPI_DOUBLE, 0, world);
  }
  for (int i = 1; i <= n; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp, "%d %g %g %g\n", i, k_b[i], delta0[i], RAD2DEG * alpha0[i]);
}

/* ---------------------------------------------------------------------- */

double AngleDirectorBend::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;
  double **mu = atom->mu;
  double r1[3], r2[3], f1[3], f3[3], tq2[3], e2[3];
  for (int d = 0; d < 3; d++) {
    r1[d] = x[i2][d] - x[i1][d];
    r2[d] = x[i3][d] - x[i2][d];
  }
  domain->minimum_image(FLERR, r1);
  domain->minimum_image(FLERR, r2);
  const double mlen = len3(mu[i2]);
  const double *ep = nullptr;
  if (mlen > 0.0) {
    scale3(1.0 / mlen, mu[i2], e2);
    ep = e2;
  }
  return bend(type, r1, r2, ep, f1, f3, tq2);
}

/* ---------------------------------------------------------------------- */

void *AngleDirectorBend::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "k_b") == 0) return (void *) k_b;
  if (strcmp(str, "delta0") == 0) return (void *) delta0;
  return nullptr;
}
