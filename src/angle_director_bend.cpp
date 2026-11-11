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
   angle_style director/bend d2_name|none
   angle_coeff N K_b delta0 alpha0 K_tw tau0 K_perp eps_bind

   Intrinsically curved and twisted filament whose beads carry a unit
   director e (atom->mu, the side of the membrane-binding face; rotated by
   fix nve/sphere update dipole). For the angle 1-2-3 with unit bond vectors
   t1 = (x2-x1)/|..|, t2 = (x3-x2)/|..|, bend b = t2 - t1, mean tangent
   t = (t1+t2)/|..| and q = director of atom 2 perpendicular to t (unit):

     bend     K_b |b - delta0 (cos(alpha0) q + sin(alpha0) t x q)|^2
     twist    1/2 K_tw |P2 - R_t1(tau0) P1|^2 + 1/2 K_tw |P3 - R_t2(tau0) P2|^2
              (Pj = director j perpendicular to the bond, unit; R_t(a) = rotation
              by a about t; an interior bond belongs to two angles, so its pair
              counts fully, the first and last bond of a chain by half)
     perp     K_perp/4 [(e1.t1)^2 + (e2.t1)^2 + (e2.t2)^2 + (e3.t2)^2]
              (per bond and bead; an interior bead gets K_perp/2 [(e.t_left)^2 +
              (e.t_right)^2] ~ K_perp (e.t)^2 from its three angles, an end bead
              K_perp/4 (e.t)^2, so the directors of the ends stay perpendicular too)
     binding  eps_bind (1 - e2 . n^)   (n^ = normal of atom 2 from d2_name)

   delta0 = |b| at the minimum (= 2 sin(d/2), d = 180 deg - theta), alpha0 and
   tau0 in degrees (tau0 per bond, right-handed about the tail -> head
   direction). The bend is a single smooth well in the bond vectors for any
   delta0, also near straight. Only the binding term refers to the surface:
   with eps_bind = 0 the energy is invariant under rotating the filament
   with its directors about its axis.

   Atom 2 without director: E = K_b |b|^2 only; a neighbour without director:
   its twist pair is skipped; no normal (d2_name none, |n| < 1e-6): no binding.
   Forces on the beads, torques -e x dE/de on the directors. The normal is
   taken as given (no force on what defines it). Needs atom attributes mu
   and torque (atom style dipole + sphere); with newton_bond on the angle is
   computed by the owner of atom 2, whose normal is local.
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
static constexpr double NORMAL_SMALL = 1.0e-6;

/* ---------------------------------------------------------------------- */

// v += s * w
static inline void axpy3(double s, const double *w, double *v)
{
  v[0] += s * w[0];
  v[1] += s * w[1];
  v[2] += s * w[2];
}

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
    Angle(_lmp), k_b(nullptr), delta0(nullptr), alpha0(nullptr), k_tw(nullptr), tau0(nullptr),
    k_perp(nullptr), eps_bind(nullptr)
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
    memory->destroy(k_tw);
    memory->destroy(tau0);
    memory->destroy(k_perp);
    memory->destroy(eps_bind);
  }
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal angle_style director/bend command: needs d2_<name> or none");
  std::string name = arg[0];
  if (name == "none") normal_name.clear();
  else if (utils::strmatch(name, "^d2_")) normal_name = name.substr(3);
  else error->all(FLERR, "Angle style director/bend needs a per-atom array d2_<name> or none, got {}", name);
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Angle style director/bend needs atom attributes mu and torque "
               "(atom style with dipole and sphere)");
  if (normal_name.empty()) return;
  int flag, cols, ghost;
  const int index = atom->find_custom_ghost(normal_name.c_str(), flag, cols, ghost);
  if (index < 0 || flag != 1 || cols != 3)
    error->all(FLERR, "Angle style director/bend needs a 3-column per-atom array d2_{}", normal_name);
  if (!force->newton_bond && !ghost)
    error->all(FLERR, "Angle style director/bend with newton_bond off needs d2_{} on ghost atoms "
               "(fix property/atom ... ghost yes)", normal_name);
}

/* ---------------------------------------------------------------------- */

double **AngleDirectorBend::normal_array()
{
  if (normal_name.empty()) return nullptr;
  int flag, cols;
  const int index = atom->find_custom(normal_name.c_str(), flag, cols);
  if (index < 0 || flag != 1 || cols != 3)
    error->one(FLERR, "Angle style director/bend: per-atom array d2_{} not found", normal_name);
  return atom->darray[index];
}

/* ----------------------------------------------------------------------
   1/2 K_tw |Pk - R_t(tau0) Pj|^2 for the bond j -> k with unit tangent t;
   adds the gradients with respect to ej, ek and t
------------------------------------------------------------------------- */

double AngleDirectorBend::twist_pair(int type, const double *ej, const double *ek, const double *t,
                                     double *g_ej, double *g_ek, double *g_t) const
{
  if (k_tw[type] == 0.0) return 0.0;
  const double cj = dot3(ej, t), ck = dot3(ek, t);
  double pj[3], pk[3];
  for (int d = 0; d < 3; d++) {
    pj[d] = ej[d] - cj * t[d];
    pk[d] = ek[d] - ck * t[d];
  }
  const double lj = len3(pj), lk = len3(pk);
  if (lj < SMALL || lk < SMALL) return 0.0;
  double Pj[3], Pk[3];
  scale3(1.0 / lj, pj, Pj);
  scale3(1.0 / lk, pk, Pk);

  // Q = R_t(tau) Pj = Pj c + (t x Pj) s + t (t.Pj)(1 - c)
  const double c = cos(tau0[type]), s = sin(tau0[type]);
  const double tP = dot3(t, Pj);
  double txP[3], Q[3], diff[3];
  cross3(t, Pj, txP);
  for (int d = 0; d < 3; d++) {
    Q[d] = Pj[d] * c + txP[d] * s + t[d] * tP * (1.0 - c);
    diff[d] = Pk[d] - Q[d];
  }
  const double energy = 0.5 * k_tw[type] * lensq3(diff);

  double g_Pk[3], g_Q[3];
  scale3(k_tw[type], diff, g_Pk);
  scale3(-k_tw[type], diff, g_Q);

  // adjoint of the rotation
  const double tg = dot3(t, g_Q);
  double gxt[3], Pxg[3], g_Pj[3];
  cross3(g_Q, t, gxt);
  cross3(Pj, g_Q, Pxg);
  for (int d = 0; d < 3; d++) {
    g_Pj[d] = g_Q[d] * c + gxt[d] * s + t[d] * tg * (1.0 - c);
    g_t[d] += s * Pxg[d] + (1.0 - c) * (tP * g_Q[d] + tg * Pj[d]);
  }

  double g_pj[3] = {0.0, 0.0, 0.0}, g_pk[3] = {0.0, 0.0, 0.0};
  unit_adjoint(g_Pj, Pj, lj, g_pj);
  unit_adjoint(g_Pk, Pk, lk, g_pk);
  proj_adjoint(g_pj, ej, t, g_ej, g_t);
  proj_adjoint(g_pk, ek, t, g_ek, g_t);
  return energy;
}

/* ---------------------------------------------------------------------- */

double AngleDirectorBend::bend(int type, const double *r1, const double *r2, const double *e1,
                               const double *e2, const double *e3, const double *normal,
                               double *f1, double *f3, double *tq1, double *tq2, double *tq3) const
{
  for (int d = 0; d < 3; d++) f1[d] = f3[d] = tq1[d] = tq2[d] = tq3[d] = 0.0;
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
  double g_e1[3] = {0.0, 0.0, 0.0}, g_e2[3] = {0.0, 0.0, 0.0}, g_e3[3] = {0.0, 0.0, 0.0};
  double g_b[3] = {0.0, 0.0, 0.0}, g_ts[3] = {0.0, 0.0, 0.0};

  if (!e2 || slen < SMALL) {
    // no director (or folded back): prefer straight
    energy += k_b[type] * lensq3(b);
    scale3(2.0 * k_b[type], b, g_b);
  } else {
    double ts[3];
    scale3(1.0 / slen, s, ts);
    const double cp = dot3(e2, ts);

    // intrinsic bend towards the face
    double p2[3];
    for (int d = 0; d < 3; d++) p2[d] = e2[d] - cp * ts[d];
    const double plen = len3(p2);
    if (plen < SMALL) {
      energy += k_b[type] * lensq3(b);
      scale3(2.0 * k_b[type], b, g_b);
    } else {
      double q[3], a[3], target[3], diff[3];
      scale3(1.0 / plen, p2, q);
      cross3(ts, q, a);
      const double ca = delta0[type] * cos(alpha0[type]), sa = delta0[type] * sin(alpha0[type]);
      for (int d = 0; d < 3; d++) {
        target[d] = ca * q[d] + sa * a[d];
        diff[d] = b[d] - target[d];
      }
      energy += k_b[type] * lensq3(diff);
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
    }

    // binding of the face to the surface
    if (normal && eps_bind[type] != 0.0) {
      const double nlen = len3(normal);
      if (nlen >= NORMAL_SMALL) {
        energy += eps_bind[type] * (1.0 - dot3(e2, normal) / nlen);
        axpy3(-eps_bind[type] / nlen, normal, g_e2);
      }
    }

    // ts = s/|s|, s = t1 + t2
    double g_s[3] = {0.0, 0.0, 0.0};
    unit_adjoint(g_ts, ts, slen, g_s);
    add3(g_t1, g_s, g_t1);
    add3(g_t2, g_s, g_t2);

    // perpendicularity of the directors of both atoms of each bond (a quarter per angle, see above)
    if (k_perp[type] != 0.0) {
      const double w = 0.25 * k_perp[type];
      const double *ee[4] = {e1, e2, e2, e3};
      double *ge[4] = {g_e1, g_e2, g_e2, g_e3};
      const double *tt[4] = {t1, t1, t2, t2};
      double *gt[4] = {g_t1, g_t1, g_t2, g_t2};
      for (int m = 0; m < 4; m++) {
        if (!ee[m]) continue;
        const double c = dot3(ee[m], tt[m]);
        energy += w * c * c;
        axpy3(2.0 * w * c, tt[m], ge[m]);
        axpy3(2.0 * w * c, ee[m], gt[m]);
      }
    }

    // twist of the two bonds of this angle
    if (e1) energy += twist_pair(type, e1, e2, t1, g_e1, g_e2, g_t1);
    if (e3) energy += twist_pair(type, e2, e3, t2, g_e2, g_e3, g_t2);
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

  // torque on a director e: -e x dE/de
  if (e1) cross3(g_e1, e1, tq1);
  if (e2) cross3(g_e2, e2, tq2);
  if (e3) cross3(g_e3, e3, tq3);
  return energy;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::compute(int eflag, int vflag)
{
  double r1[3], r2[3], f1[3], f3[3], tq1[3], tq2[3], tq3[3];
  double e[3][3];
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  double **normal = normal_array();
  int **anglelist = neighbor->anglelist;
  const int nanglelist = neighbor->nanglelist;
  const int nlocal = atom->nlocal;
  const int newton_bond = force->newton_bond;

  for (int n = 0; n < nanglelist; n++) {
    const int ia[3] = {anglelist[n][0], anglelist[n][1], anglelist[n][2]};
    const int type = anglelist[n][3];

    // the topology lists hold the closest images
    for (int d = 0; d < 3; d++) {
      r1[d] = x[ia[1]][d] - x[ia[0]][d];
      r2[d] = x[ia[2]][d] - x[ia[1]][d];
    }
    const double *ep[3];
    for (int m = 0; m < 3; m++) {
      const double mlen = len3(mu[ia[m]]);
      if (mlen > 0.0) {
        scale3(1.0 / mlen, mu[ia[m]], e[m]);
        ep[m] = e[m];
      } else ep[m] = nullptr;
    }

    const double eangle = bend(type, r1, r2, ep[0], ep[1], ep[2], normal ? normal[ia[1]] : nullptr,
                               f1, f3, tq1, tq2, tq3);
    const double *tq[3] = {tq1, tq2, tq3};

    for (int m = 0; m < 3; m++) {
      const int i = ia[m];
      if (!newton_bond && i >= nlocal) continue;
      for (int d = 0; d < 3; d++) {
        if (m == 0) f[i][d] += f1[d];
        else if (m == 1) f[i][d] -= f1[d] + f3[d];
        else f[i][d] += f3[d];
        torque[i][d] += tq[m][d];
      }
    }

    // ev_tally takes del1 = x1 - x2 = -r1, del2 = x3 - x2 = r2
    if (evflag)
      ev_tally(ia[0], ia[1], ia[2], nlocal, newton_bond, eangle, f1, f3, -r1[0], -r1[1], -r1[2],
               r2[0], r2[1], r2[2]);
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
  memory->create(k_tw, np1, "angle:k_tw");
  memory->create(tau0, np1, "angle:tau0");
  memory->create(k_perp, np1, "angle:k_perp");
  memory->create(eps_bind, np1, "angle:eps_bind");
  memory->create(setflag, np1, "angle:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::coeff(int narg, char **arg)
{
  if (narg != 8) error->all(FLERR, "Incorrect args for angle coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nangletypes, ilo, ihi, error);
  const double v[7] = {utils::numeric(FLERR, arg[1], false, lmp), utils::numeric(FLERR, arg[2], false, lmp),
                       utils::numeric(FLERR, arg[3], false, lmp), utils::numeric(FLERR, arg[4], false, lmp),
                       utils::numeric(FLERR, arg[5], false, lmp), utils::numeric(FLERR, arg[6], false, lmp),
                       utils::numeric(FLERR, arg[7], false, lmp)};
  if (v[1] < 0.0 || v[1] > 2.0)
    error->all(FLERR, "Angle style director/bend: delta0 must be in [0, 2] (|b| = 2 sin(d/2))");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k_b[i] = v[0];
    delta0[i] = v[1];
    alpha0[i] = v[2] * DEG2RAD;
    k_tw[i] = v[3];
    tau0[i] = v[4] * DEG2RAD;
    k_perp[i] = v[5];
    eps_bind[i] = v[6];
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
  for (double *p : {k_b, delta0, alpha0, k_tw, tau0, k_perp, eps_bind}) fwrite(&p[1], sizeof(double), n, fp);
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::read_restart(FILE *fp)
{
  allocate();
  const int n = atom->nangletypes;
  for (double *p : {k_b, delta0, alpha0, k_tw, tau0, k_perp, eps_bind}) {
    if (comm->me == 0) utils::sfread(FLERR, &p[1], sizeof(double), n, fp, nullptr, error);
    MPI_Bcast(&p[1], n, MPI_DOUBLE, 0, world);
  }
  for (int i = 1; i <= n; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::write_restart_settings(FILE *fp)
{
  const int len = normal_name.size() + 1;
  fwrite(&len, sizeof(int), 1, fp);
  fwrite(normal_name.c_str(), sizeof(char), len, fp);
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::read_restart_settings(FILE *fp)
{
  int len = 0;
  if (comm->me == 0) utils::sfread(FLERR, &len, sizeof(int), 1, fp, nullptr, error);
  MPI_Bcast(&len, 1, MPI_INT, 0, world);
  auto *name = new char[len];
  if (comm->me == 0) utils::sfread(FLERR, name, sizeof(char), len, fp, nullptr, error);
  MPI_Bcast(name, len, MPI_CHAR, 0, world);
  normal_name = name;
  delete[] name;
}

/* ---------------------------------------------------------------------- */

void AngleDirectorBend::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++)
    fprintf(fp, "%d %g %g %g %g %g %g %g\n", i, k_b[i], delta0[i], RAD2DEG * alpha0[i], k_tw[i],
            RAD2DEG * tau0[i], k_perp[i], eps_bind[i]);
}

/* ---------------------------------------------------------------------- */

double AngleDirectorBend::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;
  double **mu = atom->mu;
  double r1[3], r2[3], f1[3], f3[3], tq1[3], tq2[3], tq3[3], e[3][3];
  for (int d = 0; d < 3; d++) {
    r1[d] = x[i2][d] - x[i1][d];
    r2[d] = x[i3][d] - x[i2][d];
  }
  domain->minimum_image(FLERR, r1);
  domain->minimum_image(FLERR, r2);
  const int ia[3] = {i1, i2, i3};
  const double *ep[3];
  for (int m = 0; m < 3; m++) {
    const double mlen = len3(mu[ia[m]]);
    if (mlen > 0.0) {
      scale3(1.0 / mlen, mu[ia[m]], e[m]);
      ep[m] = e[m];
    } else ep[m] = nullptr;
  }
  double **normal = normal_array();
  return bend(type, r1, r2, ep[0], ep[1], ep[2], normal ? normal[i2] : nullptr, f1, f3, tq1, tq2, tq3);
}

/* ---------------------------------------------------------------------- */

void *AngleDirectorBend::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "k_b") == 0) return (void *) k_b;
  if (strcmp(str, "k_tw") == 0) return (void *) k_tw;
  if (strcmp(str, "k_perp") == 0) return (void *) k_perp;
  if (strcmp(str, "eps_bind") == 0) return (void *) eps_bind;
  return nullptr;
}
