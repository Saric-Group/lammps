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
   bond_style director
   bond_coeff N K r0 K_perp K_tw tau0

   Bond of a filament whose beads carry a unit director e (atom->mu, the side
   of the binding face; see angle_style director/bend for the bend). With
   r = x2 - x1, t = r/|r| and P = director perpendicular to t (unit):

     stretch  K (|r| - r0)^2
     perp     K_perp/2 [(e1 . t)^2 + (e2 . t)^2]
     twist    K_tw |P2 - R_t(tau0) P1|^2   (R_t(a) = rotation by a about t)

   tau0 in degrees, right-handed about the bond direction 1 -> 2 (tail ->
   head in BAD topologies). An interior bead gets K_perp/2 [(e.t_left)^2 +
   (e.t_right)^2] from its two bonds, a chain end K_perp/2 (e.t)^2, a dimer
   both: every bond keeps its directors perpendicular and in register, also
   before the filament has an angle. Atoms without a director (mu = 0) only
   get the stretch. Forces on both atoms (not central), torques -e x dE/de
   on the directors. Needs atom attributes mu and torque.
------------------------------------------------------------------------- */

#include "bond_director.h"

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
using namespace MathExtra;

static constexpr double SMALL = 1.0e-8;

/* ---------------------------------------------------------------------- */

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

BondDirector::BondDirector(LAMMPS *_lmp) :
    Bond(_lmp), k(nullptr), r0(nullptr), k_perp(nullptr), k_tw(nullptr), tau0(nullptr)
{
  born_matrix_enable = 0;
}

/* ---------------------------------------------------------------------- */

BondDirector::~BondDirector()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(k_perp);
    memory->destroy(k_tw);
    memory->destroy(tau0);
  }
}

/* ---------------------------------------------------------------------- */

void BondDirector::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Bond style director needs atom attributes mu and torque "
               "(atom style with dipole and sphere)");
}

/* ---------------------------------------------------------------------- */

double BondDirector::bond_energy(int type, const double *r, const double *e1, const double *e2,
                                 double *f1, double *tq1, double *tq2) const
{
  for (int d = 0; d < 3; d++) f1[d] = tq1[d] = tq2[d] = 0.0;
  const double len = len3(r);
  if (len == 0.0) return 0.0;
  double t[3];
  scale3(1.0 / len, r, t);

  const double dr = len - r0[type];
  double energy = k[type] * dr * dr;
  double g_r[3], g_t[3] = {0.0, 0.0, 0.0};
  scale3(2.0 * k[type] * dr, t, g_r);
  double g_e1[3] = {0.0, 0.0, 0.0}, g_e2[3] = {0.0, 0.0, 0.0};

  // perpendicularity
  if (k_perp[type] != 0.0) {
    const double w = 0.5 * k_perp[type];
    const double *ee[2] = {e1, e2};
    double *ge[2] = {g_e1, g_e2};
    for (int m = 0; m < 2; m++) {
      if (!ee[m]) continue;
      const double c = dot3(ee[m], t);
      energy += w * c * c;
      axpy3(2.0 * w * c, t, ge[m]);
      axpy3(2.0 * w * c, ee[m], g_t);
    }
  }

  // twist: K_tw |P2 - R_t(tau0) P1|^2
  if (k_tw[type] != 0.0 && e1 && e2) {
    const double c1 = dot3(e1, t), c2 = dot3(e2, t);
    double p1[3], p2[3];
    for (int d = 0; d < 3; d++) {
      p1[d] = e1[d] - c1 * t[d];
      p2[d] = e2[d] - c2 * t[d];
    }
    const double l1 = len3(p1), l2 = len3(p2);
    if (l1 >= SMALL && l2 >= SMALL) {
      double P1[3], P2[3];
      scale3(1.0 / l1, p1, P1);
      scale3(1.0 / l2, p2, P2);
      const double c = cos(tau0[type]), s = sin(tau0[type]);
      const double tP = dot3(t, P1);
      double txP[3], Q[3], diff[3];
      cross3(t, P1, txP);
      for (int d = 0; d < 3; d++) {
        Q[d] = P1[d] * c + txP[d] * s + t[d] * tP * (1.0 - c);
        diff[d] = P2[d] - Q[d];
      }
      energy += k_tw[type] * lensq3(diff);
      double g_P2[3], g_Q[3];
      scale3(2.0 * k_tw[type], diff, g_P2);
      scale3(-2.0 * k_tw[type], diff, g_Q);
      const double tg = dot3(t, g_Q);
      double gxt[3], Pxg[3], g_P1[3];
      cross3(g_Q, t, gxt);
      cross3(P1, g_Q, Pxg);
      for (int d = 0; d < 3; d++) {
        g_P1[d] = g_Q[d] * c + gxt[d] * s + t[d] * tg * (1.0 - c);
        g_t[d] += s * Pxg[d] + (1.0 - c) * (tP * g_Q[d] + tg * P1[d]);
      }
      double g_p1[3] = {0.0, 0.0, 0.0}, g_p2[3] = {0.0, 0.0, 0.0};
      unit_adjoint(g_P1, P1, l1, g_p1);
      unit_adjoint(g_P2, P2, l2, g_p2);
      proj_adjoint(g_p1, e1, t, g_e1, g_t);
      proj_adjoint(g_p2, e2, t, g_e2, g_t);
    }
  }

  // t = r/|r|; r = x2 - x1 -> F1 = +dE/dr
  unit_adjoint(g_t, t, len, g_r);
  for (int d = 0; d < 3; d++) f1[d] = g_r[d];
  if (e1) cross3(g_e1, e1, tq1);
  if (e2) cross3(g_e2, e2, tq2);
  return energy;
}

/* ---------------------------------------------------------------------- */

void BondDirector::compute(int eflag, int vflag)
{
  double r[3], f1[3], tq1[3], tq2[3], e[2][3];
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int **bondlist = neighbor->bondlist;
  const int nbondlist = neighbor->nbondlist;
  const int nlocal = atom->nlocal;
  const int newton_bond = force->newton_bond;

  for (int n = 0; n < nbondlist; n++) {
    const int i1 = bondlist[n][0], i2 = bondlist[n][1], type = bondlist[n][2];
    for (int d = 0; d < 3; d++) r[d] = x[i2][d] - x[i1][d];
    const double *ep[2];
    const int ia[2] = {i1, i2};
    for (int m = 0; m < 2; m++) {
      const double mlen = len3(mu[ia[m]]);
      if (mlen > 0.0) {
        scale3(1.0 / mlen, mu[ia[m]], e[m]);
        ep[m] = e[m];
      } else ep[m] = nullptr;
    }
    const double ebond = bond_energy(type, r, ep[0], ep[1], f1, tq1, tq2);

    if (newton_bond || i1 < nlocal) {
      for (int d = 0; d < 3; d++) {
        f[i1][d] += f1[d];
        torque[i1][d] += tq1[d];
      }
    }
    if (newton_bond || i2 < nlocal) {
      for (int d = 0; d < 3; d++) {
        f[i2][d] -= f1[d];
        torque[i2][d] += tq2[d];
      }
    }
    // force f1 on atom 1, del = x1 - x2 = -r
    if (evflag)
      ev_tally_xyz(i1, i2, nlocal, newton_bond, ebond, f1[0], f1[1], f1[2], -r[0], -r[1], -r[2]);
  }
}

/* ---------------------------------------------------------------------- */

void BondDirector::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;
  memory->create(k, np1, "bond:k");
  memory->create(r0, np1, "bond:r0");
  memory->create(k_perp, np1, "bond:k_perp");
  memory->create(k_tw, np1, "bond:k_tw");
  memory->create(tau0, np1, "bond:tau0");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ---------------------------------------------------------------------- */

void BondDirector::coeff(int narg, char **arg)
{
  if (narg != 6) error->all(FLERR, "Incorrect args for bond coefficients" + utils::errorurl(21));
  if (!allocated) allocate();
  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);
  const double v[5] = {utils::numeric(FLERR, arg[1], false, lmp), utils::numeric(FLERR, arg[2], false, lmp),
                       utils::numeric(FLERR, arg[3], false, lmp), utils::numeric(FLERR, arg[4], false, lmp),
                       utils::numeric(FLERR, arg[5], false, lmp)};
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = v[0];
    r0[i] = v[1];
    k_perp[i] = v[2];
    k_tw[i] = v[3];
    tau0[i] = v[4] * DEG2RAD;
    setflag[i] = 1;
    count++;
  }
  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients" + utils::errorurl(21));
}

/* ---------------------------------------------------------------------- */

double BondDirector::equilibrium_distance(int i)
{
  return r0[i];
}

/* ---------------------------------------------------------------------- */

void BondDirector::write_restart(FILE *fp)
{
  const int n = atom->nbondtypes;
  for (double *p : {k, r0, k_perp, k_tw, tau0}) fwrite(&p[1], sizeof(double), n, fp);
}

/* ---------------------------------------------------------------------- */

void BondDirector::read_restart(FILE *fp)
{
  allocate();
  const int n = atom->nbondtypes;
  for (double *p : {k, r0, k_perp, k_tw, tau0}) {
    if (comm->me == 0) utils::sfread(FLERR, &p[1], sizeof(double), n, fp, nullptr, error);
    MPI_Bcast(&p[1], n, MPI_DOUBLE, 0, world);
  }
  for (int i = 1; i <= n; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

void BondDirector::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp, "%d %g %g %g %g %g\n", i, k[i], r0[i], k_perp[i], k_tw[i], RAD2DEG * tau0[i]);
}

/* ----------------------------------------------------------------------
   energy of the bond; fforce = the central (stretch) part only, the director
   terms give non-central forces
------------------------------------------------------------------------- */

double BondDirector::single(int type, double rsq, int i, int j, double &fforce)
{
  double **x = atom->x;
  double **mu = atom->mu;
  double r[3], f1[3], tq1[3], tq2[3], e[2][3];
  for (int d = 0; d < 3; d++) r[d] = x[j][d] - x[i][d];
  domain->minimum_image(FLERR, r);
  const int ia[2] = {i, j};
  const double *ep[2];
  for (int m = 0; m < 2; m++) {
    const double mlen = len3(mu[ia[m]]);
    if (mlen > 0.0) {
      scale3(1.0 / mlen, mu[ia[m]], e[m]);
      ep[m] = e[m];
    } else ep[m] = nullptr;
  }
  const double rlen = sqrt(rsq);
  fforce = rlen > 0.0 ? -2.0 * k[type] * (rlen - r0[type]) / rlen : 0.0;
  return bond_energy(type, r, ep[0], ep[1], f1, tq1, tq2);
}

/* ---------------------------------------------------------------------- */

void *BondDirector::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "k") == 0) return (void *) k;
  if (strcmp(str, "r0") == 0) return (void *) r0;
  if (strcmp(str, "k_perp") == 0) return (void *) k_perp;
  if (strcmp(str, "k_tw") == 0) return (void *) k_tw;
  return nullptr;
}
