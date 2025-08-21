/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#include "pair_nematic_angle_soft.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;


PairNematicSoft::PairNematicSoft(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 1;
  cut_global = 0.0;
}

PairNematicSoft::~PairNematicSoft()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(Aamp);
    memory->destroy(kappa);
    memory->destroy(theta0);
    memory->destroy(c0);
    memory->destroy(s0);
    memory->destroy(c0s0);
    memory->destroy(cos2t0);
    memory->destroy(cut);
    memory->destroy(cutsq);
    memory->destroy(c_fac);
  }
}

void PairNematicSoft::allocate()
{
  if (allocated) return;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(Aamp, n + 1, n + 1, "pair:Aamp");
  memory->create(kappa, n + 1, n + 1, "pair:kappa");
  memory->create(theta0, n + 1, n + 1, "pair:theta0");
  memory->create(c0, n + 1, n + 1, "pair:c0");
  memory->create(s0, n + 1, n + 1, "pair:s0");
  memory->create(c0s0, n + 1, n + 1, "pair:c0s0");
  memory->create(cos2t0, n + 1, n + 1, "pair:cos2t0");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(c_fac, n + 1, n + 1, "pair:c_fac");

  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
}

void PairNematicSoft::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Incorrect args for pair_style nematic/angle/soft");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
}

void PairNematicSoft::coeff(int narg, char **arg)
{
  // i j A kappa theta0 [cut]
  if (narg != 5 && narg != 6) error->all(FLERR, "Incorrect args for pair_coeff in nematic/angle/soft");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double A_one = utils::numeric(FLERR, arg[2], false, lmp);
  double kappa_one = utils::numeric(FLERR, arg[3], false, lmp);
  double t0_one = utils::numeric(FLERR, arg[4], false, lmp);

  double cut_one = (narg == 6) ? utils::numeric(FLERR, arg[5], false, lmp) : cut_global;
  if (cut_one <= 0.0) error->all(FLERR, "Invalid cutoff for nematic/angle/soft");

  // cache trig
  double c0_one = cos(t0_one);
  double s0_one = sin(t0_one);
  double c0s0_one = c0_one * s0_one;
  double cos2t0_one = c0_one * c0_one - s0_one * s0_one;
  // canonicalization factor to make minimum of orientational contribution exactly 0.
  double c_fac_one = 1 / (1.0 + exp(-8.0 * kappa_one * (c0s0_one * c0s0_one)));

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      Aamp[i][j] = A_one;
      kappa[i][j] = kappa_one;
      theta0[i][j] = t0_one;
      c0[i][j] = c0_one;
      s0[i][j] = s0_one;
      c0s0[i][j] = c0s0_one;
      cos2t0[i][j] = cos2t0_one;
      c_fac[i][j] = c_fac_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair_coeff nematic/angle/soft");
}

void PairNematicSoft::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "pair_style nematic/angle/soft requires atom attributes mu and torque");
  neighbor->add_request(this);
}

double PairNematicSoft::init_one(int i, int j)
{
  if (!setflag[i][j]) {
    Aamp[i][j] = kappa[i][j] = theta0[i][j] = 0.0;
    c0[i][j] = s0[i][j] = c0s0[i][j] = cos2t0[i][j] = c_fac[i][j] = 0.0;
    cut[i][j] = 0.0;
  }

  if (i > j) {
    Aamp[i][j] = Aamp[j][i];
    kappa[i][j] = kappa[j][i];
    theta0[i][j] = theta0[j][i];
    c0[i][j] = c0[j][i];
    s0[i][j] = s0[j][i];
    c0s0[i][j] = c0s0[j][i];
    cos2t0[i][j] = cos2t0[j][i];
    cut[i][j] = cut[j][i];
    c_fac[i][j] = c_fac[j][i];
    setflag[i][j] = setflag[j][i];
  }

  return cut[i][j];
}

void PairNematicSoft::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int itype = type[i];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      int jtype = type[j];

      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;

      if (rsq >= cutsq[itype][jtype]) continue;

      double r = sqrt(rsq);
      if (r == 0.0) continue;    // avoid division by 0, no force/torque well-defined at r=0
      double rinv = 1.0 / r;

      double rc = cut[itype][jtype];
      double Aij = Aamp[itype][jtype];
      double kij = kappa[itype][jtype];
      double c0ij = c0[itype][jtype];
      double s0ij = s0[itype][jtype];
      double c0s0ij = c0s0[itype][jtype];
      double cos2t0ij = cos2t0[itype][jtype];
      double c_fac_ij = c_fac[itype][jtype];

      if (rc <= 0.0) { rc = cut_global; }
      if (rc <= 0.0) {
        error->all(FLERR, "Cutoff must be set for pair coefficients in nematic/align");
      }

      // assume mu vectors are unit and in-plane (xy)
      double c = mu[i][0] * mu[j][0] + mu[i][1] * mu[j][1];    // dot
      // z-component of cross (2D signed sine)
      double s = (mu[i][0] * mu[j][1] - mu[i][1] * mu[j][0]);

      double c2 = c * c;
      double s2 = s * s;
      double sc = s * c;

      // angular energy in stable form: Uang = 1 - 2*exp(Sth)*cosh(Rth)
      double Sth = -2.0 * kij * (s2 * c0ij * c0ij + c2 * s0ij * s0ij);    // <= 0
      double Rth = 4.0 * kij * (sc * c0s0ij);

      double eS = exp(Sth);
      double coshR = cosh(Rth);
      double sinhR = sinh(Rth);

      double Uang = 1.0 - 2.0 * eS * coshR * c_fac_ij;

      // soft radial factor and derivative
      // S(r) = A [1 + cos(pi r / rc)] for r < rc
      double xarg = M_PI * r / rc;
      double Sr = Aij * (1.0 + cos(xarg));
      double dSdr = -Aij * (M_PI / rc) * sin(xarg);

      // Total energy
      double evdwl = 0.0;
      if (eflag) evdwl = Sr * Uang;

      // F = -dU/dr * rhat = -dS/dr * Uang * rhat
      double f_over_r = (-dSdr * Uang) * rinv;
      double fx = f_over_r * delx;
      double fy = f_over_r * dely;
      double fz = f_over_r * delz;

      // torque from angular derivative: dU/dθ
      double dUang_dtheta = 8.0 * kij * eS * (sc * cos2t0ij * coshR - c0s0ij * (c2 - s2) * sinhR) * c_fac_ij;

      double tau_i_z = -Sr * dUang_dtheta;
      double tau_j_z = Sr * dUang_dtheta;

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      torque[i][2] += tau_i_z;

      if (newton_pair || j < atom->nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;
        torque[j][2] += tau_j_z;
      }

      if (eflag) evdwl = Sr * Uang;    // or: evdwl = energy; if you named it that way
      if (evflag) ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

double PairNematicSoft::single(int i, int j, int itype, int jtype,
                               double rsq, double factor_coul, double factor_lj,
                               double &fforce)
{
  // cutoff guard
  double rc = cut[itype][jtype] > 0.0 ? cut[itype][jtype] : cut_global;
  if (rc <= 0.0 || rsq >= rc*rc) { fforce = 0.0; return 0.0; }

  double Aij = Aamp[itype][jtype];
  double kij = kappa[itype][jtype];
  double c0ij = c0[itype][jtype];
  double s0ij = s0[itype][jtype];
  double c0s0ij = c0s0[itype][jtype];
  double c_fac_ij = c_fac[itype][jtype];

  double **mu = atom->mu;
  double c = mu[i][0] * mu[j][0] + mu[i][1] * mu[j][1];      // xy dot
  double s = mu[i][0] * mu[j][1] - mu[i][1] * mu[j][0];      // z-comp of cross

  double c2 = c*c, s2 = s*s, sc = s*c;

  double Sth = -2.0 * kij * (s2 * c0ij * c0ij + c2 * s0ij * s0ij);
  double Rth =  4.0 * kij * (sc * c0s0ij);
  double Uang = 1.0 - 2.0 * exp(Sth) * cosh(Rth) * c_fac_ij;

  double r = sqrt(rsq);
  double xarg = M_PI * r / rc;
  double Sr   = Aij * (1.0 + cos(xarg));
  double dSdr = -Aij * (M_PI/rc) * sin(xarg) / r;  // this is divided by r also in og pair_soft implementation

  double energy = Sr * Uang;
  fforce = (-dSdr * Uang) * factor_lj;   // -dU/dr
  return energy * factor_lj;
}