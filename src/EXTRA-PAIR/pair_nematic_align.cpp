/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
   Modified by Google Gemini
------------------------------------------------------------------------- */

#include "pair_nematic_align.h"

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

PairNematicAlign::PairNematicAlign(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
}

PairNematicAlign::~PairNematicAlign()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(epsilon);
    memory->destroy(cut);
    memory->destroy(wca_flag);
    memory->destroy(lj_sigma);
    memory->destroy(lj_epsilon);
    memory->destroy(no_radial_flag);
  }
}

void PairNematicAlign::allocate()
{
  if (allocated) return;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
  memory->create(cut, n + 1, n + 1, "pair:cut");

  memory->create(wca_flag, n + 1, n + 1, "pair:wca_flag");
  memory->create(lj_sigma, n + 1, n + 1, "pair:lj_sigma");
  memory->create(lj_epsilon, n + 1, n + 1, "pair:lj_epsilon");
  memory->create(no_radial_flag, n + 1, n + 1, "pair:no_radial_flag");

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      wca_flag[i][j] = 0;
      no_radial_flag[i][j] = 0;
    }
  }
}

void PairNematicAlign::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Incorrect args for pair_style command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

void PairNematicAlign::coeff(int narg, char **arg)
{

  int no_radial_flag_one = 0;
  int narg_eff = narg;

  // Find and conceptually remove "no_radial" keyword from arguments
  for (int i = 3; i < narg_eff; i++) {
    if (strcmp(arg[i], "no_radial") == 0) {
      no_radial_flag_one = 1;
      // Shift remaining arguments left to overwrite the keyword
      for (int j = i; j < narg_eff - 1; j++) { arg[j] = arg[j + 1]; }
      narg_eff--;
      break;    // Assume only one 'no_radial' keyword
    }
  }

  // The rest of the function uses narg_eff to validate argument count
  if (narg_eff != 3 && narg_eff != 4 && narg_eff != 6 && narg_eff != 7)
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);

  double cut_one = cut_global;
  int wca_flag_one = 0;
  double lj_sigma_one = 0.0;
  double lj_epsilon_one = 0.0;

  int wca_idx = -1;
  for (int i = 3; i < narg_eff; i++) {
    if (strcmp(arg[i], "wca") == 0) {
      wca_idx = i;
      break;
    }
  }

  if (wca_idx != -1) {
    wca_flag_one = 1;
    if (narg_eff != wca_idx + 3)
      error->all(FLERR, "Incorrect args for pair coefficients: wca requires 2 parameters");

    lj_sigma_one = utils::numeric(FLERR, arg[wca_idx + 1], false, lmp);
    lj_epsilon_one = utils::numeric(FLERR, arg[wca_idx + 2], false, lmp);

    if (wca_idx == 4) {
      cut_one = utils::numeric(FLERR, arg[3], false, lmp);
    } else if (wca_idx != 3) {
      error->all(FLERR, "Incorrect args for pair coefficients");
    }

  } else {
    if (narg_eff == 4) { cut_one = utils::numeric(FLERR, arg[3], false, lmp); }
  }

  if (cut_one <= 0.0) error->all(FLERR, "Invalid cutoff specified for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      cut[i][j] = cut_one;
      wca_flag[i][j] = wca_flag_one;
      lj_sigma[i][j] = lj_sigma_one;
      lj_epsilon[i][j] = lj_epsilon_one;
      no_radial_flag[i][j] = no_radial_flag_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

void PairNematicAlign::init_style()
{
  if (!atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Pair style 'nematic/align' requires atom attributes mu and torque");

  neighbor->add_request(this);
}

double PairNematicAlign::init_one(int i, int j)
{
  if (!setflag[i][j]) {
    epsilon[i][j] = 0.0;
    cut[i][j] = 0.0;
    wca_flag[i][j] = 0;
    lj_sigma[i][j] = 0.0;
    lj_epsilon[i][j] = 0.0;
    no_radial_flag[i][j] = 0.0;
  }

  epsilon[j][i] = epsilon[i][j];
  cut[j][i] = cut[i][j];
  wca_flag[j][i] = wca_flag[i][j];
  lj_sigma[j][i] = lj_sigma[i][j];
  lj_epsilon[j][i] = lj_epsilon[i][j];
  no_radial_flag[j][i] = no_radial_flag[i][j];

  double nematic_cut = cut[i][j];
  double wca_cut = 0.0;
  if (wca_flag[i][j] && lj_sigma[i][j] > 0.0) { wca_cut = 1.12246204831 * lj_sigma[i][j]; }

  return MAX(nematic_cut, wca_cut);
}

void PairNematicAlign::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  double r, r_over_rcut, rinv;
  double mu_dot_mu;
  double mu1_dot_rij, mu2_dot_rij;
  double energy, fx, fy, fz;
  double tix, tiy, tiz, tjx, tjy, tjz;
  double eps, rc;
  double evdwl;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int **no_radial_flag = this->no_radial_flag;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        rinv = 1.0 / r;

        energy = 0.0;
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;
        tix = 0.0;
        tiy = 0.0;
        tiz = 0.0;
        tjx = 0.0;
        tjy = 0.0;
        tjz = 0.0;

        // WCA repulsion (if enabled)
        if (wca_flag[itype][jtype]) {
          double lj_s = lj_sigma[itype][jtype];
          double wca_cut = 1.12246204831 * lj_s;
          if (r < wca_cut) {
            double lj_e = lj_epsilon[itype][jtype];
            double sr2 = lj_s * lj_s / rsq;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double wca_force_over_r = 48.0 * lj_e * (sr12 - 0.5 * sr6) * rinv * rinv;

            fx += wca_force_over_r * delx;
            fy += wca_force_over_r * dely;
            fz += wca_force_over_r * delz;
            if (eflag) energy += 4.0 * lj_e * (sr12 - sr6) + lj_e;
          }
        }

        // alignment interaction calculation
        rc = cut[itype][jtype];
        if (r < rc && mu[i][3] > 0.0 && mu[j][3] > 0.0) {
          eps = epsilon[itype][jtype];
          if (rc <= 0.0) { rc = cut_global; }
          if (rc <= 0.0) {
            error->all(FLERR, "Cutoff must be set for pair coefficients in nematic/align");
          }

          r_over_rcut = r / rc;

          mu1_dot_rij = mu[i][0] * delx + mu[i][1] * dely + mu[i][2] * delz;
          mu2_dot_rij = mu[j][0] * delx + mu[j][1] * dely + mu[j][2] * delz;

          // projections onto u and the scalar S (used for energy and force)
          const double mu1_u = mu1_dot_rij * rinv;    // (mu1·u)
          const double mu2_u = mu2_dot_rij * rinv;    // (mu2·u)
          const double S = mu1_u * mu1_u + mu2_u * mu2_u;

          // common radial prefactor
          const double one_minus = 1.0 - r_over_rcut;

          if (eflag) { energy += -eps * one_minus * one_minus * S; }

          // potential and its derivative w.r.t r
          const double g = -eps * one_minus * one_minus;
          const double gprime = (2.0 * eps / rc) * one_minus;

          const double ux = delx * rinv;
          const double uy = dely * rinv;
          const double uz = delz * rinv;

          // force calculation (split into radial + tangential)
          // components of (mu_k - (mu_k·u) u)  — purely tangential to u
          const double a1x = mu[i][0] - mu1_u * ux;
          const double a1y = mu[i][1] - mu1_u * uy;
          const double a1z = mu[i][2] - mu1_u * uz;

          const double a2x = mu[j][0] - mu2_u * ux;
          const double a2y = mu[j][1] - mu2_u * uy;
          const double a2z = mu[j][2] - mu2_u * uz;

          // force magnitudes
          const double frad_mag = -gprime * S;
          const double ftan_pref = -g * (2.0 * rinv);

          // tangential contribution (perpendicular to rij)
          double fx_tan = ftan_pref * (mu1_u * a1x + mu2_u * a2x);
          double fy_tan = ftan_pref * (mu1_u * a1y + mu2_u * a2y);
          double fz_tan = ftan_pref * (mu1_u * a1z + mu2_u * a2z);

          // tangential force
          fx += fx_tan;
          fy += fy_tan;
          fz += fz_tan;

          // add radial contribution unless 'no_radial' flag is set for this pair type
          if (!no_radial_flag[itype][jtype]) {
            fx += frad_mag * ux;
            fy += frad_mag * uy;
            fz += frad_mag * uz;
          }

          // torque calculation
          const double torque_common = -2.0 * g * rinv * rinv;

          tix = torque_common * mu1_dot_rij * (mu[i][1] * delz - mu[i][2] * dely);
          tiy = torque_common * mu1_dot_rij * (mu[i][2] * delx - mu[i][0] * delz);
          tiz = torque_common * mu1_dot_rij * (mu[i][0] * dely - mu[i][1] * delx);

          tjx = torque_common * mu2_dot_rij * (mu[j][1] * delz - mu[j][2] * dely);
          tjy = torque_common * mu2_dot_rij * (mu[j][2] * delx - mu[j][0] * delz);
          tjz = torque_common * mu2_dot_rij * (mu[j][0] * dely - mu[j][1] * delx);
        }

        // total force and torque accumulation ---
        if (eflag) evdwl = energy;

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        torque[i][0] += tix;
        torque[i][1] += tiy;
        torque[i][2] += tiz;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
          torque[j][0] += tjx;
          torque[j][1] += tjy;
          torque[j][2] += tjz;
        }

        if (evflag)
          ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy, fz, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

double PairNematicAlign::single(int i, int j, int itype, int jtype, double rsq, double factor_coul,
                                double factor_lj, double &fforce)
{
  // cutoff guard
  double rc = cut[itype][jtype] > 0.0 ? cut[itype][jtype] : cut_global;
  if (rsq >= rc * rc) {
    fforce = 0.0;
    return 0.0;
  }

  double energy = 0.0;
  double wca_force_over_r = 0.0;

  // WCA repulsion (if enabled)
  if (wca_flag[itype][jtype]) {
    double lj_s = lj_sigma[itype][jtype];
    double wca_cut = 1.12246204831 * lj_s;
    double r = sqrt(rsq);
    double rinv = 1 / r;
    if (r < wca_cut) {
      double lj_e = lj_epsilon[itype][jtype];
      double sr2 = lj_s * lj_s / rsq;
      double sr6 = sr2 * sr2 * sr2;
      double sr12 = sr6 * sr6;

      wca_force_over_r += 48.0 * lj_e * (sr12 - 0.5 * sr6) * rinv * rinv;
      energy += 4.0 * lj_e * (sr12 - sr6) + lj_e;
    }
  }

  fforce = -wca_force_over_r * factor_lj;    // -dU/dr
  return energy * factor_lj;
}