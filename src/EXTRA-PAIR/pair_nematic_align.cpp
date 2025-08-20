/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
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

#include <cmath>

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
    memory->destroy(k_exp);
    memory->destroy(cut);
  }
}

void PairNematicAlign::allocate()
{
  fprintf(screen, "DEBUG: PairNematicAlign::allocate() is being called!\n");

  if (allocated) return;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
  memory->create(k_exp, n + 1, n + 1, "pair:k_exp");
  memory->create(cut, n + 1, n + 1, "pair:cut");
}

void PairNematicAlign::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Incorrect args for pair_style command");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset cutoffs for pairs that have already been set
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

void PairNematicAlign::coeff(int narg, char **arg)
{

  fprintf(screen, "DEBUG: In PairNematicAlign::coeff() with narg = %d\n", narg);

  // Expected format: pair_coeff I J epsilon k r_cut
  if (narg < 4 || narg > 5) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double k_one = utils::numeric(FLERR, arg[3], false, lmp);

  double cut_one = cut_global;
  if (narg == 5) cut_one = utils::numeric(FLERR, arg[4], false, lmp);
  if (cut_one <= 0.0) error->all(FLERR, "Invalid cutoff specified for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      k_exp[i][j] = k_one;
      cut[i][j] = cut_one;
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

  if (!allocated) allocate();

  neighbor->add_request(this);
}

double PairNematicAlign::init_one(int i, int j)
{

  if (i > j) {
    epsilon[i][j] = epsilon[j][i];
    k_exp[i][j] = k_exp[j][i];
    cut[i][j] = cut[j][i];
  }

  return cut[i][j];
}


void PairNematicAlign::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  double r, r_over_rcut, rinv;
  double mu_dot_mu, abs_mu_dot_mu;
  double mu1_dot_rij, mu2_dot_rij;
  double energy, force_magnitude_over_r;
  double fx, fy, fz;
  double torque_common, tix, tiy, tiz, tjx, tjy, tjz;
  double eps, k, rc;
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

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors
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
        eps = epsilon[itype][jtype];
        k = k_exp[itype][jtype];
        rc = cut[itype][jtype];

        // I have no idea why this is needed, cutoff should be set correctly by coeffs()
        if (rc <= 0.0) { rc = cut_global; }

        if (rc <= 0.0) {
          error->all(FLERR, "Cutoff must be set for pair coefficients in nematic/align");
        }

        // interaction applies only to particles with dipoles
        if (mu[i][3] > 0.0 && mu[j][3] > 0.0) {

          mu_dot_mu = mu[i][0] * mu[j][0] + mu[i][1] * mu[j][1] + mu[i][2] * mu[j][2];
          mu1_dot_rij = mu[i][0] * delx + mu[i][1] * dely + mu[i][2] * delz;
          mu2_dot_rij = mu[j][0] * delx + mu[j][1] * dely + mu[j][2] * delz;
          // abs_mu_dot_mu = fabs(mu_dot_mu);
          r_over_rcut = r / rc;

          // double mu_term = pow(abs_mu_dot_mu, k);
          double mu_term = mu_dot_mu * mu_dot_mu;
          double mu1_uij_term = mu1_dot_rij * mu1_dot_rij * rinv * rinv;
          double mu2_uij_term = mu2_dot_rij * mu2_dot_rij * rinv * rinv;
          double full_2nd_term = mu1_uij_term * mu2_uij_term;

          // energy calculation: U = -epsilon * (1 - r/r_cut)^2 * [(mu_1 . mu_2)^2 + (mu_1 . uij)^2 (mu_2 . uij)^2]
          if (eflag) {
            energy = -eps * (1.0 - r_over_rcut) * (1.0 - r_over_rcut) * (mu_term + full_2nd_term);
            evdwl = energy;
          } else {
            energy = 0.0;
          }

          // force calculation: F = -grad(U)
          double force_magnitude_over_r_term1 =
              -2 * (eps / rc) * (1.0 - r_over_rcut) * mu_term * rinv;

          double grad_term2_part1_magnitude =
              -2 * rinv * rinv * full_2nd_term * eps * (1.0 - r_over_rcut) * (2.0 - r_over_rcut);

          double grad_term2_part2_i_magnitude = 2 * eps * (1.0 - r_over_rcut) *
              (1.0 - r_over_rcut) * rinv * rinv * mu2_uij_term * mu1_dot_rij;

          double grad_term2_part2_j_magnitude = 2 * eps * (1.0 - r_over_rcut) *
              (1.0 - r_over_rcut) * rinv * rinv * mu1_uij_term * mu2_dot_rij;

          fx = (force_magnitude_over_r_term1 + grad_term2_part1_magnitude) * delx +
              grad_term2_part2_i_magnitude * mu[i][0] + grad_term2_part2_j_magnitude * mu[j][0];

          fy = (force_magnitude_over_r_term1 + grad_term2_part1_magnitude) * dely +
              grad_term2_part2_i_magnitude * mu[i][1] + grad_term2_part2_j_magnitude * mu[j][1];

          fz = (force_magnitude_over_r_term1 + grad_term2_part1_magnitude) * delz +
              grad_term2_part2_i_magnitude * mu[i][2] + grad_term2_part2_j_magnitude * mu[j][2];

          // torque calculation: T_i = eps*(1-r/rc)*k*|s|^(k-1)*sgn(s)*(mu_i x mu_j)
          // with s = mu_i . mu_j
          tix = tiy = tiz = tjx = tjy = tjz = 0.0;

          // Common prefactor for all torque terms
          double torque_common = 2 * eps * (1.0 - r_over_rcut) * (1.0 - r_over_rcut);

          // --- Calculate Torque on Particle i (Ti = Ti,1 + Ti,2) ---

          // Term 1 for particle i (Ti,1)
          double ti1_mag = torque_common * mu_dot_mu;
          double tix1 = ti1_mag * (mu[i][1] * mu[j][2] - mu[i][2] * mu[j][1]);
          double tiy1 = ti1_mag * (mu[i][2] * mu[j][0] - mu[i][0] * mu[j][2]);
          double tiz1 = ti1_mag * (mu[i][0] * mu[j][1] - mu[i][1] * mu[j][0]);

          // Term 2 for particle i (Ti,2) - THIS WAS MISSING
          double ti2_mag = torque_common * rinv * rinv * mu2_uij_term * mu1_dot_rij;
          double tix2 = ti2_mag * (mu[i][1] * delz - mu[i][2] * dely);
          double tiy2 = ti2_mag * (mu[i][2] * delx - mu[i][0] * delz);
          double tiz2 = ti2_mag * (mu[i][0] * dely - mu[i][1] * delx);

          // Total torque on i
          double tix = tix1 + tix2;
          double tiy = tiy1 + tiy2;
          double tiz = tiz1 + tiz2;


          // --- Calculate Torque on Particle j (Tj = Tj,1 + Tj,2) ---

          // Term 1 for particle j is the negative of Term 1 for i (Tj,1 = -Ti,1)
          double tjx1 = -tix1;
          double tjy1 = -tiy1;
          double tjz1 = -tiz1;

          // Term 2 for particle j (Tj,2)
          double tj2_mag = torque_common * rinv * rinv * mu1_uij_term * mu2_dot_rij;
          double tjx2 = tj2_mag * (mu[j][1] * delz - mu[j][2] * dely);
          double tjy2 = tj2_mag * (mu[j][2] * delx - mu[j][0] * delz);
          double tjz2 = tj2_mag * (mu[j][0] * dely - mu[j][1] * delx);

          // Total torque on j
          double tjx = tjx1 + tjx2;
          double tjy = tjy1 + tjy2;
          double tjz = tjz1 + tjz2;


          // --- Accumulate forces and torques ---
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
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
