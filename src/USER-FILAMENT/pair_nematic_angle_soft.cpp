/* ----------------------------------------------------------------------
   Updated by @andraz-gnidovec
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
  orientation_flag = 1;
  intermol_flag = 0; // Default: 0 = Calculate everything (Intra + Inter)
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
    memory->destroy(cut);
    memory->destroy(cutsq);
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
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 0;
    }
}

void PairNematicSoft::settings(int narg, char **arg)
{
  // Syntax: pair_style nematic/angle/soft <cut_global> [intermol]
  if (narg < 1) error->all(FLERR, "Incorrect args for pair_style nematic/angle/soft");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  
  intermol_flag = 0; // Reset default
  
  if (narg > 1) {
      if (strcmp(arg[1], "intermol") == 0) {
          intermol_flag = 1; 
      } else {
          error->all(FLERR, "Unknown keyword for pair_style nematic/angle/soft. Only 'intermol' is allowed.");
      }
  }
}

void PairNematicSoft::coeff(int narg, char **arg)
{
  // Arguments: i j A kappa theta0 [cut]
  if (narg != 5 && narg != 6)
    error->all(FLERR, "Incorrect args for pair_coeff in nematic/angle/soft");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double A_one = utils::numeric(FLERR, arg[2], false, lmp);
  double kappa_one = utils::numeric(FLERR, arg[3], false, lmp);
  double t0_one = utils::numeric(FLERR, arg[4], false, lmp);

  double cut_one = cut_global;
  if (narg == 6) {
    cut_one = utils::numeric(FLERR, arg[5], false, lmp);
  }

  // cache trig
  double c0_one = cos(t0_one);
  double s0_one = sin(t0_one);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      Aamp[i][j] = A_one;
      kappa[i][j] = kappa_one;
      theta0[i][j] = t0_one;
      c0[i][j] = c0_one;
      s0[i][j] = s0_one;
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
    c0[i][j] = s0[i][j] = 0.0;
    cut[i][j] = 0.0;
  }

  Aamp[j][i] = Aamp[i][j];
  kappa[j][i] = kappa[i][j];
  theta0[j][i] = theta0[i][j];
  c0[j][i] = c0[i][j];
  s0[j][i] = s0[i][j];
  cut[j][i] = cut[i][j];
  setflag[j][i] = setflag[i][j];

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
  int *molecule = atom->molecule;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int itype = type[i];
    int imol = molecule[i];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      int jtype = type[j];
      int jmol = molecule[j];

      // Logic:
      // If intermol_flag is 1: We ONLY want intermolecular. SKIP if imol == jmol.
      // If intermol_flag is 0: We want ALL interactions. Never skip based on molecule ID.
      if (intermol_flag && (imol == jmol)) {
        continue;
      }

      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;

      if (rsq >= cutsq[itype][jtype]) continue;

      double energy = 0.0;
      double fx = 0.0, fy = 0.0, fz = 0.0;
      double tau_i_z = 0.0;

      double r = sqrt(rsq);
      double rinv = 1.0 / r;

      double rc = cut[itype][jtype];
      double Aij = Aamp[itype][jtype];
      double kij = kappa[itype][jtype];
      double c0ij = c0[itype][jtype];
      double s0ij = s0[itype][jtype];

      if (rc <= 0.0) { rc = cut_global; }

      // assume mu vectors are unit and in-plane (xy)
      double c = mu[i][0] * mu[j][0] + mu[i][1] * mu[j][1];
      double s = mu[i][0] * mu[j][1] - mu[i][1] * mu[j][0];

      double a = s * c0ij;
      double b = c * s0ij;
      double sm = a - b;
      double sp = a + b;

      double em = exp(-2.0 * kij * sm * sm);
      double ep = exp(-2.0 * kij * sp * sp);

      double Uang = (1.0 - em) * (1.0 - ep);

      double xarg = M_PI * r / rc;
      double Sr = Aij * (1.0 + cos(xarg));
      double dSdr = -Aij * (M_PI / rc) * sin(xarg);

      if (eflag) energy += Sr * Uang;

      double f_over_r = (-dSdr * Uang) * rinv;
      fx += f_over_r * delx;
      fy += f_over_r * dely;
      fz += f_over_r * delz;

      // torque
      double cm = c * c0ij + s * s0ij;
      double cp = c * c0ij - s * s0ij;
      double dfm = 4.0 * kij * em * sm * cm;
      double dfp = 4.0 * kij * ep * sp * cp;
      double dUang_dtheta = dfm * (1.0 - ep) + (1.0 - em) * dfp;

      tau_i_z = Sr * dUang_dtheta;

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      torque[i][2] += tau_i_z;

      if (newton_pair || j < atom->nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;
        torque[j][2] -= tau_i_z;
      }

      if (evflag)
        ev_tally_xyz(i, j, nlocal, newton_pair, energy, 0.0, fx, fy, fz, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

double PairNematicSoft::single(int i, int j, int itype, int jtype, double rsq, double factor_coul,
                               double factor_lj, double &fforce)
{
  // cutoff guard
  double rc = cut[itype][jtype] > 0.0 ? cut[itype][jtype] : cut_global;
  if (rsq >= rc * rc) {
    fforce = 0.0;
    return 0.0;
  }

  double energy = 0.0;
  fforce = 0.0;

  double Aij = Aamp[itype][jtype];
  
  double r = sqrt(rsq);
  double xarg = M_PI * r / rc;
  double Sr = Aij * (1.0 + cos(xarg));
  double dSdr = -Aij * (M_PI / rc) * sin(xarg) / r;

  energy += Sr;
  fforce += -dSdr;

  energy *= factor_lj;
  fforce *= factor_lj;

  return energy;
}

double PairNematicSoft::single_orientation(int itype, int jtype, double rsq, double factor_lj,
                                           const double *mu_i, const double *mu_j)
{
  double energy = 0.0;
  double rc = cut[itype][jtype] > 0.0 ? cut[itype][jtype] : cut_global;

  if (rc > 0.0 && rsq < rc * rc) {
    double Aij = Aamp[itype][jtype];
    double kij = kappa[itype][jtype];
    double c0ij = c0[itype][jtype];
    double s0ij = s0[itype][jtype];

    double c = mu_i[0] * mu_j[0] + mu_i[1] * mu_j[1];
    double s = mu_i[0] * mu_j[1] - mu_i[1] * mu_j[0];
    double sp = s * c0ij + c * s0ij;
    double sm = s * c0ij - c * s0ij;

    double ep = exp(-2.0 * kij * sp * sp);
    double Uang = (1.0 - ep);

    double r = sqrt(rsq);
    double xarg = M_PI * r / rc;
    double Sr = Aij * (1.0 + cos(xarg));

    energy += Sr * Uang;
  }

  return energy * factor_lj;
}

void PairNematicSoft::write_restart(FILE *fp)
{
  Pair::write_restart(fp);

  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&intermol_flag, sizeof(int), 1, fp); // Save the global flag

  int n = atom->ntypes;
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&Aamp[i][j], sizeof(double), 1, fp);
        fwrite(&kappa[i][j], sizeof(double), 1, fp);
        fwrite(&theta0[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

void PairNematicSoft::read_restart(FILE *fp)
{
  Pair::read_restart(fp);

  allocate();

  fread(&cut_global, sizeof(double), 1, fp);
  fread(&intermol_flag, sizeof(int), 1, fp); // Read the global flag

  int n = atom->ntypes;
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      fread(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fread(&Aamp[i][j], sizeof(double), 1, fp);
        fread(&kappa[i][j], sizeof(double), 1, fp);
        fread(&theta0[i][j], sizeof(double), 1, fp);
        fread(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}