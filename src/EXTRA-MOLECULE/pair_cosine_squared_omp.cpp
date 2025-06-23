// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
   Miguel Amaral (ISTA)
------------------------------------------------------------------------- */

#include "pair_cosine_squared_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "math_const.h"
#include "neigh_list.h"
#include "suffix.h"

#include "omp_compat.h"
using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */


PairCosineSquaredOMP::PairCosineSquaredOMP(LAMMPS *lmp) :
  PairCosineSquared(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */


void PairCosineSquaredOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }
    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairCosineSquaredOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type = atom->type;
  const double * _noalias const special_lj = force->special_lj;
  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;

  double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double rsq,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair;

  const int nlocal = atom->nlocal;
  int j,jj,jnum,jtype;

  evdwl = 0.0;

  double r,force_cos,cosone;

  // loop over neighbors of my atoms

  for (int ii = iifrom; ii < iito; ++ii) {
    const int i = ilist[ii];
    const int itype = type[i];
    const int    * _noalias const jlist = firstneigh[i];

    // cosine_squared coeffs
    const double * _noalias const epsiloni = epsilon[itype];
    const double * _noalias const sigmai = sigma[itype];
    const double * _noalias const wi = w[itype];
    const double * _noalias const cuti = cut[itype];
    const double * _noalias const cutsqi = cutsq[itype];
    
    const int * _noalias const wcaflagi = wcaflag[itype];

    const double * _noalias const lj12_ei = lj12_e[itype];
    const double * _noalias const lj6_ei = lj6_e[itype];
    const double * _noalias const lj12_fi = lj12_f[itype];
    const double * _noalias const lj6_fi = lj6_f[itype];

    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsqi[jtype]) {

        r = sqrt(rsq);

        if (r <= sigmai[jtype]) {
          if (wcaflagi[jtype]) {
            r2inv = 1.0/rsq;
            r6inv = r2inv*r2inv*r2inv;
            forcelj = r6inv*(lj12_fi[jtype]*r6inv - lj6_fi[jtype]);
            fpair = factor_lj*forcelj*r2inv;
            if (EFLAG) {
              evdwl = factor_lj*r6inv*
                      (lj12_ei[jtype]*r6inv - lj6_ei[jtype]);
              if (sigmai[jtype] == cuti[jtype]) {
                // this is the WCA-only case (it requires this shift by definition)
                evdwl += factor_lj*epsiloni[jtype];
              }
            }
          } else {
            fpair = 0.0;
            if (EFLAG) {
              evdwl = -factor_lj*epsiloni[jtype];
            }
          }
        } else {
          force_cos = -(MY_PI*epsiloni[jtype] / (2.0*wi[jtype])) *
                      sin(MY_PI*(r-sigmai[jtype]) / wi[jtype]);
          fpair = factor_lj*force_cos / r;
          if (EFLAG) {
            cosone = cos(MY_PI*(r-sigmai[jtype]) / (2.0*wi[jtype]));
            evdwl = -factor_lj*epsiloni[jtype]*cosone*cosone;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx*fpair;
          f[j].y -= dely*fpair;
          f[j].z -= delz*fpair;
        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,fpair,delx,dely,delz,thr);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairCosineSquaredOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairCosineSquared::memory_usage();

  return bytes;
}
