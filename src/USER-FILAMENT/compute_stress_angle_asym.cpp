/* ----------------------------------------------------------------------
   Added by @andraz-gnidovec
------------------------------------------------------------------------- */

#include "compute_stress_angle_asym.h"

#include "angle.h"
#include "angle_hybrid.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr double SMALL = 0.001;    // as in angle_harmonic.cpp

/* ----------------------------------------------------------------------
   harmonic or a host-side accelerated variant (harmonic/omp, harmonic/intel),
   which keep k and theta0 in the AngleHarmonic arrays; not Kokkos, whose
   positions live on the device
------------------------------------------------------------------------- */

static bool is_harmonic(const char *style)
{
  if (strcmp(style, "harmonic") == 0) return true;
  return (strncmp(style, "harmonic/", 9) == 0) && !utils::strmatch(style, "/kk");
}

/* ---------------------------------------------------------------------- */

ComputeStressAngleAsym::ComputeStressAngleAsym(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), stress(nullptr), k(nullptr), theta0(nullptr), angle(nullptr)
{
  if (narg != 3) error->all(FLERR, "Illegal compute stress/angle/asym command: no arguments allowed");
  if (domain->dimension != 2) error->all(FLERR, "Compute stress/angle/asym requires a 2D simulation");
  if (!atom->molecular || atom->avec->angles_allow == 0)
    error->all(FLERR, "Compute stress/angle/asym requires an atom style with angles");

  peratom_flag = 1;
  size_peratom_cols = 4;
  comm_reverse = 4;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeStressAngleAsym::~ComputeStressAngleAsym()
{
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

void ComputeStressAngleAsym::init()
{
  // fetch here, not in the constructor: the angle style may be defined after the compute
  // and coefficients may change between runs

  angle = nullptr;
  if (force->angle && force->angle_style) {
    if (is_harmonic(force->angle_style))
      angle = force->angle;
    else if (utils::strmatch(force->angle_style, "^hybrid")) {
      auto *hybrid = dynamic_cast<AngleHybrid *>(force->angle);
      for (int m = 0; hybrid && m < hybrid->nstyles; m++)
        if (is_harmonic(hybrid->keywords[m])) {
          angle = hybrid->styles[m];
          break;
        }
    }
  }
  if (!angle) error->all(FLERR, "Compute stress/angle/asym requires angle_style harmonic");

  int dim;
  k = (double *) angle->extract("k", dim);
  theta0 = (double *) angle->extract("theta0", dim);
  if (!k || !theta0) error->all(FLERR, "Compute stress/angle/asym could not extract k and theta0");
}

/* ---------------------------------------------------------------------- */

void ComputeStressAngleAsym::compute_peratom()
{
  int i, n, i1, i2, i3, type;
  double delx1, dely1, delx2, dely2;
  double rsq1, rsq2, r1, r2, c, s, a, a11, a12, a22;
  double f1x, f1y, f3x, f3y;
  double wa[4], wb[4];

  invoked_peratom = update->ntimestep;

  // grow local stress array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(stress);
    nmax = atom->nmax;
    memory->create(stress, nmax, 4, "stress/angle/asym:stress");
    array_atom = stress;
  }

  // zero local and ghost (if newton_bond) atoms

  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  int ntotal = nlocal + (newton_bond ? atom->nghost : 0);

  for (i = 0; i < ntotal; i++) stress[i][0] = stress[i][1] = stress[i][2] = stress[i][3] = 0.0;

  // recompute the forces of AngleHarmonic::compute from the positions; the angle list
  // already uses the closest images, so no minimum image

  double **x = atom->x;
  int *setflag = angle->setflag;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;

  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // angles turned off (bond/react, delete_bonds) or of another hybrid substyle

    if (type <= 0 || !setflag[type]) continue;

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    rsq1 = delx1 * delx1 + dely1 * dely1;
    r1 = sqrt(rsq1);

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    rsq2 = delx2 * delx2 + dely2 * dely2;
    r2 = sqrt(rsq2);

    c = delx1 * delx2 + dely1 * dely2;
    c /= r1 * r2;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    s = sqrt(1.0 - c * c);
    if (s < SMALL) s = SMALL;
    s = 1.0 / s;

    a = -2.0 * k[type] * (acos(c) - theta0[type]) * s;
    a11 = a * c / rsq1;
    a12 = -a / (r1 * r2);
    a22 = a * c / rsq2;

    f1x = a11 * delx1 + a12 * delx2;
    f1y = a11 * dely1 + a12 * dely2;
    f3x = a22 * delx2 + a12 * delx1;
    f3y = a22 * dely2 + a12 * dely1;

    // pair A: +f1 on i1, -f1 on i2; pair B: +f3 on i3, -f3 on i2
    // half of each pair's dipole to each of its atoms, columns xx, yy, xy, yx

    wa[0] = 0.5 * delx1 * f1x;
    wa[1] = 0.5 * dely1 * f1y;
    wa[2] = 0.5 * delx1 * f1y;
    wa[3] = 0.5 * dely1 * f1x;

    wb[0] = 0.5 * delx2 * f3x;
    wb[1] = 0.5 * dely2 * f3y;
    wb[2] = 0.5 * delx2 * f3y;
    wb[3] = 0.5 * dely2 * f3x;

    if (newton_bond || i1 < nlocal)
      for (int j = 0; j < 4; j++) stress[i1][j] += wa[j];
    if (newton_bond || i2 < nlocal)
      for (int j = 0; j < 4; j++) stress[i2][j] += wa[j] + wb[j];
    if (newton_bond || i3 < nlocal)
      for (int j = 0; j < 4; j++) stress[i3][j] += wb[j];
  }

  // communicate ghost contributions between neighbor procs

  if (newton_bond) comm->reverse_comm(this);

  // zero virial of atoms not in group, only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) stress[i][0] = stress[i][1] = stress[i][2] = stress[i][3] = 0.0;

  // convert to stress*volume units = -pressure*volume

  double nktv2p = -force->nktv2p;
  for (i = 0; i < nlocal; i++)
    for (int j = 0; j < 4; j++) stress[i][j] *= nktv2p;
}

/* ---------------------------------------------------------------------- */

int ComputeStressAngleAsym::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = stress[i][0];
    buf[m++] = stress[i][1];
    buf[m++] = stress[i][2];
    buf[m++] = stress[i][3];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeStressAngleAsym::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    stress[j][0] += buf[m++];
    stress[j][1] += buf[m++];
    stress[j][2] += buf[m++];
    stress[j][3] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeStressAngleAsym::memory_usage()
{
  double bytes = (double) nmax * 4 * sizeof(double);
  return bytes;
}
