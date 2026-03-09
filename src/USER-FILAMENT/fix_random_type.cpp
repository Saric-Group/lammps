#include "fix_random_type.h"
#include <cstring>
#include "atom.h"
#include "update.h"
#include "random_park.h"
#include "error.h"
#include "comm.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixRandomType::FixRandomType(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg != 8) error->all(FLERR,"Syntax: fix ID group random/type Nevery seed fraction type_from type_to");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  int seed = utils::inumeric(FLERR,arg[4],false,lmp);
  fraction = utils::numeric(FLERR,arg[5],false,lmp);
  type_from = utils::inumeric(FLERR,arg[6],false,lmp);
  type_to = utils::inumeric(FLERR,arg[7],false,lmp);

  if (nevery <= 0 || fraction < 0.0 || fraction > 1.0)
    error->all(FLERR,"Illegal fix random/type command");

  // Initialize RNG uniquely per MPI rank to prevent identical selections
  random = new RanPark(lmp, seed + comm->me);
}

FixRandomType::~FixRandomType() {
  delete random;
}

int FixRandomType::setmask() {
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

void FixRandomType::end_of_step() {
  if (update->ntimestep % nevery != 0) return;

  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // Revert any existing active sites
      if (type[i] == type_to) {
        type[i] = type_from; 
      }
      // Assign for new active sites
      if (type[i] == type_from) {
        if (random->uniform() < fraction) {
          type[i] = type_to;
        }
      }
    }
  }
}