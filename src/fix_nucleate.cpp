#include "fix_nucleate.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "math_extra.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

#include <set>

using namespace LAMMPS_NS;
using namespace FixConst;

FixNucleate::FixNucleate(class LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  int iarg = 3;

  // necessary args
  nevery = utils::inumeric(FLERR, arg[iarg++], false, lmp);
  seed = utils::inumeric(FLERR, arg[iarg++], false, lmp);

  // standard values for kwargs
  prob = 1.;
  r_surf = 1.0;
  overlap = 0; overlapsq=0;

  // parse kwargs
  while (iarg < narg) {
    if (strcmp(arg[iarg], "prob") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Missing numeric parameter after prob kwarg.");
            
      prob = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"Rsurf") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Missing numeric parameter after Rsurf kwarg.");
            
      r_surf = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"Roverlap") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Missing numeric parameter after Roverlap kwarg.");
            
      overlap = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      overlapsq = overlap*overlap;
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix nucleate command.");
    }
  }

  random = new RanMars(lmp, seed + comm->me);
}

FixNucleate::~FixNucleate() {}

void FixNucleate::init() {
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;

  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR, "Pair style ylz requires atom style ellipsoid");

  // need a half neighbor list, built every Nevery steps
  // addlipid does the same thing
  // but for some reason self implements add_request
  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);  
}

int FixNucleate::setmask() {
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

void FixNucleate::post_integrate() {

  int n_nucleate_local = prob * atom->nlocal; // number of nucleation events to attempt this step
  int max_nucleate_global;
  MPI_Allreduce(&n_nucleate_local, &max_nucleate_global, 1, MPI_INT, MPI_MAX, world);

  memory->create(insert_coords, comm->nprocs*max_nucleate_global, 6, "fix_nucleate:insert_coords");
  memory->create(filled_coords_flags, comm->nprocs*max_nucleate_global, "fix_nucleate:filled_coords_flags");

  // create a set of unique random indices to nucleate at
  std::set<int> nucleate_indices;
  while (nucleate_indices.size() < n_nucleate_local) {
    int index = static_cast<int>(random->uniform() * atom->nlocal);
    nucleate_indices.insert(index);
  }

  // not sure if necessary but both addlipid and bond/react do this
  comm->forward_comm();

  neighbor->build_one(list);

  int nglobal = atom->natoms; // TODO: generate a random array of atom indices to pass to all processes
  int nlocal = list->inum;
  int *ilist = list->ilist;

  // get maximum atom tag on all processors
  tagint *tag = atom->tag;
  tagint max_atomtag = 0;
  for (int i = 0; i < nlocal; i++) max_atomtag = MAX(max_atomtag,tag[i]); // max tag on this proc
  MPI_Allreduce(MPI_IN_PLACE,&max_atomtag,1,MPI_LMP_TAGINT,MPI_MAX,world); // max tag on all procs
  tagint maxmol_all = 0;
  for (int i = 0; i < atom->nlocal; i++) maxmol_all = MAX(maxmol_all, atom->molecule[i]);
  MPI_Allreduce(MPI_IN_PLACE, &maxmol_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  
  addatoms.clear();

  int n_added_local = 0;
  for(int nucleate_index : nucleate_indices) {
    int ilocal = ilist[nucleate_index];
    
    // make sure we have an atom from target group
    if (!(group->bitmask[igroup] & atom->mask[ilocal])) continue;

    int itype = atom->type[ilocal];
    double *x = atom->x[ilocal];

    double normal[3];
    double rotation[3][3];

    double* iquat = avec->bonus[ilocal].quat;
    MathExtra::quat_to_mat_trans(iquat, rotation);
    // taken from pair_ylz.cpp
    // does this mean longest axis has to be x?
    // or are the ellipsoid axis sorted by length?
    normal[0] = rotation[0][0];
    normal[1] = rotation[0][1];
    normal[2] = rotation[0][2];

    double x_insert[3];
    x_insert[0] = x[0] + r_surf*normal[0];
    x_insert[1] = x[1] + r_surf*normal[1];
    x_insert[2] = x[2] + r_surf*normal[2];
    // apply PBC
    domain->minimum_image(x_insert[0], x_insert[1], x_insert[2]);
    insert_coords[comm->me*max_nucleate_global + n_added_local][0] = x_insert[0];
    insert_coords[comm->me*max_nucleate_global + n_added_local][1] = x_insert[1];
    insert_coords[comm->me*max_nucleate_global + n_added_local][2] = x_insert[2];
    filled_coords_flags[comm->me*max_nucleate_global + n_added_local] = 1;
    n_added_local++;
  }

  // communicate all inserted coords to all procs
  MPI_Allreduce(MPI_IN_PLACE, insert_coords, comm->nprocs*max_nucleate_global*6, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, filled_coords_flags, comm->nprocs*max_nucleate_global, MPI_INT, MPI_SUM, world);

  // now that candidates have been communicated, check for overlaps and insert
  int owned_by_proc = 0;
  int overlapflag = 0;
  for (int icoord=0; icoord < comm->nprocs*max_nucleate_global; icoord++) {
    if (! filled_coords_flags[icoord]) continue; // make sure some process wrote coords here

    // check if this proc owns the coords to be inserted
    check_ownership(insert_coords[icoord], owned_by_proc);
    if (!owned_by_proc) continue;

    // check if there is an overlap with existing atoms
    check_overlap(insert_coords[icoord], overlapflag);
    if (overlapflag) continue;

    atom->avec->clear_bonus(); // no clue what this is for

    atom->avec->create_atom(2, insert_coords[icoord]);
    int newind = atom->nlocal - 1;
    atom->type[newind] = 2; // TODO: make variable
    MPI_Allreduce(MPI_IN_PLACE,&max_atomtag,1,MPI_LMP_TAGINT,MPI_MAX,world); // max tag on all procs
    atom->tag[newind] = ++max_atomtag;
    
    MPI_Allreduce(MPI_IN_PLACE, &maxmol_all, 1, MPI_LMP_TAGINT, MPI_MAX, world);
    atom->molecule[newind] = ++maxmol_all;

    initialise_v(atom->v[newind], atom->rmass[newind]);
    atom->natoms += 1;
    if (atom->natoms < 0)
      error->all(FLERR,"Too many total atoms");
    if (max_atomtag >= MAXTAGINT)
      error->all(FLERR,"New atom IDs exceed maximum allowed ID");

    atom->mask[newind] = 1 | groupbit;
    atom->image[newind] = 0;
    modify->create_attribute(newind);

    comm->forward_comm(); // not sure if needed
    // reset atom->map, no idea what this does
    if (atom->map_style != Atom::MAP_NONE) {
      atom->map_init();
      atom->map_set();
    }
  }
}

void FixNucleate::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa - 1) post_integrate();
}

void FixNucleate::initialise_v(double *v, const double mass) {
  // initialise velocity from Maxwell-Boltzmann distribution
  // based on Chris' bugfix of fix bond/react
  double temperature = force->temperature; // insert at system temperature, TODO: make variable?
  double vtnorm = sqrt( ( 12 * temperature * force->boltz ) / (mass * force->mvv2e ) );
  v[0] = vtnorm*(0.5-(random->uniform()));     // Chris 21/07/2023 added "0.5-"
  v[1] = vtnorm*(0.5-(random->uniform()));     // Chris 21/07/2023 added "0.5-"
  if (domain->dimension < 3) {
    v[2] = 0.0;
  }
  else {
    v[2] = vtnorm*(0.5-(random->uniform()));     // Chris 21/07/2023 added "0.5-"
  }
}

void FixNucleate::average_normals() {
  // calculate average normal vector for each atom in group
  // store in an atom variable
  // probably requires one neighborlist pass
  error->all(FLERR, "FixNucleate::average_normals is not implemented yet");

  // make sure nlist is up-to-date
  neighbor->build_one(list);

  int nlocal = list->inum;
  int *ilist = list->ilist;

  for(int ii=0; ii<nlocal; ii++) {
    int ilocal = ilist[ii];
    int itype = atom->type[ilocal];
    double *x = atom->x[ilocal];

    int numneigh = list->numneigh[ilocal];
    int* jlist = list->firstneigh[ilocal];
    for(int jj=0; jj<numneigh; jj++) {
      int jlocal = jlist[jj];
      jlocal &= NEIGHMASK;
      int jtype = atom->type[jlocal];
      double *xj = atom->x[jlocal];
    }
  }
}

void FixNucleate::check_overlap(double* coords, int& abortflag) {
  double delx, dely, delz, rsq;
  abortflag = 0;
  if (overlapsq > 0) {
    for (int i = 0; i < atom->nlocal; i++) {
      delx = coords[0] - atom->x[i][0];
      dely = coords[1] - atom->x[i][1];
      delz = coords[2] - atom->x[i][2];
      domain->minimum_image(delx,dely,delz);
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < overlapsq) {
        abortflag = 1;
        return;
      }
    }
  }
}

void FixNucleate::check_ownership(double* coords, int& flag) {
  double lamda[3];
  double* newcoord;

  double *sublo,*subhi;
  if (domain->triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  if (domain->triclinic) {
    domain->x2lamda(coords,lamda);
    newcoord = lamda;
  } else newcoord = coords;

  flag = 0;
  if (newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
      newcoord[1] >= sublo[1] && newcoord[1] < subhi[1] &&
      newcoord[2] >= sublo[2] && newcoord[2] < subhi[2]) flag = 1;
  else if (domain->dimension == 3 && newcoord[2] >= domain->boxhi[2]) {
    if (comm->layout != Comm::LAYOUT_TILED) {
      if (comm->myloc[2] == comm->procgrid[2]-1 &&
          newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
          newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
    } else {
      if (comm->mysplit[2][1] == 1.0 &&
          newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
          newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
    }
  } else if (domain->dimension == 2 && newcoord[1] >= domain->boxhi[1]) {
    if (comm->layout != Comm::LAYOUT_TILED) {
      if (comm->myloc[1] == comm->procgrid[1]-1 &&
          newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;
    } else {
      if (comm->mysplit[1][1] == 1.0 &&
          newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;
    }
  }
}

void insert_atoms() {
  check_overlap(x_insert, abortflag);
    if (abortflag) continue;

    // insert new atom
    AddAtom newatom;
    newatom.type = 2;
    newatom.mask = 1 | groupbit; // default group + this fix's group
    newatom.rmass = 1.0;
    newatom.x[0] = x_insert[0];
    newatom.x[1] = x_insert[1];
    newatom.x[2] = x_insert[2];
    initialise_v(newatom.v, newatom.rmass);
    newatom.image = 0; // image flags are set to 0 when atom is created
    // figure out tag when inserting

    n_added_local++;
  }
}