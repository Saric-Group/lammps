#include "fix_nucleate.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

#include <set>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum { WARN, NOWARN }; // warnflag values

FixNucleate::FixNucleate(class LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  int iarg = 3;
  force_reneighbor = 1;
  next_reneighbor = -1;

  // necessary args
  nevery = utils::inumeric(FLERR, arg[iarg++], false, lmp);
  seed = utils::inumeric(FLERR, arg[iarg++], false, lmp);

  // standard values for kwargs
  prob = 1.;
  r_surf = 1.0;
  overlap = 0; overlapsq=0;
  warnflag = WARN;
  insert_sigma = 1.0;

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
    } else if (strcmp(arg[iarg],"insert_sigma") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Missing numeric parameter after insert_sigma kwarg.");
      
      insert_sigma = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"bond_type") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Missing numeric parameter after bond_type kwarg.");
      
      bond_type = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"nowarn") == 0) {
      warnflag = NOWARN;
      iarg += 1;
    } else if (strcmp(arg[iarg], "noffset") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Missing numeric parameter after offset kwarg.");

      noffset = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix nucleate command.");
    }
  }

  random = new RanMars(lmp, seed + comm->me);
}

FixNucleate::~FixNucleate() {
  delete random;
}

void FixNucleate::post_constructor() {
  if(!modify->get_fix_by_id("normal_tracking"))
    Fix* fix2 = modify->add_fix("normal_tracking all property/atom d_aligned");
}

void FixNucleate::init() {
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;

  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR, "Pair style ylz requires atom style ellipsoid");

  // need a half neighbor list, built every Nevery steps
  // addlipid does the same thing
  // but for some reason self implements add_request
  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
  force_reneighbor = 1;
}

int FixNucleate::setmask() {
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

void FixNucleate::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

void FixNucleate::post_integrate() {
  // TODO: initialise a lot of variables here outside of loops
  // TODO: remove alignment atom/property
  // TODO: forward_comm somehow breaks everything, would be nice if this could work at some point
  if ((update->ntimestep-noffset) % nevery) return;

  // figure out how many nucleation events to attempt this step
  int n_nucleate_group = 0;
  for (int i=0; i<atom->nlocal; i++) if (group->bitmask[igroup] & atom->mask[i]) n_nucleate_group++;
  int n_nucleate_local = prob * n_nucleate_group; // number of nucleation events to attempt this step
   // if prob is small, do a random check for probability
  if (n_nucleate_local == 0)
    if (random->uniform() < prob * n_nucleate_group) n_nucleate_local = 1;
  
  // pass through all MPI processes to find max number of nucleation events
  int max_nucleate_global;
  MPI_Allreduce(&n_nucleate_local, &max_nucleate_global, 1, MPI_INT, MPI_MAX, world);

  if (max_nucleate_global == 0) {
    if ((comm->me == 0) && (warnflag == WARN)) error->warning(FLERR,"Fix nucleate: no nucleation events this step at rate %g", prob);
    return; // nothing to do this step
  }

  // make arrays for communicating coordinates
  // already initialised to 0.0
  memory->create(insert_coords, comm->nprocs*max_nucleate_global, 6, "fix_nucleate:insert_coords");
  memory->create(filled_coords_flags, comm->nprocs*max_nucleate_global, "fix_nucleate:filled_coords_flags");
  
  // initialize arrays so the MPI_Allreduce works correctly
  for(int i=0;i<comm->nprocs*max_nucleate_global;i++) {
    insert_coords[i][0] = insert_coords[i][1] = insert_coords[i][2] = 0.0;
    insert_coords[i][3] = insert_coords[i][4] = insert_coords[i][5] = 0.0;
    filled_coords_flags[i] = 0;
  }
  
  // create a set of unique random indices to nucleate at
  std::set<int> nucleate_indices;
  while (nucleate_indices.size() < n_nucleate_local) {
    int index = static_cast<int>(random->uniform() * n_nucleate_group);
    nucleate_indices.insert(index);
  }
  std::set<int>::iterator nuc_iterator = nucleate_indices.begin();

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
  
  int flag,cols;
  int index1 = atom->find_custom("aligned",flag,cols);
  double *d_aligned = atom->dvector[index1];

  int n_added_local = 0;
  int nright_group = -1; // start at -1 for so it serves as indices

  for(int iatom=0; iatom<nlocal; iatom++) {
    int ilocal = ilist[iatom];
    // make sure we have an atom from target group
    if (!(group->bitmask[igroup] & atom->mask[ilocal])) continue;
    nright_group++;
    // check if we're at the right nucleation index
    if (nright_group != *nuc_iterator) {
      continue;
    }
    
    int itype = atom->type[ilocal];
    double *x = atom->x[ilocal];

    double normal[3]; 
    double rotation[3][3];

    double* iquat = avec->bonus[atom->ellipsoid[ilocal]].quat;
    MathExtra::quat_to_mat_trans(iquat, rotation);
    // taken from pair_ylz.cpp
    // does this mean longest axis has to be x?
    // or are the ellipsoid axis sorted by length?
    normal[0] = rotation[0][0];
    normal[1] = rotation[0][1];
    normal[2] = rotation[0][2];

    double x_insert[6], x_starting[3], orientation_vec[3];
    x_starting[0] = x[0];
    x_starting[1] = x[1];
    x_starting[2] = x[2];
    x_insert[0] = x[0] - r_surf*normal[0]; // ylz normals point out, so negative sign
    x_insert[1] = x[1] - r_surf*normal[1];
    x_insert[2] = x[2] - r_surf*normal[2];
    random_orientation_on_plane(normal, orientation_vec);

    // shift dimer so center of mass is at equilibrium distance from surface
    x_insert[0] -= 0.5*insert_sigma*orientation_vec[0];
    x_insert[1] -= 0.5*insert_sigma*orientation_vec[1];
    x_insert[2] -= 0.5*insert_sigma*orientation_vec[2];
    x_insert[3] = x_insert[0] + insert_sigma*orientation_vec[0];
    x_insert[4] = x_insert[1] + insert_sigma*orientation_vec[1];
    x_insert[5] = x_insert[2] + insert_sigma*orientation_vec[2];

    double norm = sqrt(x_starting[0]*x_starting[0] + x_starting[2]*x_starting[2]);
    d_aligned[ilocal] = x_starting[0] * normal[0] / norm + x_starting[2] * normal[2] / norm;
    
    // apply PBC
    domain->minimum_image(FLERR, x_insert[0], x_insert[1], x_insert[2]);
    domain->minimum_image(FLERR, x_insert[3], x_insert[4], x_insert[5]);

    // add to global vector of candidate coords
    insert_coords[comm->me*max_nucleate_global + n_added_local][0] = x_insert[0];
    insert_coords[comm->me*max_nucleate_global + n_added_local][1] = x_insert[1];
    insert_coords[comm->me*max_nucleate_global + n_added_local][2] = x_insert[2];
    insert_coords[comm->me*max_nucleate_global + n_added_local][3] = x_insert[3];
    insert_coords[comm->me*max_nucleate_global + n_added_local][4] = x_insert[4];
    insert_coords[comm->me*max_nucleate_global + n_added_local][5] = x_insert[5];

    filled_coords_flags[comm->me*max_nucleate_global + n_added_local] = 1;
    n_added_local++;
    ++nuc_iterator;

    if (n_added_local == n_nucleate_local) break; // break if we've added enough atoms this step
  }

  // communicate all inserted coords to all procs
  for (int i=0; i < comm->nprocs*max_nucleate_global; i++) {
    MPI_Allreduce(MPI_IN_PLACE, insert_coords[i], 6, MPI_DOUBLE, MPI_SUM, world);
  }
  MPI_Allreduce(MPI_IN_PLACE, filled_coords_flags, comm->nprocs*max_nucleate_global, MPI_INT, MPI_SUM, world);
  
  // now that candidates have been communicated, check for overlaps and insert
  int owned_by_proc = 0;
  int overlapflag = 0;
  int is_mine[comm->nprocs*max_nucleate_global];
  int owned_by_me_left = 0, owned_by_me_right = 0;
  for (int icoord=0; icoord < comm->nprocs*max_nucleate_global; icoord++) {
    is_mine[icoord] = 0;
    if (!filled_coords_flags[icoord]) continue; // make sure some process wrote coords here

    check_ownership(insert_coords[icoord], owned_by_me_left);
    check_ownership(insert_coords[icoord]+3, owned_by_me_right);

    if (!(owned_by_me_left && owned_by_me_right)) continue;

    is_mine[icoord] = 1;
  }

  if (atom->map_style != Atom::MAP_NONE) atom->map_clear();
  atom->nghost = 0; // taken from bond/react and adsorb
  atom->avec->clear_bonus(); // no clue what this is for

  int my_insertions = 0, global_insertions = 0, nlocal_prev=atom->nlocal;
  for (int icoord=0; icoord < comm->nprocs*max_nucleate_global; icoord++) {
    if (!is_mine[icoord]) continue;
    
    // check if there is an overlap with existing atoms
    overlapflag = 0;
    check_overlap(insert_coords[icoord], overlapflag);
    check_overlap(insert_coords[icoord]+3, overlapflag);
    if (overlapflag) continue;
    
    atom->avec->create_atom(2, insert_coords[icoord]);
    atom->avec->create_atom(3, insert_coords[icoord]+3);

    for (int newind=atom->nlocal-2; newind<atom->nlocal; newind++) {
      initialise_v(atom->v[newind], atom->rmass[newind]);
      atom->mask[newind] = 1 | groupbit;
      atom->image[newind] = ((imageint) IMGMAX << IMG2BITS) |
          ((imageint) IMGMAX << IMGBITS) | IMGMAX;
      modify->create_attribute(newind);
    }

    my_insertions++;
  }

  // send around how many insertions each proc made
  MPI_Scan(&my_insertions, &global_insertions, 1, MPI_INT, MPI_SUM, world);
  int n_insert_before = global_insertions - my_insertions, n_bonds = atom->num_bond[0]; // TODO: should be num_bond[itype]?
  // and use that knowledge to properly set atom tags and molecule IDs
  for (int iinsert=0; iinsert<my_insertions; iinsert++) {
    atom->tag[nlocal_prev+2*iinsert] = max_atomtag + (2 * n_insert_before) + 2 * iinsert + 1;
    atom->tag[nlocal_prev+(2*iinsert)+1] = max_atomtag + (2 * n_insert_before) + 2 * iinsert + 2;
    atom->molecule[nlocal_prev+2*iinsert] = maxmol_all + n_insert_before + iinsert + 1;
    atom->molecule[nlocal_prev+(2*iinsert)+1] = maxmol_all + n_insert_before + iinsert + 1;
  }

  // if (atom->tag_enable) atom->tag_extend(); // <- alternative to handling myself
  atom->tag_check(); // if this fails, I did something wrong

  // communicate total number of atoms in system to all procs
  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (atom->natoms < 0)
    error->all(FLERR,"Too many total atoms");

  // reset atom->map, no idea what this does
  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }

  // now that map exists again, add bonds back in
  for (int iinsert=0; iinsert<my_insertions; iinsert++) {
    // as taken from create_bonds.cpp
    atom->bond_type[atom->map(atom->tag[nlocal_prev+2*iinsert])][0] = 1; // TODO: should be a variable bond type
    atom->bond_atom[atom->map(atom->tag[nlocal_prev+2*iinsert])][0] = atom->tag[nlocal_prev+(2*iinsert)+1];
    atom->num_bond[atom->map(atom->tag[nlocal_prev+2*iinsert])] = 1;
    
    // and, importantly, also add marker for 1-2 special interaction (bonds)
    atom->nspecial[nlocal_prev+2*iinsert][0] = 1;
    atom->special[nlocal_prev+2*iinsert][0] = atom->tag[nlocal_prev+(2*iinsert)+1];
    atom->nspecial[nlocal_prev+2*iinsert+1][0] = 1;
    atom->special[nlocal_prev+2*iinsert+1][0] = atom->tag[nlocal_prev+(2*iinsert)];

    // this is vital and I have no idea what it does
    // bond info doesn't need to be communicated since we're only adding local bonds
    // taken from fix_bond_create.cpp
    rebuild_special_one(nlocal_prev+2*iinsert);
    rebuild_special_one(nlocal_prev+2*iinsert+1);

    if (!force->newton_bond) {
      atom->bond_type[atom->map(atom->tag[nlocal_prev+2*iinsert])][bond_type] = 1;
      atom->bond_atom[atom->map(atom->tag[nlocal_prev+2*iinsert])][bond_type] = atom->tag[nlocal_prev+2*iinsert];
      atom->num_bond[atom->map(atom->tag[nlocal_prev+2*iinsert])] = 1;
    }
  }

  // also all taken from create_bonds.cpp
  bigint nbonds = 0;
  for (int i = 0; i < nlocal; i++) nbonds += atom->num_bond[i];
  MPI_Allreduce(MPI_IN_PLACE,&nbonds,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (!force->newton_bond) atom->nbonds /= 2;

  next_reneighbor = update->ntimestep;

  memory->destroy(insert_coords);
  memory->destroy(filled_coords_flags);
}

void FixNucleate::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa - 1) post_integrate();
}

void FixNucleate::initialise_v(double *v, const double mass) {
  // initialise velocity from Maxwell-Boltzmann distribution
  // based on Chris' bugfix of fix bond/react
  double temperature = 1.; // insert at system temperature, TODO: make variable?
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

void FixNucleate::check_overlap(double* coords, int& overlapflag) {
  double delx, dely, delz, rsq;
  if (overlapsq > 0) {
    for (int i = 0; i < atom->nlocal; i++) {
      delx = coords[0] - atom->x[i][0];
      dely = coords[1] - atom->x[i][1];
      delz = coords[2] - atom->x[i][2];
      domain->minimum_image(FLERR, delx,dely,delz);
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < overlapsq) {
        overlapflag = 1;
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

void FixNucleate::random_orientation_on_plane(double* normal_vec, double *out_vec)
{
  // TODO: this function allocates a lot of memory, optimise to use preallocated variables
  double phi = 2*M_PI*random->uniform(); // random angle on cylinder

  // normalise normal vector just in case
  double norm = 1./sqrt(normal_vec[0]*normal_vec[0] + normal_vec[1]*normal_vec[1] + normal_vec[2]*normal_vec[2]);
  for(uint i=0;i<3;i++) normal_vec[i] *= norm;

  // make a trial vector in y direction to rotate
  double* buff_vec = new double[3];
  buff_vec[0] = 0;
  buff_vec[1] = 1;
  buff_vec[2] = 0;

  // now use shortened version of Rodrigues' rotation formula (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
  // compute (v cos(phi)) + (k x v sin(phi))
  out_vec[0] = buff_vec[0] * cos(phi) + (normal_vec[1] * buff_vec[2] - normal_vec[2] * buff_vec[1]) * sin(phi);
  out_vec[1] = buff_vec[1] * cos(phi) + (normal_vec[2] * buff_vec[0] - normal_vec[0] * buff_vec[2]) * sin(phi);
  out_vec[2] = buff_vec[2] * cos(phi) + (normal_vec[0] * buff_vec[1] - normal_vec[1] * buff_vec[0]) * sin(phi);
  
  // technically this shouldn't be necessary since both vectors are unit vectors
  norm = 1./sqrt(out_vec[0]*out_vec[0] + out_vec[1]*out_vec[1] + out_vec[2]*out_vec[2]);
  for(uint i=0;i<3;i++) out_vec[i] *= norm;
  
  delete [] buff_vec;
}

void FixNucleate::rebuild_special_one(int m)
{
  // FelixWodaczek: taken from fix_bond_create.cpp, don't fully understand it
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int maxspecial = atom->maxspecial;
  tagint* copy = new tagint[maxspecial*maxspecial + maxspecial];

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR, Error::NOLASTLINE, "Fix {} needs ghost atoms from further away", style);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR, Error::NOLASTLINE, "Special list size exceeded in fix {}", style);

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR, Error::NOLASTLINE, "Fix {} needs ghost atoms from further away", style);
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR, Error::NOLASTLINE, "Special list size exceeded in fix {}", style);

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
  
  delete [] copy;
}

int FixNucleate::dedup(int nstart, int nstop, tagint *copy)
{
  // FelixWodaczek: taken from fix_bond_create.cpp, don't fully understand it
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}
