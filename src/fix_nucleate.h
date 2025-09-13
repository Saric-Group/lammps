#ifdef FIX_CLASS

FixStyle(nucleate, FixNucleate)

#else

#ifndef LMP_FIX_NUCLEATE_H
#define LMP_FIX_NUCLEATE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNucleate : public Fix {
  public:
    FixNucleate(class LAMMPS *, int, char **);
    ~FixNucleate();

    int setmask() override;
    void init() override;
    void init_list(int, class NeighList *) override;
    void post_integrate() override;
    void post_integrate_respa(int, int) override;
    void post_constructor() override;

    void average_normals();
    void initialise_v(double *, const double);
    void check_ownership(double*, int&);
    void check_overlap(double*, int&);
    void random_orientation_on_plane(double*, double*);

  private:
    int nlevels_respa;
    int seed; // seed for random number generator
    int warnflag; // whether to warn if no nucleation events occur
    double prob; // probability (0-1) of nucleation event occurring per particle
    double r_surf; // distance of particle from surface
    double overlap, overlapsq; // minimum distance from existing atoms
    double insert_sigma; // bond length of inserted dimer
    int bond_type; // bond type of inserted dimer

    double** insert_coords; // store coords of inserted atoms to check overlap
    int* filled_coords_flags; // flags to indicate which insert_coords are filled
    
    class RanMars *random; // random number generator

    class NeighList *list;

    class AtomVecEllipsoid *avec;
};
}

#endif
#endif