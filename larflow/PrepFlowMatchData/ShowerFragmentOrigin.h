#ifndef __LARFLOW_PREP_SHOWER_FRAGMENT_ORIGIN_H__
#define __LARFLOW_PREP_SHOWER_FRAGMENT_ORIGIN_H__

#include <vector>

namespace larflow {
namespace prep {

class ShowerFragmentOrigin {

public:

  ShowerFragmentOrigin() {};
  ~ShowerFragmentOrigin() {};

  typedef std::vector< std::vector<float> > shcluster_t;

  std::vector< shcluster_t > shower_fragments_v; ///< set of 3d points
  std::vector< shcluster_t > shower_edep_v;      ///< wire plane edep for each pt in the cluster
  std::vector< std::vector<long> > shower_pointindices_v; ///< list of indices for the spacepoints
  std::vector< int >       shower_trackid_v;     ///< geant4 trackid of shower fragment
  std::vector< int >       shower_pid_v;         ///< pdg code of shower fragment
  std::vector< int >       shower_istrunk_v;     ///< 1 if cluster is start of EM shower, 2 if cluster is secondary fragment
  std::vector< int >       shower_type_v;        ///< 0: inside-neutrino, 1: outside, 2: inside-cosmic-origin
  std::vector< std::vector<float> > shower_startpt_v;  ///< start of fragment
  std::vector< std::vector<float> > shower_originpt_v; ///< origin of fragment
  std::vector< std::vector<float> > shower_pret0shiftedstart_v;  ///< origin of fragment from geant4 truth, no t0-shift applied

  void clear();

};


}
}

#endif