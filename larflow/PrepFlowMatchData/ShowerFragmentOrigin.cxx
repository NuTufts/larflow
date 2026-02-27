#include "ShowerFragmentOrigin.h"

namespace larflow {
namespace prep {

void ShowerFragmentOrigin::clear() 
{
  shower_fragments_v.clear(); ///< set of 3d points
  shower_edep_v.clear();      ///< plane edep for each point
  shower_pointindices_v.clear(); ///< list of point indices in the cluster
  shower_trackid_v.clear();   ///< geant4 trackid of shower fragment
  shower_pid_v.clear();       ///< pdg code of shower fragment
  shower_istrunk_v.clear();   ///< 1 if cluster is start of EM shower, 2 if cluster is secondary fragment
  shower_type_v.clear();
  shower_startpt_v.clear();   
  shower_originpt_v.clear();
  shower_pret0shiftedstart_v.clear();

}

}
}