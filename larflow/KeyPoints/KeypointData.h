#ifndef __LARFLOW_KEYPOINT_KEYPOINTDATA_H__
#define __LARFLOW_KEYPOINT_KEYPOINTDATA_H__

#include "larcv/core/Base/larcv_base.h"

namespace larflow {
namespace keypoints {

  class KeypointData : public larcv::larcv_base {
  public:
    KeypointData();
    virtual ~KeypointData() {};

    int tpcid;
    int cryoid;
    std::vector< std::vector<float> > _kppos_v[6]; ///< container of true keypoint 3D positions in cm, for each of the 6 classes
    std::vector< std::vector<int> >   _kp_pdg_trackid_v[6]; ///< each entry maps (pdg, trackid) for truth meta-data matching    
    std::vector< std::vector<float> > _match_proposal_labels_v[6]; ///< provides the labels for triplet proposals made by larflow::prep::PrepMatchTriplets
    
    
  };

}
}

#endif
