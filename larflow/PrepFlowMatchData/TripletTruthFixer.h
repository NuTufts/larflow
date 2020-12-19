#ifndef __LARFLOW_PREP_TRIPLETTRUTHFIXER_H__
#define __LARFLOW_PREP_TRIPLETTRUTHFIXER_H__

#include "larcv/core/DataFormat/IOManager.h"

#include "larflow/Reco/cluster_functions.h"
#include "PrepMatchTriplets.h"

namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData
   * @class TripletTruthFixer
   * @brief Uses clustering and larlite truth to repair triplet spacepoint instance labels
   */
  class TripletTruthFixer {

  public:

    TripletTruthFixer() {};
    virtual ~TripletTruthFixer() {};
    
    void calc_reassignments( PrepMatchTriplets& tripmaker,
                             larcv::IOManager& iolcv );

    void _cluster_same_showerpid_spacepoints( std::vector<larflow::reco::cluster_t>& cluster_v,
                                              std::vector<int>& pid_v,
                                              larflow::prep::PrepMatchTriplets& tripmaker,
                                              bool reassign_instance_labels );

    void _reassignSmallTrackClusters( larflow::prep::PrepMatchTriplets& tripmaker,
                                      const std::vector< larcv::Image2D >& instanceimg_v,
                                      const float threshold );
      
    
  };

}
}

#endif
