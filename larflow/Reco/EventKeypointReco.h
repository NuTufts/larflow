#ifndef __LARFLOW_RECO_EVENT_KEYPOINTRECO_H__
#define __LARFLOW_RECO_EVENT_KEYPOINTRECO_H__

#include <iostream>
#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"

namespace larflow {
namespace reco {
  /**
   * @class EventKeypointReco
   * @brief Process keypoint scores and form keypoints
   */
  
  class EventKeypointReco : public larcv::larcv_base {

  public:
    EventKeypointReco()
      : larcv::larcv_base("EventKeypointReco")
      {};
    virtual ~EventKeypointReco() {};

    void process_larmatch_v2( larlite::storage_manager& ioll,
			      std::string hittree );
    

  };

  
}
}

#endif
