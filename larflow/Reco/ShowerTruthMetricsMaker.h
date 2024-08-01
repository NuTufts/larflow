#ifndef __LARFLOW_RECO_SHOWER_TRUTH_METRICS_MAKER_H__
#define __LARFLOW_RECO_SHOWER_TRUTH_METRICS_MAKER_H__

/**
 * @class ShowerTruthMetricsMaker
 * 
 * @brief Calculations which we use to characterize/tag photons
 *
 * For example, we want to know where a photon started leaving energy in the detector
 * that we can observe. We also want some measure of how much energy was left behind.
 * and what a perfect reconstruction might label its start point and shower-direction to be.
 *
 * These quantities are intended to included in the Gen2 ntuple and for analysis.
 *
 */


#include "ublarcvapp/MCTools/MCPixelPGraph.h"


namespace larflow {
namespace reco {

  class ShowerTruthMetricsMaker {

  public:

    ShowerTruthMetricsMaker(){};
    virtual ~ShowerTruthMetricsMaker() {};

    static std::vector<float> getPhotonFirstEDepPosition( ublarcvapp::mctools::MCPixelPGraph& mcpg, const int trackid_of_shower );
    
    static std::vector<float> getPhotonTrunkLineSegment( ublarcvapp::mctools::MCPixelPGraph& mcpg, const int trackid_of_shower );
    
    //static std::vector<float> getPhotonFirstEDepTrunkDirection( const ublarcvapp::mctools::MCPixelPGraph& mcpg, const int trackid_of_shower );
    static std::vector<float> getPhotonTrunkPlanePixelSum( ublarcvapp::mctools::MCPixelPGraph& mcpg, const int trackid_of_shower );


  };
  
}
}

#endif
