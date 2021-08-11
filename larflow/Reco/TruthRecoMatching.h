#ifndef __LARFLOW_RECO_TRUTH_RECO_MATCHING_H__
#define __LARFLOW_RECO_TRUTH_RECO_MATCHING_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @class TruthRecoMatching
   * @brief Match neutrino candidate reco 
   *
   * We quantify reco completeness by pixel overlap.
   * 
   */
  
  class TruthRecoMatching : public larcv::larcv_base {

  public:
    
    TruthRecoMatching();
    virtual ~TruthRecoMatching() {};

    /* typedef struct Match_t { */
    /*   int istrack;  //1: is track, 0: is shower */
    /*   int recoidx;  // reco index in track container */
    /*   int matchid;  // 1: matched to track, 0: matched to mcshower, -1: no match */
    /*   int matchidx; // index of mctrack or mcshower object */
    /*   float match_score; */
    /* }; */

    struct PixelMatch_t {
      PixelMatch_t() {};
      std::vector<int> true_visible_pixels;
      std::vector<int> reco_visible_pixels;
      std::vector<int> misreco_visible_pixels;    
      std::vector<float> frac_visible_pixels;      

      std::vector<float> true_visible_charge;
      std::vector<float> reco_visible_charge;
      std::vector<float> misreco_visible_charge;    
      std::vector<float> frac_visible_charge;      
    };
    
    PixelMatch_t calculate_pixel_completeness( const std::vector<larcv::Image2D>& wire_v,
                                               const std::vector<larcv::Image2D>& instance_v,
                                               const larflow::reco::NuVertexCandidate& nuvtx,
                                               const float adc_threshold=10.0 );
  };

}
}

#endif
