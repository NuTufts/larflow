#ifndef __LARFLOW_RECO_TRACK_OTFIT_H__
#define __LARFLOW_RECO_TRACK_OTFIT_H__

#include <vector>

namespace larcv {  
  class Image2D;
}

namespace larlite {
  class larflow3dhit;
}

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class TrackOTFit
   * @brief Least squares fit of vertex location to prong clusters
   *
   */
  class TrackOTFit {

  public:

    TrackOTFit() {};
    virtual ~TrackOTFit() {};


    static void fit_segment( std::vector< std::vector<float> >& initial_segment,
                             std::vector< std::vector<float> >& track_pts_w_feat_v,
                             const int _maxiters_=100,
                             const float lr=0.1 );

    
    static float d2_segment_point( const std::vector<float>& seg_start,
                            const std::vector<float>& seg_end,
                            const std::vector<float>& testpt );
    
    static std::vector<float> grad_d2_wrt_segend( const std::vector<float>& seg_start,
                                                  const std::vector<float>& seg_end,
                                                  const std::vector<float>& testpt );

    static void getLossAndGradient(  const std::vector< std::vector<float> >& initial_track,
                                     const std::vector< std::vector<float> >& track_pts_w_feat_v,
                                     float& loss,
                                     std::vector<float>& grad );

    static void getWeightedLossAndGradient(  const std::vector< std::vector<float> >& initial_track,
                                             const std::vector< std::vector<float> >& track_pts_w_feat_v,
                                             float& loss,
                                             float& totweight,
                                             std::vector<float>& grad );

    static void addLarmatchScoreAndChargeFeatures( std::vector< std::vector<float> >& point_v,
                                                   const std::vector<larlite::larflow3dhit>& lfhit_v,                                                   
                                                   const std::vector<larcv::Image2D>& adc_v );

    
  };
  
}
}

#endif
