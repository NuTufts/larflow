#ifndef __LARFLOW_RECO_TRACK_OTFIT_H__
#define __LARFLOW_RECO_TRACK_OTFIT_H__

#include <vector>

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


    void fit( std::vector< std::vector<float> >& initial_track,
              std::vector< std::vector<float> >& track_pts_w_feat_v );

    
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
    
  };
  
}
}

#endif