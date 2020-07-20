#ifndef __LARFLOW_RECO_TRACK_OTFIT_H__
#define __LARFLOW_RECO_TRACK_OTFIT_H__

#include <vector>

namespace larflow {
namespace reco {

  class TrackOTFit {

  public:

    TrackOTFit() {};
    virtual ~TrackOTFit() {};


    void fit( std::vector< std::vector<float> >& initial_track,
              std::vector< std::vector<float> >& track_pts_w_feat_v );

    
    float d2_segment_point( const std::vector<float>& seg_start,
                            const std::vector<float>& seg_end,
                            const std::vector<float>& testpt );
    
    std::vector<float> grad_d2_wrt_segend( const std::vector<float>& seg_start,
                                           const std::vector<float>& seg_end,
                                           const std::vector<float>& testpt );

    void getLossAndGradient(  const std::vector< std::vector<float> >& initial_track,
                              const std::vector< std::vector<float> >& track_pts_w_feat_v,
                              float& loss,
                              std::vector<float>& grad );
    

    
    
  };
  
}
}

#endif
