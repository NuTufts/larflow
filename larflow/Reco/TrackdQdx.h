#ifndef __LARFLOW_RECO_TRACKDQDX_H__
#define __LARFLOW_RECO_TRACKDQDX_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class TrackdQdx
   * @brief Calculates the dqdx for a given track cluster
   *
   */  
  class TrackdQdx : public larcv::larcv_base {

  public:

    TrackdQdx() {};
    ~TrackdQdx() {};

    /** 
     * @struct TrackPt_t
     * @brief info we collect for each space point matched to a track's set of line segments 
     */
    struct TrackPt_t {
      int hitidx; ///< index in the space point container
      int pid;  ///< assigned pid to points
      float s;  ///< distance along line segment path from the start of the track
      float res; ///< distance along line segment path from the end of the track
      float r; ///< distance of space point to closest line segment of track
      float q; ///< charge of space point on collection plane
      float dqdx; ///< dqdx of space point on collection plane
      float q_med; ///< median charge of all three planes
      float dqdx_med; ///< median dqdx of all three planes
      float lm; ///< muon-likelihood value
      float ll; ///< likelihood
      float llw; ///< likelihood weight
      std::vector<float> linept;  ///< point on current track line
      std::vector<float> pt;      ///< space point location
      std::vector<float> dir;     ///< direction of track line segment
      std::vector<float> err_v;   ///< vector from linept to space point (pt)
      std::vector<double> dqdx_v; ///< dqdx on all three planes
      /** @brief comparison operator used to sort by path along track */
      bool operator<( const TrackPt_t& rhs ) const
      {
        if ( s>rhs.s) return true;
        return false;
      };
    };

    typedef std::vector<TrackPt_t> TrackPtList_t;  ///< simple container for TrackPt_T
    
  public:
    
    larlite::track calculatedQdx( const larlite::track& track,
                                  const larlite::larflowcluster& trackhits,
                                  const std::vector<const larcv::Image2D*>& adc_v ) const;

    larlite::track calculatedQdx( const larlite::track& track,
                                  const larlite::larflowcluster& trackhits,
                                  const std::vector<larcv::Image2D>& adc_v ) const;
    
    std::vector< std::vector<float> > calculatedQdx2D( const larlite::track& lltrack,
                                                       const std::vector<const larcv::Image2D*>& adc_v,
                                                       const float stepsize ) const;
    

    protected:

    void _makeTrackPtInfo( const std::vector<float>& start,
                           const std::vector<float>& end,
                           const std::vector<float>& pt,
                           const std::vector<int>& imgcoord,
                           const std::vector<const larcv::Image2D*>& adc_v,
                           const int hitidx, 
                           const float r,
                           const float local_s,
                           const float global_s,                           
                           const float lm_score,
			   const int tpcid,
			   const int cryoid,
                           TrackdQdx::TrackPt_t& trkpt ) const;
    
    

  };


}
}
    

#endif
