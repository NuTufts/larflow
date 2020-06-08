#ifndef __PCA_TRACKER_H__
#define __PCA_TRACKER_H__

#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  class PCATracker : public larcv::larcv_base {

  public:

    PCATracker() {};
    virtual ~PCATracker() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );


    struct StartSeed_t {

      std::vector<float> start;
      std::vector< std::vector<float> > pca_v;
      std::vector< float > eigenval_v;
      int _cluster_idx;

      float getpca12ratio() const { return eigenval_v[1]/eigenval_v[0]; };
      float getpca13ratio() const { return eigenval_v[2]/eigenval_v[0]; };

      bool operator<( const StartSeed_t& rhs ) const {
        if ( getpca12ratio() < rhs.getpca12ratio() ) return true;
        return false;
      };
      
    };

    struct SegmentHit_t {
      int index;
      float r;
      float s;
      bool operator<( const SegmentHit_t& rhs ) const {
        if ( s<rhs.s ) return true;
        else if (s==rhs.s && r<rhs.r) return true;
        return false;
      };
      SegmentHit_t( float ss, float rr, int idx )
      : index(idx),
        r(rr),
        s(ss)
      {};
    };

    struct Segment_t {
      std::vector<float> start_v;
      float len;
      std::vector<float> end_v;
      std::vector<SegmentHit_t> seg_v;
      std::vector< std::vector<float> > pca_v;
      std::vector< float > eigenval_v;      
    };
    
  protected:

    std::vector< StartSeed_t > _trackseeds_v;
    void _make_keypoint_seeds( larlite::storage_manager& ioll );

  protected:
    
    Segment_t _get_next_segment( const std::vector<float> start,
                                 const std::vector<float> start_dir,
                                 const larlite::event_larflow3dhit& hit_v,
                                 std::vector<int>& used_v );

    std::vector<Segment_t> _build_track( const StartSeed_t& seed,
                                         const larlite::event_larflow3dhit& hit_v,
                                         std::vector<int>& used_v );

    bool _decide_action( Segment_t& seg,
                         const larlite::event_larflow3dhit& hit_v,
                         std::vector<int>& used_v );
    

  };

    

}
}

#endif
