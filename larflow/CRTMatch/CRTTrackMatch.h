#ifndef __LARFLOW_CRTTRACK_MATCH_H__
#define __LARFLOW_CRTTRACK_MATCH_H__

#include <vector>

#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/crttrack.h"
#include "larlite/core/DataFormat/opflash.h"
#include "larlite/core/LArUtil/SpaceChargeMicroBooNE.h"

#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace crtmatch {

  class CRTTrackMatch {

  public:

    CRTTrackMatch();
    virtual ~CRTTrackMatch();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void set_max_iters( int tries ) { _max_iters = tries; };

    // struct to assemble info
    struct crttrack_t {

      int crt_track_index;      
      const larlite::crttrack* pcrttrack;

      std::vector< std::vector< int > >    pixellist_vv[3]; // (row,col) of pixels near line
      std::vector< float >                 pixelrad_vv[3];  // pixel radius from crt-track line
      std::vector< std::vector< double > > pixelpos_vv;     // pixel radius from crt-track line
      std::vector< std::vector<int> >      pixelcoord_vv;   // pixel coordinates
      std::vector< float >                 totalq_v;        // total charge along path
      std::vector< float >                 toterr_v;
      float                                len_intpc_sce;
      float                                t0_usec;

      int opflash_index;
      const larlite::opflash*  opflash;

      crttrack_t( int idx, const larlite::crttrack* ptrack )
      : crt_track_index(idx),
        pcrttrack( ptrack ),
        totalq_v( {0,0,0} ),
        toterr_v( {0,0,0} ),
        len_intpc_sce(0.0)
      {};
      
    };

    // methods
    crttrack_t _find_optimal_track( const larlite::crttrack& crt,
                                    const std::vector<larcv::Image2D>& adc_v );

    
    crttrack_t _collect_chargepixels_for_track( const std::vector<double>& hit1_pos,
                                                const std::vector<double>& hit2_pos,
                                                const float t0_usec,
                                                const std::vector<larcv::Image2D>& adc_v,
                                                const float max_step_size,
                                                const int col_neighborhood );
    
    bool _propose_corrected_track( const CRTTrackMatch::crttrack_t& old,
                                   std::vector< double >& hit1_new,
                                   std::vector< double >& hit2_new );

    bool _crt_intersections( larflow::reco::cluster_t& track_cluster,
                             const larlite::crttrack& orig_crttrack,
                             std::vector< std::vector<double> >& panel_pos_v,
                             std::vector< double >& dist2_original_hits );
    


    std::string _str( const CRTTrackMatch::crttrack_t& data );
    

    larutil::SpaceChargeMicroBooNE* _sce;
    larutil::SpaceChargeMicroBooNE* _reverse_sce;
    int _max_iters;

  };
  
}
}

#endif
