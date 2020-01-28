#ifndef __LARFLOW_CRTTRACK_MATCH_H__
#define __LARFLOW_CRTTRACK_MATCH_H__

#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/crttrack.h"
#include "larlite/core/DataFormat/opflash.h"
#include "larlite/core/LArUtil/SpaceChargeMicroBooNE.h"

#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace crtmatch {

  class CRTTrackMatch : public larcv::larcv_base {

  public:

    CRTTrackMatch();
    virtual ~CRTTrackMatch();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void set_max_iters( int tries ) { _max_iters = tries; };
    void set_col_neighborhood( int cols )   { _col_neighborhood = cols; };
    void set_max_fit_step_size( float cm )  { _max_fit_step_size = cm; };
    void set_max_last_step_size( float cm ) { _max_last_step_size = cm; };
    void make_debug_images( bool make_debug ) { _make_debug_images = make_debug; };

    // struct to assemble info
    struct crttrack_t {

      int crt_track_index;                 
      const larlite::crttrack* pcrttrack;  // pointer to original CRT track object

      std::vector< std::vector<double> >   hit_pos_vv;      // modified CRT hit positions
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

    // algorithm parameters
    int _max_iters;            //< max attempts to improve CRT track fit
    int _col_neighborhood;     //< window to search for charge around track
    float _max_fit_step_size;  //< max step size when fittering
    float _max_last_step_size; //< max step size for final fit
    std::string _adc_producer; //< name of ADC image tree
    std::string _crttrack_producer; //< name of CRT track producer
    std::vector<std::string> _opflash_producer_v;  //< name of opflash producer
    bool _make_debug_images;

  };
  
}
}

#endif
