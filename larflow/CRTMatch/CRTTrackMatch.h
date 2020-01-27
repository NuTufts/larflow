#ifndef __LARFLOW_CRTTRACK_MATCH_H__
#define __LARFLOW_CRTTRACK_MATCH_H__

#include <vector>

#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/crttrack.h"
#include "larlite/core/DataFormat/opflash.h"
#include "larlite/core/LArUtil/SpaceChargeMicroBooNE.h"

namespace larflow {
namespace crtmatch {

  class CRTTrackMatch {

  public:

    CRTTrackMatch();
    virtual ~CRTTrackMatch();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    // struct to assemble info
    struct crttrack_t {

      int crt_track_index;      
      const larlite::crttrack* pcrttrack;

      std::vector< std::vector< int > >    pixellist_vv[3]; // (row,col) of pixels near line
      std::vector< float >                 pixelrad_vv[3];  // pixel radius from crt-track line
      std::vector< std::vector< double > > pixelpos_vv;     // pixel radius from crt-track line
      std::vector< std::vector<int> >      pixelcoord_vv;   // pixel coordinates
      std::vector< float >                 totalq_v;        // total charge along path
      float                                len_intpc_sce;

      int opflash_index;
      const larlite::opflash*  opflash;

      crttrack_t( int idx, const larlite::crttrack* ptrack )
      : crt_track_index(idx),
        pcrttrack( ptrack ),
        totalq_v( {0,0,0} ),
        len_intpc_sce(0.0)
      {};
      
    };

    // methods
    crttrack_t _collect_chargepixels_for_track( const larlite::crttrack& crt,      
                                                const std::vector<larcv::Image2D>& adc_v );

    larutil::SpaceChargeMicroBooNE* _sce;

  };
  
}
}

#endif
