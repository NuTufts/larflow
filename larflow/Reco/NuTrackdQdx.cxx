#include "NuTrackdQdx.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/RecoTools/DetUtils.h"

#include "TrackdQdx.h"

namespace larflow {
namespace reco {

  int NuTrackdQdx::process_nuvertex_tracks( larcv::IOManager& iolcv,
					    larflow::reco::NuVertexCandidate& nuvtx )
  {

    if ( nuvtx.track_v.size()==0 )
      return 0;
    
    const int tpcid  = nuvtx.tpcid;
    const int cryoid = nuvtx.cryoid;
    
    std::vector< larlite::track > track_dqdx_v;
    track_dqdx_v.reserve( nuvtx.track_v.size() );

    // wire plane images for getting dqdx later
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();
    
    std::vector< const larcv::Image2D* > ptpc_adc_v
      = ublarcvapp::recotools::DetUtils::getTPCImages( adc_v, tpcid, cryoid );    
    
    for (int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++) {
      auto& fitted = nuvtx.track_v.at(itrack);
      auto& hitcluster = nuvtx.track_hitcluster_v.at(itrack);

      // fitted with dqdx
      larflow::reco::TrackdQdx dqdx_algo;
      larlite::track track_dqdx;
      bool success = false;
      try {
	track_dqdx = dqdx_algo.calculatedQdx( fitted, hitcluster, ptpc_adc_v );
	success = true;
      }
      catch (...) {
	success = false;
      }
      if ( success ) {
	track_dqdx_v.emplace_back( std::move(track_dqdx) );
      }
      else {
	track_dqdx_v.push_back( fitted );
      }
    }
    std::swap( nuvtx.track_v, track_dqdx_v );

    return 0;
  }

}
}
