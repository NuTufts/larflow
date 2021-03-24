#include "NuSelUnrecoCharge.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPixel2D.h"

#include "ClusterImageMask.h"

namespace larflow {
namespace reco {

  void NuSelUnrecoCharge::analyze( larcv::IOManager& iolcv,
                                   larlite::storage_manager& ioll,
                                   larflow::reco::NuVertexCandidate& nuvtx,
                                   larflow::reco::NuSelectionVariables& output )
  {

    // what are the measures?
    // (1) define nearby clusters:  clusters with hits some distance
    //     from the vertex, track end-points.
    // (2) can count number of hits in these clusters.
    // (3) can count total charge in these clusters on the three planes.

    // we do we get the clusters?
    // we've also lost track of where our clusters came from?

    // first thing to do is make a mask of where our charge is.
    
    
    std::vector< std::string > cluster_producers
      = { "trackprojsplit_wcfilter", "showergoodhit" };

    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_img->as_vector();

    ClusterImageMask masker;
    std::vector<larcv::Image2D> nuvtx_mask_v = masker.makeChargeMask( nuvtx, adc_v );

    LARCV_DEBUG() << "Number of pixels masked: " << masker._npix << std::endl;

    larcv::EventImage2D* evout_mask
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "nuvtx_mask" );
    larcv::EventPixel2D* evout_pix
      = (larcv::EventPixel2D*)iolcv.get_data(larcv::kProductPixel2D, "nuvtx_mask" );
    for ( auto& img : nuvtx_mask_v ) {

      // make pixel cluster
      larcv::Pixel2DCluster pixcluster;
      for (int c=0; c<(int)img.meta().cols(); c++) {
        for (int r=0; r<(int)img.meta().rows(); r++) {
          if ( img.pixel(r,c)>0 ) {
            larcv::Pixel2D pix(c,r);
            pixcluster += pix;
          }
        }
      }
      evout_pix->Append( img.meta().plane(), pixcluster );
      
      evout_mask->Emplace( std::move(img) );
    }
    
  }
                                   
  
  
  
}
}
