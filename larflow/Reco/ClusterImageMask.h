#ifndef __LARFLOW_RECO_CLUSTER_IMAGE_MASK_H__
#define __LARFLOW_RECO_CLUSTER_IMAGE_MASK_H__

#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class ClusterImageMask : public larcv::larcv_base {

  public:

    ClusterImageMask()
      : _npix(0),
	_store_pixel_value(false)
      {
	_cluster_mask_v.clear();	
      };
    virtual ~ClusterImageMask() {};


    std::vector< larcv::Image2D >
      makeChargeMask( NuVertexCandidate& nuvtx,
                      const std::vector<larcv::Image2D>& adc_v );
    
    
    void maskCluster( const larlite::larflowcluster& cluster,
                      const std::vector<larcv::Image2D>& adc_v,
                      std::vector<larcv::Image2D>& mask_v,
                      const float thresh,
                      const int dpix=2 );

    void maskClusterAndStore( const larlite::larflowcluster& cluster,
			      const std::vector<larcv::Image2D>& adc_v,
			      const float thresh,
			      const int dpix=2,
			      const bool clear_tracking_image=true);

    void maskWithImage( const std::vector<larcv::Image2D>& adc_v,
			std::vector<larcv::Image2D>& mask_v,
			const float thresh,
			const bool invert );
    
    void maskStoredImage( const std::vector<larcv::Image2D>& adc_v,
			  const float thresh,
			  const bool invert );
    
    float getPlaneMaskSum( int plane );
    
    std::vector<float> getMaskSums();
    
    void maskTrack( const larlite::track& track,
                    const std::vector<larcv::Image2D>& adc_v,
                    std::vector<larcv::Image2D>& mask_v,
                    const float thresh,
                    const int dcol=2,
                    const int drow=2,
                    const float minstepsize=0.1,
                    const float maxstepsize=1.0 );

    void storePixelValue( bool doit=true ) { _store_pixel_value=doit; };

    int _npix;
    bool _store_pixel_value;
    std::vector<larcv::Image2D> _cluster_mask_v; ///< carries a copy of the image whose pixels correspond to the projection of the 3D clusters
    
  };

}
}


#endif
