#ifndef __FILL_PIXEL_MASK_H__
#define __FILL_PIXEL_MASK_H__

#include <vector>

// larcv2
#include "larcv/core/DataFormat/Image2D.h"

// larlite
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pixelmask.h"

namespace larflow {

  class FillPixelMask {

  public:

    FillPixelMask() {};
    virtual ~FillPixelMask() {};


    std::vector< std::vector<larlite::pixelmask> >
      fillMasks( const std::vector<larcv::Image2D>& adc_v, const larlite::event_larflowcluster& evcluster_v,
		 const std::vector<const larlite::event_pixelmask*>& evmask_vv);
    
    std::vector<larlite::pixelmask> fillMask( const std::vector<larcv::Image2D>& adc_v,
					      const larlite::larflowcluster& cluster3d,
					      const std::vector<larlite::pixelmask>& origmask_v );
    
  };
  

}

#endif
