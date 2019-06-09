#ifndef __make_flow_pixels_h__
#define __make_flow_pixels_h__

#include "DataFormat/hit.h"

#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {

  /**
   * create hits from pixels in the image.
   */
  larlite::event_hit makeHitsFromWholeImagePixels( const larcv::Image2D& src_adc, const float threshold );
  
}

#endif
