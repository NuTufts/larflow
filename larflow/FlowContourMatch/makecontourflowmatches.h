#ifndef __make_contour_flow_matches_h__
#define __make_contour_flow_matches_h__

#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {

  ContourFlowMatchDict_t createMatchData( const ublarcvapp::ContourClusterAlgo& contour_data,
                                          const larcv::Image2D& flow_img,
                                          const larcv::Image2D& src_adc,
                                          const larcv::Image2D& tar_adc );

}

#endif
