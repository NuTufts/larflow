#ifndef __make_contour_flow_matches_h__
#define __make_contour_flow_matches_h__

#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "ContourFlowMatch.h"

namespace larflow {

  void createMatchData( const ublarcvapp::ContourClusterAlgo& contour_data,
                        const larcv::Image2D& src_adc_full,
                        const larcv::Image2D& tar_adc_full,
                        const larcv::Image2D& flow_img_crop,
                        const larcv::Image2D& src_adc_crop,
                        const larcv::Image2D& tar_adc_crop,
                        ContourFlowMatchDict_t& matchdict,
                        const float threshold,
                        const float max_dist_to_target_contour=30.0,
                        bool visualize=false );
  
}

#endif
