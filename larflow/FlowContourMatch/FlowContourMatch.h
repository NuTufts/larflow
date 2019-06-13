#ifndef __FLOWCONTOURMATCH__
#define __FLOWCONTOURMATCH__

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// larcv
#include "larcv/core/Base/larcv_logger.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/SparseImage.h"
#include "larcv/core/DataFormat/EventChStatus.h"

// larlite data product
#include "DataFormat/larflow3dhit.h"

namespace larflow {

  
  std::vector<larlite::larflow3dhit> makeFlowHitsFromSparseCrops( const std::vector<larcv::Image2D>& adc_crop_v,
                                                                  const std::vector<larcv::SparseImage>& flowdata,
                                                                  const float threshold,
                                                                  const std::string cropcfg,
                                                                  const larcv::msg::Level_t verbosity=larcv::msg::kNORMAL,
                                                                  const bool visualize=false );
    
  std::vector<larlite::larflow3dhit> makeTrueFlowHitsFromWholeImage( const std::vector<larcv::Image2D>& adc_v,
                                                                     const larcv::EventChStatus& chstatus,
                                                                     const std::vector<larcv::Image2D>& larflow_v,
                                                                     const float threshold,
                                                                     const std::string cropcfg,
                                                                     const larcv::msg::Level_t verbosity=larcv::msg::kNORMAL,
                                                                     const bool visualize=false );

  // hack to get functions to load
  class load_flow_contour_match {
  public:
    load_flow_contour_match() {};
    ~load_flow_contour_match() {};
  };
}

#endif
