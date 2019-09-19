#ifndef __make_simlpe_flow_hits_h__
#define __make_simlpe_flow_hits_h__

#include <vector>

#include "DataFormat/larflow3dhit.h"

#include "larcv/core/DataFormat/Image2D.h"

#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"

#include "ContourFlowMatch.h"

namespace larflow {

  std::vector<larlite::larflow3dhit> makeSimpleFlowHits( const std::vector<larcv::Image2D>& adc_full_v,
                                                         const ublarcvapp::ContourClusterAlgo& contours,
                                                         const std::vector<ContourFlowMatchDict_t>& matchdict );

}

#endif
