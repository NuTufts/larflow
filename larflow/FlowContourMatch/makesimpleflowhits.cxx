#include "makesimpleflowhits.h"

namespace larflow {

  /**
   * make hits from flow data using very simple algo.
   *
   * we use the data in the ConturFlowMatchDict(s) to make hits using
   * a very simple algorithm. we do not attempt a sophisticated
   * method to correct flow predictions using the contour-matches.
   *
   * @param[in] adc_full_v Image2D with ADC values over a full image (not expecting crops)
   * @param[in] contours   ContourClusterAlgo which holds the contours found in each plane
   * @param[in] matchdict  ContorFlowMatchDict_t for each flow direction
   *
   */
  std::vector<larlite::larflow3dhit> makeSimpleFlowHits( const std::vector<larcv::Image2D>& adc_full_v,
                                                         const ublarcvapp::ContourClusterAlgo& contours,
                                                         const std::vector<ContourFlowMatchDict_t>& matchdict_v )
  {

    // we loop through the source contours
    // and choose the best flow for each source pixel.
    // we choose based on distance of source pixel to center.
    // also, we try to pix one which flows into a target contour

    auto const& src_ctr_v = contours.m_plane_atomics_v.at(2);

    for ( size_t src_ctr_idx=0; src_ctr_idx<src_ctr_v.size(); src_ctr_idx++ ) {

      auto const& src_ctr = src_ctr_v.at(src_ctr_idx);
      const std::vector<int>& src_ctr_pixel_v = matchdict_v.front().src_ctr_pixel_v.at( src_ctr_idx );
      std::cout << "src ctr[" << src_ctr_idx << "] number of pixels in contour: " << src_ctr_pixel_v.size() << std::endl;
      
      
    }

    
    std::vector<larlite::larflow3dhit> flowhits_v;
    return flowhits_v;
  }
  
  


}
