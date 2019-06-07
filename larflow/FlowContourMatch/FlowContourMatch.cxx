#include "FlowContourMatch.h"

#include "makehitsfrompixels.h"

// ublarcvapp
#include "ublarcvapp/ContourTools/ContourShapeMeta.h"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"


namespace larflow {

  /**
   * create 3D spacepoints using the LArFlow network outputs
   *
   * @param[in] adc_crop_v vector of Image2D containing ADC values. Whole view.
   * @param[in] flowy2u vector of SparseImage crops containing Y->U flow predictions
   * @param[in] flowy2v vector of SparseImage crops containing Y->V flow predictions
   * @param[in] threshold ignore information where pixel value is below threshold
   *
   * return vector of larflow3dhit
   */
  std::vector<larlite::larflow3dhit> makeFlowPixels( std::vector<larcv::Image2D>& adc_v,
                                                     std::vector<larcv::SparseImage>& flowy2u,
                                                     std::vector<larcv::SparseImage>& flowy2v,
                                                     const float threshold ) {

    /// step: make hits from pixels in image, for each plane
    std::vector<larlite::event_hit> hit_vv(adc_v.size());
    for ( auto const& adc : adc_v ) {
      auto hit_v = makeHitsFromPixels( adc, threshold );
      hit_vv.emplace_back( std::move(hit_v) );
    }
    
    /// step: make contours on whole view images.
    ublarcvapp::ContourClusterAlgo contours;
    contours.analyzeImages( adc_v );

    
    
    // loop over flow crops, for each crop,
    // in each crop, for each source pixel,
    //   accumulate number of flows between unique (src/tar) index pairs
    // repeat this for all flow directions

    // for each source contour, score matching target contours
    // choose between conflicting target contours using ???

    // for all source pixels, make 3D points
    //   decide which flow to use to make point


    std::vector<larlite::larflow3dhit> flowhits_v;
    return flowhits_v;
  }
  

}
