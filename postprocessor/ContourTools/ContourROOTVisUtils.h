#ifndef __CONTOUR_ROOTVISUTILS_H__
#define __CONTOUR_ROOTVISUTILS_H__

/**
 * Utilities for visualizing the contours. For ROOT plots.
 *
 */

#include <vector>

// larcv2
#include "larcv/core/DataFormat/ImageMeta.h"

// ROOT
#include "TGraph.h"

#include "ContourShapeMeta.h"

namespace larlitecv {

  class ContourROOTVisUtils {
  protected:

    ContourROOTVisUtils() {};
    virtual ~ContourROOTVisUtils() {};

  public:
    
    static TGraph contour_as_tgraph( const std::vector<cv::Point>& contour, const larcv::ImageMeta* meta=nullptr );

    static std::vector< TGraph > contour_as_tgraph( const std::vector< std::vector<cv::Point> >& contour_v,
						    const larcv::ImageMeta* meta=nullptr );

    /* static TGraph contourmeta_as_tgraph( const ContourShapeMeta& contour, const larcv::ImageMeta* meta=nullptr ); */

    /* static std::vector< TGraph > contourmeta_as_tgraph( const std::vector< ContourShapeMeta >& contour_v, */
    /* 							const larcv::ImageMeta* meta=nullptr ); */
    
    
  };
  
}

#endif
