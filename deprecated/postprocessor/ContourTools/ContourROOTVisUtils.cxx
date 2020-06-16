#include "ContourROOTVisUtils.h"

namespace larlitecv {

  TGraph ContourROOTVisUtils::contour_as_tgraph( const std::vector< cv::Point >& contour, const larcv::ImageMeta* meta  ) {
    TGraph g( contour.size()+ 1);

    size_t pt=0;
    for ( auto const& cvpoint : contour ) {
      float x = cvpoint.x;
      float y = cvpoint.y;
      
      if ( meta ) {
	x = meta->pos_x( x );
	y = meta->pos_y( y );
      }
      g.SetPoint( pt, x, y );      
      pt++;
    }

    // add the first point to close the contour
    if ( meta ) {
      g.SetPoint( pt, meta->pos_x( contour.front().x ), meta->pos_y( contour.front().y ) );
    }
    else {
      g.SetPoint( pt, contour.front().x, contour.front().y );
    }
    
    return g;
  }
  
  std::vector< TGraph > ContourROOTVisUtils::contour_as_tgraph( const std::vector< std::vector< cv::Point > >& contour_v,
								const larcv::ImageMeta* meta ) {
    std::vector< TGraph > g_v;
    
    for ( auto const& contour : contour_v ) {
      g_v.emplace_back( std::move( contour_as_tgraph(contour,meta)  ) );
    }
    
    return g_v;
  }

}
