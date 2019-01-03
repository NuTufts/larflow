#include "SSNetStitcher.h"

namespace dlcosmictag {

  SSNetStitcher::SSNetStitcher( const std::vector<larcv::Image2D>& outimgtemplate_v, ScoreChoiceMethod_t scoremethod )
    : ImageStitcherBase(),
      m_scoremethod(scoremethod)
  {

    // set the meta
    setWholeViewMeta( outimgtemplate_v );
    
  }

  void SSNetStitcher::setWholeViewMeta( const std::vector< larcv::Image2D >& wholeviewimg_v ) {
    ImageStitcherBase::setWholeViewMeta( wholeviewimg_v );

    // define metric image
    m_metric_image_v.clear();
    m_metric_image_v.reserve( wholeviewimg_v.size() );
    for ( auto const& meta : WholeViewMeta() ) {
      larcv::Image2D metric(meta);
      metric.paint(-1);
      m_metric_image_v.emplace_back( std::move(metric) );
    }
  }
  
  
  int SSNetStitcher::addSubImage( const larcv::Image2D& subimg, int plane, float threshold ) {

    auto const& subimg_meta   = subimg.meta();
    auto const& stitched_meta = WholeViewMeta().at(plane);
    
    auto& stitched = Stitched_mutable().at(plane);
    auto& metric   = m_metric_image_v.at(plane);


    float center_row = subimg_meta.rows()/2;
    float center_col = subimg_meta.cols()/2;

    for ( size_t r=0; r<subimg_meta.rows(); r++ ) {

      float subimg_y = subimg_meta.pos_y(r);
      if ( subimg_y < stitched_meta.min_y() || subimg_y>=stitched_meta.max_y() )
        continue;
      
      int stitched_row = stitched_meta.row( subimg_y );
      
      for ( size_t c=0; c<subimg_meta.cols(); c++ ) {

        float subimg_x = subimg_meta.pos_x(c);
        if ( subimg_x < stitched_meta.min_x() || subimg_x>=stitched_meta.max_x() )
          continue;
        
        float subimg_pixval = subimg.pixel(r,c);
        if ( subimg_pixval<threshold ) continue;

        int stitched_col = stitched_meta.col( subimg_x );
        float metric_val = metric.pixel( stitched_row, stitched_col );

        if ( metric_val>-1.0 ) {
          // we must have had a value here. decide to keep old score or new score.
          if ( m_scoremethod==kCENTER ) {
            // replace value if location is closer to subimg center
            float dist =
              ((float)r-center_row)*((float)r-center_row) + ((float)c-center_col)*((float)c-center_col);
            
            if ( dist<metric_val ) {
              stitched.set_pixel( stitched_row, stitched_col, subimg_pixval );
              metric.set_pixel( stitched_row, stitched_col, dist );
            }
          }
          else if ( m_scoremethod==kCONFIDENCE ) {
            // replace value if it's higher than the previous one
            if ( subimg_pixval > metric_val ) {
              stitched.set_pixel( stitched_row, stitched_col, subimg_pixval );
              metric.set_pixel( stitched_row, stitched_col, subimg_pixval );
            }
          }
        }//end of metric value indicates previous value was set in this pixel
      }//end of subimg cols
    }//end of subimg rows
  }//end of addSubImage

  void SSNetStitcher::clear() {
    ImageStitcherBase::clear();
    m_metric_image_v.clear();
  }

  
  
}
