#include "ImageStitcherBase.h"


namespace dlcosmictag {

  /**
   * clear internal data vectors
   *
   */
  void ImageStitcherBase::clear() {
    m_stitched_v.clear();
    m_wholeview_meta_v.clear();
  }
  
  /**
   * defines the output image meta. 
   *
   * we copy the metas from the input images.
   * then we use the metas to create blank images.
   *
   * @param[in] wholeviewimg_v Vector of Images that will define the output dimensions and coordinates
   *
   */
  void ImageStitcherBase::setWholeViewMeta( const std::vector< larcv::Image2D >& wholeviewimg_v ) {
    m_wholeview_meta_v.clear();
    m_wholeview_adc_v.clear();

    for ( auto const& img : wholeviewimg_v ) {
      m_wholeview_meta_v.push_back( img.meta() );
      m_wholeview_adc_v.push_back( &img );
    }

    for ( auto const& meta : m_wholeview_meta_v ) {
      larcv::Image2D img( meta );
      img.paint(0.0);
      m_stitched_v.emplace_back( std::move(img) );
    }
    
    std::cout << "[" << __FILE__ << "::" << __FUNCTION__ << "] "
              << " len(m_wholview_meta_v)=" << m_wholeview_meta_v.size()
              << " len(m_stitched_v)=" << m_stitched_v.size()
              << std::endl;
  }

  /**
   *
   *  output the stitched image as a larlite::pixelmask
   *
   *  @param[in] threshold Optional pixel value threshold, below which pixel is not stored. Default is >=0.
   *  @param[in] label Optional integer label for output pixel mask(s). Default is 0.
   *
   *  return Vector of pixelmasks, one for each plane
   *
   */
  std::vector<larlite::pixelmask> ImageStitcherBase::as_pixel_mask( float threshold, int label ) const {

          
    const int values_per_pixel = 2 + ValuesPerPixel(); // (r,c) + values
    std::vector<larlite::pixelmask> mask_v;

    for ( auto const& img : m_stitched_v ) {

      auto const& meta = img.meta();

      std::vector< std::vector<float> > pixeldata_v;
      pixeldata_v.reserve( meta.rows()*meta.cols() );

      for (size_t r=0; r<meta.rows(); r++) {
        for ( size_t c=0; c<meta.cols(); c++ ) {
          if ( img.pixel(r,c)<threshold ) continue;

          std::vector<float> pixeldata( values_per_pixel, 0 );
          pixeldata[0] = (float)meta.pos_x(c);
          pixeldata[1] = (float)meta.pos_y(r);
          size_t idx=0;
          for ( auto const& val : getPixelValue( img, r, c ) ) {
            pixeldata[2+idx] = val;
            idx++;
          }
          pixeldata_v.push_back( pixeldata );
        } // end of col loop
      }//end of row loop
      
      larlite::pixelmask mask( 0, pixeldata_v,
                               meta.min_x(), meta.min_y(), meta.max_x(), meta.max_y(),
                               meta.cols(), meta.rows(), values_per_pixel );
      
      mask_v.emplace_back( std::move(mask) );
    }//end of loop over each plane image

    
    return mask_v;
  }

}
