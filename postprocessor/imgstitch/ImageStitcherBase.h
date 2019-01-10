#ifndef __ImageStitcherBase_h__
#define __ImageStitcherBase_h__

/**
 *
 * \class ImageStitcherBase
 *
 * \defgroup imgstitch
 *
 * \brief Abstract base class for stitcher classes
 *
 * defines interface to stitchers, which is to
 * 1) define the output image meta using setWholeViewMeta
 * 2) provide subimages using addSubImage, which the user must define for each type of image
 * 3) output can be larcv image or larlite pixelmask object
 *
 * User MUST define:
 *   addSubImage. The idea is to be given a subimage and fill the appropriate plane image in m_stitched_v.
 *   user can retrieve the stitched images using Stitched_mutable().
 * User has the option of defining:
 *   1) ValuesPerPixel(): returns the number of values to store per pixel (default is 1)
 *   2) getPixelValue(): returns pixel value for given image and (row,col) position. 
 *                       Default is the pixel value from the input image.
 */

#include <vector>

// larcv2
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"

// larlite
#include "DataFormat/pixelmask.h"

namespace dlcosmictag {

  class ImageStitcherBase {

  public:
    
    ImageStitcherBase() {};
    virtual ~ImageStitcherBase() {};

    // define the output stitched image meta
    virtual void setWholeViewMeta( const std::vector< larcv::Image2D >& wholeviewimg_v );

    // add subimage to be incorporated into the final stitched image
    virtual int addSubImage( const larcv::Image2D& subimg, int plane, float threshold=0 ) = 0; // child class defines stitching action

    // retrieve the current wholeview meta (const)
    const std::vector< larcv::ImageMeta >& WholeViewMeta() const { return m_wholeview_meta_v; };

    // retrieve current whole view ADC image (const)
    const larcv::Image2D& WholeViewADC( const int plane ) { return *m_wholeview_adc_v.at(plane); };
        
    // retrieve the current state of the stitched image (const)
    const std::vector< larcv::Image2D >& Stitched() const { return m_stitched_v; };

    // retrieve the current state of the stitched image (mutable)
    std::vector< larcv::Image2D >& Stitched_mutable() { return m_stitched_v; };
    
    /// return stitched result as larlite pixel mask
    std::vector<larlite::pixelmask> as_pixel_mask( float threshold=0, int label=0 ) const;
    
    /// clear stored images
    virtual void clear();
    
  protected:

    // definition of the wholeview meta
    std::vector< larcv::ImageMeta > m_wholeview_meta_v;
    std::vector< const larcv::Image2D* > m_wholeview_adc_v;

    // output stitched image
    std::vector< larcv::Image2D > m_stitched_v;

    // these functions are used to fill the pixel mask object. User can override this behavior
    // to have the pixelmask store more than just the pixel value in the m_stitched_v images
    virtual int ValuesPerPixel() const { return 1; }; // defaults, but user can override
    virtual std::vector<float> getPixelValue( const larcv::Image2D& img, const int row, const int col ) const {
      std::vector<float> pixdata(1,img.pixel(row,col));
      return pixdata;
    };
    
  };


}

#endif
