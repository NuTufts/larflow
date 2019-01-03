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
 *
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
    void setWholeViewMeta( const std::vector< larcv::Image2D >& wholeviewimg_v );

    // add subimage to be incorporated into the final stitched image
    virtual int addSubImage( const larcv::Image2D& subimg, int plane ) = 0; // child class defines stitching action

    // retrieve the current wholeview meta (const)
    const std::vector< larcv::ImageMeta >& WholeViewMeta() const { return m_wholeview_meta_v; };
    
    // retrieve the current wholeview meta (mutable)
    std::vector< larcv::ImageMeta >& WholeViewMeta_mutable() { return m_wholeview_meta_v; };
    
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

    // output stitched image
    std::vector< larcv::Image2D > m_stitched_v;


  };


}

#endif
