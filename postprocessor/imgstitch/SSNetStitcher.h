#ifndef __SSNET_STITCHER_BASE_H__
#define __SSNET_STITCHER_BASE_H__

/**
 *
 * \class SSNetStitcher
 *
 * \ingroup imgstitch
 *
 * \brief Stitchers together cropped SSNet images
 *
 * For overlapping regions, we provide a couple of stitching methods.
 *   CENTER: we take the score closest to the center of the image.
 *   CONFIDENCE: score with the highest non-background confidence.
 */

#include "ImageStitcherBase.h"

namespace dlcosmictag {

  class SSNetStitcher : public ImageStitcherBase {

  public:
    
    typedef enum { kCENTER, kCONFIDENCE } ScoreChoiceMethod_t;

    SSNetStitcher( const std::vector<larcv::Image2D>& outimgtemplate_v, ScoreChoiceMethod_t scoremethod );

    int addSubImage( const larcv::Image2D& subimg, int plane, float threshold=0 ) override; // child class defines stitching action

    void clear() override;

  protected:

    ScoreChoiceMethod_t m_scoremethod;

    // images to keep track of metric of pixel used to fill value
    // this value is used to determine which pixel to use if there is an overlap
    std::vector< larcv::Image2D > m_metric_image_v; 

  };

}

#endif
