#ifndef __LARFLOW_RECO_PROJPIXFITTER_H__
#define __LARFLOW_RECO_PROJPIXFITTER_H__

#include <vector>

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class ProjPixFitter
   * @brief Fit 3D line segments to cluster of 3D space points assumed to be a track
   *
   * This provides a static function which calculates the gradient
   * and mean squared-error when fitting a cluster of 3D points to a 3D line segment.
   */    
  class ProjPixFitter {

  public:

    ProjPixFitter() {};
    virtual ~ProjPixFitter() {};

    static void grad_and_d2_pixel( const std::vector<float>& pt1, 
                                   const std::vector<float>& pt2, 
                                   const float tick,
                                   const float wire,
                                   const int plane,
                                   float& d2,
                                   std::vector<float>& grad );
    
  };

}
}

#endif
