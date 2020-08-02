#ifndef __LARFLOW_RECO_PROJPIXFITTER_H__
#define __LARFLOW_RECO_PROJPIXFITTER_H__

#include <vector>

namespace larflow {
namespace reco {

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
