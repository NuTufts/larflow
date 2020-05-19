#include "geofuncs.h"
#include <cmath>

namespace larflow {
namespace reco {

  float pointLineDistance( const std::vector<float>& linept1,
                           const std::vector<float>& linept2,
                           const std::vector<float>& pt )
  {
    
    // get distance of point from pca-axis
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    std::vector<float> d1(3);
    std::vector<float> d2(3);

    float len1 = 0.;
    float linelen = 0.;
    for (int i=0; i<3; i++ ) {
      d1[i] = pt[i] - linept1[i];
      d2[i] = pt[i] - linept2[i];
      len1 += d1[i]*d1[i];
      linelen += (linept1[i]-linept2[i])*(linept1[i]-linept2[i]);
    }
    len1 = sqrt(len1);
    linelen = sqrt(linelen);

    if ( linelen<1.0e-4 ) {
      // short cluster, use distance to end point
      return len1;
    }

    // cross-product
    std::vector<float> d1xd2(3);
    d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
    d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
    d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
    float len1x2 = 0.;
    for ( int i=0; i<3; i++ ) {
      len1x2 += d1xd2[i]*d1xd2[i];
    }
    len1x2 = sqrt(len1x2);
    float r = len1x2/linelen;
    return r;
  }

  
}
}
