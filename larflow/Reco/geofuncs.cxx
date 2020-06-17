#include "geofuncs.h"
#include <cmath>
#include <stdexcept>

namespace larflow {
namespace reco {

  template <class T>
  T pointLineDistance( const std::vector<T>& linept1,
                           const std::vector<T>& linept2,
                           const std::vector<T>& pt )
  {
    
    // get distance of point from pca-axis
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    std::vector<T> d1(3);
    std::vector<T> d2(3);

    T len1 = 0.;
    T linelen = 0.;
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
    std::vector<T> d1xd2(3);
    d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
    d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
    d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
    T len1x2 = 0.;
    for ( int i=0; i<3; i++ ) {
      len1x2 += d1xd2[i]*d1xd2[i];
    }
    len1x2 = sqrt(len1x2);
    T r = len1x2/linelen;
    return r;
  }

  template <class T>
  T pointRayProjection( const std::vector<T>& start,
                        const std::vector<T>& dir,
                        const std::vector<T>& testpt )
  {

    T len = 0.;
    T proj = 0.;
    for ( size_t v=0; v<3; v++ ) {
      len += dir[v]*dir[v];
      proj += dir[v]*( testpt[v]-start[v] );
    }
    len = sqrt(len);
    if (len>0)
      proj /= len;
    else {
      throw std::runtime_error("geofuncs.cxx:pointRayProjection: zero-length direction vector given");
    }

    return proj;
    
  }

  float pointLineDistance3f( const std::vector<float>& linept1,
                             const std::vector<float>& linept2,
                             const std::vector<float>& testpt ){
    return pointLineDistance<float>( linept1, linept2, testpt );
  }

  float pointRayProjection3f( const std::vector<float>& start,
                              const std::vector<float>& dir,
                              const std::vector<float>& testpt )
  {
    return pointRayProjection<float>( start, dir, testpt );
  }

  double pointLineDistance3d( const std::vector<double>& linept1,
                              const std::vector<double>& linept2,
                              const std::vector<double>& testpt ){
    return pointLineDistance<double>( linept1, linept2, testpt );
  }

  double pointRayProjection3d( const std::vector<double>& start,
                               const std::vector<double>& dir,
                               const std::vector<double>& testpt )
  {
    return pointRayProjection<double>( start, dir, testpt );
  }
  
  
}
}
