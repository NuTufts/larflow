#include "geofuncs.h"
#include <cmath>
#include <stdexcept>

namespace larflow {
namespace reco {

  /**
   * @brief template function that gets distance of test point from line defined by two points.
   *
   *  calculation from:  http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
   * 
   * @param[in] linept1 Point on line
   * @param[in] linept2 Point on line
   * @param[in] pt Test point
   * @return distance from line
   */
  template <class T>
  T pointLineDistance( const std::vector<T>& linept1,
                       const std::vector<T>& linept2,
                       const std::vector<T>& pt )
  {
    
    
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

  /**
   * @brief get projected distance from start of ray to test point
   *
   *  calculation from:  http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
   * 
   * @param[in] start 3D start point of ray
   * @param[in] dir 3D direction of ray (doesn't need to be unit normalized)
   * @param[in] testpt Test point
   * @return projected distance
   */  
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

  /**
   * @brief float version of pointLineDistance()
   *
   * @param[in] linept1 Point on line
   * @param[in] linept2 Point on line
   * @param[in] testpt  Test point
   * @return distance from line
   */    
  float pointLineDistance3f( const std::vector<float>& linept1,
                             const std::vector<float>& linept2,
                             const std::vector<float>& testpt ){
    return pointLineDistance<float>( linept1, linept2, testpt );
  }

  /**
   * @brief float version of pointRayProjection()
   *
   * for use in python
   * 
   * @param[in] start  3D start point of ray
   * @param[in] dir    3D direction of ray (doesn't need to be unit normalized)
   * @param[in] testpt Test point
   * @return projected distance
   */      
  float pointRayProjection3f( const std::vector<float>& start,
                              const std::vector<float>& dir,
                              const std::vector<float>& testpt )
  {
    return pointRayProjection<float>( start, dir, testpt );
  }

  /**
   * @brief double version of pointLineDistance()
   *
   * @param[in] linept1 Point on line
   * @param[in] linept2 Point on line
   * @param[in] testpt  Test point
   * @return distance from line
   */  
  double pointLineDistance3d( const std::vector<double>& linept1,
                              const std::vector<double>& linept2,
                              const std::vector<double>& testpt ){
    return pointLineDistance<double>( linept1, linept2, testpt );
  }

  /**
   * @brief double version of pointRayProjection()
   *
   * for use in python
   * 
   * @param[in] start  3D start point of ray
   * @param[in] dir    3D direction of ray (doesn't need to be unit normalized)
   * @param[in] testpt Test point
   * @return projected distance
   */      
  double pointRayProjection3d( const std::vector<double>& start,
                               const std::vector<double>& dir,
                               const std::vector<double>& testpt )
  {
    return pointRayProjection<double>( start, dir, testpt );
  }
  
  
}
}
