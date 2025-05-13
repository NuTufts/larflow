#include "geofuncs.h"
#include <cmath>
#include <stdexcept>

namespace larflow {
namespace recoutils {

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
  
  /**
   * @brief smallest distance between skew lines
   *
   * https://mathworld.wolfram.com/Line-LineDistance.html
   * 
   * @param[in] line1_x1 point on line 1
   * @param[in] line1_x2 point on line 1
   * @param[in] line2_x3 point on line 2
   * @param[in] line2_x4 point on line 2
   * @return projected distance
   */      
  float lineLineDistance3f( const std::vector<float>& line1_x1,
			    const std::vector<float>& line1_x2,
			    const std::vector<float>& line2_x3,
			    const std::vector<float>& line2_x4 )
  {
    std::vector<float> a(3,0);
    std::vector<float> b(3,0);
    std::vector<float> c(3,0);
    for (int i=0; i<3; i++) {
      a[i] = line1_x2[i] - line1_x1[i];
      b[i] = line2_x4[i] - line2_x3[i];
      c[i] = line2_x3[i] - line1_x1[i];
    }

    std::vector<float> axb(3,0);
    axb[0] = a[1]*b[2]-a[2]*b[1];
    axb[1] = a[2]*b[0]-a[0]*b[2];
    axb[2] = a[0]*b[1]-a[1]*b[0];

    float lenaxb = 0.;
    for (int i=0; i<3; i++)
      lenaxb += axb[i]*axb[i];
    lenaxb = sqrt(lenaxb);
    if ( lenaxb<1.0e-3 )
      return -1.0;
    
    float lentripprod = 0.;
    for (int i=0; i<3; i++) {
      float tripprod = c[i]*axb[i];
      lentripprod += tripprod;
    }
    lentripprod = fabs(lentripprod);

    return lentripprod/lenaxb;
  }
  
  /**
   * @brief smallest distance between skew lines
   *
   * from Claude
   * 
   * @param[in] line1_x1 point on line 1
   * @param[in] line1_x2 point on line 1
   * @param[in] line2_x3 point on line 2
   * @param[in] line2_x4 point on line 2
   * @return projected distance
   */      
  float lineLineDistance3f_claude( const std::vector<float>& x1,
				   const std::vector<float>& fdir1,
				   const std::vector<float>& x2,
				   const std::vector<float>& fdir2,
				   std::vector<float>& minseg_pt1,
				   std::vector<float>& minseg_pt2)				   
  {
    // Convert points to vectors for easier calculation
    GeoFuncVector3D p1v(x1[0],x1[1],x1[2]);
    GeoFuncVector3D p2v(x2[0],x2[1],x2[2]);
    GeoFuncVector3D dir1(fdir1[0],fdir1[1],fdir1[2]);
    GeoFuncVector3D dir2(fdir2[0],fdir2[1],fdir2[2]);
    
    // Normalize direction vectors
    double len1 = dir1.magnitude();
    double len2 = dir2.magnitude();
    GeoFuncVector3D d1( dir1.x/len1, dir1.y/len1, dir1.z/len1 );
    GeoFuncVector3D d2( dir2.x/len2, dir2.y/len2, dir2.z/len2 );

    // Get vector connecting two starting points
    GeoFuncVector3D r = p1v-p2v;

    // Compute various dot products we'll need
    double a = d1.dot(d1);
    double b = d1.dot(d2);
    double c = d2.dot(d2);
    double d = d1.dot(r);
    double e = d2.dot(r);

    double denom = a*c - b*b;
    if ( std::fabs(denom)<1.0e-10 ) {
      // parallel lines
      // project r onto dir1 to find closest point
      double t1 = -d/a;
      GeoFuncVector3D closest1 = p1v + d1*t1;
      GeoFuncVector3D diff = closest1-p2v;
      double t2 = diff.dot( d2 );
      GeoFuncVector3D closest2 = p2v + d2*t2;

      GeoFuncVector3D mindiff = closest1-closest2;
      
      minseg_pt1 = closest1.as_vectorf();
      minseg_pt2 = closest2.as_vectorf();
      double mindist = mindiff.magnitude();
      return mindist;
    }

    double t1 = (b*e - c*d )/denom;
    double t2 = (a*e - b*d )/denom;

    GeoFuncVector3D closest1 = p1v + d1*t1;
    GeoFuncVector3D closest2 = p2v + d2*t2;
    GeoFuncVector3D mindiff = closest1-closest2;
      
    minseg_pt1 = closest1.as_vectorf();
    minseg_pt2 = closest2.as_vectorf();
    double mindist = mindiff.magnitude();
    return mindist;

  }

}
}
