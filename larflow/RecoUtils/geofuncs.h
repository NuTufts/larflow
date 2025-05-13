#ifndef __LARFLOW_GEOFUNCS_H__
#define __LARFLOW_GEOFUNCS_H__

#include <vector>
#include <cmath>
#include <array>

namespace larflow {
namespace recoutils {

#ifndef __CINT__
#ifndef __CLING__
  template <class T>
    T pointLineDistance( const std::vector<T>& linept1,
                            const std::vector<T>& linept2,
                            const std::vector<T>& testpt );

  template <class T>
    T pointRayProjection( const std::vector<T>& start,
                          const std::vector<T>& dir,
                          const std::vector<T>& testpt );
#endif
#endif

  float pointLineDistance3f( const std::vector<float>& linept1,
                             const std::vector<float>& linept2,
                             const std::vector<float>& testpt );
  
  float pointRayProjection3f( const std::vector<float>& start,
                              const std::vector<float>& dir,
                              const std::vector<float>& testpt );

  double pointLineDistance3d( const std::vector<double>& linept1,
                              const std::vector<double>& linept2,
                              const std::vector<double>& testpt );
  
  double pointRayProjection3d( const std::vector<double>& start,
                               const std::vector<double>& dir,
                               const std::vector<double>& testpt );

  float lineLineDistance3f( const std::vector<float>& line1_x1,
			    const std::vector<float>& line1_x2,
			    const std::vector<float>& line2_x3,
			    const std::vector<float>& line2_x4 );


  class GeoFuncVector3D {
    public:

    GeoFuncVector3D( double xx, double yy, double zz )
      : x(xx),
	y(yy),
	z(zz)
    {};

    double x;
    double y;
    double z;
      
    GeoFuncVector3D operator-(const GeoFuncVector3D& other) const {
      return {x - other.x, y - other.y, z - other.z};
    };

    GeoFuncVector3D operator+(const GeoFuncVector3D& other) const {
      return {x + other.x, y + other.y, z + other.z};
    };

    GeoFuncVector3D operator*(const double scalar ) const {
      return { x*scalar, y*scalar, z*scalar };
    };
          
    double dot(const GeoFuncVector3D& other) const {
      return x * other.x + y * other.y + z * other.z;
    };
    
    GeoFuncVector3D cross(const GeoFuncVector3D& other) const {
      return {
              y * other.z - z * other.y,
              z * other.x - x * other.z,
              x * other.y - y * other.x
      };
    };
    
    double magnitude() const {
      return std::sqrt(dot(*this));
    };

    std::vector<double> as_vectord() const {
      std::vector<double> out(3,0);
      out[0] = x;
      out[1] = y;
      out[2] = z;
      return out;
    };
    
    std::vector<float> as_vectorf() const {
      std::vector<float> out(3,0);
      out[0] = (float)x;
      out[1] = (float)y;
      out[2] = (float)z;
      return out;
    };
    
  };

  float lineLineDistance3f_claude( const std::vector<float>& x1,
				   const std::vector<float>& dir1,
				   const std::vector<float>& x2,
				   const std::vector<float>& dir2,
				   std::vector<float>& minseg_pt1,
				   std::vector<float>& minseg_pt2);
  
}
}

#endif
