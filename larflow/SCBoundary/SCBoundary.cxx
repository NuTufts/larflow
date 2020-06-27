#include "SCBoundary.h"

#include <cmath>

namespace larflow {
namespace scb {

  
  
  template float  SCBoundary::dist2boundary<float>( const std::vector<float>& pos ) const;
  template double SCBoundary::dist2boundary<double>( const std::vector<double>& pos ) const;
  template float  SCBoundary::pointLineDistance( const std::vector<float>& linept1,
                                                 const std::vector<float>& linept2,
                                                 const std::vector<float>& testpt ) const;
  template double SCBoundary::pointLineDistance( const std::vector<double>& linept1,
                                                 const std::vector<double>& linept2,
                                                 const std::vector<double>& testpt ) const;
  template float  SCBoundary::XatBoundary<float>( const std::vector<float>& pos ) const;
  template double SCBoundary::XatBoundary<double>( const std::vector<double>& pos ) const;
  
  double SCBoundary::YX_TOP_x1_array[10] = {150.00, 132.56, 122.86, 119.46, 114.22, 110.90, 115.85, 113.48, 126.36, 144.21};
  double SCBoundary::YX_TOP_y2_array[10] = {110.00, 108.14, 106.77, 105.30, 103.40, 102.18, 101.76, 102.27, 102.75, 105.10};
  double SCBoundary::YX_BOT_x1_array[10] = {115.71, 98.05, 92.42, 91.14, 92.25, 85.38, 78.19, 74.46, 78.86, 108.90};
  double SCBoundary::YX_BOT_y2_array[10] = {-101.72, -99.46, -99.51, -100.43, -99.55, -98.56, -98.00, -98.30, -99.32, -104.20};
  double SCBoundary::ZX_Dw_x1_array[10]  = {120.00, 115.24, 108.50, 110.67, 120.90, 126.43, 140.51, 157.15, 120.00, 120.00};
  double SCBoundary::ZX_Dw_z2_array[10]  = {1029.00, 1029.12, 1027.21, 1026.01, 1024.91, 1025.27, 1025.32, 1027.61, 1026.00, 1026.00};
  
  template <class T>
  T SCBoundary::dist2boundary( const std::vector<T>& pos ) const
  {

    // first calculate the dist to the TPC boundaries
    T dx1 = pos[0];
    T dx2 = 256-pos[0];
    T dy1 = 116.0-pos[1];
    T dy2 = pos[1]+116.0;
    T dz1 = pos[2];
    T dz2 = 1037.0-pos[2];

    T dwall = 1.0e9;
    int boundary_type = -1;

    if ( fabs(dy1)<fabs(dwall) ) {
      dwall = dy1;
      boundary_type = 0; // top
    }
    if ( fabs(dy2)<fabs(dwall) ) {
      dwall = dy2;
      boundary_type = 1; // bottom
    }
    if ( fabs(dz1)<fabs(dwall) ) {
      dwall = dz1;
      boundary_type = 2; // upstream
    }
    if ( fabs(dz2)<fabs(dwall) ) {
      dwall = dz2;
      boundary_type = 3; // downstream
    }
    if ( fabs(dx1)<fabs(dwall) ) {
      dwall = dx1;
      boundary_type = 4; // anode
    }
    if ( fabs(dx2)<fabs(dwall) ) {
      dwall = dx2;
      boundary_type = 5; // cathode
    }

    // if outside the box, we just return the distance from the TPC wall
    if ( dwall<0 )
      return dwall;
    
    // find the box to evaluate the XY view
    int zbox = pos[2]/100.0;
    if ( zbox>9 ) zbox = 9;
    std::vector<double> xyproj = { (double)pos[0], (double)pos[1], 0.0 };
    std::vector<double> xy_topline1 = { YX_TOP_x1_array[zbox], YX_TOP_y1_array, 0.0 };
    std::vector<double> xy_topline2 = { YX_TOP_x2_array, YX_TOP_y2_array[zbox], 0.0 };
    double topxy = pointLineDistance<double>( xy_topline1, xy_topline2, xyproj );
    float dtop = (pos[0]-YX_TOP_x1_array[zbox])*(YX_TOP_y2_array[zbox]-YX_TOP_y1_array)
      - (pos[1]-YX_TOP_y1_array)*(YX_TOP_x2_array-YX_TOP_x1_array[zbox]);
    if ( dtop<0 ) topxy *= -1.0; // outside boundary
    
    std::vector<double> xy_botline1 = { YX_BOT_x1_array[zbox], YX_BOT_y1_array, 0.0 };
    std::vector<double> xy_botline2 = { YX_BOT_x2_array, YX_BOT_y2_array[zbox], 0.0 };
    double botxy = pointLineDistance<double>( xy_botline1, xy_botline2, xyproj );
    float dbot = (pos[0]-YX_BOT_x1_array[zbox])*(YX_BOT_y2_array[zbox]-YX_BOT_y1_array)
      - (pos[1]-YX_BOT_y1_array)*(YX_BOT_x2_array-YX_BOT_x1_array[zbox]);
    if ( dbot>0 ) botxy *= -1.0; // outside boundary
    
    // evlaute xy-view and xz-view boundary dist
    if ( pos[2]>1024 ) {
      int ybox = (int)(pos[1]-(-116.0))/24.0;
      if ( ybox>9 ) ybox = 9;
      std::vector<double> xz_proj = { (double)pos[0], 0.0, (double)pos[2] };
      std::vector<double> xz_line1 = { ZX_Dw_x1_array[ybox], 0.0, ZX_Dw_z1_array };
      std::vector<double> xz_line2 = { ZX_Dw_x2_array, 0.0, ZX_Dw_z2_array[ybox] };
      double topxz = pointLineDistance<double>( xz_line1, xz_line2, xz_proj );
      double dtopxz = (pos[0]-ZX_Dw_x1_array[ybox])*(ZX_Dw_z2_array[ybox]-ZX_Dw_z1_array)
        - (pos[2]-ZX_Dw_z1_array)*(ZX_Dw_x2_array-ZX_Dw_x1_array[ybox]);
      if ( dtopxz<0.0 ) topxz *= -1.0;

      if ( topxz<0 ) return topxz;
      if ( botxy<0 ) return botxy;
      if ( topxy<0 ) return topxy;
      
      // put it all together
      T dists[4] = { (T)dwall, (T)topxy, (T)botxy, (T)topxz };
      T mindist = 1.0e9;
      int minidx = 0;
      for (int i=0; i<4; i++) {
        if ( fabs(dists[i])<mindist ) {
          mindist = fabs(dists[i]);
          minidx = i;
        }
      }
      return dists[minidx];
    }
    else if ( pos[2]<11.0 ) {

      int ybox = (int)(pos[1]-(-116.0))/24.0;
      if ( ybox>9 ) ybox = 9;
      std::vector<double> xz_proj = { (double)pos[0], 0.0, (double)pos[2] };
      std::vector<double> xz_line1 = { ZX_Up_x1_array, 0.0, ZX_Up_z1_array };
      std::vector<double> xz_line2 = { ZX_Up_x2_array, 0.0, ZX_Up_z2_array };
      double topxz = pointLineDistance<double>( xz_line1, xz_line2, xz_proj );
      double dtopxz = ( pos[0]-ZX_Up_x1_array )*( ZX_Up_z2_array-ZX_Up_z1_array )
        - ( pos[2]-ZX_Up_z1_array )*( ZX_Up_x2_array-ZX_Up_x1_array );
      if ( dtopxz>0.0 ) topxz *= -1.0;

      if ( topxz<0 ) return topxz;
      if ( botxy<0 ) return botxy;
      if ( topxy<0 ) return topxy;
      
      // put it all together
      T dists[4] = { (T)dwall, (T)topxy, (T)botxy, (T)topxz };
      T mindist = 1.0e9;
      int minidx = 0;
      for (int i=0; i<4; i++) {
        if ( fabs(dists[i])<mindist ) {
          mindist = fabs(dists[i]);
          minidx = i;
        }
      }
      return dists[minidx];
      
    }

    // return the right dwall distance

    // if outside the SCboundary, return distance to that line
    if ( topxy<0 )
      return topxy;
    if ( botxy<0 )
      return botxy;
    
    // else we are inside the boundary, so chose closest distance
    T dists[3] = { (T)dwall, (T)topxy, (T)botxy };
    T mindist = 1.0e9;
    int minidx = 0;
    for (int i=0; i<3; i++) {
      if ( fabs(dists[i])<mindist ) {
        mindist = fabs(dists[i]);
        minidx = i;
      }
    }
    return dists[minidx];
    
  }

  template <class T>
  T SCBoundary::pointLineDistance( const std::vector<T>& linept1,
                                   const std::vector<T>& linept2,
                                   const std::vector<T>& pt ) const
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

  float SCBoundary::dist2boundary( float x, float y, float z ) const {
    std::vector<float> pos = { x, y, z };
    return dist2boundary<float>(pos);
  }

  double SCBoundary::dist2boundary( double x, double y, double z ) const {
    std::vector<double> pos = { x, y, z };
    return dist2boundary<double>(pos);
  }


  template<class T>
  T SCBoundary::XatBoundary( const std::vector<T>& pos ) const
  {

    T z = pos[2];
    if ( z<0 ) z = 0.0;
    else if ( z>1036.0 ) z = 1036.;

    T y = pos[1];
    if ( y<-116 ) y = -116.0;
    else if ( y>116.0 ) y = 116.0;
    
    int zbox = z/100.0;
    if ( zbox>9 ) zbox = 9;

    T x_y = (T)YX_TOP_x2_array;
    if ( y > YX_TOP_y2_array[zbox] ) {
      x_y = (T)( (y-YX_TOP_y1_array)*(YX_TOP_x2_array-YX_TOP_x1_array[zbox])/(YX_TOP_y2_array[zbox]-YX_TOP_y1_array) + YX_TOP_x1_array[zbox] );
    }
    else if ( y < YX_BOT_y2_array[zbox] ) {
      x_y = (T)( (y-YX_BOT_y1_array)*(YX_BOT_x2_array-YX_BOT_x1_array[zbox])/(YX_BOT_y2_array[zbox]-YX_BOT_y1_array) + YX_BOT_x1_array[zbox] );      
    }

    int ybox = (int)(pos[1]-(-116.0))/24.0;
    if ( ybox>9 ) ybox = 9;
    
    T x_z = (T)ZX_Dw_x2_array;    
    if ( z>1024.0 ) {
      x_z = (T)( (z-ZX_Dw_z1_array)*( ZX_Dw_x2_array-ZX_Dw_x1_array[ybox] )/(ZX_Dw_z2_array[ybox]-ZX_Dw_z1_array) + ZX_Dw_x1_array[ybox] );
    }
    else if ( z<11.0 ) {
      x_z = (T)( (z-ZX_Up_z1_array)*( ZX_Up_x2_array-ZX_Up_x1_array )/(ZX_Up_z2_array-ZX_Up_z1_array) + ZX_Up_x1_array );      
    }

    if ( x_z>(T)ZX_Dw_x2_array ) x_z = (T)ZX_Dw_x2_array;

    if ( x_y<x_z )
      return x_y;
    else
      return x_z;

    // should not reach here
    return pos[0];
    
  }

  float SCBoundary::XatBoundary( float x, float y, float z ) const
  {
    std::vector<float> pos = {x,y,z};
    return XatBoundary<float>(pos);
  }

  double SCBoundary::XatBoundary( double x, double y, double z ) const
  {
    std::vector<double> pos = {x,y,z};
    return XatBoundary<double>(pos);
  }
  
}
}
