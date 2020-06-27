#ifndef __LARFLOW_SCB_SCBOUNDARY_H__
#define __LARFLOW_SCB_SCBOUNDARY_H__

#include <vector>

namespace larflow {
namespace scb {

  class SCBoundary {

  public:

    SCBoundary() {};
    virtual ~SCBoundary() {};

    template <class T>
      T dist2boundary( const std::vector<T>& pos ) const; //< if inside tpc, return distance to space charge boundary

    float  dist2boundary( float  x, float y, float z ) const;
    double dist2boundary( double x, double y, double z ) const;

    template <class T>
      T XatBoundary( const std::vector<T>& pos ) const;     ///< value of x if we shift point along-x to boundary near cathode
    float  XatBoundary( float x,  float y,  float z ) const;
    double XatBoundary( double x, double y, double z ) const;

  protected:
    
    // The units are cm
    // index from 1 to 10 for position dependece. See the talk
    /* https://microboone-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=26423 */
    
    /// YX view has Z dependence: Z sub-range from 0 to 10m per 1m
    static constexpr double YX_TOP_y1_array     = 116;
    static double YX_TOP_x1_array[10];
    static double YX_TOP_y2_array[10];
    static constexpr double YX_TOP_x2_array = 256;
    
    static constexpr double YX_BOT_y1_array     = -115;
    static double YX_BOT_x1_array[10];
    static double YX_BOT_y2_array[10];
    static constexpr double YX_BOT_x2_array = 256;
    
    /// ZX view has Y dependence: Y sub-range from -116 to 116cm per 24cm
    static constexpr double ZX_Up_z1_array = 0;
    static constexpr double ZX_Up_x1_array = 120;
    static constexpr double ZX_Up_z2_array = 10;
    static constexpr double ZX_Up_x2_array = 256;
    
    static constexpr double ZX_Dw_z1_array     = 1037;
    static double ZX_Dw_x1_array[10];
    static double ZX_Dw_z2_array[10]; 
    static constexpr double ZX_Dw_x2_array     = 256;

    template <class T>
      T pointLineDistance( const std::vector<T>& linept1,
                           const std::vector<T>& linept2,
                           const std::vector<T>& testpt ) const;
    
  };


  /* template <> float  SCBoundary::dist2boundary<float>( const std::vector<float>& pos ) const; */
  /* template <> double SCBoundary::dist2boundary<double>( const std::vector<double>& pos ) const; */
  /* template <>float  SCBoundary::pointLineDistance( const std::vector<float>& linept1, */
  /*                                                  const std::vector<float>& linept2, */
  /*                                                  const std::vector<float>& testpt ); */
  /* template <> double SCBoundary::pointLineDistance( const std::vector<double>& linept1, */
  /*                                                   const std::vector<double>& linept2, */
  /*                                                   const std::vector<double>& testpt ); */
  
  
}
}

#endif
