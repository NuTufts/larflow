#ifndef __LARFLOW_SCB_SCBOUNDARY_H__
#define __LARFLOW_SCB_SCBOUNDARY_H__

#include <vector>

namespace larflow {
namespace scb {

  /**
   * @ingroup SCBoundary
   * @class SCBoundary
   * @brief Calculate Distances to Space Charge Boundary
   *
   * Ions accumulating in the TPC volume create distortions in the electric field.
   * This causes distortions in the reconstructed location of charge.
   * For tracks entering the detector, instead of charge appearing at the edge of the 
   * TPC volume, space charge effects will make that track look as if it started 
   * some distance inside the TPC. Measurements of the space charge effect has allowed
   * the experiment to define a "space charge boundary". This is apparent boundary inside
   * the detector where entering tracks appear to start. (Or where exiting tracks seem 
   * to stop.)
   *
   * More info on the boundary implemented can be found at:
   * https://microboone-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=26423
   *
   */
  class SCBoundary {

  public:

    /** @brief Labels for the different boundaries of the TPC */
    typedef enum { kTop=0, kBottom, kUpstream, kDownstream, kAnode, kCathode, kNumBoundaries } Boundary_t;
    
    SCBoundary() {};
    virtual ~SCBoundary() {};

    template <class T>
      T dist2boundary( const std::vector<T>& pos, Boundary_t& btype ) const;

    float  dist2boundary( float  x, float y, float z ) const;
    double dist2boundary( double x, double y, double z ) const;
    float  dist2boundary(  float x,  float y,  float z, int& ibtype ) const;
    double dist2boundary( double x, double y, double z, int& ibtype ) const;        

    template <class T>
      T XatBoundary( const std::vector<T>& pos ) const; 
    float  XatBoundary( float x,  float y,  float z ) const;
    double XatBoundary( double x, double y, double z ) const;

  protected:
    
    // The units are cm
    // index from 1 to 10 for position dependece. See the talk
    /* https://microboone-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=26423 */
    
    static constexpr double YX_TOP_y1_array     = 116; ///< y-position defining start of top space charge boundary
    static double YX_TOP_x1_array[10]; ///< x-position defining start of top space charge boundary for 10 subsections
    static double YX_TOP_y2_array[10]; ///< y-position defining end of top space charge boundary for 10 subsections
    static constexpr double YX_TOP_x2_array = 256; ///< x-position defining end of top space charge boundary
    
    static constexpr double YX_BOT_y1_array     = -115; ///< y-position defining start of bottom space charge boundary
    static double YX_BOT_x1_array[10];                  ///< x-position defining start of bottom space charge boundary for 10 z-subsections
    static double YX_BOT_y2_array[10];                  ///< y-position defining start of bottom space charge boundary for 10 z-subsections
    static constexpr double YX_BOT_x2_array = 256;      ///< x-position defining end of bottom space charge boundary
    
    /// ZX view has Y dependence: Y sub-range from -116 to 116cm per 24cm
    static constexpr double ZX_Up_z1_array = 0;   ///< z-position defining start of upstream space charge boundary
    static constexpr double ZX_Up_x1_array = 120; ///< x-position defining start of upstream space charge boundary
    static constexpr double ZX_Up_z2_array = 10;  ///< z-position defining end of upstream space charge boundary
    static constexpr double ZX_Up_x2_array = 256; ///< x-position defining end of upstream space charge boundary
    
    static constexpr double ZX_Dw_z1_array     = 1037; ///< z-position defining start of downstream space charge boundary
    static double ZX_Dw_x1_array[10];                  ///< x-position defining start of downstream space charge boundary in 10 y-subsections
    static double ZX_Dw_z2_array[10];                  ///< z-position defining end of downstream space charge boundary in 10 y-subsections
    static constexpr double ZX_Dw_x2_array     = 256;  ///< x-position defining end of downstream space charge boundary

    template <class T>
      T pointLineDistance( const std::vector<T>& linept1,
                           const std::vector<T>& linept2,
                           const std::vector<T>& testpt ) const;
    
  };
  
  
}
}

#endif
