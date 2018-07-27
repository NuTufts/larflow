#ifndef __FLOW_MATCH_HIT_3D_H__
#define __FLOW_MATCH_HIT_3D_H__

#include <vector>

namespace larflow {
  
  class FlowMatchHit3D: public std::vector<float>  {
    // class stores 3d location, but also provides us information
    // on the source and target contours it belongs to.
    // produced by FlowContourMatch::
    
  public:

    FlowMatchHit3D() {};
    ~FlowMatchHit3D() {};

    typedef enum { kQandCmatch=0, kCmatch, kClosestC } MatchQuality_t; // quality of match
    
    int tick;        // row
    int srcwire;     // column in source image
    int targetwire;  // column in target image
    //int src_ctrid;   // contour index in source image
    //int tar_ctrid;   // contour index in target image
    int idxhit;      // index in eventhit vector
    MatchQuality_t matchquality; // quality of plane-correspondence match
    float center_y_dist;  // distance to center of y-image used for flow prediction
    
  };


}


#endif
