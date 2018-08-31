#ifndef __FLOW_MATCH_HIT_3D_H__
#define __FLOW_MATCH_HIT_3D_H__

#include <vector>

namespace larflow {
  
  class FlowMatchHit3D: public std::vector<float>  {
    // class stores 3d location, but also provides us information
    // on the source and target contours it belongs to.
    // produced by FlowContourMatch::
    
  public:

    FlowMatchHit3D() {
      targetwire.resize(2,-1);
    };
    ~FlowMatchHit3D() {};

    typedef enum { kQandCmatch=0, kCmatch, kClosestC, kNoMatch } MatchQuality_t; // quality of match
<<<<<<< HEAD
    typedef enum { kIn5mm=0, kIn10mm, kIn50mm, kOut50mm, kNoValue } Consistency_t; // distance b/n y2u and y2v
=======
    typedef enum { kIn5mm=0, kIn10mm, kIn50mm, kOut50mm, kNoValue } Consistency_t; // quality of match
>>>>>>> master
    
    int tick;        // row
    int srcwire;     // column in source image
    std::vector<int> targetwire;  // column in target image
    //int src_ctrid;   // contour index in source image
    //int tar_ctrid;   // contour index in target image
    int idxhit;      // index in eventhit vector
    MatchQuality_t matchquality; // quality of plane-correspondence match
    Consistency_t consistency3d; //flag for distance b/n y2u and y2v predicted spacepoints
    float center_y_dist;  // distance to center of y-image used for flow prediction
    float X[3]; //3d position. 
    float dy; //distance in y coord. between y2u and y2v predicted spacepoints
    float dz; //distance in z coord. between y2u and y2v predicted spacepoints
  };


}


#endif
