#ifndef __LARFLOW_PREP_MCKEYPOINT_H__
#define __LARFLOW_PREP_MCKEYPOINT_H__


#include <vector>
#include <string>

namespace larflow {
namespace prep {

  /**
   * @ingroup MCTools
   * @class MCKeypoint
   * @brief Represents Keypoints in active areas of the detector
   *
   * Produced by ublarcvapp::mctools::MCKeypointFinder.
   * 
   */
  class MCKeypoint {

  public:


    enum KPType_t { kTrackStart=0, kTrackEnd, kShowerStart, kMichel, kDelta, kNumKPTypes, kUnitialized };
    
    std::vector<float> keypt_true;   ///< 3D position of keypoint in cm
    std::vector<float> keypt_appear; ///< 3D position of keypoint as it appears in the wire data

    int tick;
    int row;
    std::vector<int> imgcoord; ///< (U col, V col, Y col)
    
    int trackid;               ///< ID of track or shower by which this keypoint data was made
    int pid;                   ///< particle ID of track or shower making keypoint
    int is_shower;             ///< if =1, then keypoint came from shower
    int origin;                ///< if =1, origin is cosmics; if =2 origin is from neutrino interaction generator
    KPType_t kptype;           ///< Keypoint type
    
    MCKeypoint()
    : keypt_true({0,0,0}),
      keypt_appear({0,0,0}),
      tick(0),
      row(0),
      imgcoord({0,0,0,0}),
      trackid(-1),
      pid(-1),
      is_shower(-1),
      origin(-1),
      kptype(kUnitialized)
    {};
    ~MCKeypoint() {};

    std::string str() const;
    
  };
  
}
}

#endif
