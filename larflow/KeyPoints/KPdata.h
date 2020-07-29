#ifndef __KPdata_h__
#define __KPdata_h__


#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include <vector>
#include <string>

namespace larflow {
namespace keypoints {

  /**
   * @ingroup Keypoints
   * @class KPdata
   * @brief Info representing true keypoints in the images
   *
   * Produced by larflow::keypoints::PrepKeypointData.
   * 
   */
  class KPdata {

  public:
    
    int crossingtype; ///< crossing type: 0=entering; 1=exiting; 2=through-going; -1=not crossing; 
    std::vector<int> imgcoord; ///< (row, U col, V col, Y col)
    std::vector<float> keypt;  ///< 3D position of keypoint in cm
    int trackid;               ///< ID of track or shower by which this keypoint data was made
    int pid;                   ///< particle ID of track or shower making keypoint
    int vid;                   ///< vector index of container from which the track or shower truth object came
    int is_shower;             ///< if =1, then keypoint came from shower
    int origin;                ///< if =1, origin is cosmics; if =2 origin is from neutrino interaction generator
    KeyPoint_t kptype;         ///< class of keypoint label, KeyPoint_t
    KPdata() {
      crossingtype = -1;
      imgcoord.clear();
      keypt.clear();
      trackid = 0;
      pid = 0;
      vid = 0;
      is_shower = 0;
      origin = -1;
      kptype = larflow::kNumKeyPoints; // sentinal value
    };
    ~KPdata() {};

    std::string str() const;
    
  };

  bool kpdata_compare_x( const KPdata* lhs, const KPdata* rhs );
  bool kpdata_compare_y( const KPdata* lhs, const KPdata* rhs );
  bool kpdata_compare_z( const KPdata* lhs, const KPdata* rhs );
  
}
}

#endif
