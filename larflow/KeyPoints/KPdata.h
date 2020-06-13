#ifndef __KPdata_h__
#define __KPdata_h__

/**
 * Struct representing true keypoints in the images
 *
 */

#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include <vector>

namespace larflow {
namespace keypoints {

  struct KPdata {
    int crossingtype;
    std::vector<int> imgcoord;
    std::vector<float> keypt;
    int trackid;
    int pid;
    int vid;
    int is_shower;
    int origin;
    KeyPoint_t kptype;
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

  };

  bool kpdata_compare_x( const KPdata* lhs, const KPdata* rhs );
  bool kpdata_compare_y( const KPdata* lhs, const KPdata* rhs );
  bool kpdata_compare_z( const KPdata* lhs, const KPdata* rhs );
  
}
}

#endif
