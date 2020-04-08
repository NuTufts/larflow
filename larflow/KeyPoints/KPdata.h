#ifndef __KPdata_h__
#define __KPdata_h__

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
    KPdata() {
      crossingtype = -1;
      imgcoord.clear();
      keypt.clear();
      trackid = 0;
      pid = 0;
      vid = 0;
      is_shower = 0;
      origin = -1;
    };
    ~KPdata() {};
  };
  
  
}
}

#endif
