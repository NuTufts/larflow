#ifndef __KPdata_h__
#define __KPdata_h__

#include <vector>

namespace larflow {
namespace keypoints {

  struct KPdata {
    int crossingtype;
    std::vector<int> imgcoord_start;
    std::vector<int> imgcoord_end;
    std::vector<float> startpt;
    std::vector<float> endpt;
    int trackid;
    int pid;
    int vid;
    int is_shower;
    int origin;
    KPdata() {
      crossingtype = -1;
      imgcoord_start.clear();
      imgcoord_end.clear();
      startpt.clear();
      endpt.clear();
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
