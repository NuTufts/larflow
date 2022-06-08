#include "KeypointData.h"

namespace larflow {
namespace keypoints {

  KeypointData::KeypointData()
    : larcv::larcv_base("KeypointData"),
      tpcid(0),
      cryoid(0)
  {
  }
  
}
}
