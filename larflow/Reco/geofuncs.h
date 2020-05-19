#ifndef __LARFLOW_GEOFUNCS_H__
#define __LARFLOW_GEOFUNCS_H__

#include <vector>

namespace larflow {
namespace reco {

  float pointLineDistance( const std::vector<float>& linept1,
                           const std::vector<float>& linept2,
                           const std::vector<float>& testpt );
  
}
}

#endif
