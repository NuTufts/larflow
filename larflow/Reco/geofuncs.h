#ifndef __LARFLOW_GEOFUNCS_H__
#define __LARFLOW_GEOFUNCS_H__

#include <vector>

namespace larflow {
namespace reco {


  template <class T>
    T pointLineDistance( const std::vector<T>& linept1,
                            const std::vector<T>& linept2,
                            const std::vector<T>& testpt );

  template <class T>
    T pointRayProjection( const std::vector<T>& start,
                          const std::vector<T>& dir,
                          const std::vector<T>& testpt );

  float pointLineDistance3f( const std::vector<float>& linept1,
                             const std::vector<float>& linept2,
                             const std::vector<float>& testpt );
  
  float pointRayProjection3f( const std::vector<float>& start,
                              const std::vector<float>& dir,
                              const std::vector<float>& testpt );

  double pointLineDistance3d( const std::vector<double>& linept1,
                              const std::vector<double>& linept2,
                              const std::vector<double>& testpt );
  
  double pointRayProjection3d( const std::vector<double>& start,
                               const std::vector<double>& dir,
                               const std::vector<double>& testpt );
  
}
}

#endif
