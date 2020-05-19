#ifndef __LARFLOW_RECO_KPTRACKFIT_H__
#define __LARFLOW_RECO_KPTRACKFIT_H__

/**
 *
 * class that implements dijkstra's algorithm using space points
 * the start and end node are determined by KeyPoint candidates
 *
 */

#include <vector>
#include <map>
#include "larcv/core/Base/larcv_base.h"

namespace larflow {
namespace reco {

  class KPTrackFit : public larcv::larcv_base {

  public:

    KPTrackFit()
      : larcv::larcv_base("KPTrackFit")
      {};
    virtual ~KPTrackFit() {};

    std::vector<int> fit( const std::vector< std::vector<float> >& point_v,
                          int start, int end );

    void defineEdges( const std::vector< std::vector<float> >& point_v, // list of points
                      std::map< std::pair<int,int>, float >& distmap,   // map between point index and distance between points
                      const float max_neighbor_dist );                  // max distance between points
    
  };

}
}
    

#endif
