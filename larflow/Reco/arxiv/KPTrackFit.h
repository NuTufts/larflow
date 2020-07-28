#ifndef __LARFLOW_RECO_KPTRACKFIT_H__
#define __LARFLOW_RECO_KPTRACKFIT_H__

#include <vector>
#include <map>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {
namespace reco {

  /**
   *
   * @ingroup Reco
   * @class KPTrackFit
   * @brief track reconstruction using dijkstra's algorithm to navigate between two spacepoints
   *
   * Class that implements dijkstra's algorithm using space points.
   * The start and end node are determined by KeyPoint candidates.
   *
   */  
  class KPTrackFit : public larcv::larcv_base {

  public:

    KPTrackFit()
      : larcv::larcv_base("KPTrackFit")
      {};
    virtual ~KPTrackFit() {};

    std::vector<int> fit( const std::vector< std::vector<float> >& point_v,
                          const std::vector< larcv::Image2D >& badch_v,
                          int start, int end );

    void defineEdges( const std::vector< std::vector<float> >& point_v, // list of points
                      std::map< std::pair<int,int>, float >& distmap,   // map between point index and distance between points
                      const float max_neighbor_dist );                  // max distance between points

    void addBadChCrossingConnections( const std::vector< std::vector<float> >& point_v,
                                      const std::vector< larcv::Image2D >& badch_v,
                                      const float min_gap, const float max_gap,                                      
                                      std::map< std::pair<int,int>, float >& distmap );
    
    void addConnectGapsViaClusterPCA( const std::vector< std::vector<float> >& point_v,
                                      const float min_gap, const float max_gap,
                                      std::map< std::pair<int,int>, float >& distmap );
    
  };

}
}
    

#endif
