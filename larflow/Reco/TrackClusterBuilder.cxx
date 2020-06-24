#include "TrackClusterBuilder.h"

namespace larflow {
namespace reco {

  void TrackClusterBuilder::process( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll )
  {

    std::string producer = "trackprojsplit_full";
    
    larlite::event_larflowcluster* ev_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);

    loadClusterLibrary( *ev_cluster, *ev_pcaxis );
    
  }

  void TrackClusterBuilder::loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                                                const larlite::event_pcaxis& pcaxis_v )
  {

    for (int i=0; i<cluster_v.size(); i++) {
      const larlite::larflowcluster& cluster = cluster_v.at(i);
      const larlite::pcaxis& pca = pcaxis_v.at(i);

      // create a segment object
      std::vector<float> start(3,0);
      std::vector<float> end(3,0);
      for (int v=0; v<3; v++){
        start[v] = pca.getEigenVectors()[3][v];
        end[v]   = pca.getEigenVectors()[4][v];
      }

      Segment_t seg( start, end );
      _segment_v.push_back(seg);
      _segment_v.back().idx = (int)_segment_v.size()-1;
    }

    LARCV_INFO() << "Stored " << _segment_v.size() << " track segments" << std::endl;
  }

}
}
