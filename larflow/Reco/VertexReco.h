#ifndef __LARFLOW_RECO_VERTEX_RECO_H__
#define __LARFLOW_RECO_VERTEX_RECO_H__

#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  class VertexReco {

  public:

    typedef enum { kUndefined=0,
                   kConnectedTrackShower,
                   kUnconnectedTrackShower,
                   kConnectedTrackTrack,
                   kUnconnectedTrackTrack,
                   kShowerWithVertexActivity } Type_t ;

    typedef enum { kTrackProng=0,
                   kShowerProng,
                   kActivityProng } Prong_t;
    
    struct Candidate_t {

      Type_t type;
      std::vector<float> pos;
      std::vector<int>   imgcoord;
      std::vector<cluster_t> cluster_v;
      std::vector<Prong_t> prong_v;
      
    };
    
    std::vector<Candidate_t> findVertices( larcv::IOManager& iolcv,
                                           larlite::storage_manager& ioll );
    

    std::vector<Candidate_t> trackShowerIntersections( const larlite::event_larflowcluster& track_v,
                                                       const larlite::event_larflowcluster& shower_v,
                                                       const std::vector<larcv::Image2D>& adc_v,
                                                       const float max_end_dist,
                                                       const float max_inter_dist );

    void dumpCandidates2json( const std::vector< Candidate_t >& vtx_v, std::string outfile );
    
  };
  
}
}

#endif
