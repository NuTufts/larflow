#ifndef __LARFLOW_RECO_VERTEX_RECO_H__
#define __LARFLOW_RECO_VERTEX_RECO_H__

#include <vector>

#include "nlohmann/json.hpp"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class VertexReco
   * @brief Make Vertex space points from keypoint network scores
   */
  class VertexReco {

  public:

    /**
     * labels for types of vertices
     */
    typedef enum { kUndefined=0,  ///< undefined type
                   kConnectedTrackShower, ///< vertex with track and shower cluster attached to vertex
                   kUnconnectedTrackShower, ///< vertex with track and shower unconnected to vertex
                   kConnectedTrackTrack,  ///< vertex with two connected tracks
                   kUnconnectedTrackTrack, ///< vertex with two unconnected tracks
                   kShowerWithVertexActivity ///< vertex on a shower with vertex activity at the start of the shower
    } Type_t ;

    /**
     * Prong types
     */
    typedef enum { kTrackProng=0, ///< track-like prong cluster
                   kShowerProng,  ///< shower-like prong cluster
                   kActivityProng ///< vertex activity prong cluster
    } Prong_t;

    /**
     * @struct Candidate_t
     * @brief internal class to store info for a vertex candidate
     */
    struct Candidate_t {

      Type_t type; ///< vertex type
      std::vector<float> pos; ///< 3d position of vertex
      std::vector<int>   imgcoord; ///< position of vertex in the images (u col,v col,y col, tick)
      std::vector<cluster_t> cluster_v; ///< clusters assigned to the vertex
      std::vector<Prong_t> prong_v;  ///< prongs assigned to the vertex
      
    };
    
    std::vector<Candidate_t> findVertices( larcv::IOManager& iolcv,
                                           larlite::storage_manager& ioll );
    

    std::vector<Candidate_t> trackShowerIntersections( const larlite::event_larflowcluster& track_v,
                                                       const larlite::event_larflowcluster& shower_v,
                                                       const std::vector<larcv::Image2D>& adc_v,
                                                       const float max_end_dist,
                                                       const float max_inter_dist );

    std::vector<Candidate_t> trackShowerIntersections( const std::vector<cluster_t>& track_v,
                                                       const std::vector<cluster_t>& shower_v,
                                                       const std::vector<larcv::Image2D>& adc_v,
                                                       const float max_end_dist,
                                                       const float max_inter_dist );

    std::vector< Candidate_t > showerEndActivity( const std::vector<cluster_t>& track_v,
                                                  const std::vector<cluster_t>& shower_v,
                                                  const std::vector<larcv::Image2D>& adc_v );

    std::vector< cluster_t > _mergeShowerClusters( const std::vector<cluster_t>& shower_v,
                                                   const float max_endpt_dist );
    std::vector< cluster_t > _mergeShowerAndTrackClusters( const std::vector<cluster_t>& shower_v,
                                                           const std::vector<cluster_t>& track_v );
    
    std::vector<float> _findMaxChargePixel( const std::vector<float>& pt3d,
                                            const larcv::Image2D& adc,
                                            const int boxradius,
                                            const float pixel_threshold );
    

    void dumpCandidates2json( const std::vector< Candidate_t >& vtx_v, std::string outfile );
    nlohmann::json dump2json( const std::vector< VertexReco::Candidate_t >& vtx_v );    
    
  };
  
}
}

#endif
