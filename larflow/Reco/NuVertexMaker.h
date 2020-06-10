#ifndef __NU_VERTEX_MAKER_H__
#define __NU_VERTEX_MAKER_H__

/**
 * Form Neutrino Vertex Candidates.
 * Approach is to use Keypoints and rank candidates
 * based on number of track and shower segments that point back to it.
 *
 * Goal is to form seed to build particle hypothesis graph.
 *
 */

#include <string>
#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace reco {

  class NuVertexMaker : public larcv::larcv_base {

  public:

    NuVertexMaker();
    virtual ~NuVertexMaker() {};

    void process( larcv::IOManager& ioman, larlite::storage_manager& ioll );

    /**
       structure representing particle cluster that is close enough to
       vertex
    */
    typedef enum { kTrack=0, kShowerKP, kShower } ClusterType_t;
    struct VtxCluster_t {
      std::string producer;
      int index;
      std::vector<float> dir;
      std::vector<float> pos;
      float gap;
      float impact;
      int npts;
      ClusterType_t type;
    };

    struct Vertex_t {
      std::string keypoint_producer;      
      int keypoint_index;
      std::vector<float> pos;
      std::vector< VtxCluster_t > cluster_v;
      float score;
      bool operator<(const Vertex_t& rhs) const {
        if ( score>rhs.score ) return true;
        return false;
      };
    };

  protected:

    std::vector<Vertex_t> _vertex_v;

  protected:

    std::map<std::string, larlite::event_larflow3dhit* > _keypoint_producers;
    std::map<std::string, larlite::event_pcaxis* >       _keypoint_pca_producers;

    std::map<std::string, larlite::event_larflowcluster* > _cluster_producers;
    std::map<std::string, larlite::event_pcaxis* >         _cluster_pca_producers;
    std::map<std::string, ClusterType_t >                  _cluster_type;
    std::map<ClusterType_t, float>                         _cluster_type_max_impact_radius;
    std::map<ClusterType_t, float>                         _cluster_type_max_gap;
    
  public:

    void add_keypoint_producer( std::string name ) {
      _keypoint_producers[name] = nullptr;
      _keypoint_pca_producers[name] = nullptr;
    };

    void add_cluster_producer( std::string name, NuVertexMaker::ClusterType_t ctype ) {
      _cluster_producers[name] = nullptr;
      _cluster_pca_producers[name] = nullptr;
      _cluster_type[name] = ctype;
    };

    void clear();
    
  protected:

    void _createCandidates();
    void _set_defaults();
    void _score_vertex( Vertex_t& vtx ); 
    
  };
  
}
}

#endif
