#ifndef __LARFLOW_RECO_NUVERTEX_SHOWER_RECO_H__
#define __LARFLOW_RECO_NUVERTEX_SHOWER_RECO_H__

#include <vector>
#include <map>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "NuVertexCandidate.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /** 
   * @ingroup Reco
   * @class NuVertexShowerReco
   * @brief Build showers by making simple cone to shower fragments starting from vertex.
   *
   * Take shower clusters assigned to vertex and build final shower objects for vertex.
   * Do this by simple cone algorithm. Provide dE/dx measure.
   *
   */
  class NuVertexShowerReco : public larcv::larcv_base {

  public:

    NuVertexShowerReco()
      : larcv::larcv_base("NuVertexShowerReco")
    {};
    virtual ~NuVertexShowerReco() {};


    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  std::vector<NuVertexCandidate>& nu_candidate_v );

  protected:

    std::map<std::string, larlite::event_larflowcluster* > _cluster_producers;     ///< map from tree name to event container for larflowcluster
    std::map<std::string, larlite::event_pcaxis* >         _cluster_pca_producers; ///< map from tree name to pca info for cluster
    std::map<std::string, NuVertexCandidate::ClusterType_t > _cluster_type;        ///< cluster type
    
  public:
    
    /** @brief add name of tree to get shower clusters from. call before running process. */
    void add_cluster_producer( std::string name, NuVertexCandidate::ClusterType_t ctype ) {
      _cluster_producers[name] = nullptr;
      _cluster_pca_producers[name] = nullptr;
      _cluster_type[name] = ctype;      
    };
    
  protected:

    void _build_vertex_showers( NuVertexCandidate& nuvtx,
                                larcv::IOManager& iolcv, 
                                larlite::storage_manager& ioll );
    
    void _make_trunk_cand( const std::vector<float>& pos,
                           const larlite::larflowcluster& lfcluster,
                           std::vector<float>& shower_start,
                           std::vector<float>& shower_dir,
                           float& shower_ll );


  };

}
}


#endif
