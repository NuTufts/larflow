#ifndef __LARFLOW_RECO_VETO_HIT_CLUSTERING_H__
#define __LARFLOW_RECO_VETO_HIT_CLUSTERING_H__

#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "NuVertexCandidate.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class VetoHitClustering
   * @brief Cluster vetoed hits around keypoints
   *
   * Hits are either appended to prongs or new clusters are created.
   *
   */
  class VetoHitClustering : public larcv::larcv_base {
  public:

    VetoHitClustering()
      : larcv::larcv_base("VetoHitClustering")
    {};
    virtual ~VetoHitClustering() {};

    void process( larlite::storage_manager& io,
		  larcv::IOManager& iolcv,
		  larflow::reco::NuVertexCandidate& nuvtx );
		  
    void _merge_hits_into_prongs( const larlite::event_larflow3dhit& inputhits,
				  std::vector<int>& close_hits_v,
				  larflow::reco::NuVertexCandidate& nuvtx );

    void _findVetoClusters( const larlite::event_larflow3dhit& inputhits,
			    const std::vector<int>& close_hits_v,
			    larflow::reco::NuVertexCandidate& nuvtx,
			    std::vector<larflow::reco::cluster_t>& output_cluster_v );
    
  };  

}
}



#endif
