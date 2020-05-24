#ifndef __LARFLOW_SHOWER_RECO_KEYPOINT_H__
#define __LARFLOW_SHOWER_RECO_KEYPOINT_H__

#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  class ShowerRecoKeypoint : public larcv::larcv_base {

  public:

    ShowerRecoKeypoint()
      : larcv::larcv_base("ShowerRecoKeypoint"),
      _ssnet_lfhit_tree_name("showerhit")
      {};    
    virtual ~ShowerRecoKeypoint() {};
    
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  protected:

    struct ShowerTrunk_t {
      int idx_keypoint;
      const larlite::larflow3dhit* keypoint;      
      std::vector< float > pcaxis_v;
      std::vector< float > center_v;      
      float pca_eigenval_ratio;
      int npts;
      float gapdist;
      float impact_par;
      ShowerTrunk_t() {
        idx_keypoint = -1;
        keypoint = nullptr;        
        pcaxis_v.resize(3,0);
        center_v.resize(3,0);
        pca_eigenval_ratio = -1;
        npts = -1;
        gapdist = -1;
        impact_par = -1;
      };
    };
    
    struct ShowerCandidate_t {
      int cluster_idx;
      const cluster_t* cluster;
      std::vector< ShowerTrunk_t > trunk_candidates_v;
    };

    std::vector< ShowerCandidate_t > _shower_cand_v;
    void _reconstructClusterTrunks( const std::vector<const cluster_t*>&    showercluster_v,
                                    const std::vector<const larlite::larflow3dhit*>& keypoint_v );    
    void _buildShowers();
    
  protected:
    
    // PARAMETERS
    std::string _ssnet_lfhit_tree_name;

  public:

    void set_ssnet_lfhit_tree_name( std::string name ) { _ssnet_lfhit_tree_name=name; };
    
    
    
  };
  

}
}

#endif
