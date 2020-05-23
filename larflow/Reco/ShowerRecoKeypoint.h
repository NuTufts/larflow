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

    void _reconstructClusterTrunks( const std::vector<const cluster_t*>&    showercluster_v,
                                    const std::vector<const larlite::larflow3dhit*>& keypoint_v );
    
    
    /* std::vector<cluster_t> findTrunkCandidates( const cluster_t& showerclust, */
    /*                                             const std::vector<larcv::Image2D>& adc_v ); */


  protected:
    
    // PARAMETERS
    std::string _ssnet_lfhit_tree_name;

  public:

    void set_ssnet_lfhit_tree_name( std::string name ) { _ssnet_lfhit_tree_name=name; };
    
    
    
  };
  

}
}

#endif
