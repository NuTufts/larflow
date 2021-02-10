#ifndef __SHORT_PROTON_CLUSTER_RECO_H__
#define __SHORT_PROTON_CLUSTER_RECO_H__

#include <string>

#include "DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  class ShortProtonClusterReco : public larcv::larcv_base {

  public:

    ShortProtonClusterReco()
      : larcv::larcv_base("ShortProtonClusterReco"),
      _input_hit_treename("ssnetsplit_wcfilter_trackhit")
      {};
    virtual ~ShortProtonClusterReco() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& io );

    void add_clustertree_forcheck( std::string name ) { _input_cluster_tree_checklist_v.push_back(name); };
    void clear_clustertree_checklist() { _input_cluster_tree_checklist_v.clear(); };
    
  protected:

    void checkForOverlap( larlite::storage_manager& io,
                          std::vector< larflow::reco::cluster_t >& proton_cluster_v,                          
                          std::vector< std::string >& cluster_overlap_list );

    std::string _input_hit_treename; ///< input name of hits to use
    std::vector<std::string> _input_cluster_tree_checklist_v; ///< name of trees containing clusters we should check against to ensure no overlap
    
  };
  
}
}

#endif
