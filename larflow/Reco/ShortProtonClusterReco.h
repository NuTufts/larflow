#ifndef __SHORT_PROTON_CLUSTER_RECO_H__
#define __SHORT_PROTON_CLUSTER_RECO_H__

#include <string>

#include "DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

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

  protected:

    std::string _input_hit_treename;
    
  };
  
}
}

#endif
