#ifndef __LARFLOW_SHOWER_RECO_H__
#define __LARFLOW_SHOWER_RECO_H__

#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  class ShowerReco {
  public:

    ShowerReco() {};
    virtual ~ShowerReco() {};
    
    void process( larcv::IOManager& iolc, larlite::storage_manager& ioll );

  protected:

    std::vector<cluster_t> findTrunkCandidates( const cluster_t& showerclust,
                                                const std::vector<larcv::Image2D>& adc_v );
    
    void dump2json( const std::vector<cluster_t>& shower_v, std::string outfile );
    
  };
  

}
}

#endif
