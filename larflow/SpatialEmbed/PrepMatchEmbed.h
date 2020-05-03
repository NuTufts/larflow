#ifndef __PrepMatchEmbed_h__
#define __PrepMatchEmbed_h__

/**
 * This class uses the ancestor image to make 
 * truth information for triplets for
 * training a spatialembedding network
 * for clustering
 *
 */

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larcv {
  class IOManager;
}
namespace larlite {
  class storage_manager;
}


namespace larflow {
  
namespace spatialembed {

  class PrepMatchEmbed {
  public:

    PrepMatchEmbed() {};
    virtual ~PrepMatchEmbed() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll, const PrepMatchTriplets& triplets );

    void _collect_instance_stats();
    
    
  };
  
}
}
    

#endif
