#ifndef __QCLUSTER_CORE_H__
#define __QCLUSTER_CORE_H__

#include <vector>

// larlite
#include "DataFormat/pcaxis.h"

#include "FlashMatchTypes.h"


namespace larflow {

  class QClusterCore {
    // because this information about the qcluster we want for each qcluster-flash pair
    //  but we dont want to calculate it more than once per core cluster, we break it
    //  out as its own class
  public:
    
    QClusterCore( const QCluster_t& qcluster );
    virtual ~QClusterCore() {};
    
    const QCluster_t* _cluster;
    
    QCluster_t  _core;
    larlite::pcaxis _pca_core;
    std::vector< QCluster_t > _noncore;
    std::vector< larlite::pcaxis > _pca_noncore;
    
    QCluster_t _gapfill_qcluster;
    
  protected:
    
    void buildCore();
    void fillGaps();
    
  };


}

#endif
