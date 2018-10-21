#ifndef __FLASHMATCHCANDIDATE_H__
#define __FLASHMATCHCANDIDATE_H__

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
    
  };
  
  class FlashMatchCandidate {
  public:

    FlashMatchCandidate( const FlashData_t& fdata, const QClusterCore& qcoredata );
    virtual ~FlashMatchCandidate() {};

    const FlashData_t* _flashdata; // source flash data
    const QCluster_t*  _cluster;   // source cluster data
    const QClusterCore* _core;

    QCluster_t _entering_qcluster;
    QCluster_t _exiting_qcluster;
    QCluster_t _gapfill_qcluster;
    
    FlashHypo_t _corehypo;
    FlashHypo_t _noncorehypo;
    FlashHypo_t _entering_hypo;
    FlashHypo_t _exiting_hypo;
    FlashHypo_t _gapfill_hypo;

  protected:

    // Flash Hypothesis Building Routines
    // ----------------------------------

    // define the core, non-core, and their respective principle components
    void defineCore();

    // fill gap within the core
    //void fillGap();

    // extend the core to the detector edge from the most likely entering end
    //  and past the TPC boundaries in so far as it improves the flash-match
    //void ExtendEnteringEnd();

    // extend core on the exiting end
    //   only continues as long as flash-match improved
    
    
    
  };

}

#endif
