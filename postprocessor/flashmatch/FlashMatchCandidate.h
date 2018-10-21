#ifndef __FLASHMATCHCANDIDATE_H__
#define __FLASHMATCHCANDIDATE_H__

// larlite
#include "DataFormat/pcaxis.h"

#include "FlashMatchTypes.h"
#include "QClusterCore.h"

namespace larflow {

  
  class FlashMatchCandidate {
  public:

    FlashMatchCandidate( const FlashData_t& fdata, const QClusterCore& qcoredata );
    virtual ~FlashMatchCandidate() {};

    const FlashData_t* _flashdata; // source flash data
    const QCluster_t*  _cluster;   // source cluster data
    const QClusterCore* _core;
    float _xoffset;

    QCluster_t _entering_qcluster;
    QCluster_t _exiting_qcluster;
    
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
