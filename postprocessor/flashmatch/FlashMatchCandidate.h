#ifndef __FLASHMATCHCANDIDATE_H__
#define __FLASHMATCHCANDIDATE_H__

#include "FlashMatchTypes.h"

namespace larflow {
  
  class FlashMatchCandidate {
  public:

    FlashMatchCandidate( const FlashData_t& fdata, const QCluster_t& qdata );
    virtual ~FlashMatchCandidate() {};

    const FlashData_t* _flashdata;
    const QCluster_t*  _cluster;

    FlashHypo_t _corehypo;
    FlashHypo_t _noncorehypo;
    FlashHypo_t _entering_hypo;
    FlashHypo_t _exiting_hypo;
    FlashHypo_t _gapfill_hypo;

  protected:

    // Flash Hypothesis Building Routines
    

    
  };

}

#endif
