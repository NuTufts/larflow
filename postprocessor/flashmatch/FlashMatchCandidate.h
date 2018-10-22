#ifndef __FLASHMATCHCANDIDATE_H__
#define __FLASHMATCHCANDIDATE_H__

// LArCV2 data
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/ChStatus.h"

// larlite
#include "DataFormat/pcaxis.h"

// larflow
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
    float _maxch_pe;
    int   _maxch;

    const larcv::EventChStatus* _evstatus;

    QCluster_t _offset_qcluster; // basically a copy of the core, but with the xoffset applied
    QCluster_t _entering_qcluster;
    QCluster_t _exiting_qcluster;
    
    FlashHypo_t _corehypo;
    FlashHypo_t _noncorehypo;
    FlashHypo_t _entering_hypo;
    FlashHypo_t _exiting_hypo;
    FlashHypo_t _gapfill_hypo;

    static FlashHypo_t buildFlashHypothesis( const FlashData_t& flashdata, const QCluster_t&  qcluster, const float xoffset );
    void dumpMatchImage();
    void setChStatusData( const larcv::EventChStatus* evstatus ) { _evstatus = evstatus; };
    
  protected:

    // Flash Hypothesis Building Routines
    // ----------------------------------
    
    // define the core, non-core, and their respective principle components
    void defineCore();

    // extend the core to the detector edge from the most likely entering end
    //  and past the TPC boundaries in so far as it improves the flash-match
    int _usefront; // use the positive eigen vector end to extend, use the negative end
    void ExtendEnteringEnd();
    void ExtendExitingEnd();
    
    // extend core on the exiting end
    //   only continues as long as flash-match improved
    
    
    
  };

}

#endif