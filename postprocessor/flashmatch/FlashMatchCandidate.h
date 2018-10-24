#ifndef __FLASHMATCHCANDIDATE_H__
#define __FLASHMATCHCANDIDATE_H__

// eigen
#include <Eigen/Dense>

// LArCV2 data
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/ChStatus.h"

// larlite
#include "DataFormat/pcaxis.h"
#include "DataFormat/mctrack.h"

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

    bool _hasevstatus;
    const larcv::EventChStatus* _evstatus;

    QCluster_t _offset_qcluster; // basically a copy of the core, but with the xoffset applied
    QCluster_t _offset_qgap;     // copy of the core cluster gap points, but with the xoffset applied    
    QCluster_t _entering_qcluster;
    QCluster_t _exiting_qcluster;
    Eigen::Vector3f _topend;
    Eigen::Vector3f _botend;
    float _toptick;
    float _bottick;
    
    FlashHypo_t _corehypo;
    FlashHypo_t _noncorehypo;
    FlashHypo_t _entering_hypo;
    FlashHypo_t _exiting_hypo;
    FlashHypo_t _gapfill_hypo;

    static FlashHypo_t buildFlashHypothesis( const FlashData_t& flashdata, const QCluster_t&  qcluster, const float xoffset );
    void dumpMatchImage();
    void setChStatusData( const larcv::EventChStatus* evstatus ) { _evstatus = evstatus; _hasevstatus=true; };

    void addMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v );
    
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
    
    // MCTrack info for flash-mctrack and cluster-mctrack matching
    const std::vector< larlite::mctrack >* _mctrack_v;
    int _flash_mctrackid;
    int _flash_mctrackidx;
    int _cluster_mctrackid;
    int _cluster_mctrackidx;    
    const larlite::mctrack* _flash_mctrack;
    const larlite::mctrack* _cluster_mctrack;

    
  };

}

#endif
