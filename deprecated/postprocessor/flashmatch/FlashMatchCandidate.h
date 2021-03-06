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
#include "QClusterComposite.h"

namespace larflow {

  
  class FlashMatchCandidate {
  public:

    FlashMatchCandidate( const FlashData_t& fdata, const QClusterComposite& qcomposite );
    virtual ~FlashMatchCandidate() {};

    const FlashData_t* _flashdata; // source flash data
    const QCluster_t*  _cluster;   // source cluster data
    const QClusterComposite* _qcomposite;
    
    float _xoffset;
    float _maxch_pe;
    int   _maxch;
    void getFlashClusterIndex( int& flashidx, int& clustidx ) { flashidx=_flashdata->idx; clustidx=_cluster->idx; };

    bool _hasevstatus;
    const larcv::EventChStatus* _evstatus;
    
    FlashHypo_t _core_hypo;
    FlashHypo_t _gapfill_hypo;    
    FlashHypo_t _entering_hypo;
    FlashHypo_t _exiting_hypo;

    static FlashHypo_t buildFlashHypothesis( const FlashData_t& flashdata, const QCluster_t&  qcluster, const float xoffset );
    static float getMaxDist( const FlashData_t& flashdata, const FlashHypo_t& flashhypo, bool isnormed=true );
    static float getPERatio( const FlashData_t& flashdata, const FlashHypo_t& flashhypo );
    void dumpMatchImage();
    void setChStatusData( const larcv::EventChStatus* evstatus ) { _evstatus = evstatus; _hasevstatus=true; };
    void addMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v );
    FlashHypo_t getHypothesis( bool withextensions, bool suppresscosmicdisc, float cosmicdiscthresh=10.0 );
    bool isTruthMatch();

    
  protected:

    // Flash Hypothesis Building Routines
    // ----------------------------------
    
    // extend the core to the detector edge from the most likely entering end
    //  and past the TPC boundaries in so far as it improves the flash-match
    /* int _usefront; // use the positive eigen vector end to extend, use the negative end */
    /* void ExtendEnteringEnd(); */
    /* void ExtendExitingEnd(); */
    
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
