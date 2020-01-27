#ifndef __LARFLOW_RECO_CRTMATCH_H__
#define __LARFLOW_RECO_CRTMATCH_H__

/**
 * goal of class is to match larflow track clusters with CRT hits and then optical flashes
 *
 */

#include <vector>
#include "DataFormat/storage_manager.h"
#include "DataFormat/crthit.h"
#include "DataFormat/crttrack.h"
#include "DataFormat/opflash.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace crtmatch {

  class CRTMatch {

  public:

    CRTMatch() {};
    virtual ~CRTMatch() {};
    
    void addIntimeOpFlashes( const larlite::event_opflash& opflash_v );
    void addCosmicOpFlashes( const larlite::event_opflash& opflash_v );
    void addCRThits( const larlite::event_crthit& crthit_v );
    void addCRTtracks( const larlite::event_crttrack& crttrack_v );
    void addLArFlowClusters( const larlite::event_larflowcluster& lfcluster_v, const larlite::event_pcaxis& pcaxis );

    void clear();
    void match( larlite::storage_manager& llio, larlite::storage_manager& outio );
    void printHitInfo();

    float makeOneMatch( const larlite::pcaxis& lfcluster_axis, const larlite::crthit& hit, std::vector<float>& panel_pos );
    float getLength( const larlite::pcaxis& pca );    

    // data stores
    std::vector<larlite::opflash> _intime_opflash_v;  ///< flashes in unbiased beam readout
    std::vector<larlite::opflash> _outtime_opflash_v; ///< flashes from cosmic disc readout

    // crt hits
    std::vector<larlite::crthit>   _crthit_v;
    std::vector<larlite::crttrack> _crttrack_v;
    
    // clusters
    std::vector< const larlite::larflowcluster* > _lfcluster_v;
    std::vector< const larlite::pcaxis* >         _pcaxis_v;


    std::vector< std::vector<int> >               _hit2trackidx_v;
    
  };
  
}
}

#endif
