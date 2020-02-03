#ifndef __LARFLOW_CRTMATCH_CRTMATCH_H__
#define __LARFLOW_CRTMATCH_CRTMATCH_H__

#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/core/DataFormat/storage_manager.h"
#include "CRTTrackMatch.h"
#include "CRTHitMatch.h"

namespace larflow {
namespace crtmatch {

  class CRTMatch {

  public:
    
    CRTMatch() {};
    virtual ~CRTMatch() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll);
    void store_output( larlite::storage_manager& outll, bool remove_if_no_flash=true );
    
    CRTTrackMatch _track_matcher;
    CRTHitMatch   _hit_matcher;

  };
  
}
}

#endif
