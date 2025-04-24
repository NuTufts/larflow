#ifndef __LARFLOW_RECOUTIL_COMPRESS_RECOTRACK_H__
#define __LARFLOW_RECOUTIL_COMPRESS_RECOTRACK_H__

#include "larlite/DataFormat/track.h"

namespace larflow {
namespace recoutils {

  class CompressRecoTrack {
    
  public:

    CompressRecoTrack() {};
    ~CompressRecoTrack() {};

    larlite::track compress( const larlite::track& input_track, float max_saggita_cm=0.3, float max_step_size=10.0 ) const;
    
  };
  
}
}

#endif
