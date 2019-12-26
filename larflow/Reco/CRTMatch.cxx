#include "CRTMatch.h"

namespace larflow {
namespace reco {

  void CRTMatch::clear() {
    _intime_opflash_v.clear();
    _outtime_opflash_v.clear();
    _crthit_v.clear();
    _crttrack_v.clear();
    _lfcluster_v.clear();
    _pcaxis_v.clear();
  }
  
  void CRTMatch::addIntimeOpFlashes( const larlite::event_opflash& opflash_v ) {
    for ( auto const& opf : opflash_v )
      _intime_opflash_v.push_back( opf );
  }

  void CRTMatch::addCosmicOpFlashes( const larlite::event_opflash& opflash_v ) {
    for ( auto const& opf : opflash_v )
      _outtime_opflash_v.push_back( opf );
  }

  void CRTMatch::addCRThits( const larlite::event_crthit& crthit_v ) {
    for ( auto const& hit : crthit_v )
      _crthit_v.push_back( hit );
  }
  
  void CRTMatch::addCRTtracks( const larlite::event_crttrack& crttrack_v ) {
    for ( auto const& track : _crttrack_v )
      _crttrack_v.push_back( track );
  }
  
  void CRTMatch::addLArFlowClusters( const larlite::event_larflowcluster& lfcluster_v,
                                     const larlite::event_pcaxis& pcaxis_v ) {
    for ( auto const& cluster : lfcluster_v ) {
      _lfcluster_v.push_back( &cluster );
    }
    
    for ( auto const& pca : pcaxis_v ) {
      _pcaxis_v.push_back( &pca );
    }
  }
  
  
}
}
