#include "CRTMatch.h"

namespace larflow {
namespace crtmatch {

  /**
   *
   * iomanager inputs:
   *   adc images (typicall 'wire')
   * 
   * larlite inputs:
   *   opflash: typically 'simpleflashbeam' and 'simpleflashcosmic'
   *   crttrack: typically 'crttrack'
   *   crthit: typically 'crthit'
   *   larflowcluster (typically) pcacluster
   *   pcaxis for larflow cluster: typically 'pcacluster'
   *
   * larlite output: correllated among the following
   *   crttrack: records crt info, including crttrack matches AND crthit. For crt hit, only first hit information listed.
   *   larflowcluster: list of larflow3dhits that relate space point to image coordinates
   *   opflash: matching opflash to crt track (empty flash with totpe=0.0 if no flash matched)
   * 
   * larcv output: 
   *   image2d: with matched CRT pixels masked out
   * 
   */
  void CRTMatch::process( larcv::IOManager& iocv, larlite::storage_manager& ioll ) {

    _hit_matcher.set_verbosity(larcv::msg::kINFO);
    
    _hit_matcher.process( iocv, ioll );
    _track_matcher.process( iocv, ioll );
  }

  void CRTMatch::store_output( larlite::storage_manager& outll, bool remove_if_no_flash ) {
    
    _track_matcher.save_to_file( outll, remove_if_no_flash );
    _hit_matcher.save_to_file( outll, remove_if_no_flash );
    
  }

}
}
