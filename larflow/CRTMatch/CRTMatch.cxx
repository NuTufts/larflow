#include "CRTMatch.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/core/DataFormat/larflowcluster.h"

namespace larflow {
namespace crtmatch {

  /**
   * @brief Process data for one event
   *
   * iomanager inputs:
   * \verbatim embed:rst:leading-asterisks
   *   * adc images (typicall 'wire')
   * \endverbatim
   * 
   * larlite inputs:
   * \verbatim embed:rst:leading-asterisks
   *   * opflash: typically 'simpleflashbeam' and 'simpleflashcosmic'
   *   * crttrack: typically 'crttrack'
   *   * crthit: typically 'crthit'
   *   * larflowcluster (typically) pcacluster
   *   * pcaxis for larflow cluster: typically 'pcacluster'
   * \endverbatim
   *
   * larlite output: correlated among the following
   * \verbatim embed:rst:leading-asterisks
   *   * crttrack: records crt info, including crttrack matches AND crthit. For crt hit, only first hit information listed.
   *   * larflowcluster: list of larflow3dhits that relate space point to image coordinates
   *   * opflash: matching opflash to crt track (empty flash with totpe=0.0 if no flash matched)
   * \endverbatim
   * 
   * larcv output: 
   * \verbatim embed:rst:leading-asterisks
   *   * image2d: with matched CRT pixels masked out
   * \endverbatim
   * 
   * @param[in] iocv larcv::IOManager from which data will be retrieved and output to
   * @param[in] ioll larlite::storage_manager from which data will be retrieved and output to
   *
   */
  void CRTMatch::process( larcv::IOManager& iocv, larlite::storage_manager& ioll ) {

    _hit_matcher.set_verbosity(larcv::msg::kINFO);

    _hit_matcher.process( iocv, ioll );
    _track_matcher.process( iocv, ioll );

    larlite::event_larflowcluster* ev_lfcluster
      = (larlite::event_larflowcluster*) ioll.get_data( larlite::data::kLArFlowCluster, "pcacluster" );

    // we now tag image, and match lfcluster to crttrack cluster, finally labeling untagged
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iocv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector<larcv::Image2D>& adc_v = ev_adc->Image2DArray();
    
    untagged_v.clear(); // image where matched pixels are removed
    track_index_v.clear(); // image where crt track index labels image, so we can match larflow clusters to it.
    for ( auto const& img : adc_v ) {

      // make a copy to mask
      untagged_v.push_back( img );

      // make a blank to tag with index
      larcv::Image2D hempty( img.meta() );
      hempty.paint(-1.0);
      track_index_v.emplace_back( std::move(hempty) );

    }

    // mark used clusters
    std::vector<int> used_clusters_v( ev_lfcluster->size(), 0 );
    for ( size_t i=0; i<ev_lfcluster->size(); i++ ) {
      if ( !_hit_matcher.was_cluster_used( i ) ) continue;
      
      used_clusters_v[i] = 1;

      for ( auto const& lfhit : ev_lfcluster->at(i) ) {
        int row = adc_v[0].meta().row( lfhit.tick );
        for ( size_t p=0; p<3; p++ ) {
          untagged_v[p].set_pixel( row, lfhit.targetwire[p], 0.0 );
        }
      }
    }
    
    // loop over constructed CRT tracks
    float cidx = 0;
    for ( auto const& cluster : _track_matcher.getClusters() ) {
      for ( auto const& lfhit : cluster ) {
        int row = adc_v[0].meta().row( lfhit.tick );
        for ( size_t p=0; p<3; p++ ) {
          track_index_v[p].set_pixel( row, lfhit.targetwire[p], cidx );
          untagged_v[p].set_pixel( row, lfhit.targetwire[p], 0 );
        }
      }
      cidx += 1.0;
    }
    
    
    // now match to clusters
    for ( size_t idx=0; idx<ev_lfcluster->size(); idx++ ) {
      if ( used_clusters_v[idx] ) continue;

      std::vector<int> index_counter( _track_matcher.getClusters().size(), 0 );
      for ( auto const& lfhit : ev_lfcluster->at(idx) ) {
        int row = track_index_v[0].meta().row( lfhit.tick );
        for ( size_t p=0; p<3; p++ ) {
          int index = (int)track_index_v[p].pixel( row, lfhit.targetwire[p] );
          if ( index>=0 && index<(int)index_counter.size() )
            index_counter[index]++;
        }
      }

      // if we match 50% we absorb the cluster
      int nmatched = 0;
      for (size_t i=0; i<index_counter.size(); i++ ) {
        nmatched += index_counter[i];
      }

      float frac_matched = float(nmatched)/float(ev_lfcluster->at(idx).size());

      if ( frac_matched>0.3 ) {
        used_clusters_v[idx] = 1;

        // more complete mask
        for ( auto const& lfhit : ev_lfcluster->at(idx) ) {
          int row = adc_v[0].meta().row( lfhit.tick );
          for ( size_t p=0; p<3; p++ )
            untagged_v[p].set_pixel( row, lfhit.targetwire[p], 0 );
        }
        
      }
    }
    
    _unmatched_clusters_v.clear();
    for ( size_t idx=0; idx<ev_lfcluster->size(); idx++ ) {
      if ( used_clusters_v[idx] ) continue;

      _unmatched_clusters_v.push_back( ev_lfcluster->at(idx) );
    }
    
  }

  /**
   * @brief Copy output in algo containers to larcv and larlite IO managers
   *
   * larlite output: correlated among the following
   * \verbatim embed:rst:leading-asterisks
   *   * crttrack: records crt info, including crttrack matches AND crthit. For crt hit, only first hit information listed.
   *   * larflowcluster: list of larflow3dhits that relate space point to image coordinates
   *   * opflash: matching opflash to crt track (empty flash with totpe=0.0 if no flash matched)
   * \endverbatim
   * 
   * larcv output: 
   * \verbatim embed:rst:leading-asterisks
   *   * image2d: with matched CRT pixels masked out
   * \endverbatim
   *
   * @param[in] outlcv Output larcv::IOManager
   * @param[in] outll  Output larlite::storage_manager
   * @param[in] remove_if_no_flash If true, for both CRT hit and CRT track matcher,
   *                               TPC track to CRT object matches are skipped when no
   *                               optical flash could be associated to the pair.
   */
  void CRTMatch::store_output( larcv::IOManager& outlcv,
                               larlite::storage_manager& outll,
                               bool remove_if_no_flash ) {

    
    _track_matcher.save_to_file( outll, remove_if_no_flash );
    _hit_matcher.save_to_file( outll, remove_if_no_flash );

    larlite::event_larflowcluster* ev_unused =
      (larlite::event_larflowcluster*)outll.get_data( larlite::data::kLArFlowCluster, "crtunmatched" );
    
    for ( auto const& cluster : _unmatched_clusters_v )
      ev_unused->push_back( cluster );

    larcv::EventImage2D* ev_untagged = (larcv::EventImage2D*)outlcv.get_data(larcv::kProductImage2D, "crtmasked" );
    for ( auto const& img : untagged_v ) {
      ev_untagged->Append(img);
    }
    
  }

}
}
