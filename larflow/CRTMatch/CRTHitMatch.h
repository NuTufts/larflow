#ifndef __LARFLOW_RECO_CRTHITMATCH_H__
#define __LARFLOW_RECO_CRTHITMATCH_H__

/**
 * goal of class is to match larflow track clusters with CRT hits and then optical flashes
 *
 */

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/crthit.h"
#include "DataFormat/crttrack.h"
#include "DataFormat/opflash.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflowcluster.h"

#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace crtmatch {

  class CRTHitMatch : public larcv::larcv_base {

  public:

    struct match_t {
      int hitidx;
      int trackidx;
      // variables to rank matches
      float dist2hit;
      float tracklen;
      bool operator<( const match_t& rhs ) {
        // for distances over 10 cm, we rank by distance
        if ( dist2hit<rhs.dist2hit )
          return true;
        return false;
      };

    };
    
    CRTHitMatch()
      : larcv::larcv_base("CRTHitMatch")
      {};
    virtual ~CRTHitMatch() {};
    
    void addIntimeOpFlashes( const larlite::event_opflash& opflash_v );
    void addCosmicOpFlashes( const larlite::event_opflash& opflash_v );
    void addCRThits( const larlite::event_crthit& crthit_v );
    void addCRTtracks( const larlite::event_crttrack& crttrack_v );
    void addLArFlowClusters( const larlite::event_larflowcluster& lfcluster_v, const larlite::event_pcaxis& pcaxis );

    void clear();
    bool process( larcv::IOManager& iocv, larlite::storage_manager& ioll );
    void save_to_file( larlite::storage_manager& outll, bool remove_if_no_flash=true );
    
    bool makeMatches();
    void compilematches();
    void printHitInfo();

    float makeOneMatch( const larlite::pcaxis& lfcluster_axis, const larlite::crthit& hit, std::vector<float>& panel_pos );
    float getLength( const larlite::pcaxis& pca );

    bool was_cluster_used( int idx );

    // data stores
    std::vector<larlite::opflash> _intime_opflash_v;  ///< flashes in unbiased beam readout
    std::vector<larlite::opflash> _outtime_opflash_v; ///< flashes from cosmic disc readout

    // crt hits
    std::vector<larlite::crthit>   _crthit_v;
    std::vector<larlite::crttrack> _crttrack_v;
    
    // clusters
    std::vector< const larlite::larflowcluster* > _lfcluster_v;
    //std::vector< larlite::larflowcluster >        _lfcluster_v;    
    std::vector< const larlite::pcaxis* >         _pcaxis_v;


    std::vector< std::vector< match_t > >         _hit2track_rank_v; // matches for each hit
    std::vector< match_t >                        _all_rank_v;       // all matches


    void _matchOpflashes( const std::vector< const larlite::opflash* >& flash_v,
                          const std::vector< const larlite::crthit* >&  hit_v,
                          const std::vector< larlite::larflowcluster >& cluster_v,
                          std::vector< larlite::opflash >& matched_opflash_v );
    
    
    larlite::larflowcluster _merge_matched_cluster( const std::vector< CRTHitMatch::match_t >& hit_match_v,
                                                    std::vector<int>& used_in_merge,
                                                    bool& merged );

    // output containers
    std::vector<int>                       _matched_hitidx;
    std::vector< const larlite::opflash* > _flash_v;    
    std::vector< const larlite::crthit* >  _match_hit_v;    
    std::vector<larlite::larflowcluster>   _matched_cluster;
    std::vector<larflow::reco::cluster_t>  _matched_cluster_t;
    std::vector< larlite::opflash >        _matched_opflash_v;
    std::vector<int>                       _used_tracks_v;
    
  };
  
}
}

#endif
