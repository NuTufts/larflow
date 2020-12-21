#ifndef __LARFLOW_RECO_CRTHITMATCH_H__
#define __LARFLOW_RECO_CRTHITMATCH_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/crthit.h"
#include "DataFormat/crttrack.h"
#include "DataFormat/opflash.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/track.h"

#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace crtmatch {

  /**
   * @ingroup CRTMatch
   * @class CRTHitMatch
   * @brief match larflow track clusters with CRT hits and optical flashes
   *
   * This algorithms relies on track candidate clusters to first have been made.
   * There are algorithms in larflow/Reco/ for track reconstruction.
   *
   * Given a track, it's direction is used to project an intersection point with
   * the Cosmic Ray Tagger (CRT).  This intersection point is compared with
   * the CRT hits to see if any are nearby.
   *
   * A match of a track to a CRT hit allows us to assign the time of the CRT hit
   * to the track. This time can be used to associate an optical flash to 
   * the track as well.
   *
   * There are two ways to run this algorithm.
   * The first is to input the ingredients to match using 
   * \verbatim embed:rst:leading-asterisks
   *   * addIntimeOpFlashes
   *   * addCosmicOpFlashes
   *   * addCRThits
   *   * addCRTtracks
   *   * addLArFlowClusters
   * \endverbatim
   * and then call `makeMatches()`. One can then save the output 
   * by calling `save_to_file(...)`.
   *
   * The second way is to use the `process(...)` method.
   *
   */  
  class CRTHitMatch : public larcv::larcv_base {

  public:

    /**
     * @brief internal struct used to compile info about a TPC track to CRT hit match
     *
     */
    struct match_t {
      int hitidx;   ///< index of CRT hit in the event container
      int trackidx; ///< index of track in the event container
      // variables to rank matches
      float dist2hit; ///< distance to closest CRT hit
      float tracklen; ///< length of track reconstructed

      /** 
       * @brief comparator used to sort match candidates by distance to CRT hit
       */
      bool operator<( const match_t& rhs ) {
        // for distances over 10 cm, we rank by distance
        if ( dist2hit<rhs.dist2hit )
          return true;
        return false;
      };

    };
    
    CRTHitMatch()
      : larcv::larcv_base("CRTHitMatch"),
      _kInputDataType(kInputCluster),
      _input_cluster_treename("pcacluster"),
      _input_pcaxis_treename("pcacluster"),
      _input_track_treename("cosmictrack"),
      _input_trackcluster_treename("__null__")
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

    float makeOneMatch( const larlite::track& track,
                        const larlite::crthit& hit,
                        std::vector<float>& panel_pos,
                        std::vector<float>& endpt_matched );
    float getLength( const larlite::pcaxis& pca );
    float getLength( const larlite::track& t );

    bool was_cluster_used( int idx );

    // data stores   
    std::vector<larlite::opflash> _intime_opflash_v;  ///< flashes in unbiased beam readout
    std::vector<larlite::opflash> _outtime_opflash_v; ///< flashes from cosmic disc readout
    std::vector< const larlite::opflash* > _flash_v;  ///< container holding pointers to all input flashes (from both in-time and out-of-time sets)
    
    // crt hits
    std::vector<larlite::crthit>   _crthit_v;    ///< CRT hits in the event
    std::vector<larlite::crttrack> _crttrack_v;  ///< CRT tracks in the event

    // the input data type to match can either be clusters or larlite tracks
    // we will convert clusters into a track representation in the end
    // because we need a line to point back to the CRT
    /** @brief defines kind of input class use */
    typedef enum { kInputCluster=0, kInputTrack } InputDataType_t;
    InputDataType_t _kInputDataType; ///< sets the input type we are parsing
    
    // clusters, if using clusters
    std::string _input_cluster_treename; ///< root tree name from which we get track clusters to match
    std::string _input_pcaxis_treename;  ///< root tree name from which we get the principle component axes for clusters from _input_cluster_treename
    /** @brief set the name of the root tree from which we should get larlite::larflowcluster from */
    void setInputClusterTreename( std::string name ) { _input_cluster_treename = name; _kInputDataType=kInputCluster; };
    /** @brief set the name of the root tree from which we should get larlite::larflowcluster from */
    void setInputPCAxisTreename( std::string name ) { _input_pcaxis_treename = name; _kInputDataType=kInputCluster; };
    std::vector< const larlite::larflowcluster* > _lfcluster_v;   ///< 3d space point clusters making up a track
    std::vector< const larlite::pcaxis* >         _pcaxis_v;      ///< the principle component axes for the clusters in _lfcluster_v
    std::vector< larlite::track >                 _cluster_as_track_v; ///< track representation of tracks
    void _convertClusterToTracks( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    // tracks, if using tracks to match
    std::string _input_track_treename;        ///< name of larlite::track tree to use for input track objects
    std::string _input_trackcluster_treename; ///< name of larlite::larflowcluster tree associated with track objects
    std::vector< const larlite::track* >           _track_input_v;        ///< container for tracks to match to CRT objects
    std::vector< const larlite::larflowcluster* >  _trackcluster_input_v; ///< container for tracks to match to CRT objects    
    /** @brief set the name of the root tree from which we should get the input larlite::track */
    void setInputTrackTreename( std::string name ) { _input_track_treename = name; _kInputDataType=kInputTrack; };
    /** @brief set the name of the root trees from which we should get the pairs of larlite::track and larlite::larflowcluster */
    void setInputTrackTreename( std::string track, std::string trackcluster ) {
      _input_track_treename = track;
      _input_trackcluster_treename = trackcluster;
      _kInputDataType=kInputTrack;
    };    
    void _loadTrackInput( larlite::storage_manager& ioll );

    std::vector< std::vector< match_t > >         _hit2track_rank_v; ///< for each input crt hit (outer vector), match info to each track
    std::vector< match_t >                        _all_rank_v;       ///< all matches


    void _matchOpflashes( const std::vector< const larlite::opflash* >& flash_v,
                          const std::vector< const larlite::crthit* >&  hit_v,
                          const std::vector< larlite::larflowcluster >& cluster_v,
                          std::vector< larlite::opflash >& matched_opflash_v );
    void _matchOpflashes( const std::vector< const larlite::opflash* >& flash_v,
                          const std::vector< const larlite::crthit* >&  hit_v,
                          const std::vector< larlite::track >& track_v,
                          std::vector< larlite::opflash >& matched_opflash_v );
    
    
    larlite::larflowcluster _merge_matched_cluster( const std::vector< CRTHitMatch::match_t >& hit_match_v,
                                                    std::vector<int>& used_in_merge,
                                                    bool& merged );

    // output containers
    std::vector<int>                       _matched_hitidx;    ///< indices of CRT hits in _crthit_v which found a good track match
    std::vector< const larlite::crthit* >  _match_hit_v;       ///< pointer to CRT hit object that was matched
    std::vector<int>                       _match_hitidx_v;    ///< container index to CRT hit object that was matched, for debugging
    std::vector<larlite::larflowcluster>   _matched_cluster;   ///< matching cluster to the CRT hit
    std::vector<larflow::reco::cluster_t>  _matched_cluster_t; ///< matching cluster in larflow::reco::cluster_t form
    std::vector<larlite::track>            _matched_track;     ///< matching track to the CRT hit
    std::vector<larlite::larflowcluster>   _matched_trackcluster; ///< matching track cluster to the CRT hit    
    std::vector< larlite::opflash >        _matched_opflash_v; ///< matched opflash to CRT hit + track pair
    std::vector<int>                       _used_tracks_v;     ///< flag indicating if track in _lfcluster_v has been matched. 1: matched, 0: unmatched
    
  };

  
  
}
}

#endif
