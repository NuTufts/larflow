#include "CosmicParticleReconstruction.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
//#include "ublarcvapp/Reco3D/TrackReverser.h"
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "TrackdQdx.h"
#include "SplitHitsBySSNet.h"
#include "KeypointFilterByWCTagger.h"
#include "ChooseMaxLArFlowHit.h"
#include "KeypointReco.h"
#include "ProjectionDefectSplitter.h"
#include "CosmicTrackBuilder.h"

namespace larflow {
namespace reco {

  CosmicParticleReconstruction::CosmicParticleReconstruction()
    : larcv::larcv_base("CosmicParticleReconstruction"),
    _ana_file(nullptr),
    _ana_tree(nullptr),
    _save_flashmatchdata_tree(true),
    _flashmatchdata_tree(nullptr)
  {
    set_default_param_values();
  }

  void CosmicParticleReconstruction::set_default_param_values()
  {

    _flash_producer   = "simpleFlashCosmic";
    _wireimg_producer = "wire";
    _outoftime_tagged_pixels_producer = "thrumu";
    _larmatch_hit_producer   = "larmatch";
    _ana_output_file  = "test_cosmicreco.root";

    _ana_run = 0;
    _ana_subrun = 0;
    _ana_event = 0;
    _reco_status = 0;
    _t_event_elapsed = 0.;   
    
  }

  void CosmicParticleReconstruction::clear()
  {
    _cosmic_candidates_v.clear();
  }

  /**
   * @brief create ana file and define output tree
   *
   * The tree created is `KPSRecoManagerTree`.
   *
   */
  void CosmicParticleReconstruction::make_reco_output_file()
  {

    if ( _ana_output_file=="" ) {
      LARCV_ERROR() << "Did not specify an output file." << std::endl;
    }
    
    
    LARCV_NORMAL() << "Create Ana Output File: " << _ana_output_file << std::endl;
    
    _ana_file = new TFile(_ana_output_file.c_str(), "recreate");
    _ana_tree = new TTree("KPSCosmicTree","Output of CosmicParticleReconstruction algorithms");

    // event book-keeping indicies: run, subrun, event
    _ana_tree->Branch("run",&_ana_run,"run/I");
    _ana_tree->Branch("subrun",&_ana_subrun,"subrun/I");
    _ana_tree->Branch("event",&_ana_event,"event/I");
    _ana_tree->Branch("reco_status", &_reco_status, "reco_status/I");
    _ana_tree->Branch("telapsed", &_t_event_elapsed, "telapsed/F" );    
    //_ana_tree->Branch("error_messages", &_error_messages);

    _event_kpc_track_start_v.clear(); 
    _event_kpc_track_end_v.clear();    
    _ana_tree->Branch( "kpc_track_start_v",  &_event_kpc_track_start_v );
    _ana_tree->Branch( "kpc_track_end_v",    &_event_kpc_track_end_v );

    
    if ( _save_flashmatchdata_tree ) {
      // Create flashmatchdata_tree and setup branches
      _flashmatchdata_tree = new TTree("FlashMatchData", "Cosmic tracks, optical flashes, and CRT information for flash matching");
      
      // Event info branches
      _flashmatchdata_tree->Branch("run", &_ana_run, "run/I");
      _flashmatchdata_tree->Branch("subrun", &_ana_subrun, "subrun/I");
      _flashmatchdata_tree->Branch("event", &_ana_event, "event/I");
      
      // Data container branches
      _flashmatchdata_tree->Branch("track_v", &_flashmatchdata_track_v);
      _flashmatchdata_tree->Branch("opflash_v", &_flashmatchdata_opflash_v);
      _flashmatchdata_tree->Branch("crttrack_v", &_flashmatchdata_crttrack_v);
      _flashmatchdata_tree->Branch("crthit_v", &_flashmatchdata_crthit_v);
    }

  }

  /**
   * @brief process event data to find stopping muons
   *
   * we find this sample for calibration purposes.
   * 
   * start by seeding possible vertices using
   * @verbatim embed:rst:leading-asterisk
   *  * keypoints 
   *  * intersections of particle clusters (not yet implemented)
   *  * vertex activity near ends of partice clusters (not yet implemented)
   * @endverbatim
   *
   * output:
   * @verbatim embed:rst:leading-asterisk
   *  * vertex candidates stored in _vertex_v
   *  * need to figure out way to store in larcv or larlite iomanagers
   * @endverbatim
   *
   * @param[in] iolcv Instance of LArCV IOManager with event data
   * @param[in] ioll  Instance of larlite storage_manager containing event data
   *
   */
  void CosmicParticleReconstruction::process( larcv::IOManager& iolcv,
                                              larlite::storage_manager& ioll )
  {
    
    clear();

    // Setup what trees out write for the larlite file
    //ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_offtrigger_trackhit" ); /// track-like and out of time
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit,   "offtrigger_maxtrackhit" ); /// track-like and out of time
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit,   "keypoint_all" );   /// all track start and end keypoints
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit,   "keypointcosmic" ); /// cosmic keypoints
    ioll.set_data_to_write( larlite::data::kCRTTrack,       "crttrack");
    ioll.set_data_to_write( larlite::data::kCRTHit,         "crthitcorr");
    ioll.set_data_to_write( larlite::data::kLArFlowCluster, "trackprojsplit_offtrigger" );
    ioll.set_data_to_write( larlite::data::kPCAxis,         "trackprojsplit_offtrigger" );
    ioll.set_data_to_write( larlite::data::kTrack,          "cosmictrack");
    ioll.set_data_to_write( larlite::data::kLArFlowCluster, "cosmictrack");
    ioll.set_data_to_write( larlite::data::kOpFlash,        "simpleFlashCosmic");
    ioll.set_data_to_write( larlite::data::kOpFlash,        "simpleFlashBeam");

    // Stages

    // PrepSpacepoints: isolate out-of-time spacepoints using the out-of-time tagger using 
    // passing spacepoints are stored in the larlite storage_manager with the treename 'cosmicreco'
    prepSpacepoints( iolcv, ioll );

    // Reconstruct Track-Start and Track-End Keypoints using the larmatch info in the spacepoints
    recoKeypoints( iolcv, ioll );

    // isolate track-like spacepoints and reconstruct into line-like segments
    buildTrackFragments( iolcv, ioll );

    // // use the CosmicTrackBuilder to make muon candidates
    buildCosmicTracks( iolcv, ioll );

    // // make flash predictions and make possible matches
    // makeFlashPredictionAndMatches();

    // // make CRT connections
    // makeCRTConnections();

    // Set run, subrun, event indices in ana tree
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _larmatch_hit_producer );
    _ana_run    = ev_larmatch->run();
    _ana_subrun = ev_larmatch->subrun();
    _ana_event  = ev_larmatch->event_id();

    if ( _ana_tree )
      _ana_tree->Fill(); 

    if ( _save_flashmatchdata_tree ) {
      fillFlashMatchData( ioll );
    }
  }
  
  /**
   * @brief algorithms for splitting up and filtering larmatch space points
   *
   */
  void CosmicParticleReconstruction::prepSpacepoints( larcv::IOManager& iolcv,
                                                      larlite::storage_manager& ioll )
  {

    LARCV_NORMAL() << "Select and sort spacepoints for reconstruction" << std::endl;
    LARCV_NORMAL() << " input container: " << _larmatch_hit_producer << std::endl;
    
    // PREP: make sure the larmatch points have their idxhit index set
    // we can use this as a way to trace identity of hits back to original set
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _larmatch_hit_producer );
    LARCV_NORMAL() << "Number of input spacepoints: " << ev_larmatch->size() << std::endl;    
    for (size_t ihit=0; ihit<ev_larmatch->size(); ihit++) {
      auto& hit = ev_larmatch->at(ihit);
      hit.idxhit = (int)ihit;
    }
    
    // PREP: LABEL larmatch POINTS WITH 2D SSNET SHOWER SCORE
    // input:
    //  * image2d_ubspurn_planeX: ssnet (track,shower) scores
    //  * larflow3dhit_taggerfilterhit_tree: WC in-time space points
    // output:
    //  * process_labelonly only modifies larmatch hits to have shower score

    larflow::reco::SplitHitsBySSNet _splithits_wcfilter;
    _splithits_wcfilter.set_verbosity( logger().level() );
    _splithits_wcfilter.set_larmatch_tree_name( _larmatch_hit_producer );
    _splithits_wcfilter.process_labelonly( iolcv, ioll );   // the 2d shower score is added to larflow3dhit::renormed_shower_score. hits modified.

    // PREP: ALTER THRUMU IMAGE TO INCLUDE SSNET CLUSTERS OF A CERTAIN SIZE
    
    // PREP WC-FILTERED HITS
    // filters raw larmatch hits using wire cell thrumu tagger image
    // input:
    //  larflow3dhit_larmatch_tree: raw larmatch deploy output
    // output(s):
    //  larflow3dhit_taggerfilterhit_tree: in-time hits
    //  larflow3dhit_taggerrejecthit_tree: out-of-time/cosmic-tagged hits
    larflow::reco::KeypointFilterByWCTagger _wcfilter;
    _wcfilter.set_verbosity( logger().level() );
    _wcfilter.set_input_larmatch_tree_name( _larmatch_hit_producer );
    _wcfilter.set_output_filteredhits_tree_name( "taggerfilterhit" );
    _wcfilter.set_save_rejected_hits( true );
    _wcfilter.process_hits( iolcv, ioll );

    // PREP: SPLIT WC-FILTERED HITS INTO TRACK/SHOWER
    // input:
    //  * image2d_ubspurn_planeX: ssnet (track,shower) scores
    //  * larflow3dhit_taggerfilterhit_tree: WC in-time space points
    // output:
    //  * larflow3dhit_ssnetsplit_wcfilter_showerhit_tree: in-time shower hits
    //  * larflow3dhit_ssnetsplit_wcfilter_trackhit_tree:  in-time track hits
    _splithits_wcfilter.set_larmatch_tree_name( "taggerrejecthit"  );
    _splithits_wcfilter.set_output_tree_stem_name( "ssnetsplit_offtrigger" );
    _splithits_wcfilter.process_splitonly( iolcv, ioll );    

    // PREP: MAX-SCORE REDUCTION ON COSMIC HITS
    // input:
    //  * larflow3dhit_ssnetsplit_full_trackhit_tree: out-of-time track hits
    // output:
    //  *  larflow3dhit_full_maxtrackhit_tree: reduced out-of-time track hits
    larflow::reco::ChooseMaxLArFlowHit _choosemaxhit;
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_offtrigger_trackhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "offtrigger_maxtrackhit" );
    _choosemaxhit.process( iolcv, ioll );


  }

  /**
   * @brief make keypoints for use to help make particle track and nu interaction candidates
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void CosmicParticleReconstruction::recoKeypoints( larcv::IOManager& iolcv,
                                                    larlite::storage_manager& ioll )
  {

    // KEYPOINT RECO: make keypoint candidates
    //  * larflow3dhit_larmatch_tree: output of KPS larmatch network
    // output:
    //  * _kpreco.output_pt_v: container of KPCluster objects
    //LARCV_NORMAL() << "reco keypoints version=" << _reco_version << std::endl;

    // we take advantage of the fact that we dont want anything stored by the keypoint reco class
    // after it runs. everything we need downstream is saved to a larlite tree.
    // so we simply re-run the algorithms to work with the additional vertex types.

    // clear past results
    _event_kpc_track_start_v.clear();
    _event_kpc_track_end_v.clear();  

    // Keypoint Reco algorithm
    larflow::reco::KeypointReco  _kpreco_trackstart; ///< reconstruct keypoints from network scores for track class
    _kpreco_trackstart.set_keypoint_type( (int)larflow::kTrackStart );
    _kpreco_trackstart.set_lfhit_score_index( 18 );
    _kpreco_trackstart.clear_output();

    larflow::reco::KeypointReco  _kpreco_trackend;   ///< reconstruct keypoints from network scores for track class
    _kpreco_trackend.set_keypoint_type( (int)larflow::kTrackEnd );
    _kpreco_trackend.set_lfhit_score_index( 19 );
    _kpreco_trackend.clear_output();

    // neutrino interaction track: we have track starts and ends
    std::vector< larflow::reco::KeypointReco* > _kpreco_track_v
      = { &_kpreco_trackstart, &_kpreco_trackend };
    for ( auto& pkpreco_track : _kpreco_track_v ) {	
      pkpreco_track->clear_output();
      pkpreco_track->set_verbosity( logger().level() );
      pkpreco_track->set_num_passes(1);
      pkpreco_track->set_input_larmatch_tree_name( _larmatch_hit_producer ); 
      pkpreco_track->set_sigma( 10.0 );
      pkpreco_track->set_max_dbscan_dist( 0.7 );
      pkpreco_track->set_larmatch_threshold( 0.5 );      
      pkpreco_track->set_min_cluster_size(   10, 0 );
      pkpreco_track->set_keypoint_threshold( 0.2, 0 );
      pkpreco_track->set_output_tree_name( "keypoint_all" );
      pkpreco_track->process( ioll );
    }

    larlite::event_larflow3dhit* ev_kpall
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint_all" );
    LARCV_INFO() << "Number of total track-start + track-end keypoints reconstructed: " << ev_kpall->size() << std::endl;

    for ( auto& pkprecotype : _kpreco_track_v ) {
	    for ( auto& kpc : pkprecotype->output_pt_v ) {

        if ( kpc._cluster_type==1 )
	        _event_kpc_track_start_v.push_back( kpc );
        else if (kpc._cluster_type==2 )
          _event_kpc_track_start_v.push_back( kpc );

	    }
    } 
    
  }

  void CosmicParticleReconstruction::buildTrackFragments( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) 
  {
    // PRIMITIVE TRACK FRAGMENTS: OFF-TRIGGER TRACK HITS
    const float _maxdist = 1.0;
    const float _minsize = 10;
    const float _maxkd   = 100;

    LARCV_INFO() << "RUN PROJ-SPLITTER applied to 'offtrigger_maxtrackhit' (out-of-time hits)" << std::endl;
    larflow::reco::ProjectionDefectSplitter _projsplitter_cosmic;
    _projsplitter_cosmic.set_verbosity( logger().level() );     
    _projsplitter_cosmic.set_dbscan_pars( 5.0, _minsize, _maxkd ); // cosmic parameters, courser maxdist to reduce number of cosmic fragments
    _projsplitter_cosmic.doClusterVetoHits(false);
    _projsplitter_cosmic.set_input_larmatchhit_tree_name( "offtrigger_maxtrackhit" );
    _projsplitter_cosmic.set_fit_line_segments_to_clusters( true ); // can be slow
    _projsplitter_cosmic.set_output_tree_name("trackprojsplit_offtrigger");
    _projsplitter_cosmic.process( iolcv, ioll );

  }

  /**
   * @brief Perform cosmic ray reconstruction
   *
   * At some point, execute Mask-RCNN here
   *
   */
  void CosmicParticleReconstruction::buildCosmicTracks( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) 
  {

    LARCV_INFO() << "reco cosmic tracks" << std::endl;

    // PREP: make bad channel image
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();

    ublarcvapp::EmptyChannelAlgo _badchmaker;
    try {
    
      larcv::EventChStatus* ev_chstatus =
        (larcv::EventChStatus*)iolcv.get_data(larcv::kProductChStatus, "wire");
      // std::vector<larcv::Image2D> gapch_v =
      //   _badchmaker.makeGapChannelImage( adc_v, *ev_chstatus,
      //                                    4, 3, 2400, 6*1008, 3456, 6, 1,
      //                                    5.0, 50, -1.0 );
      std::vector<larcv::Image2D> gapch_v =
        _badchmaker.makeOverlayedBadChannelImage( adc_v, *ev_chstatus, 4, 15.0 );
    
      LARCV_INFO() << "Number of badcv images made: " << gapch_v.size() << std::endl;
      
      larcv::EventImage2D* evout_badch =
        (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"badch");
      for ( auto& gap : gapch_v ) {
        evout_badch->Emplace( std::move(gap) );
      }
    }
    catch (std::exception& e ) {
      std::stringstream msg;
      msg << "KPSRecoManager.cxx:L." << __LINE__ << " error running : makeOverlayedBadChannelImage() - "
          << '\n'
	        << e.what()
	        << std::endl;
      throw std::runtime_error(msg.str());
    }

    // filter keypoints, split into start and end
    // sort by keypoint score
    larlite::event_larflow3dhit* ev_kpall
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint_all" );

    struct KeypointInfo_t {
      float score;
      int index;
      KeypointInfo_t( float s, int idx )
      : score(s), index(idx)
      {};

      bool operator<(const KeypointInfo_t& rhs) const {
        if ( score > rhs.score ) 
          return true;
        return false;
      };
    };

    std::vector< KeypointInfo_t > kp_by_score;
    kp_by_score.reserve( ev_kpall->size() );

    larlite::event_larflow3dhit* ev_kpstart
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint_cosmic_start" );
    for ( size_t i=0; i<ev_kpall->size(); i++ ) {
      auto& pkp = ev_kpall->at(i);

      int kp_type    = (int)pkp.at(3);
      float kp_score = pkp.at(4);


      if ( kp_type==1 && kp_score>0.8 ) {
        kp_by_score.push_back( KeypointInfo_t(kp_score, (int)i) );
      }
    }

    std::sort( kp_by_score.begin(), kp_by_score.end() );
    LARCV_INFO() << "Sorted cosmic track-start keypoints: " << std::endl;
    for ( auto& info : kp_by_score ) {
      auto& pkp = ev_kpall->at( info.index );
      LARCV_INFO() << "  [" << info.index << "] score=" << info.score << " type=" << (int)pkp.at(3) << std::endl;
      ev_kpstart->push_back( pkp );
    }

    LARCV_INFO() << "Number of track-start keypoints to seed cosmic reconstruction: " << ev_kpstart->size() << std::endl;
    
    larflow::reco::CosmicTrackBuilder  _cosmic_track_builder;
    _cosmic_track_builder.clear();
    _cosmic_track_builder.set_verbosity( logger().level() );
    _cosmic_track_builder.do_boundary_analysis( false );
    _cosmic_track_builder.add_cluster_treename( "trackprojsplit_offtrigger" );
    _cosmic_track_builder.set_keypoint_treename( "keypoint_cosmic_start" );
    _cosmic_track_builder.process( iolcv, ioll );

    // filter repeats
    larlite::event_larflowcluster* ev_trackclusters
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "trackprojsplit_offtrigger" );
    std::vector< int > cluster_used_v( ev_trackclusters->size(), 0 );

    larlite::event_track* ev_cosmic_tracks
      = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, "simplecosmictrack" );
    larlite::event_larflowcluster* ev_cosmic_trackcluster
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "simplecosmictrack" );

    larlite::event_track* ev_filtered_tracks
      = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, "cosmictrack" );
    larlite::event_larflowcluster* ev_filtered_trackcluster
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "cosmictrack" );

    for ( size_t i=0; i<ev_cosmic_tracks->size(); i++ ) {
      std::vector<int> segment_idx = _cosmic_track_builder.getProposalSegmentContainerIndices( i );
      bool hasrepeat = false;
      for (auto& idx : segment_idx) {
        if ( idx >=0 && idx<(int)cluster_used_v.size() && cluster_used_v[idx]==1 ) {
          hasrepeat = true;
        }
      }
      if ( !hasrepeat ) {
        ev_filtered_tracks->push_back(       ev_cosmic_tracks->at(i) );
        ev_filtered_trackcluster->push_back( ev_cosmic_trackcluster->at(i) );

        for (auto& idx : segment_idx) {
          cluster_used_v[idx]=1;
        }

      }
    }
    ev_cosmic_tracks->clear();
    ev_cosmic_trackcluster->clear();

    //_cosmic_proton_finder.set_verbosity( larcv::msg::kDEBUG );
    // _cosmic_proton_finder.set_verbosity( logger().level() );    
    // _cosmic_proton_finder.process( iolcv, ioll );
    
  }

  /**
   * @brief Fill flash match data tree with cosmic tracks, optical flashes, and CRT information
   *
   * This method collects reconstructed cosmic tracks, optical flashes (both cosmic and beam),
   * CRT tracks, and CRT hits from the larlite storage manager and stores them in the
   * flashmatchdata_tree for downstream flash matching analysis.
   *
   * @param[in] ioll larlite storage_manager containing event data
   */
  void CosmicParticleReconstruction::fillFlashMatchData( larlite::storage_manager& ioll )
  {
    // Clear the containers first
    _flashmatchdata_track_v.clear();
    _flashmatchdata_opflash_v.clear();
    _flashmatchdata_crttrack_v.clear();
    _flashmatchdata_crthit_v.clear();
    
    // Fill cosmic tracks
    larlite::event_track* ev_cosmic_tracks = 
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "cosmictrack");
    if (ev_cosmic_tracks) {
      for (const auto& track : *ev_cosmic_tracks) {
        _flashmatchdata_track_v.push_back(track);
      }
    }
    
    // Fill optical flashes (both cosmic and beam)
    larlite::event_opflash* ev_cosmic_flashes = 
      (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash, "simpleFlashCosmic");
    if (ev_cosmic_flashes) {
      for (const auto& flash : *ev_cosmic_flashes) {
        _flashmatchdata_opflash_v.push_back(flash);
      }
    }
    
    larlite::event_opflash* ev_beam_flashes = 
      (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash, "simpleFlashBeam");
    if (ev_beam_flashes) {
      for (const auto& flash : *ev_beam_flashes) {
        _flashmatchdata_opflash_v.push_back(flash);
      }
    }
    
    // Fill CRT tracks
    larlite::event_crttrack* ev_crt_tracks = 
      (larlite::event_crttrack*)ioll.get_data(larlite::data::kCRTTrack, "crttrack");
    if (ev_crt_tracks) {
      for (const auto& crttrack : *ev_crt_tracks) {
        _flashmatchdata_crttrack_v.push_back(crttrack);
      }
    }
    
    // Fill CRT hits
    larlite::event_crthit* ev_crt_hits = 
      (larlite::event_crthit*)ioll.get_data(larlite::data::kCRTHit, "crthitcorr");
    if (ev_crt_hits) {
      for (const auto& crthit : *ev_crt_hits) {
        _flashmatchdata_crthit_v.push_back(crthit);
      }
    }
    
    // Fill the tree
    if (_flashmatchdata_tree) {
      _flashmatchdata_tree->Fill();
    }
  }

}
}
