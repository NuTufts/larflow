
#include "KPSRecoManager.h"

#include <ctime>
#include <chrono>

// larlite
#include "larlite/DataFormat/opflash.h"

#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include "SplitHitsByParticleSSNet.h"
#include "TrackFindBadConnection.h"

namespace larflow {
namespace reco {

  /** 
   * @brief constructor where an output file is made 
   *
   * @param[in] inputfile_name Name of output file for non-larcv and non-larlite reco products
   */
  KPSRecoManager::KPSRecoManager( std::string inputfile_name, int reco_ver, std::string basename )
    : larcv::larcv_base(basename),
    _spacepoint_input_container_name("larmatch"),
    _spacepoint_input_datatype("larflow3dhit"),
    _save_event_mc_info(false),    
    _ana_output_file(inputfile_name),
    _ana_tree(nullptr),
    _nuvertexmaker_tree(nullptr),
    _t_event_elapsed(0),
    _save_selected_only(false),
    _save_keypoints_in_anafile(false),
    _save_nustream_hits(false),
    _save_attachable_clusters(false),      
    _mcphoton_tree(nullptr),
    _event_mcshower_v(nullptr),
    _kMinize_outputfile_size(false),
    _reco_version(reco_ver),
    _stop_after_prepspacepoints(false),
    _stop_after_keypointreco(false),
    _stop_after_subclustering(false),
    _stop_after_nutracker(false),
    _run_perfect_mcreco(false)
  {
    make_ana_file();
    _nuvertexmaker.add_nuvertex_branch( _ana_tree );
    _ana_tree->Branch( "nu_sel_v", &_nu_sel_v );
    _ana_tree->Branch( "telapsed", &_t_event_elapsed, "telapsed/F" );
    _ana_tree->Branch( "nu_perfect_v", &_nu_perfect_v );

    _nuvertex_shower_reco.activateMCanalysisMode( true );
    
  }

  KPSRecoManager::~KPSRecoManager()
  {
  }

  /**
   * @brief process event data in larcv and larlite IO managers 
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void KPSRecoManager::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll )
  {

    if ( _reco_version!=1 && _reco_version!=2  ) {
      LARCV_ERROR() << "Did not set a proper reco version. Choices: [1,2]. Use proper constructor or call set_reco_version(int)." << std::endl;
    }
    
    std::clock_t start_event = std::clock_t();

    _nu_sel_v.clear(); ///< clear vertex selection variable container
    _nu_perfect_v.clear(); ///< clear perfect reco

    // clear storage of mcdetectable photons (might be filled by NuVertexShowerReco
    _event_mcshower_v->clear();
    // _nustream_shower_hits_v.clear();
    // _nustream_track_hits_v.clear();
    _nuvertexmaker_track_v.clear();
    _nuvertexmaker_track_pcaxis_v.clear();
    _nuvertexmaker_shower_v.clear();
    _nuvertexmaker_shower_pcaxis_v.clear();
    _event_kpc_nu_v.clear();
    _event_kpc_track_v.clear();
    _event_kpc_shower_v.clear();
    _event_kpc_cosmic_v.clear();

    _reco_status = 0;
    _t_event_elapsed = 0.0;
    _error_messages.clear();

    // PREP: make bad channel image
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();

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

      // make five particle ssnet images
      larflow::reco::SplitHitsByParticleSSNet fiveparticlealgo;
      //fiveparticlealgo.set_verbosity( larcv::msg::kDEBUG );
      fiveparticlealgo.set_verbosity( logger().level() );
      try {
	fiveparticlealgo.process( iolcv, ioll );
      }
      catch (std::exception& e ) {
	std::stringstream msg;
	msg << "KPSRecoManager.cxx:L." << __LINE__ << " error running SplitHitsByParticleSSNet fiveparticlealgo: "
	    << '\n'
	    << e.what()
	    << std::endl;
	throw std::runtime_error(msg.str());
      }

      // Set run, subrun, event indices in ana tree
      _ana_run = ev_adc->run();
      _ana_subrun = ev_adc->subrun();
      _ana_event  = ev_adc->event();
    
      // PREP SETS OF HITS
      // ------------------
      prepSpacepoints( iolcv, ioll  );
      if ( _stop_after_prepspacepoints ) {
	// early stoppage to debug (and visualize) prepared spacepoints
	_ana_tree->Fill();
	_mcphoton_tree->Fill();
	return;
      }

      // Make keypoint candidates from larmatch vertex
      // ---------------------------------------------
      recoKeypoints( iolcv, ioll );

      if ( _stop_after_keypointreco ) {
	// early stoppage to debug (and visualize) prepared keypoints
	_ana_tree->Fill();
	_mcphoton_tree->Fill();      
	return;
      }
      
      // PARTICLE FRAGMENT RECO
      clusterSubparticleFragments( iolcv, ioll );
      if ( _stop_after_subclustering ) {
	// early stopping to debug (and visualize) subclusters
	_ana_tree->Fill();
	_mcphoton_tree->Fill();      
	return;
      }
    
      // COSMIC RECO
      //cosmicTrackReco( iolcv, ioll );
    
      // MULTI-PRONG INTERNAL RECO
      multiProngReco( iolcv, ioll );
      // if ( _stop_after_nutracker ) {
      //   _ana_tree->Fill();
      //   return;
      // }

      // if ( _stop_after_prongreco ) {
      //   _ana_tree->Fill();
      //   return;      
      // }

      // kinematics
      runBasicKinematics( iolcv, ioll );

      // dqdx
      runBasicPID( iolcv, ioll );
    
      // Copy larlite contents
      // in-time opflash
      larlite::event_opflash* ev_input_opflash_beam =
	(larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"simpleFlashBeam");
      larlite::event_opflash* evout_opflash_beam =
	(larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"simpleFlashBeam");
      for ( auto const& flash : *ev_input_opflash_beam )
	evout_opflash_beam->push_back( flash );

      if ( _save_event_mc_info ) {
	_event_mcinfo_maker.process( ioll );      
      }
      if ( _save_event_mc_info && _run_perfect_mcreco ) {

	LARCV_DEBUG() << "Run perfect reco." << std::endl;
	//_perfect_reco.set_verbosity( larcv::msg::kDEBUG );
	NuVertexCandidate nuperfect = _perfect_reco.makeNuVertex( iolcv, ioll );
	_nu_perfect_v.emplace_back( std::move(nuperfect) );
	//truthAna( iolcv, ioll );
      }

      // make selection variables
      makeNuCandidateSelectionVariables( iolcv, ioll );
    

      if ( _kMinize_outputfile_size ) {
	// save only fitted vertex candidates
	_nuvertexmaker.get_mutable_nu_candidates().clear();
	_nuvertexmaker.get_mutable_vetoed_candidates().clear();
	_nuvertexmaker.get_mutable_merged_candidates().clear();            
      }

    }
    catch ( std::exception& e ) {
      std::stringstream errmsg;
      errmsg << "reco error: " << e.what() << std::endl;
      _error_messages.push_back( errmsg.str() );
      LARCV_WARNING() << "Caught Reco error: " << e.what() << std::endl;
    }
      
    // Fill Ana Tree
    _ana_run = ev_adc->run();
    _ana_subrun = ev_adc->subrun();
    _ana_event  = ev_adc->event();

    std::clock_t end_event = std::clock_t();
    _t_event_elapsed = (end_event-start_event)/CLOCKS_PER_SEC;
    LARCV_NORMAL() << "Save entry [" << _ana_run << ", " << _ana_subrun << ", " << _ana_event << "]" << std::endl;
    _ana_tree->Fill();
    _mcphoton_tree->Fill();    
    LARCV_NORMAL() << "Finished Event" << std::endl;
    return;
  }

  /**
   * @brief algorithms for splitting up and filtering larmatch space points
   *
   */
  void KPSRecoManager::prepSpacepoints( larcv::IOManager& iolcv,
                                        larlite::storage_manager& ioll )
  {

    LARCV_NORMAL() << "Select and sort spacepoints for reconstruction" << std::endl;
    LARCV_NORMAL() << " input container: " << _spacepoint_input_container_name << std::endl;
    LARCV_NORMAL() << " input datatype: " <<  _spacepoint_input_datatype << std::endl;
    
    // PREP: make sure the larmatch points have their idxhit index set
    // we can use this as a way to trace identity of hits back to original set
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _spacepoint_input_container_name );
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
    _splithits_wcfilter.set_verbosity( logger().level() );
    _splithits_wcfilter.set_larmatch_tree_name( _spacepoint_input_container_name );
    _splithits_wcfilter.process_labelonly( iolcv, ioll );   // the 2d shower score is added to larflow3dhit::renormed_shower_score. hits modified.

    // PREP: ALTER THRUMU IMAGE TO INCLUDE SSNET CLUSTERS OF A CERTAIN SIZE
    
    // PREP WC-FILTERED HITS
    // filters raw larmatch hits using wire cell thrumu tagger image
    // input:
    //  larflow3dhit_larmatch_tree: raw larmatch deploy output
    // output(s):
    //  larflow3dhit_taggerfilterhit_tree: in-time hits
    //  larflow3dhit_taggerrejecthit_tree: out-of-time/cosmic-tagged hits
    _wcfilter.set_verbosity( larcv::msg::kINFO );
    _wcfilter.set_input_larmatch_tree_name( _spacepoint_input_container_name );
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
    //_splithits_wcfilter.set_larmatch_tree_name( _spacepoint_input_container_name ); //< why by-pass cosmic removal?
    _splithits_wcfilter.set_larmatch_tree_name( "taggerfilterhit"  );
    _splithits_wcfilter.set_output_tree_stem_name( "ssnetsplit_wcfilter" );
    _splithits_wcfilter.process_splitonly( iolcv, ioll );    

    // PREP: ENFORCE UNIQUE PIXEL PREDICTION USING MAX SCORE FOR TRACK HITS
    // a method to downsample hits: for hits that land on the same plane,
    //  choose the highest score hit. Return the union of hits on all planes.
    // input:
    //  * larflow3dhit_ssnetsplit_wcfilter_trackhit_tree: in-time track hits
    // output:
    //  * larflow3dhit_maxtrackhit_wcfilter_tree: in-time track hits after filter
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_wcfilter_trackhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "maxtrackhit_wcfilter" );
    _choosemaxhit.set_verbosity( logger().level() );
    _choosemaxhit.process( iolcv, ioll );
    // input:
    //  * larflow3dhit_ssnetsplit_wcfilter_showerhit_tree: in-time shower hits
    // output:
    //  * larflow3dhit_maxshowerhit_tree: in-time shower hits after filter
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_wcfilter_showerhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "maxshowerhit" );
    _choosemaxhit.set_verbosity( logger().level() );
    _choosemaxhit.process( iolcv, ioll );

    // PREP: SPLIT SHOWER/TRACK FOR COSMIC HITS
    // input:
    //  * larflow3dhit_taggerrejecthit_tree: out-of-time hits
    // output:
    //  * larflow3dhit_ssnetsplit_full_showerhit_tree: out-of-time shower hits
    //  * larflow3dhit_ssnetsplit_full_trackhit_tree:  out-of-time track hits
    _splithits_full.set_verbosity( logger().level() );
    _splithits_full.set_larmatch_tree_name( "taggerrejecthit" );
    _splithits_full.set_output_tree_stem_name( "ssnetsplit_offtrigger" );
    _splithits_full.process_splitonly( iolcv, ioll );

    // PREP: MAX-SCORE REDUCTION ON COSMIC HITS
    // input:
    //  * larflow3dhit_ssnetsplit_full_trackhit_tree: out-of-time track hits
    // output:
    //  *  larflow3dhit_full_maxtrackhit_tree: reduced out-of-time track hits
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_offtrigger_trackhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "offtrigger_maxtrackhit" );
    _choosemaxhit.process( iolcv, ioll );

    // if we're stopping at this stage for debugging/plotting,
    // we force the saving of all the intermediate hit containers
    if ( _stop_after_prepspacepoints ) {
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, _spacepoint_input_container_name ); ///< save all original hits (now ssnet-labeled)
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "maxtrackhit_wcfilter" ); ///< final set of in-time track hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "maxshowerhit" );         ///< final set of in-time shower hits      
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "offtrigger_maxtrackhit" );     ///< final set of out-of-time track hits

      // intermediate hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_offtrigger_trackhit" );      //< pre-max out-of-time track hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_offtrigger_showerhit" );     //< pre-max out-of-time shower hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_wcfilter_trackhit" );  //< pre-max in-time track hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_wcfilter_showerhit" ); //< pre-max in-time track hits
      
    }

    if ( _save_nustream_hits ) {
      // add to branch (to do)
    }
    
  }
  
  /**
   * @brief make keypoints for use to help make particle track and nu interaction candidates
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void KPSRecoManager::recoKeypoints( larcv::IOManager& iolcv,
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
    _event_kpc_nu_v.clear();
    _event_kpc_track_v.clear();
    _event_kpc_shower_v.clear();
    _event_kpc_cosmic_v.clear();      
    
    // neutrino
    _kpreco_nu.clear_output();
    _kpreco_nu.set_num_passes(1);
    _kpreco_nu.set_verbosity( logger().level() );
    _kpreco_nu.set_input_larmatch_tree_name( "larmatch" ); // previous: taggerfilterhit
    _kpreco_nu.set_sigma( 10.0 );
    _kpreco_nu.set_max_dbscan_dist( 0.7 );
    _kpreco_nu.set_larmatch_threshold( 0.5 );      
    _kpreco_nu.set_min_cluster_size(   10, 0 );
    _kpreco_nu.set_keypoint_threshold( 0.2, 0 );
    _kpreco_nu.set_output_tree_name( "keypoint_all" );
    _kpreco_nu.set_keypoint_type( (int)larflow::kNuVertex );
    _kpreco_nu.set_lfhit_score_index( 17 ); // (v2 larmatch-minkowski network neutrino-score index in hit)
    _kpreco_nu.process( ioll );
      
    // neutrino interaction track: we have track starts and ends
    std::vector< larflow::reco::KeypointReco* > _kpreco_track_v
      = { &_kpreco_trackstart, &_kpreco_trackend };
    for ( auto& pkpreco_track : _kpreco_track_v ) {	
      pkpreco_track->clear_output();
      pkpreco_track->set_verbosity( logger().level() );
      pkpreco_track->set_num_passes(1);
      pkpreco_track->set_input_larmatch_tree_name( "larmatch" ); // previous: taggerfilterhit
      pkpreco_track->set_sigma( 10.0 );
      pkpreco_track->set_max_dbscan_dist( 0.7 );
      pkpreco_track->set_larmatch_threshold( 0.5 );      
      pkpreco_track->set_min_cluster_size(   10, 0 );
      pkpreco_track->set_keypoint_threshold( 0.2, 0 );
      pkpreco_track->set_output_tree_name( "keypoint_all" );
    }
	
    // neutrino interaction track start
    _kpreco_trackstart.clear_output(); // clears kpdata containers      
    _kpreco_trackstart.set_keypoint_type( (int)larflow::kTrackStart );
    _kpreco_trackstart.set_lfhit_score_index( 18 ); // (v2 larmatch-minkowski network track-start-score index in hit)
    _kpreco_trackstart.process( ioll );
    // neutrino interaction track end
    _kpreco_trackend.clear_output(); // clears kpdata containers
    _kpreco_trackend.set_keypoint_type( (int)larflow::kTrackEnd );
    _kpreco_trackend.set_lfhit_score_index( 19 ); // (v2 larmatch-minkowski network track-end-score index in hit)
    _kpreco_trackend.process( ioll );
      
    // neutrino interaction shower
    std::vector< larflow::reco::KeypointReco* > _kpreco_shower_v
      = { &_kpreco_shower,
	  &_kpreco_michel,
	  &_kpreco_deltas };
    for ( auto& pkpreco : _kpreco_shower_v )  {								      
      pkpreco->clear_output();
      pkpreco->set_verbosity( logger().level() );
      pkpreco->set_input_larmatch_tree_name( "larmatch" ); // previous: taggerfilterhit
      pkpreco->set_sigma( 10.0 );
      pkpreco->set_larmatch_threshold( 0.5 );
      pkpreco->set_min_cluster_size(   10, 0 );
      pkpreco->set_keypoint_threshold( 0.2, 0 );
      pkpreco->set_output_tree_name( "keypoint_all" );
    }
    _kpreco_shower.set_keypoint_type( (int)larflow::kShowerStart );
    _kpreco_shower.set_lfhit_score_index( 20 ); // (v2 larmatch-minkowski network nu-shower-score index in hit)
    _kpreco_shower.process( ioll );
    // neutrino+cosmic interaction michel
    _kpreco_michel.set_keypoint_type( (int)larflow::kShowerMichel );
    _kpreco_michel.set_lfhit_score_index( 21 ); // (v2 larmatch-minkowski network michel-shower-score index in hit)
    _kpreco_michel.process( ioll );
    // neutrino+cosmic interaction delta
    _kpreco_deltas.set_keypoint_type( (int)larflow::kShowerDelta );
    _kpreco_deltas.set_lfhit_score_index( 22 ); // (v2 larmatch-minkowski network delta-shower-score index in hit)
    _kpreco_deltas.process( ioll );


    
    // filter out keypoints by in-time and cosmic
    larlite::event_larflow3dhit* ev_kpintime = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint" );
    larlite::event_pcaxis* ev_kp_pca = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "keypoint" );    
    larlite::event_larflow3dhit* ev_kpcosmic = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypointcosmic" );
    larlite::event_pcaxis* ev_kp_pca_cosmic = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "keypointcosmic" );

    larcv::EventImage2D* ev_image2d_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "thrumu" );
    int nplanes = ev_image2d_v->as_vector().size();

    std::vector< larflow::reco::KeypointReco* > kpreco_v
      = { &_kpreco_nu,
	        &_kpreco_trackstart,
	        &_kpreco_trackend,
	        &_kpreco_shower, // showers go another route
	        &_kpreco_michel,
	        &_kpreco_deltas };
    
    // loop over algos for each keypoint class
    int intime_cluster_index = 0;
    int cosmic_cluster_index = 0;
    for ( auto& pkpreco : kpreco_v ) {
      // loop over reco keypoints
      for ( auto const& kpc : pkpreco->output_pt_v ) {
	      // cut on max value keypoint score
	      if ( kpc.max_score < 0.5 ) // 0.7 too strong?
	        continue;
	      
	      // get if near a cosmic-tagged pixel
	      float thrumu_pixsum_allplanes = 0.;
	      std::vector<float> thrumu_pixsum(nplanes,0);
	      for (int p=0; p<3; p++) {
	        thrumu_pixsum[p] = _pt_image_projection.getPixelSumAroundProjPoint( kpc.max_pt_v, ev_image2d_v->as_vector().at(p), 2, 10.0 );
	        thrumu_pixsum_allplanes += thrumu_pixsum[p];
	      }
      
	      /// make larflow3dhit version and add thrumu projection info.
	      larlite::larflow3dhit kphit = kpc.as_larflow_hit();
	      kphit.push_back( thrumu_pixsum_allplanes );	
	      for (int p=0; p<3; p++)
	        kphit.push_back( thrumu_pixsum[p] );
      
	      if ( thrumu_pixsum_allplanes < 50.0 ) {
	        // then ok to pass on as potential nu candidate
	        ev_kpintime->push_back( kphit );
	        ev_kp_pca->push_back( kpc.get_pcaxis( intime_cluster_index ) );
	        intime_cluster_index++;
	      }
	      else {
	        // assign as comics
	        ev_kpcosmic->push_back( kphit );
	        ev_kp_pca_cosmic->push_back( kpc.get_pcaxis( cosmic_cluster_index ) );
	        cosmic_cluster_index++;
	      }
      }
    }

    // filter duplicates for intime
    std::vector<int> intime_kp_status( ev_kpintime->size(), 1 );
    
    for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {

      auto const& hit = ev_kpintime->at(ikp);
      int kp_type = int(hit[3]);
      
      // recursive check with those before
      for (int jkp=0; jkp<ikp; jkp++) {
	      if ( intime_kp_status[jkp]==0 ) {
	        // already filtered. skip.
	        continue;
	      }
	      auto const& past_hit = ev_kpintime->at(jkp);
	      int past_type = int(past_hit[3]);
	      
	      // if the same type, don't do the duplicate removal test
	      if ( kp_type==past_type ) {
	        continue;
	      }
      
	      float dist = 0.;
	      float dx = 0.;
	      for (int i=0; i<3; i++) {
	        dx = (past_hit[i]-hit[i]);
	        dist += dx*dx;
	      }
	      dist = sqrt(dist);
      
	      if ( dist>3.0 ) {
	        // no overlap
	        continue;
	      }
      
	      if ( past_type==0 && kp_type!=0 ) {
	        // past type is nu vertex. we de-activate in favor of that vertex
	        intime_kp_status[ikp] = 0;
	        break;
	      }
	      else if ( kp_type==0 && past_type!=0 ) {
	        // current keypoint is nu-type. deactivate past vertex
	        intime_kp_status[jkp] = 0;
	        // keep going
	      }
	      else if ( (kp_type==1 && past_type==2 )
	      	  || (kp_type==2 && past_type==1 ) ) {
	        // comparison between track start and track end
	        // if we're really close, then go with start label. will use to seed neutrino.
	        if ( dist<0.7 ) {
	          if ( kp_type==2 ) {
	            intime_kp_status[ikp] = 0;
	            break; // current kp has been deactivated. stop.
	          }
	          else if (past_type==2) {
	            intime_kp_status[jkp] = 0;
	            // keep going
	          }
	        }
	      }
	      else if ( (kp_type==3 && (past_type==1 || past_type==2))
	      	  || (past_type==3 && (kp_type==1 || kp_type==2)) ) {
          // shower keypoints override and remove track end and track start keypoints
	        if ( dist<0.7 ) {
	          if ( kp_type!=3 ) {
	            intime_kp_status[ikp] = 0;
	            break; // current kp has been deactivated. stop.
	          }
	          else if ( past_type!=3 ) {
	            intime_kp_status[jkp] = 0;
	            // keep-going
	          }
	        }
	      }//end of case overlap loop
      }
    }
    
    //std::vector<int> intime_kp_status( ev_kpintime->size(), 1 );
    int num_deactivated = 0;
    for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {
      if ( intime_kp_status[ikp]==0 ) {
	      num_deactivated++;
	      break;
      }
    }

    if ( num_deactivated>0 ) {
      
      std::vector< larlite::larflow3dhit > passing_keypoints;
      std::vector< larlite::pcaxis > passing_pcaxis;
      for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {
	       if ( intime_kp_status[ikp]==1 ) {
	         passing_keypoints.push_back( ev_kpintime->at(ikp) );
	         passing_pcaxis.push_back( ev_kp_pca->at(ikp) );
	       }
      }

      ev_kpintime->clear();
      ev_kp_pca->clear();
      for (int ikp=0; ikp<(int)passing_keypoints.size(); ikp++) {
	      ev_kpintime->push_back( passing_keypoints.at(ikp) );
	      ev_kp_pca->push_back( passing_pcaxis.at(ikp) );
      }
      LARCV_NORMAL() << "After cross-type duplicate filter. Number of intime keypoints: " << ev_kpintime->size() << std::endl;
    }
    
    // cosmic keypoints
    // _kpreco_track_cosmic.clear_output();
    // _kpreco_track_cosmic.set_input_larmatch_tree_name( "taggerrejecthit" );
    // _kpreco_track_cosmic.set_output_tree_name( "keypointcosmic" );
    // _kpreco_track_cosmic.set_sigma( 50.0 );    
    // _kpreco_track_cosmic.set_min_cluster_size(   50.0, 0 );
    // _kpreco_track_cosmic.set_max_dbscan_dist( 10.0 );
    // _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 0 );
    // _kpreco_track_cosmic.set_min_cluster_size(   20.0, 1 );    
    // _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 1 );    
    // _kpreco_track_cosmic.set_larmatch_threshold( 0.5 );

    // _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackStart );
    // _kpreco_track_cosmic.set_lfhit_score_index( 18 ); // (v2 larmatch network track-start-score index in hit)
    // _kpreco_track_cosmic.process( ioll );
    
    // _kpreco_track_cosmic.clear_output();
    // _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackEnd );
    // _kpreco_track_cosmic.set_lfhit_score_index( 19 ); // (v2 larmatch network track-end-score index in hit)
    // _kpreco_track_cosmic.process( ioll );
      

    LARCV_NORMAL() << "Num nu vertex candidates [keypoint]: " << ev_kpintime->size() << std::endl;
    LARCV_NORMAL() << "Num cosmic vertex candidates [keypoint]: " << ev_kpcosmic->size() << std::endl;
    for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {
      auto const& kphit = ev_kpintime->at(ikp);
      int kptype = -1;
      float maxkpscore = -1;
      float thrumupixsum = -1;
      if ( kphit.size()>=4 ) {
        kptype = kphit.at(3);
	      maxkpscore = kphit.at(4);
	      thrumupixsum = kphit.at(5);
      }
      LARCV_NORMAL() << " [" << ikp << "] type=" << kptype << " maxscore=" << maxkpscore << " cosmicpixsum=" << thrumupixsum << std::endl;
    }
    
    if ( _save_keypoints_in_anafile ) {
      for ( auto& pkprecotype : kpreco_v ) {
	      for ( auto& kpc : pkprecotype->output_pt_v ) {
	        if ( kpc._cluster_type==0 )
	          _event_kpc_nu_v.push_back( kpc );
	        else if ( kpc._cluster_type==1 || kpc._cluster_type==2 )
	          _event_kpc_track_v.push_back( kpc );
	        else if ( kpc._cluster_type>=3 )
	          _event_kpc_shower_v.push_back( kpc );
	      }
        // for ( auto& kpc : _kpreco_track_cosmic.output_pt_v  )
        //_event_kpc_cosmic_v.push_back( kpc );
      }
    }
    
  }

  /**
   * @brief form sub-particle clusters
   *
   * Form the subclusters we will piece back together to form track and shower clusters.
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void KPSRecoManager::clusterSubparticleFragments( larcv::IOManager& iolcv,
                                                    larlite::storage_manager& ioll )
  {

    
    // TRACK 2-KP RECO: make tracks using pairs of keypoints
    // input:
    // * larflow3dhit_trackhit_tree: track hits from  SplitLArMatchHitsBySSNet
    // output:
    // * track_track2kp_tree: output tracks
    // * larflow3dhit_keypoint_tree: copy of hits passed into algorithm
    // _tracker2kp.set_verbosity( larcv::msg::kDEBUG );
    // _tracker2kp.set_larflow3dhit_tree_name( "trackhit" );
    // _tracker2kp.set_keypoint_tree_name( "keypoint_bigcluster" );
    // _tracker2kp.process( iolcv, ioll );
    
    // TRACK PCA-CLUSTER: act on remaining clusters
    //_pcacluster.set_input_larmatchhit_tree_name( "track2kpunused" );
    //_pcacluster.set_input_larmatchhit_tree_name( "trackhit" );
    //_pcacluster.process( iolcv, ioll );

    // PRIMITIVE TRACK FRAGMENTS: WC-FILTER
    const float _maxdist = 1.0;
    const float _minsize = 10;
    const float _maxkd   = 100;
    LARCV_INFO() << "RUN PROJ-SPLITTER ON: maxtrackhit_wcfilter (in-time track hits)" << std::endl;
    //_projsplitter.set_verbosity( larcv::msg::kDEBUG );
    _projsplitter.set_verbosity( logger().level() );    
    _projsplitter.set_dbscan_pars( _maxdist, _minsize, _maxkd );
    _projsplitter.doClusterVetoHits(false);
    _projsplitter.set_fit_line_segments_to_clusters( true );
    _projsplitter.set_input_larmatchhit_tree_name( "maxtrackhit_wcfilter" );
    //output of wctagger: taggerfilterhit
    //_projsplitter.set_input_larmatchhit_tree_name( "ssnetsplit_wcfilter_trackhit" );    
    _projsplitter.add_input_keypoint_treename_for_hitveto( "keypoint" );
    _projsplitter.set_output_tree_name("trackprojsplit_wcfilter");
    _projsplitter.set_output_kpvetoed_tree_name( "projsplitvetoed" );
    _projsplitter.process( iolcv, ioll );


    // PRIMITIVE TRACK FRAGMENTS: OFF-TRIGGER TRACK HITS
    LARCV_INFO() << "RUN PROJ-SPLITTER ON: offtrigger_maxtrackhit (out-of-time hits)" << std::endl;
    _projsplitter_cosmic.set_verbosity( logger().level() );    
    _projsplitter_cosmic.set_verbosity( larcv::msg::kINFO );
    //_projsplitter_cosmic.set_verbosity( larcv::msg::kDEBUG );    
    _projsplitter_cosmic.set_dbscan_pars( 5.0, _minsize, _maxkd ); // cosmic parameters, courser maxdist to reduce number of cosmic fragments
    _projsplitter_cosmic.doClusterVetoHits(false);
    _projsplitter_cosmic.set_input_larmatchhit_tree_name( "offtrigger_maxtrackhit" );
    _projsplitter_cosmic.set_fit_line_segments_to_clusters( true ); // can be slow
    _projsplitter_cosmic.set_output_tree_name("trackprojsplit_offtrigger");
    _projsplitter_cosmic.process( iolcv, ioll );

    // SHOWER 1-KP RECO: make shower using clusters and single keypoint
    // class: larflow::reco::ShowerRecoKeypoint
    _showerkp.setShowerRadiusThresholdcm( 5.0 );
    _showerkp.set_ssnet_lfhit_tree_name( "maxshowerhit" );
    //_showerkp.set_ssnet_lfhit_tree_name( "ssnetsplit_wcfilter_showerhit" );    
    //_showerkp.set_verbosity( larcv::msg::kDEBUG );
    _showerkp.set_verbosity( logger().level() );    
    _showerkp.process( iolcv, ioll ); // output are larflowclusters+trunk-pcaxis+keypoint in showerkp

    // SHORT HIP FRAGMENTS
    //_short_proton_reco.set_verbosity( larcv::msg::kDEBUG );
    _short_proton_reco.set_verbosity( logger().level() );    
    _short_proton_reco.clear_clustertree_checklist();
    _short_proton_reco.add_clustertree_forcheck( "trackprojsplit_wcfilter" );
    _short_proton_reco.process( iolcv, ioll );
    
    // TRACK CLUSTER-ONLY RECO: make tracks without use of keypoints

    // SHOWER CLUSTER-ONLY RECO: make showers without use of keypoints

    if ( _stop_after_subclustering ) {
      // we're going to stop here. save key intermediate products from this stage.
      ioll.set_data_to_write( larlite::data::kLArFlowCluster, "trackprojsplit_wcfilter" ); // in-time track clusters
      ioll.set_data_to_write( larlite::data::kPCAxis, "trackprojsplit_wcfilter" );         // in-time track clusters
      
      ioll.set_data_to_write( larlite::data::kLArFlowCluster, "trackprojsplit_offtrigger" ); // out-of-time track clusters
      ioll.set_data_to_write( larlite::data::kPCAxis, "trackprojsplit_offtrigger" );         // out-of-time track clusters
      
      ioll.set_data_to_write( larlite::data::kLArFlowCluster, "showerkp" ); // shower in-time clusters
      ioll.set_data_to_write( larlite::data::kPCAxis, "showerkp" );         // shower in-time clusters

      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "projsplitnoise" ); // unused hits in cluster splitter
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "projsplitvetoed" ); // unused hits in cluster splitter      
    }
  }

  /**
   * @brief reconstruct tracks and showers attached to vertices
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */  
  void KPSRecoManager::multiProngReco( larcv::IOManager& iolcv,
                                       larlite::storage_manager& ioll )
  {


    _nuvertexactivity.set_verbosity( logger().level() );
    //_nuvertexactivity.set_verbosity( larcv::msg::kDEBUG );    

    // configure to use shower and in-time hits
    std::vector<std::string> input_cluster_track_list
      = { "trackprojsplit_wcfilter" }; // in-time track clusters
    std::vector<std::string> input_cluster_shower_list
      = { "showergoodhit" }; // in-time track clusters
    std::vector<std::string> input_cluster_showerkp_list
      = { "showerkp" }; // showers made with keypoint clustering

    // std::vector<std::string> input_hit_list
    //   = {"taggerfilterhit",            // all in-time hits
    //      "ssnetsplit_offtrigger_showerhit"}; // out-of-time shower hits    
    //_nuvertexactivity.set_input_hit_list( input_hit_list );    
    //_nuvertexactivity.set_input_cluster_list( input_cluster_list );
    //_nuvertexactivity.set_output_treename( "keypoint" );
    //_nuvertexactivity.process( iolcv, ioll );

    larlite::event_larflow3dhit* ev_keypoint
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "keypoint");
    larlite::event_larflow3dhit* ev_keypoint_showerkp
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "showerkp");
    larlite::event_larflow3dhit* ev_keypoint_nuvtxseed
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "keypoint_nuvtxseed");
    
    for (auto& kp : *ev_keypoint ) {
      if ( kp.at(3)>=0 && kp.at(3)<=1 ) {
        ev_keypoint_nuvtxseed->push_back( kp );
      }
    }
    for (auto& kp : *ev_keypoint_showerkp ) {
      ev_keypoint_nuvtxseed->push_back(kp);
    }

    //_nuvertexmaker.set_verbosity( larcv::msg::kDEBUG );
    //_nuvertexmaker.set_verbosity( larcv::msg::kINFO );        
    _nuvertexmaker.set_verbosity( logger().level() );
    _nuvertexmaker.clear();
    _nuvertexmaker.add_keypoint_producer( "keypoint_nuvtxseed" );
    for ( auto& name : input_cluster_track_list ) {
      _nuvertexmaker.add_cluster_producer( name, NuVertexCandidate::kTrack);
    } 
    for ( auto& name : input_cluster_showerkp_list )
      _nuvertexmaker.add_cluster_producer(name, NuVertexCandidate::kShowerKP );
    for ( auto& name : input_cluster_shower_list )
      _nuvertexmaker.add_cluster_producer(name, NuVertexCandidate::kShower );
    //_nuvertexmaker.add_cluster_producer("cosmicproton", NuVertexCandidate::kTrack );
    ////_nuvertexmaker.add_cluster_producer("hip", NuVertexCandidate::kTrack ); 
    
    _nuvertexmaker.apply_cosmic_veto( true );
    _nuvertexmaker.setOutputStage( larflow::reco::NuVertexMaker::kVetoed );    
    _nuvertexmaker.process( iolcv, ioll );

    // if ( _save_attachable_clusters ) {
    //   LARCV_NORMAL() << "Saving attachable clusters for debug" << std::endl;
    //   for (auto& name : input_cluster_track_list ) {
    // 	auto ev_cluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, name );
    // 	auto ev_pcaxis  = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, name);
    // 	int nclusters = ev_cluster->size();
    // 	for (int ic=0; ic<nclusters; ic++) {
    // 	  _nuvertexmaker_track_v.push_back( ev_cluster->at(ic) );
    // 	  _nuvertexmaker_track_pcaxis_v.push_back( ev_pcaxis->at(ic) );
    // 	}
    //   }
    //   for (auto& name : input_cluster_shower_list ) {
    // 	auto ev_cluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, name );
    // 	auto ev_pcaxis  = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, name);
    // 	int nclusters = ev_cluster->size();
    // 	for (int ic=0; ic<nclusters; ic++) {
    // 	  _nuvertexmaker_shower_v.push_back( ev_cluster->at(ic) );
    // 	  _nuvertexmaker_shower_pcaxis_v.push_back( ev_pcaxis->at(ic) );
    // 	}
    //   }
    // }

    LARCV_NORMAL() << "Cluster-book summary after [NuVertexMaker]" << std::endl;
    for (int ivtx=0; ivtx<(int)_nuvertexmaker.get_mutable_output_candidates().size(); ivtx++) {
      auto const& nucand = _nuvertexmaker.get_mutable_output_candidates().at(ivtx);
      LARCV_NORMAL() << "vertex[" << ivtx << ", kptype=" << nucand.keypoint_type << "] nclusters used: "
		     << _nuvertexmaker.get_candidate_cluster_book().at(ivtx).numUsed()
		     << std::endl;
      LARCV_NORMAL() << "  pos: (" << nucand.pos[0] << ", " << nucand.pos[1] << ", " << nucand.pos[2] << ")" << std::endl;
    }

    
    // NuTrackBuilder class
    LARCV_NORMAL() << "Build out track prongs" << std::endl;
    _nu_track_builder.clear();
    if ( _stop_after_nutracker )
      _nu_track_builder.set_verbosity( logger().level() );    
    else 
      _nu_track_builder.set_verbosity( logger().level() );
    _nu_track_builder.process( iolcv, ioll,
			       _nuvertexmaker.get_mutable_output_candidates(),
			       _nuvertexmaker.get_candidate_cluster_book() );

    // compress track representation
    larflow::recoutils::CompressRecoTrack track_compressor;
    float max_saggita=0.3;
    float max_step_size=5.0;
    for ( auto& nuvtx : _nuvertexmaker.get_mutable_output_candidates() ) {
      for ( int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++ ) {
	      auto& track = nuvtx.track_v.at(itrack);
	      if ( track.NumberTrajectoryPoints() > 50 ) {
	        larlite::track compressed = track_compressor.compress( track, max_saggita, max_step_size );
	        LARCV_INFO() << "nuvtx:track[" << itrack << "] "
	      	       << "compressed npts=" << track.NumberTrajectoryPoints() << " --> "
	      	       << "npts=" << compressed.NumberTrajectoryPoints()
	      	       << std::endl;
	        if ( compressed.NumberTrajectoryPoints()<track.NumberTrajectoryPoints() ){
	          std::swap(track,compressed); // danger?
	        }
	      }
      }
    }

    LARCV_NORMAL() << "Cluster-book summary after [NuTrackBuilder]" << std::endl;    
    for (int ivtx=0; ivtx<(int)_nuvertexmaker.get_mutable_output_candidates().size(); ivtx++) {
      auto const& nucand = _nuvertexmaker.get_mutable_output_candidates().at(ivtx);
      LARCV_NORMAL() << "vertex[" << ivtx << ", kptype=" << nucand.keypoint_type << "] nclusters used: "
		     << _nuvertexmaker.get_candidate_cluster_book().at(ivtx).numUsed()
		     << std::endl;
      LARCV_NORMAL() << "  pos: (" << nucand.pos[0] << ", " << nucand.pos[1] << ", " << nucand.pos[2] << ")" << std::endl;      
    }
    
    // larflow::reco::TrackFindBadConnection track_splitter;
    // track_splitter.set_verbosity( larcv::msg::kINFO );
    // for (auto& nuvtx : _nuvertexmaker.get_mutable_fitted_candidates() )
    //   int nsplit = track_splitter.processNuVertexTracks( nuvtx, iolcv );
    if ( _stop_after_nutracker ) {
      _nu_track_builder.saveConnections( ioll, "tcb_connections" );
      ioll.set_data_to_write( larlite::data::kTrack, "tcb_connections" );
    }

    // first attempt
    // _nu_shower_builder.set_verbosity( larcv::msg::kDEBUG );
    // _nu_shower_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    // simpler, cone-based reco
    //_nuvertex_shower_reco.set_verbosity( larcv::msg::kDEBUG );
    _nuvertex_shower_reco.set_verbosity( logger().level() );    
    //_nuvertex_shower_reco.activateMCanalysisMode(true);
    _nuvertex_shower_reco.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    _nuvertex_shower_reco.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    _nuvertex_shower_reco.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );    
    //_nuvertex_shower_reco.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );
    _nuvertex_shower_reco.process( iolcv, ioll,
				   _nuvertexmaker.get_mutable_output_candidates(),
				   _nuvertexmaker.get_candidate_cluster_book() );

    if ( _nuvertex_shower_reco.isMCanaModeActive() ) {
      _nuvertex_shower_reco.save_detectable_photon_info( ioll );
      // transfer the detectable photon info into _mcphoton_tree via the _event_mcphoton_v container
      larlite::event_mcshower* ev_detshower
        = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcdetectableshower" );
      for (size_t ishower=0; ishower<ev_detshower->size(); ishower++) {
	      _event_mcshower_v->push_back( ev_detshower->at(ishower) );
      }
      LARCV_NORMAL() << "Saved " << _event_mcshower_v->size() << " MC detectable photon information" << std::endl;
    }

    LARCV_NORMAL() << "Cluster-book summary after [NuVertexShowerReco]" << std::endl;
    for (int ivtx=0; ivtx<(int)_nuvertexmaker.get_mutable_output_candidates().size(); ivtx++) {
      auto const& nucand = _nuvertexmaker.get_mutable_output_candidates().at(ivtx);
      LARCV_NORMAL() << "vertex[" << ivtx << ", kptype=" << nucand.keypoint_type << "] nclusters used: "
		     << _nuvertexmaker.get_candidate_cluster_book().at(ivtx).numUsed()
		     << std::endl;
      LARCV_NORMAL() << "  pos: (" << nucand.pos[0] << ", " << nucand.pos[1] << ", " << nucand.pos[2] << ")" << std::endl;      
    }    

    // // - repair shower trunks by absorbing tracks or creating hits
    // //_nuvertex_shower_trunk_check.set_verbosity( larcv::msg::kDEBUG );
    // _nuvertex_shower_trunk_check.set_verbosity( logger().level() );
    // int ivtx = 0;
    // //for ( auto& vtx : _nuvertexmaker.get_mutable_fitted_candidates() ) {
    // for ( auto& vtx : _nuvertexmaker.get_mutable_output_candidates() ) {
    //   LARCV_DEBUG() << "Run shower trunk check on vertex candidate [" << ivtx << "]" << std::endl;
    //   _nuvertex_shower_trunk_check.checkNuCandidateProngs( vtx );
    //   //_nuvertex_shower_trunk_check.checkNuCandidateProngsForMissingCharge( vtx, iolcv, ioll );
    //   ivtx++;
    // }

    // post-neutrino-candidate processing:
    // - remove tracks from neutrino candidates that significantly overlap with showers
    _nuvertex_postcheck_showertrunkoverlap.set_verbosity( larcv::msg::kDEBUG );
    //_nuvertex_postcheck_showertrunkoverlap.process( _nuvertexmaker.get_mutable_fitted_candidates() );
    //_nuvertex_postcheck_showertrunkoverlap.set_verbosity( logger().level() );
    _nuvertex_postcheck_showertrunkoverlap.process( _nuvertexmaker.get_mutable_output_candidates() );

    // // - repair shower trunks again by absorbing tracks or creating hits for new near-vertex tracks
    // ivtx = 0;
    // //for ( auto& vtx : _nuvertexmaker.get_mutable_fitted_candidates() ) {
    // _nuvertex_shower_trunk_check.set_verbosity( logger().level() );
    // for ( auto& vtx : _nuvertexmaker.get_mutable_output_candidates() ) {
    //   LARCV_DEBUG() << "Run shower trunk check on vertex candidate [" << ivtx << "]" << std::endl;
    //   _nuvertex_shower_trunk_check.checkNuCandidateProngs( vtx );
    //   //_nuvertex_shower_trunk_check.checkNuCandidateProngsForMissingCharge( vtx, iolcv, ioll );
    //   ivtx++;
    // }

    // - add secondaries
    _nuvertex_add_secondaries.set_verbosity( logger().level() );
    //_nuvertex_add_secondaries.set_verbosity( larcv::msg::kDEBUG );
    LARCV_NORMAL() << "ADDING SECONDARIES" << std::endl;
    _nuvertex_add_secondaries.init_trackbuilder_for_event( iolcv, ioll );
    for ( size_t ivtx=0; ivtx<_nuvertexmaker.get_mutable_output_candidates().size(); ivtx++ ) {
      LARCV_NORMAL() << "Try to add secondaries to VTX[" << ivtx << "]" << std::endl;
      auto& nuvtx = _nuvertexmaker.get_mutable_output_candidates().at(ivtx);
      auto& book  = _nuvertexmaker.get_candidate_cluster_book().at(ivtx);
      _nuvertex_add_secondaries.process( nuvtx, book, iolcv, ioll );
    }    


    _nuvertex_restore_kphits.set_verbosity( logger().level() );
    _nuvertex_restore_kphits.process( _nuvertexmaker.get_mutable_output_candidates(), ioll, iolcv );
    
    // - add dq/dx information
    //_nuvertex_trackdqdx.set_verbosity( larcv::msg::kDEBUG );
    /*
    LARCV_NORMAL() << "calculate Track dQ/dx" << std::endl;
    for ( auto& vtx : _nuvertexmaker.get_mutable_output_candidates() ) {        
      _nuvertex_trackdqdx.process_nuvertex_tracks( iolcv, vtx );
    }
    */
    
    //_cosmic_vertex_builder.set_verbosity( larcv::msg::kDEBUG );
    //_cosmic_vertex_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    
    
  }

  /**
   * @brief Perform cosmic ray reconstruction
   *
   * At some point, execute Mask-RCNN here
   *
   */
  void KPSRecoManager::cosmicTrackReco( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    LARCV_INFO() << "reco cosmic tracks" << std::endl;
    
    _cosmic_track_builder.clear();
    //_cosmic_track_builder.set_verbosity( larcv::msg::kDEBUG );
    _cosmic_track_builder.set_verbosity( logger().level() );    
    _cosmic_track_builder.do_boundary_analysis( true );
    _cosmic_track_builder.process( iolcv, ioll );

    //_cosmic_proton_finder.set_verbosity( larcv::msg::kDEBUG );
    _cosmic_proton_finder.set_verbosity( logger().level() );    
    _cosmic_proton_finder.process( iolcv, ioll );
    
  }
  

  /**
   * @brief create ana file and define output tree
   *
   * The tree created is `KPSRecoManagerTree`.
   *
   */
  void KPSRecoManager::make_ana_file()
  {

    if ( _ana_output_file=="" ) {
      LARCV_ERROR() << "Did not specify an output file." << std::endl;
    }
    
    
    LARCV_NORMAL() << "Create Ana Output File: " << _ana_output_file << std::endl;
    
    _ana_file = new TFile(_ana_output_file.c_str(), "recreate");
    _ana_tree = new TTree("KPSRecoManagerTree","Ana Output of KPSRecoManager algorithms");

    // event book-keeping indicies: run, subrun, event
    _ana_tree->Branch("run",&_ana_run,"run/I");
    _ana_tree->Branch("subrun",&_ana_subrun,"subrun/I");
    _ana_tree->Branch("event",&_ana_event,"event/I");
    _ana_tree->Branch("reco_status", &_reco_status, "reco_status/I");
    _ana_tree->Branch("error_messages", &_error_messages);
    _ana_tree->Branch("elapsed_time", &_t_event_elapsed, "elapsed_time/F");

    _event_kpc_nu_v.clear();
    _event_kpc_track_v.clear();
    _event_kpc_shower_v.clear();
    _event_kpc_cosmic_v.clear();    
    _ana_tree->Branch( "kpc_nu_v",     &_event_kpc_nu_v );
    _ana_tree->Branch( "kpc_track_v",  &_event_kpc_track_v );
    _ana_tree->Branch( "kpc_shower_v", &_event_kpc_shower_v );
    _ana_tree->Branch( "kpc_cosmic_v", &_event_kpc_cosmic_v );
    
    // _nustream_shower_hits_v.clear();
    // _nustream_track_hits_v.clear();
    // _ana_tree->Branch( "nustream_shower_hits", &_nustream_shower_hits_v );
    // _ana_tree->Branch( "nustream_track_hits",  &_nustream_track_hits_v );

    // _nuvertexmaker_tree = new TTree("kps_nuvertex_tree","store nuvertexmaker clusters for debug");
    // _nuvertexmaker_track_v.clear();
    // _nuvertexmaker_shower_v.clear();
    // _nuvertexmaker_track_pcaxis_v.clear();
    // _nuvertexmaker_shower_pcaxis_v.clear();
    // _nuvertexmaker_tree->Branch( "nuvertexmaker_track", &_nuvertexmaker_track_v );
    // _nuvertexmaker_tree->Branch( "nuvertexmaker_shower", &_nuvertexmaker_shower_v );    
    // _nuvertexmaker_tree->Branch( "nuvertexmaker_track_pcaxis", &_nuvertexmaker_track_pcaxis_v );
    // _nuvertexmaker_tree->Branch( "nuvertexmaker_shower_pcaxis", &_nuvertexmaker_shower_pcaxis_v );    
    
    _nuvertex_shower_reco.createMCAnalysisTree( _ana_file );

    _mcphoton_tree = new TTree("kps_mcphoton_tree","store modified mcshower objects");
    _event_mcshower_v = new std::vector< larlite::mcshower >;
    _event_mcshower_v->clear();
    _mcphoton_tree->Branch( "mcshower_v", &_event_mcshower_v );

  }

  /**
   * @brief Close ana file
   *
   */
  void KPSRecoManager::close_ana_file()
  {
    _ana_file->Close();
    _ana_file = nullptr;
    _ana_tree = nullptr;
  }
  
  /**
  *  @brief if savemc set to true, save MC event summary to output ttree
  * 
  * @param[in] savemc If true, save MC truth information to output TTree
  * @param[in] activate_nuvertexshowerreco_mcanamode  If true, turn on MC analysis mode, which will save variables and ground truth label for finding starts of nu interaction showers.
  * 
  */  
  void KPSRecoManager::saveEventMCinfo( bool savemc, bool activate_nuvertexshowerreco_mcanamode )
  {
    if ( !_save_event_mc_info && savemc )  {
      //_track_truthreco_ana.bindAnaVariables( _ana_tree );
      _event_mcinfo_maker.bindAnaVariables( _ana_tree );
    }
    _save_event_mc_info = savemc;
    if (activate_nuvertexshowerreco_mcanamode) {
      _nuvertex_shower_reco.activateMCanalysisMode( savemc );
    }
    else {
      _nuvertex_shower_reco.activateMCanalysisMode( false );      
    }
  };

  /**
  *  @brief run Truth-Reco analyses for studying performance 
  */
  void KPSRecoManager::truthAna( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    ublarcvapp::mctools::LArbysMC truthdata;
    truthdata.process( ioll );    
    truthdata.process( iolcv, ioll );
    truthdata.printInteractionInfo();

    //std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_fitted_candidates();
    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_output_candidates();    
    std::vector<float> true_vtx = { truthdata._vtx_detx, truthdata._vtx_sce_y, truthdata._vtx_sce_z };

    if ( nuvtx_v.size()!=_nu_sel_v.size() ) {
      LARCV_CRITICAL() << "Number of NuSelectionVariable instances (" << _nu_sel_v.size() <<  ") "
                       << "does not match the number of neutrino candidates (" << nuvtx_v.size() << ")" 
                       << std::endl;
    }
    
    for ( size_t ivtx=0; ivtx<nuvtx_v.size(); ivtx++ ) {
      larflow::reco::NuVertexCandidate& nuvtx    = nuvtx_v[ivtx];
      larflow::reco::NuSelectionVariables& nusel = _nu_sel_v[ivtx];

      nusel.dist2truevtx = 0.;
      for (int i=0; i<3; i++)
        nusel.dist2truevtx += ( nuvtx.pos[i]-true_vtx[i] )*( nuvtx.pos[i]-true_vtx[i] );
      nusel.dist2truevtx = sqrt( nusel.dist2truevtx );
      
      if (nusel.dist2truevtx<3.0)
        nusel.isTruthMatchedNu = 1;
      else
        nusel.isTruthMatchedNu = 0;

      larflow::reco::NuSelTruthOnNuPixel nupix;
      nupix.analyze( iolcv, ioll, nuvtx, nusel );
    }
      
    // _track_truthreco_ana.set_verbosity( larcv::msg::kDEBUG );
    // _track_truthreco_ana.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );
  }

  /**
   * @brief run modules to produce selection variables for nu selection
   *
   */
  void KPSRecoManager::makeNuCandidateSelectionVariables( larcv::IOManager& iolcv,
                                                          larlite::storage_manager& ioll )
  {

    //std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_fitted_candidates();
    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_output_candidates();
    LARCV_INFO() << "Make Selection Variables for " << nuvtx_v.size() << " candidates" << std::endl;

    // NuSelProngVars prongvars;
    // NuSelVertexVars vertexvars;
    // NuSelShowerTrunkAna showertrunkvars;
    // NuSelWCTaggerOverlap wcoverlapvars;
    // NuSelShowerGapAna2D showergapana2d;
    // NuSelUnrecoCharge   unrecocharge;
    // NuSelCosmicTagger   cosmictagger;
    // TrackForwardBackwardLL muvsproton;

    _prongvars.set_verbosity(logger().level());
    _vertexvars.set_verbosity(logger().level());
    _wcoverlapvars.set_verbosity(logger().level());
    _showergapana2d.set_verbosity(logger().level());
    _unrecocharge.setSaveMask(false);
    _unrecocharge.set_verbosity(logger().level());
    _cosmictagger.set_verbosity(logger().level());
    _muvsproton.set_verbosity(logger().level());

    _nu_sel_v.clear();
    _nu_sel_v.reserve( nuvtx_v.size() );
    
    for ( size_t ivtx=0; ivtx<nuvtx_v.size(); ivtx++ ) {

      // nu candidate
      larflow::reco::NuVertexCandidate& nuvtx = nuvtx_v[ivtx];
      
      // make selection variables
      larflow::reco::NuSelectionVariables nusel;

      LARCV_INFO() << "===[ VERTEX " << ivtx << " ]===" << std::endl;
      LARCV_INFO() << "  source: " << nuvtx.keypoint_producer << std::endl;
      LARCV_INFO() << "  type: " << nuvtx.keypoint_type << std::endl;      
      LARCV_INFO() << "  pos (" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
      LARCV_INFO() << "  number of tracks: "  << nuvtx.track_v.size() << std::endl;
      LARCV_INFO() << "  number of showers: " << nuvtx.shower_v.size() << std::endl;

      float tot_tracklen = 0.;
      float tot_showermev = 0.;
      for (int i=0; i<(int)nuvtx.track_v.size(); i++) {
	      tot_tracklen += nuvtx.track_len_v[i];
      }
      for (int i=0; i<(int)nuvtx.shower_plane_pixsum_vv.size(); i++) {
	      auto const& plane_pixsum = nuvtx.shower_plane_pixsum_vv.at(i);
	      float maxpixsum = 0.;
	      for (auto const& pixsum : plane_pixsum ) {
	        if ( pixsum>maxpixsum )
	          maxpixsum = pixsum;
	      }
	      tot_showermev += maxpixsum*0.0162;
      }
      LARCV_INFO() << "  Total track length: " << tot_tracklen << " cm (" << tot_tracklen*2.2 << " MeV)" << std::endl;
      LARCV_INFO() << "  Shower pixel sum: " << tot_showermev << " MeV" << std::endl;

      float tot_vis_energy = tot_tracklen*2.2+tot_showermev;
      LARCV_INFO() << "  Tot. approx visible energy: " << tot_vis_energy << std::endl;
      nusel.approx_vis_energy_MeV = tot_vis_energy;
      
      // check if showers are connected to vertex      
      //_showergapana2d.analyze( iolcv, ioll, nuvtx, nusel );

      // // if so, check for need of repair
      // if ( nusel.nplanes_connected>=2 )
      //   _nuvertex_shower_trunk_check.checkNuCandidateProngsForMissingCharge( nuvtx, iolcv, ioll );

      // nusel.max_proton_pid = 1e3; // more proton, the more value is negative
      // for (int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++) {

      //   auto& lltrack = nuvtx.track_v.at(itrack);
      //   LARCV_INFO() << "  [track " << itrack << "]" << std::endl;
      //   LARCV_INFO() << "    npts: " << lltrack.NumberTrajectoryPoints() << std::endl;

      //   larflow::reco::NuSelectionVariables::TrackVar_t trackvars;

      //   trackvars.proton_ll = _sel_llpmu.calculateLL( lltrack, nuvtx.pos );
      //   if ( trackvars.proton_ll<nusel.max_proton_pid )
      //     nusel.max_proton_pid = trackvars.proton_ll;
      //   LARCV_INFO() << "    proton-ll: " << trackvars.proton_ll << std::endl;

      //   // proton ID variables        

      //   // muon ID variables

      //   // muon ID variables
        
      //   // pion ID variables

      //   nusel._track_var_v.emplace_back( std::move(trackvars) );
        
      // }//end of track loop
        

      // for (int ishower=0; ishower<(int)nuvtx.shower_v.size(); ishower++) {

      //   auto& llshower = nuvtx.shower_v.at(ishower);
        
      //   // electron ID variables
      
      //   // pi-zero ID variables

      // }

      _unrecocharge.analyze( iolcv, ioll, nuvtx, nusel );
      _unrecocharge.analyze_with_spacepoints( iolcv, ioll, nuvtx, nusel );            
      // _prongvars.analyze( nuvtx, nusel );
      // _showertrunkvars.analyze( nuvtx, nusel, iolcv, ioll );
      // _vertexvars.analyze( iolcv, ioll, nuvtx, nusel );
      // _wcoverlapvars.analyze( nuvtx, nusel, iolcv );
      // _cosmictagger.analyze( nuvtx, nusel );
      // _muvsproton.analyze( nuvtx, nusel );
      
      // LARCV_INFO() << "  minshowergap: " << nusel.min_shower_gap << std::endl;
      // LARCV_INFO() << "  maxshowergap: " << nusel.max_shower_gap << std::endl;      
      
      // nu kinematic variables
      _nu_sel_v.emplace_back( std::move(nusel) );

      
    }//end of vertex loop

    LARCV_INFO() << "Selection variables made: " << _nu_sel_v.size() << std::endl;
    
  }

  /**
   * @brief Produce baseline kinematics variables
   *
   */
  void KPSRecoManager::runBasicKinematics( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    
    LARCV_INFO() << "Calculate prong kinematics" << std::endl;
    _nu_track_kine.set_verbosity( logger().level() );
    _nu_shower_kine.set_verbosity( logger().level() );

    _nu_track_kine.clear();
    _nu_shower_kine.clear();
    
    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_output_candidates();

    for (auto& nuvtx : nuvtx_v ) {
      
      larflow::reco::NuSelectionVariables nusel;
      
      // prong kinematic calculators
      _nu_track_kine.clear();
      _nu_track_kine.analyze( nuvtx );
      
      nuvtx.track_len_v      = _nu_track_kine._track_length_v;
      nuvtx.track_kemu_v     = _nu_track_kine._track_mu_ke_v;
      nuvtx.track_keproton_v = _nu_track_kine._track_p_ke_v;
      nuvtx.track_pmu_v      = _nu_track_kine._track_mu_mom_v;
      nuvtx.track_pproton_v  = _nu_track_kine._track_p_mom_v;

      _nu_shower_kine.clear();
      _nu_shower_kine.analyze( nuvtx, nusel, iolcv );
      nuvtx.shower_plane_pixsum_vv = _nu_shower_kine._shower_plane_pixsum_v;
      nuvtx.shower_plane_mom_vv    = _nu_shower_kine._shower_mom_v;
	
    }
      
  }

  /**
   * @brief calculate basline PID-related variables for the prongs
   */
  void KPSRecoManager::runBasicPID( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    
    LARCV_INFO() << "Calculate baseline prong dq/dx-based PID metrics" << std::endl;
    
    //std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_fitted_candidates();
    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_output_candidates();
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();
    
    for (auto& nuvtx : nuvtx_v ) {

      nuvtx.track_muid_v.resize( nuvtx.track_v.size(), 0 );
      nuvtx.track_protonid_v.resize( nuvtx.track_v.size(), 0 );
      nuvtx.track_mu_vs_proton_llratio_v.resize( nuvtx.track_v.size(), 0 );

      nuvtx.shower_plane_dqdx_vv.clear();
      
      larflow::reco::NuSelectionVariables nusel;
      
      // track dq/dx-based likelihoods
      for (size_t itrack=0; itrack<nuvtx.track_v.size(); itrack++) {

        try {        
          std::vector<double> ll_results = _sel_llpmu.calculateLLseparate( nuvtx.track_v[itrack], nuvtx.pos );
          nuvtx.track_muid_v[itrack] = ll_results[2];
          nuvtx.track_protonid_v[itrack] = ll_results[1];
          nuvtx.track_mu_vs_proton_llratio_v[itrack] = ll_results[0];
        }
        catch ( const std::exception& e ) {
          LARCV_INFO() << "error running track likelihoood: " << e.what() << std::endl;
        }
      }//end of track loop

      // shower dq/dx
      for (size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++) {
        bool dqdxok = true;

        std::vector<float> shower_plane_pixsum_v(adc_v.size(),0);
        
        try {
          _sel_showerdqdx.processShower( nuvtx.shower_v[ishower],
                                         nuvtx.shower_trunk_v[ishower],
                                         nuvtx.shower_pcaxis_v[ishower],
                                         ev_adc->as_vector(), nuvtx );
        }
        catch( const std::exception& e ) {
          dqdxok = false;
          LARCV_INFO() << "error running showerdqdx: " << e.what() << std::endl;
        }
        
        // set values
        if ( !dqdxok ) {
          nuvtx.shower_plane_dqdx_vv.emplace_back( std::move(shower_plane_pixsum_v) );
        }
        else {
          // good reco
          nuvtx.shower_plane_dqdx_vv.push_back( _sel_showerdqdx._pixsum_dqdx_v );
        }
      }//end of shower loop
      
    }//end of vertex loop
      
  }
  
  void KPSRecoManager::clear()
  {
    _nu_sel_v.clear();
    _nu_perfect_v.clear();
    
  }
  
}
}
