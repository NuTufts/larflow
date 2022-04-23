#include "KPSRecoManager.h"

#include <ctime>

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
  KPSRecoManager::KPSRecoManager( std::string inputfile_name, int reco_ver )
    : larcv::larcv_base("KPSRecoManager"),
    _save_event_mc_info(false),    
    _ana_output_file(inputfile_name),
    _t_event_elapsed(0),
    _save_selected_only(false),
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

    std::clock_t start_event = std::clock_t();

    _nu_sel_v.clear(); ///< clear vertex selection variable container
    _nu_perfect_v.clear(); ///< clear perfect reco
    
    // PREP: make bad channel image
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();
    
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
    fiveparticlealgo.set_verbosity( larcv::msg::kINFO );
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
      return;
    }

    // Make keypoint candidates from larmatch vertex
    // ---------------------------------------------
    recoKeypoints( iolcv, ioll );

    if ( _stop_after_keypointreco ) {
      // early stoppage to debug (and visualize) prepared keypoints
      _ana_tree->Fill();
      return;
    }
      
    // PARTICLE FRAGMENT RECO
    clusterSubparticleFragments( iolcv, ioll );
    if ( _stop_after_subclustering ) {
      // early stopping to debug (and visualize) subclusters
      _ana_tree->Fill();
      return;
    }
    
    // COSMIC RECO
    cosmicTrackReco( iolcv, ioll );
    
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

    // make selection variables
    //makeNuCandidateSelectionVariables( iolcv, ioll );
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

    // run selection and filter events    
    //runNuVtxSelection();    

    if ( _kMinize_outputfile_size ) {
      // save only fitted vertex candidates
      _nuvertexmaker.get_mutable_nu_candidates().clear();
      _nuvertexmaker.get_mutable_vetoed_candidates().clear();
      _nuvertexmaker.get_mutable_merged_candidates().clear();            
    }
    
    // Fill Ana Tree
    _ana_run = ev_adc->run();
    _ana_subrun = ev_adc->subrun();
    _ana_event  = ev_adc->event();

    std::clock_t end_event = std::clock_t();
    _t_event_elapsed = (end_event-start_event)/CLOCKS_PER_SEC;

    _ana_tree->Fill();    
    
  }

  /**
   * @brief algorithms for splitting up and filtering larmatch space points
   *
   */
  void KPSRecoManager::prepSpacepoints( larcv::IOManager& iolcv,
                                        larlite::storage_manager& ioll )
  {

    // PREP: make sure the larmatch points have their idxhit index set
    // we can use this as a way to trace identity of hits back to original set
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "larmatch" );
    for (size_t ihit=0; ihit<ev_larmatch->size(); ihit++) {
      auto& hit = ev_larmatch->at(ihit);
      hit.idxhit = (int)ihit;
    }
    
    // PREP: LABEL larmatch POINTS WITH 2D SSNET SHOWER SCORE
    // input:
    //  * image2d_ubspurn_planeX: ssnet (track,shower) scores
    //  * larflow3dhit_taggerfilterhit_tree: WC in-time space points
    // output:
    //  * larflow3dhit_ssnetsplit_wcfilter_showerhit_tree: in-time shower hits
    //  * larflow3dhit_ssnetsplit_wcfilter_trackhit_tree:  in-time track hits
    _splithits_wcfilter.set_larmatch_tree_name( "larmatch" );
    _splithits_wcfilter.process_labelonly( iolcv, ioll );    

    // PREP: ALTER THRUMU IMAGE TO INCLUDE SSNET CLUSTERS OF A CERTAIN SIZE
    
    // PREP WC-FILTERED HITS
    // filters raw larmatch hits using wire cell thrumu tagger image
    // input:
    //  larflow3dhit_larmatch_tree: raw larmatch deploy output
    // output(s):
    //  larflow3dhit_taggerfilterhit_tree: in-time hits
    //  larflow3dhit_taggerrejecthit_tree: out-of-time/cosmic-tagged hits
    _wcfilter.set_verbosity( larcv::msg::kINFO );
    _wcfilter.set_input_larmatch_tree_name( "larmatch" );
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
    _splithits_wcfilter.set_larmatch_tree_name( "taggerfilterhit" );
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
    _choosemaxhit.set_verbosity( larcv::msg::kINFO );
    _choosemaxhit.process( iolcv, ioll );
    // input:
    //  * larflow3dhit_ssnetsplit_wcfilter_showerhit_tree: in-time shower hits
    // output:
    //  * larflow3dhit_maxshowerhit_tree: in-time shower hits after filter
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_wcfilter_showerhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "maxshowerhit" );
    _choosemaxhit.set_verbosity( larcv::msg::kINFO );
    _choosemaxhit.process( iolcv, ioll );

    // PREP: SPLIT SHOWER/TRACK FOR COSMIC HITS
    // input:
    //  * larflow3dhit_taggerrejecthit_tree: out-of-time hits
    // output:
    //  * larflow3dhit_ssnetsplit_full_showerhit_tree: out-of-time shower hits
    //  * larflow3dhit_ssnetsplit_full_trackhit_tree:  out-of-time track hits
    _splithits_full.set_larmatch_tree_name( "taggerrejecthit" );
    _splithits_full.set_output_tree_stem_name( "ssnetsplit_full" );
    _splithits_full.process_splitonly( iolcv, ioll );

    // PREP: MAX-SCORE REDUCTION ON COSMIC HITS
    // input:
    //  * larflow3dhit_ssnetsplit_full_trackhit_tree: out-of-time track hits
    // output:
    //  *  larflow3dhit_full_maxtrackhit_tree: reduced out-of-time track hits
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_full_trackhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "full_maxtrackhit" );
    _choosemaxhit.process( iolcv, ioll );

    // if we're stopping at this stage for debugging/plotting,
    // we force the saving of all the intermediate hit containers
    if ( _stop_after_prepspacepoints ) {
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "larmatch" ); ///< save all original hits (now ssnet-labeled)
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "maxtrackhit_wcfilter" ); ///< final set of in-time track hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "maxshowerhit" );         ///< final set of in-time shower hits      
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "full_maxtrackhit" );     ///< final set of out-of-time track hits

      // intermediate hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_full_trackhit" );      //< pre-max out-of-time track hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_full_showerhit" );     //< pre-max out-of-time shower hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_wcfilter_trackhit" );  //< pre-max in-time track hits
      ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_wcfilter_showerhit" ); //< pre-max in-time track hits
      
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

    if ( _reco_version==1 ) {
      // neutrino
      _kpreco_nu.set_input_larmatch_tree_name( "taggerfilterhit" );
      _kpreco_nu.set_output_tree_name( "keypoint" );    
      _kpreco_nu.set_sigma( 10.0 );
      _kpreco_nu.set_min_cluster_size(   50.0, 0 );
      _kpreco_nu.set_keypoint_threshold( 0.5, 0 );
      _kpreco_nu.set_min_cluster_size(   20.0, 1 );    
      _kpreco_nu.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_nu.set_larmatch_threshold( 0.5 );
      _kpreco_nu.set_keypoint_type( (int)larflow::kNuVertex );
      _kpreco_nu.set_lfhit_score_index( 13 ); // (v1 larmatch network score index in hit)
      _kpreco_nu.process( ioll );
      
      _kpreco_track.set_input_larmatch_tree_name( "taggerfilterhit" );
      _kpreco_track.set_output_tree_name( "keypoint" );        
      _kpreco_track.set_sigma( 10.0 );    
      _kpreco_track.set_min_cluster_size(   50.0, 0 );
      _kpreco_track.set_keypoint_threshold( 0.5, 0 );
      _kpreco_track.set_min_cluster_size(   20.0, 1 );    
      _kpreco_track.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_track.set_larmatch_threshold( 0.5 );
      _kpreco_track.set_keypoint_type( (int)larflow::kTrackEnd );
      _kpreco_track.set_lfhit_score_index( 14 ); // (v1 larmatch network track-score index in hit)
      _kpreco_track.process( ioll );
      
      _kpreco_shower.set_input_larmatch_tree_name( "taggerfilterhit" );
      _kpreco_shower.set_output_tree_name( "keypoint" );
      _kpreco_shower.set_sigma( 10.0 );    
      _kpreco_shower.set_min_cluster_size(   50.0, 0 );
      _kpreco_shower.set_keypoint_threshold( 0.5, 0 );
      _kpreco_shower.set_min_cluster_size(   20.0, 1 );    
      _kpreco_shower.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_shower.set_larmatch_threshold( 0.5 );
      _kpreco_shower.set_keypoint_type( (int)larflow::kShowerStart );
      _kpreco_shower.set_lfhit_score_index( 15 ); // (v1 larmatch network shower-score index in hit)
      _kpreco_shower.process( ioll );

      _kpreco_track_cosmic.set_input_larmatch_tree_name( "taggerrejecthit" );
      _kpreco_track_cosmic.set_output_tree_name( "keypointcosmic" );
      _kpreco_track_cosmic.set_sigma( 50.0 );    
      _kpreco_track_cosmic.set_min_cluster_size(   50.0, 0 );
      _kpreco_track_cosmic.set_max_dbscan_dist( 10.0 );
      _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 0 );
      _kpreco_track_cosmic.set_min_cluster_size(   20.0, 1 );    
      _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_track_cosmic.set_larmatch_threshold( 0.5 );
      _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackEnd );
      _kpreco_track_cosmic.set_lfhit_score_index( 14 ); // (v1 larmatch network track-score index in hit)
      _kpreco_track_cosmic.process( ioll );

    }
    else if ( _reco_version==2 ) {
      // we take advantage of the fact that we dont want anything stored by the keypoint reco class
      // after it runs. everything we need downstream is saved to a larlite tree.
      // so we simply re-run the algorithms to work with the additional vertex types.

      // neutrino
      //_kpreco_nu.set_verbosity( larcv::msg::kINFO );
      _kpreco_nu.set_input_larmatch_tree_name( "taggerfilterhit" );
      _kpreco_nu.set_sigma( 10.0 );
      _kpreco_nu.set_min_cluster_size(   50.0, 0 );
      _kpreco_nu.set_keypoint_threshold( 0.5, 0 );
      _kpreco_nu.set_min_cluster_size(   20.0, 1 );    
      _kpreco_nu.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_nu.set_larmatch_threshold( 0.5 );
      _kpreco_nu.set_output_tree_name( "keypoint" );          
      _kpreco_nu.set_keypoint_type( (int)larflow::kNuVertex );
      _kpreco_nu.set_lfhit_score_index( 17 ); // (v2 larmatch-minkowski network neutrino-score index in hit)
      _kpreco_nu.process( ioll );

      // neutrino interaction track: we have track starts and ends
      _kpreco_track.set_input_larmatch_tree_name( "taggerfilterhit" );
      _kpreco_track.set_sigma( 10.0 );    
      _kpreco_track.set_min_cluster_size(   50.0, 0 );
      _kpreco_track.set_keypoint_threshold( 0.5, 0 );
      _kpreco_track.set_min_cluster_size(   20.0, 1 );    
      _kpreco_track.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_track.set_larmatch_threshold( 0.5 );
      // neutrino interaction track start
      _kpreco_track.set_output_tree_name( "keypoint" );              
      _kpreco_track.set_keypoint_type( (int)larflow::kTrackStart );
      _kpreco_track.set_lfhit_score_index( 18 ); // (v2 larmatch-minkowski network track-start-score index in hit)
      _kpreco_track.process( ioll );
      // neutrino interaction track end
      _kpreco_track.set_output_tree_name( "keypoint" );              
      _kpreco_track.set_keypoint_type( (int)larflow::kTrackEnd );
      _kpreco_track.set_lfhit_score_index( 19 ); // (v2 larmatch-minkowski network track-end-score index in hit)
      _kpreco_track.process( ioll );
      // neutrino interaction shower
      _kpreco_shower.set_input_larmatch_tree_name( "taggerfilterhit" );
      _kpreco_shower.set_output_tree_name( "keypoint" );
      _kpreco_shower.set_sigma( 10.0 );    
      _kpreco_shower.set_min_cluster_size(   50.0, 0 );
      _kpreco_shower.set_keypoint_threshold( 0.5, 0 );
      _kpreco_shower.set_min_cluster_size(   20.0, 1 );    
      _kpreco_shower.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_shower.set_larmatch_threshold( 0.5 );
      _kpreco_shower.set_keypoint_type( (int)larflow::kShowerStart );
      _kpreco_shower.set_lfhit_score_index( 20 ); // (v2 larmatch-minkowski network nu-shower-score index in hit)
      _kpreco_shower.process( ioll );
      // neutrino+cosmic interaction michel
      _kpreco_shower.set_keypoint_type( (int)larflow::kShowerMichel );
      _kpreco_shower.set_lfhit_score_index( 21 ); // (v2 larmatch-minkowski network michel-shower-score index in hit)
      _kpreco_shower.process( ioll );
      // neutrino+cosmic interaction delta
      _kpreco_shower.set_keypoint_type( (int)larflow::kShowerDelta );
      _kpreco_shower.set_lfhit_score_index( 22 ); // (v2 larmatch-minkowski network delta-shower-score index in hit)
      _kpreco_shower.process( ioll );

      // cosmic keypoints
      _kpreco_track_cosmic.set_input_larmatch_tree_name( "taggerrejecthit" );
      _kpreco_track_cosmic.set_output_tree_name( "keypointcosmic" );
      _kpreco_track_cosmic.set_sigma( 50.0 );    
      _kpreco_track_cosmic.set_min_cluster_size(   50.0, 0 );
      _kpreco_track_cosmic.set_max_dbscan_dist( 10.0 );
      _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 0 );
      _kpreco_track_cosmic.set_min_cluster_size(   20.0, 1 );    
      _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 1 );    
      _kpreco_track_cosmic.set_larmatch_threshold( 0.5 );

      _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackStart );
      _kpreco_track_cosmic.set_lfhit_score_index( 18 ); // (v2 larmatch network track-start-score index in hit)
      _kpreco_track_cosmic.process( ioll );

      _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackEnd );
      _kpreco_track_cosmic.set_lfhit_score_index( 19 ); // (v2 larmatch network track-end-score index in hit)
      _kpreco_track_cosmic.process( ioll );
      
      
    }
    else {
      std::stringstream oops;
      oops << "Reco Version unrecognized: " << _reco_version << " allowed: {1,2}" << std::endl;
      throw std::runtime_error(oops.str());
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
    _projsplitter.set_verbosity( larcv::msg::kINFO );    
    _projsplitter.set_dbscan_pars( _maxdist, _minsize, _maxkd );
    _projsplitter.doClusterVetoHits(false);
    _projsplitter.set_fit_line_segments_to_clusters( true );
    _projsplitter.set_input_larmatchhit_tree_name( "maxtrackhit_wcfilter" );
    //_projsplitter.set_input_larmatchhit_tree_name( "ssnetsplit_wcfilter_trackhit" );    
    _projsplitter.add_input_keypoint_treename_for_hitveto( "keypoint" );
    _projsplitter.set_output_tree_name("trackprojsplit_wcfilter");
    _projsplitter.process( iolcv, ioll );

    // PRIMITIVE TRACK FRAGMENTS: FULL TRACK HITS
    LARCV_INFO() << "RUN PROJ-SPLITTER ON: full_maxtrackhit (out-of-time hits)" << std::endl;    
    _projsplitter_cosmic.set_verbosity( larcv::msg::kINFO );
    //_projsplitter_cosmic.set_verbosity( larcv::msg::kDEBUG );    
    _projsplitter_cosmic.set_dbscan_pars( 5.0, _minsize, _maxkd ); // cosmic parameters, courser maxdist to reduce number of cosmic fragments
    _projsplitter_cosmic.doClusterVetoHits(false);
    _projsplitter_cosmic.set_input_larmatchhit_tree_name( "full_maxtrackhit" );
    _projsplitter_cosmic.set_fit_line_segments_to_clusters( true ); // can be slow
    _projsplitter_cosmic.set_output_tree_name("trackprojsplit_full");
    _projsplitter_cosmic.process( iolcv, ioll );

    // SHOWER 1-KP RECO: make shower using clusters and single keypoint
    // class: larflow::reco::ShowerRecoKeypoint
    _showerkp.setShowerRadiusThresholdcm( 5.0 );
    _showerkp.set_ssnet_lfhit_tree_name( "maxshowerhit" );
    //_showerkp.set_ssnet_lfhit_tree_name( "ssnetsplit_wcfilter_showerhit" );    
    //_showerkp.set_verbosity( larcv::msg::kDEBUG );
    _showerkp.set_verbosity( larcv::msg::kINFO );    
    _showerkp.process( iolcv, ioll );

    // SHORT HIP FRAGMENTS
    //_short_proton_reco.set_verbosity( larcv::msg::kDEBUG );
    _short_proton_reco.set_verbosity( larcv::msg::kINFO );    
    _short_proton_reco.clear_clustertree_checklist();
    _short_proton_reco.add_clustertree_forcheck( "trackprojsplit_wcfilter" );
    _short_proton_reco.process( iolcv, ioll );
    
    // TRACK CLUSTER-ONLY RECO: make tracks without use of keypoints

    // SHOWER CLUSTER-ONLY RECO: make showers without use of keypoints

    if ( _stop_after_subclustering ) {
      // we're going to stop here. save key intermediate products from this stage.
      ioll.set_data_to_write( larlite::data::kLArFlowCluster, "trackprojsplit_wcfilter" ); // in-time track clusters
      ioll.set_data_to_write( larlite::data::kPCAxis, "trackprojsplit_wcfilter" );         // in-time track clusters
      
      ioll.set_data_to_write( larlite::data::kLArFlowCluster, "trackprojsplit_full" ); // out-of-time track clusters
      ioll.set_data_to_write( larlite::data::kPCAxis, "trackprojsplit_full" );         // out-of-time track clusters
      
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


    _nuvertexactivity.set_verbosity( larcv::msg::kINFO );
    //_nuvertexactivity.set_verbosity( larcv::msg::kDEBUG );    

    // configure to use shower and in-time hits
    std::vector<std::string> input_hit_list
      = {"taggerfilterhit",            // all in-time hits
         "ssnetsplit_full_showerhit"}; // out-of-time shower hits
    std::vector<std::string> input_cluster_list
      = { "trackprojsplit_full"}; // in-time track clusters

    _nuvertexactivity.set_input_hit_list( input_hit_list );    
    //_nuvertexactivity.set_input_cluster_list( input_cluster_list );
    _nuvertexactivity.set_output_treename( "keypoint" );
    //_nuvertexactivity.process( iolcv, ioll );
    
    //_nuvertexmaker.set_verbosity( larcv::msg::kDEBUG );
    _nuvertexmaker.set_verbosity( larcv::msg::kINFO );
    _nuvertexmaker.clear();
    _nuvertexmaker.add_keypoint_producer( "keypoint" );
    _nuvertexmaker.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    _nuvertexmaker.add_cluster_producer("cosmicproton", NuVertexCandidate::kTrack );
    //_nuvertexmaker.add_cluster_producer("hip", NuVertexCandidate::kTrack );    
    _nuvertexmaker.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    _nuvertexmaker.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );
    
    _nuvertexmaker.apply_cosmic_veto( true );
    _nuvertexmaker.setOutputStage( larflow::reco::NuVertexMaker::kVetoed );    
    _nuvertexmaker.process( iolcv, ioll );

    // NuTrackBuilder class
    _nu_track_builder.clear();
    if ( _stop_after_nutracker )
      _nu_track_builder.set_verbosity( larcv::msg::kDEBUG );    
    else 
      _nu_track_builder.set_verbosity( larcv::msg::kINFO );
    //_nu_track_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );
    _nu_track_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_output_candidates() );
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
    _nuvertex_shower_reco.set_verbosity( larcv::msg::kINFO );    
    _nuvertex_shower_reco.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    _nuvertex_shower_reco.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    _nuvertex_shower_reco.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );    
    //_nuvertex_shower_reco.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );
    _nuvertex_shower_reco.process( iolcv, ioll, _nuvertexmaker.get_mutable_output_candidates() );
    
    // - repair shower trunks by absorbing tracks or creating hits
    //_nuvertex_shower_trunk_check.set_verbosity( larcv::msg::kDEBUG );
    int ivtx = 0;
    //for ( auto& vtx : _nuvertexmaker.get_mutable_fitted_candidates() ) {
    for ( auto& vtx : _nuvertexmaker.get_mutable_output_candidates() ) {
      LARCV_DEBUG() << "Run shower trunk check on vertex candidate [" << ivtx << "]" << std::endl;
      _nuvertex_shower_trunk_check.checkNuCandidateProngs( vtx );
      //_nuvertex_shower_trunk_check.checkNuCandidateProngsForMissingCharge( vtx, iolcv, ioll );
      ivtx++;
    }

    // post-neutrino-candidate processing:
    // - remove tracks from neutrino candidates that significantly overlap with showers
    //_nuvertex_postcheck_showertrunkoverlap.set_verbosity( larcv::msg::kDEBUG );
    //_nuvertex_postcheck_showertrunkoverlap.process( _nuvertexmaker.get_mutable_fitted_candidates() );
    _nuvertex_postcheck_showertrunkoverlap.process( _nuvertexmaker.get_mutable_output_candidates() );

    // - add hits vetod around keypoints to the ends of track prongs
    //_nuvertex_cluster_vetohits.set_verbosity( larcv::msg::kDEBUG );
    LARCV_NORMAL() << "RUN NUVERTEX CLUSTER VETOHITS" << std::endl;
    for ( auto& vtx : _nuvertexmaker.get_mutable_output_candidates() ) {    
      _nuvertex_cluster_vetohits.process( ioll, vtx );
    }

    // - add dq/dx information
    //_nuvertex_trackdqdx.set_verbosity( larcv::msg::kDEBUG );
    LARCV_NORMAL() << "calculate Track dQ/dx" << std::endl;
    for ( auto& vtx : _nuvertexmaker.get_mutable_output_candidates() ) {        
      _nuvertex_trackdqdx.process_nuvertex_tracks( iolcv, vtx );
    }

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
    _cosmic_track_builder.set_verbosity( larcv::msg::kINFO );    
    _cosmic_track_builder.do_boundary_analysis( true );
    _cosmic_track_builder.process( iolcv, ioll );

    //_cosmic_proton_finder.set_verbosity( larcv::msg::kDEBUG );
    _cosmic_proton_finder.set_verbosity( larcv::msg::kINFO );    
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
    _ana_file = new TFile(_ana_output_file.c_str(), "recreate");
    _ana_tree = new TTree("KPSRecoManagerTree","Ana Output of KPSRecoManager algorithms");
    _ana_tree->Branch("run",&_ana_run,"run/I");
    _ana_tree->Branch("subrun",&_ana_subrun,"subrun/I");
    _ana_tree->Branch("event",&_ana_event,"event/I");    
  }

  /** @brief is true, save MC event summary */  
  void KPSRecoManager::saveEventMCinfo( bool savemc )
  {
    if ( !_save_event_mc_info && savemc )  {
      //_track_truthreco_ana.bindAnaVariables( _ana_tree );
      _event_mcinfo_maker.bindAnaVariables( _ana_tree );
    }
    _save_event_mc_info = savemc;

  };

  /** @brief run Truth-Reco analyses for studying performance **/
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
    prongvars.set_verbosity(larcv::msg::kDEBUG);
    vertexvars.set_verbosity(larcv::msg::kDEBUG);
    wcoverlapvars.set_verbosity(larcv::msg::kDEBUG);
    showergapana2d.set_verbosity(larcv::msg::kDEBUG);
    unrecocharge.setSaveMask(false);
    unrecocharge.set_verbosity(larcv::msg::kDEBUG);
    cosmictagger.set_verbosity(larcv::msg::kDEBUG);
    muvsproton.set_verbosity(larcv::msg::kINFO);
    
    for ( size_t ivtx=0; ivtx<nuvtx_v.size(); ivtx++ ) {

      // nu candidate
      larflow::reco::NuVertexCandidate& nuvtx = nuvtx_v[ivtx];
      
      // make selection variables
      larflow::reco::NuSelectionVariables nusel;

      std::cout << "===[ VERTEX " << ivtx << " ]===" << std::endl;
      std::cout << "  source: " << nuvtx.keypoint_producer << std::endl;
      std::cout << "  type: " << nuvtx.keypoint_type << std::endl;      
      std::cout << "  pos (" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
      std::cout << "  number of tracks: "  << nuvtx.track_v.size() << std::endl;
      std::cout << "  number of showers: " << nuvtx.shower_v.size() << std::endl;


      // check if showers are connected to vertex      
      showergapana2d.analyze( iolcv, ioll, nuvtx, nusel );

      // if so, check for need of repair
      if ( nusel.nplanes_connected>=2 )
        _nuvertex_shower_trunk_check.checkNuCandidateProngsForMissingCharge( nuvtx, iolcv, ioll );

      nusel.max_proton_pid = 1e3; // more proton, the more value is negative
      for (int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++) {

        auto& lltrack = nuvtx.track_v.at(itrack);
        std::cout << "  [track " << itrack << "]" << std::endl;
        std::cout << "    npts: " << lltrack.NumberTrajectoryPoints() << std::endl;

        larflow::reco::NuSelectionVariables::TrackVar_t trackvars;

        trackvars.proton_ll = _sel_llpmu.calculateLL( lltrack, nuvtx.pos );
        if ( trackvars.proton_ll<nusel.max_proton_pid )
          nusel.max_proton_pid = trackvars.proton_ll;
        std::cout << "    proton-ll: " << trackvars.proton_ll << std::endl;

        // proton ID variables        

        // muon ID variables

        // muon ID variables
        
        // pion ID variables

        nusel._track_var_v.emplace_back( std::move(trackvars) );
        
      }//end of track loop
        

      for (int ishower=0; ishower<(int)nuvtx.shower_v.size(); ishower++) {

        auto& llshower = nuvtx.shower_v.at(ishower);
        
        // electron ID variables
      
        // pi-zero ID variables

      }
      
      prongvars.analyze( nuvtx, nusel );
      showertrunkvars.analyze( nuvtx, nusel, iolcv, ioll );
      vertexvars.analyze( iolcv, ioll, nuvtx, nusel );
      wcoverlapvars.analyze( nuvtx, nusel, iolcv );
      unrecocharge.analyze( iolcv, ioll, nuvtx, nusel );
      cosmictagger.analyze( nuvtx, nusel );
      muvsproton.analyze( nuvtx, nusel );
      
      // std::cout << "  minshowergap: " << nusel.min_shower_gap << std::endl;
      // std::cout << "  maxshowergap: " << nusel.max_shower_gap << std::endl;      
      
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
    _nu_track_kine.set_verbosity(larcv::msg::kDEBUG);
    _nu_shower_kine.set_verbosity(larcv::msg::kDEBUG);

    _nu_track_kine.clear();
    _nu_shower_kine.clear();
    
    //std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_fitted_candidates();
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
  
  void KPSRecoManager::runNuVtxSelection()
  {
    
    if ( !_save_selected_only )
      return;

    LARCV_INFO() << "run 1e1p development selection" << std::endl;

    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v
      //= _nuvertexmaker.get_mutable_fitted_candidates();
      = _nuvertexmaker.get_mutable_output_candidates();
    
    if ( _nu_sel_v.size()!=nuvtx_v.size() ) {
      LARCV_CRITICAL() << "mismatch in selection and vertex candidates" << std::endl;
      return;
    }
    
    std::vector<larflow::reco::NuSelectionVariables> pass_nusel_v;
    std::vector<larflow::reco::NuVertexCandidate>    pass_nuvtx_v;

    
    for ( int ivtx=0; ivtx<(int)nuvtx_v.size(); ivtx++ ) {
      auto& nusel = _nu_sel_v[ivtx];
      auto& nuvtx = nuvtx_v[ivtx];

      if ( nusel.dist2truevtx<3.0 ) {
        _eventsel_1e1p.set_verbosity( larcv::msg::kDEBUG ); // for debug
        LARCV_NORMAL() << "--------------------------------" << std::endl;
        LARCV_NORMAL() << "vtx[" << ivtx << "]  dist2true: "  << nusel.dist2truevtx << " cm" << std::endl;
      }
      else
        _eventsel_1e1p.set_verbosity( larcv::msg::kNORMAL ); // for debug
      
      int pass = _eventsel_1e1p.runSelection( nusel, nuvtx );
      if ( pass==1 ) {
        pass_nusel_v.emplace_back( std::move( nusel ) );
        pass_nuvtx_v.emplace_back( std::move( nuvtx ) );
      }
    }

    // now clear and replace
    LARCV_DEBUG() << "clear and replace vertices" << std::endl;
    nuvtx_v.clear();
    _nu_sel_v.clear();
    for ( int ivtx=0; ivtx<(int)pass_nusel_v.size(); ivtx++ ) {
      nuvtx_v.emplace_back( std::move(pass_nuvtx_v[ivtx]) );
      _nu_sel_v.emplace_back( std::move(pass_nusel_v[ivtx]) );
    }
    LARCV_INFO() << "Applied 1e1p selection. Passing vertices: " << nuvtx_v.size() << std::endl;
    
    return;
  }


  void KPSRecoManager::clear()
  {
    _nu_sel_v.clear();
    _nu_perfect_v.clear();
    
  }
}
}
