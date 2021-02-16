#include "KPSRecoManager.h"

// larlite
#include "DataFormat/opflash.h"

#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include "NuSelProngVars.h"
#include "NuSelVertexVars.h"
#include "NuSelTruthOnNuPixel.h"
#include "SplitHitsByParticleSSNet.h"

namespace larflow {
namespace reco {

  /** 
   * @brief constructor where an output file is made 
   *
   * @param[in] inputfile_name Name of output file for non-larcv and non-larlite reco products
   */
  KPSRecoManager::KPSRecoManager( std::string inputfile_name )
    : larcv::larcv_base("KPSRecoManager"),
    _save_event_mc_info(false),
    _ana_output_file(inputfile_name),
    _kMinize_outputfile_size(false)
  {
    make_ana_file();
    _nuvertexmaker.add_nuvertex_branch( _ana_tree );
    _ana_tree->Branch( "nu_sel_v", &_nu_sel_v );
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


    _nu_sel_v.clear(); ///< clear vertex selection variable container
    
    // PREP: make bad channel image
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();
    
    larcv::EventChStatus* ev_chstatus =
      (larcv::EventChStatus*)iolcv.get_data(larcv::kProductChStatus, "wire");
    std::vector<larcv::Image2D> gapch_v =
      _badchmaker.makeGapChannelImage( adc_v, *ev_chstatus,
                                       4, 3, 2400, 6*1008, 3456, 6, 1,
                                       5.0, 50, -1.0 );
    
    LARCV_INFO() << "Number of badcv images made: " << gapch_v.size() << std::endl;
    larcv::EventImage2D* evout_badch =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"badch");
    for ( auto& gap : gapch_v ) {
      evout_badch->Emplace( std::move(gap) );
    }

    // make five particle ssnet images
    larflow::reco::SplitHitsByParticleSSNet fiveparticlealgo;
    fiveparticlealgo.set_verbosity( larcv::msg::kDEBUG );
    fiveparticlealgo.process( iolcv, ioll );
    
    
    // PREP SETS OF HITS
    // ------------------
    prepSpacepoints( iolcv, ioll  );
    
    // Make keypoint candidates from larmatch vertex
    // ---------------------------------------------
    recoKeypoints( iolcv, ioll );

    if ( false ) {
      // for debug
      _ana_run = ev_adc->run();
      _ana_subrun = ev_adc->subrun();
      _ana_event  = ev_adc->event();
      _ana_tree->Fill();
      return;
    }
      
    // PARTICLE FRAGMENT RECO
    clusterSubparticleFragments( iolcv, ioll );
    
    // COSMIC RECO
    _cosmic_track_builder.clear();
    //_cosmic_track_builder.set_verbosity( larcv::msg::kDEBUG );
    _cosmic_track_builder.set_verbosity( larcv::msg::kINFO );    
    _cosmic_track_builder.do_boundary_analysis( true );
    _cosmic_track_builder.process( iolcv, ioll );

    _cosmic_proton_finder.set_verbosity( larcv::msg::kDEBUG );
    _cosmic_proton_finder.process( iolcv, ioll );

    // MULTI-PRONG INTERNAL RECO
    multiProngReco( iolcv, ioll );
    
    // Single particle interactions

    // Copy larlite contents
    // in-time opflash
    larlite::event_opflash* ev_input_opflash_beam =
      (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"simpleFlashBeam");
    larlite::event_opflash* evout_opflash_beam =
      (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"simpleFlashBeam");
    for ( auto const& flash : *ev_input_opflash_beam )
      evout_opflash_beam->push_back( flash );

    // make selection variables
    makeNuCandidateSelectionVariables( iolcv, ioll );
    
    if ( _save_event_mc_info ) {
      _event_mcinfo_maker.process( ioll );
      truthAna( iolcv, ioll );
    }

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
    _ana_tree->Fill();
    
  }

  /**
   * @brief algorithms for splitting up and filtering larmatch space points
   *
   */
  void KPSRecoManager::prepSpacepoints( larcv::IOManager& iolcv,
                                        larlite::storage_manager& ioll )
  {

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
    _kpreco_nu.set_lfhit_score_index( 13 );
    _kpreco_nu.process( ioll );

    _kpreco_track.set_input_larmatch_tree_name( "taggerfilterhit" );
    _kpreco_track.set_output_tree_name( "keypoint" );        
    _kpreco_track.set_sigma( 10.0 );    
    _kpreco_track.set_min_cluster_size(   50.0, 0 );
    _kpreco_track.set_keypoint_threshold( 0.5, 0 );
    _kpreco_track.set_min_cluster_size(   20.0, 1 );    
    _kpreco_track.set_keypoint_threshold( 0.5, 1 );    
    _kpreco_track.set_larmatch_threshold( 0.5 );
    _kpreco_track.set_keypoint_type( (int)larflow::kTrackEnds );
    _kpreco_track.set_lfhit_score_index( 14 );
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
    _kpreco_shower.set_lfhit_score_index( 15 );
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
    _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackEnds );
    _kpreco_track_cosmic.set_lfhit_score_index( 14 );    
    _kpreco_track_cosmic.process( ioll );

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
    const float _maxdist = 2.0;
    const float _minsize = 10;
    const float _maxkd   = 100;
    LARCV_INFO() << "RUN PROJ-SPLITTER ON: maxtrackhit_wcfilter (in-time track hits)" << std::endl;
    _projsplitter.set_verbosity( larcv::msg::kDEBUG );    
    //_projsplitter.set_verbosity( larcv::msg::kINFO );    
    _projsplitter.set_dbscan_pars( _maxdist, _minsize, _maxkd );
    _projsplitter.set_fit_line_segments_to_clusters( true );
    _projsplitter.set_input_larmatchhit_tree_name( "maxtrackhit_wcfilter" );
    _projsplitter.add_input_keypoint_treename_for_hitveto( "keypoint" );
    _projsplitter.set_output_tree_name("trackprojsplit_wcfilter");
    _projsplitter.process( iolcv, ioll );

    // PRIMITIVE TRACK FRAGMENTS: FULL TRACK HITS
    LARCV_INFO() << "RUN PROJ-SPLITTER ON: full_maxtrackhit (out-of-time hits)" << std::endl;    
    //_projsplitter_cosmic.set_verbosity( larcv::msg::kINFO );
    _projsplitter_cosmic.set_verbosity( larcv::msg::kDEBUG );    
    _projsplitter_cosmic.set_input_larmatchhit_tree_name( "full_maxtrackhit" );
    _projsplitter_cosmic.set_fit_line_segments_to_clusters( true );    
    _projsplitter_cosmic.set_output_tree_name("trackprojsplit_full");
    _projsplitter_cosmic.process( iolcv, ioll );

    // SHOWER 1-KP RECO: make shower using clusters and single keypoint
    _showerkp.set_ssnet_lfhit_tree_name( "maxshowerhit" );
    //_showerkp.set_verbosity( larcv::msg::kDEBUG );
    _showerkp.set_verbosity( larcv::msg::kINFO );    
    _showerkp.process( iolcv, ioll );

    // SHORT HIP FRAGMENTS
    _short_proton_reco.set_verbosity( larcv::msg::kDEBUG );
    _short_proton_reco.clear_clustertree_checklist();
    _short_proton_reco.add_clustertree_forcheck( "trackprojsplit_wcfilter" );
    _short_proton_reco.process( iolcv, ioll );
    
    // TRACK CLUSTER-ONLY RECO: make tracks without use of keypoints

    // SHOWER CLUSTER-ONLY RECO: make showers without use of keypoints
    
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
    std::vector<std::string> input_hit_list
      = {"taggerfilterhit",            // all in-time hits
         "ssnetsplit_full_showerhit"}; // out-of-time shower hits
    std::vector<std::string> input_cluster_list
      = { "trackprojsplit_full"}; // in-time track clusters

    _nuvertexactivity.set_input_hit_list( input_hit_list );    
    _nuvertexactivity.set_input_cluster_list( input_cluster_list );
    _nuvertexactivity.set_output_treename( "keypoint" );
    _nuvertexactivity.process( iolcv, ioll );
    
    _nuvertexmaker.set_verbosity( larcv::msg::kDEBUG );
    //_nuvertexmaker.set_verbosity( larcv::msg::kINFO );    
    _nuvertexmaker.clear();
    _nuvertexmaker.add_keypoint_producer( "keypoint" );
    _nuvertexmaker.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    _nuvertexmaker.add_cluster_producer("cosmicproton", NuVertexCandidate::kTrack );
    _nuvertexmaker.add_cluster_producer("hip", NuVertexCandidate::kTrack );    
    _nuvertexmaker.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    _nuvertexmaker.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );
    
    _nuvertexmaker.apply_cosmic_veto( true );
    _nuvertexmaker.process( iolcv, ioll );

    // NuTrackBuilder class
    _nu_track_builder.clear();
    _nu_track_builder.set_verbosity( larcv::msg::kDEBUG );
    _nu_track_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    // first attempt
    // _nu_shower_builder.set_verbosity( larcv::msg::kDEBUG );
    // _nu_shower_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    // simpler, cone-based reco
    _nuvertex_shower_reco.set_verbosity( larcv::msg::kDEBUG );
    _nuvertex_shower_reco.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    _nuvertex_shower_reco.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    _nuvertex_shower_reco.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );    
    _nuvertex_shower_reco.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    // repair shower trunks by absorbing tracks or creating hits
    _nuvertex_shower_trunk_check.set_verbosity( larcv::msg::kDEBUG );
    for ( auto& vtx : _nuvertexmaker.get_mutable_fitted_candidates() )
      _nuvertex_shower_trunk_check.checkNuCandidateProngs( vtx );
    
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

    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_fitted_candidates();    
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

    std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v = _nuvertexmaker.get_mutable_fitted_candidates();
    LARCV_INFO() << "Make Selection Variables for " << nuvtx_v.size() << " candidates" << std::endl;

    NuSelProngVars prongvars;
    NuSelVertexVars vertexvars;
    vertexvars.set_verbosity(larcv::msg::kDEBUG);
    
    for ( size_t ivtx=0; ivtx<nuvtx_v.size(); ivtx++ ) {

      // nu candidate
      larflow::reco::NuVertexCandidate& nuvtx = nuvtx_v[ivtx];
      
      // make selection variables
      larflow::reco::NuSelectionVariables nusel;

      std::cout << "===[ VERTEX " << ivtx << " ]===" << std::endl;
      std::cout << "  pos (" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
      std::cout << "  number of tracks: "  << nuvtx.track_v.size() << std::endl;
      std::cout << "  number of showers: " << nuvtx.shower_v.size() << std::endl;

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
      vertexvars.analyze( iolcv, ioll, nuvtx, nusel );
      
      // nu kinematic variables
      _nu_sel_v.emplace_back( std::move(nusel) );

    }//end of vertex loop

    LARCV_INFO() << "Selection variables made: " << _nu_sel_v.size() << std::endl;
    
  }
  
}
}
