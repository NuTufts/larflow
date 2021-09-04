#include "CRTTruthMatchManager.h"

// larlite
#include "larlite/DataFormat/opflash.h"
#include "larlite/DataFormat/larflowcluster.h"

#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"

namespace larflow {
namespace crtmatch {

  /** 
   * @brief constructor where an output file is made 
   *
   * @param[in] inputfile_name Name of output file for non-larcv and non-larlite reco products
   */
  CRTTruthMatchManager::CRTTruthMatchManager( std::string inputfile_name )
    : larcv::larcv_base("CRTTruthMatchManager"),
    // _save_event_mc_info(false),
    _ana_output_file(inputfile_name)    
  {
    make_ana_file();
  }

  CRTTruthMatchManager::~CRTTruthMatchManager()
  {
  }

  /**
   * @brief process event data in larcv and larlite IO managers 
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void CRTTruthMatchManager::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll )
  {


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

    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraph( iolcv, ioll );

    // larflow::reco::ShowerLikelihoodBuilder sllb;
    // sllb.process( iolcv, ioll );
    // sllb.updateMCPixelGraph( mcpg, iolcv );
    
    std::vector< larlite::larflowcluster > truth_cluster_v;
    //void makeTruthTrackClusters( iolcv, ioll, mcpg );
    
    // // PREP SETS OF HITS
    // // ------------------
    
    // // PREP: SPLIT HITS INTO TRACK/SHOWER
    // // input:
    // //  * image2d_ubspurn_planeX: ssnet (track,shower) scores
    // //  * larflow3dhit_larmatch_tree: output of KPS larmatch network
    // // output:
    // //  * larflow3dhit in 'ssnetsplit_wcfilter_trackhit'
    // //  * larflow3dhit in 'ssnetsplit_wcfilter_showerhit'
    // _splithits_ssnet.set_larmatch_tree_name( "larmatch" );
    // _splithits_ssnet.set_output_tree_stem_name( "ssnetsplitall" );    
    // _splithits_ssnet.process( iolcv, ioll );

    // // PREP: ENFORCE UNIQUE PIXEL PREDICTION USING MAX SCORE FOR TRACK HITS
    // // output:
    // //  * larflow3dhit in 'maxtrackhit_wcfilter'
    // _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplitall_trackhit" );
    // _choosemaxhit.set_output_larflow3dhit_treename( "maxtrackhit_all" );
    // _choosemaxhit.set_verbosity( larcv::msg::kINFO );
    // _choosemaxhit.process( iolcv, ioll );
    
    // // MAKE KEYPOINTS -- WILL USE 'mactrackhit_all' hits
    // recoKeypoints( iolcv, ioll );

    // if ( false ) {
    //   // for debug
    //   _ana_run = ev_adc->run();
    //   _ana_subrun = ev_adc->subrun();
    //   _ana_event  = ev_adc->event();
    //   _ana_tree->Fill();
    //   return;
    // }
      
    // // PARTICLE FRAGMENT RECO
    // recoParticles( iolcv, ioll );

    // // At this point we have track clusters in 'cosmictrackclusters' tree
    
    // // // COSMIC RECO
    // // _cosmic_track_builder.clear();
    // // //_cosmic_track_builder.set_verbosity( larcv::msg::kDEBUG );
    // // _cosmic_track_builder.set_verbosity( larcv::msg::kINFO );    
    // // _cosmic_track_builder.do_boundary_analysis( true );
    // // _cosmic_track_builder.process( iolcv, ioll );

    // // // MULTI-PRONG INTERNAL RECO
    // // multiProngReco( iolcv, ioll );
    
    // // // Single particle interactions

    // // // Copy larlite contents
    // // // in-time opflash
    // // larlite::event_opflash* ev_input_opflash_beam =
    // //   (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"simpleFlashBeam");
    // // larlite::event_opflash* evout_opflash_beam =
    // //   (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"simpleFlashBeam");
    // // for ( auto const& flash : *ev_input_opflash_beam )
    // //   evout_opflash_beam->push_back( flash );
    
    // // if ( _save_event_mc_info ) {
    // //   _event_mcinfo_maker.process( ioll );
    // //   truthAna( iolcv, ioll );
    // // }
    
    // // Fill Ana Tree
    // _ana_run = ev_adc->run();
    // _ana_subrun = ev_adc->subrun();
    // _ana_event  = ev_adc->event();
    // _ana_tree->Fill();
    
  }

  /**
   * @brief make keypoints for use to help make particle track and nu interaction candidates
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void CRTTruthMatchManager::recoKeypoints( larcv::IOManager& iolcv,
                                      larlite::storage_manager& ioll )
  {

    // KEYPOINT RECO: make keypoint candidates
    //  * larflow3dhit_larmatch_tree: output of KPS larmatch network
    // output:
    //  * _kpreco.output_pt_v: container of KPCluster objects

    // neutrino
    _kpreco_track_cosmic.set_input_larmatch_tree_name( "maxtrackhit_all" );
    _kpreco_track_cosmic.set_output_tree_name( "kpcosmic_all" );
    _kpreco_track_cosmic.set_sigma( 50.0 );    
    _kpreco_track_cosmic.set_min_cluster_size(   50.0, 0 );
    _kpreco_track_cosmic.set_max_dbscan_dist( 10.0 );
    _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 0 );
    _kpreco_track_cosmic.set_min_cluster_size(   20.0, 1 );    
    _kpreco_track_cosmic.set_keypoint_threshold( 0.5, 1 );    
    _kpreco_track_cosmic.set_larmatch_threshold( 0.5 );
    _kpreco_track_cosmic.set_keypoint_type( (int)larflow::kTrackEnd );
    _kpreco_track_cosmic.set_lfhit_score_index( 14 );    
    _kpreco_track_cosmic.process( ioll );

  }

  /**
   * @brief reconstruct tracks and showers
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void CRTTruthMatchManager::recoParticles( larcv::IOManager& iolcv,
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

    // PRIMITIVE TRACK FRAGMENTS
    const float _maxdist = 2.0;
    const float _minsize = 20;
    const float _maxkd   = 100;
    LARCV_INFO() << "RUN PROJ-SPLITTER ON: maxtrackhit_wcfilter (in-time hits)" << std::endl;
    _projsplitter.set_verbosity( larcv::msg::kDEBUG );    
    //_projsplitter.set_verbosity( larcv::msg::kINFO );    
    _projsplitter.set_dbscan_pars( _maxdist, _minsize, _maxkd );
    _projsplitter.set_fit_line_segments_to_clusters( true );
    _projsplitter.set_input_larmatchhit_tree_name( "maxtrackhit_all" );
    _projsplitter.add_input_keypoint_treename_for_hitveto( "kpcosmic_all" );
    _projsplitter.set_output_tree_name("cosmictrackclusters");
    _projsplitter.process( iolcv, ioll );
    
  }

  /**
   * @brief reconstruct tracks and showers attached to vertices
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */  
  void CRTTruthMatchManager::multiProngReco( larcv::IOManager& iolcv,
                                       larlite::storage_manager& ioll )
  {

    // //_nuvertexmaker.set_verbosity( larcv::msg::kDEBUG );
    // _nuvertexmaker.set_verbosity( larcv::msg::kINFO );    
    // _nuvertexmaker.clear();
    // _nuvertexmaker.add_keypoint_producer( "keypoint" );
    // _nuvertexmaker.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    // _nuvertexmaker.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    // _nuvertexmaker.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );
    // _nuvertexmaker.apply_cosmic_veto( true );
    // _nuvertexmaker.process( iolcv, ioll );

    // // NuTrackBuilder class
    // _nu_track_builder.clear();
    // _nu_track_builder.set_verbosity( larcv::msg::kDEBUG );
    // _nu_track_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    // // first attempt
    // // _nu_shower_builder.set_verbosity( larcv::msg::kDEBUG );
    // // _nu_shower_builder.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );

    // // simpler, cone-based reco
    // _nuvertex_shower_reco.set_verbosity( larcv::msg::kDEBUG );
    // _nuvertex_shower_reco.add_cluster_producer("trackprojsplit_wcfilter", NuVertexCandidate::kTrack );
    // _nuvertex_shower_reco.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    // _nuvertex_shower_reco.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );    
    // _nuvertex_shower_reco.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );
    
  }

  /**
   * @brief create ana file and define output tree
   *
   * The tree created is `CRTTruthMatchManagerTree`.
   *
   */
  void CRTTruthMatchManager::make_ana_file()
  {
    _ana_file = new TFile(_ana_output_file.c_str(), "recreate");
    _ana_tree = new TTree("CRTTruthMatchManagerTree","Ana Output of CRTTruthMatchManager algorithms");
    _ana_tree->Branch("run",&_ana_run,"run/I");
    _ana_tree->Branch("subrun",&_ana_subrun,"subrun/I");
    _ana_tree->Branch("event",&_ana_event,"event/I");    
  }

  /** @brief is true, save MC event summary */  
  void CRTTruthMatchManager::saveEventMCinfo( bool savemc )
  {
    if ( !_save_event_mc_info && savemc )  {
      //_track_truthreco_ana.bindAnaVariables( _ana_tree );
      _event_mcinfo_maker.bindAnaVariables( _ana_tree );
    }
    _save_event_mc_info = savemc;

  };

  /** @brief run Truth-Reco analyses for studying performance **/
  void CRTTruthMatchManager::truthAna( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    //_track_truthreco_ana.set_verbosity( larcv::msg::kDEBUG );
    //_track_truthreco_ana.process( iolcv, ioll, _nuvertexmaker.get_mutable_fitted_candidates() );
  }

  // void CRTTruthMatchManager::makeTruthTrackClusters( larcv::IOManager& iolcv,
  //                                                    larlite::storage_manager& ioll,
  //                                                    ublarcvapp::mctools::MCPixelPGraph& mcpg )
  // {
  //   // do dump loop for now. (larmatch hit x primary graph).
  //   // but at least can build bounding boxes to speed comparisons.

  //   // build primary bounding box lists
  //   typedef std::vector< std::vector<float> > BBox_t;
  //   std::vector< BBox_t > primary_bbox_v;

    
  // }

  /** @brief store match information into larcv and larlite format
   *
   * not yet implemented.
   *
   * @param[out] outlcv Store data into larcv IO manager
   * @param[out] outll  Store data into larlite IO manager
   * @param[in]  remove_if_no_flash Does not store tracks if no flash-match made to CRT hit/track [default: true]
   */
  void CRTTruthMatchManager::store_output( larcv::IOManager& outlcv,
                                           larlite::storage_manager& outll,
                                           bool remove_if_no_flash )
  {
  }
  
  
}
}
