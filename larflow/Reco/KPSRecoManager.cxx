#include "KPSRecoManager.h"

// larlite
#include "DataFormat/opflash.h"

#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"

namespace larflow {
namespace reco {

  KPSRecoManager::KPSRecoManager( std::string inputfile_name )
    : larcv::larcv_base("KPSRecoManager"),
    _ana_input_file(inputfile_name)
  {
    make_ana_file();
    _nuvertexmaker.add_nuvertex_branch( _ana_tree );
  }

  KPSRecoManager::~KPSRecoManager()
  {
  }
  
  void KPSRecoManager::process( larcv::IOManager& iolcv,
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

    // PREP SETS OF HITS
    // ------------------
    
    // PREP WC-FILTERED HITS
    _wcfilter.set_verbosity( larcv::msg::kINFO );
    _wcfilter.set_input_larmatch_tree_name( "larmatch" );
    _wcfilter.set_output_filteredhits_tree_name( "taggerfilterhit" );
    _wcfilter.set_save_rejected_hits( true );
    _wcfilter.process_hits( iolcv, ioll );

    // PREP: SPLIT WC-FILTERED HITS INTO TRACK/SHOWER
    // input:
    //  * image2d_ubspurn_planeX: ssnet (track,shower) scores
    //  * larflow3dhit_larmatch_tree: output of KPS larmatch network
    // output:
    //  * larflow3dhit_showerhit_tree
    //  * larflow3dhit_trackhit_tree
    _splithits_wcfilter.set_larmatch_tree_name( "taggerfilterhit" );
    _splithits_wcfilter.set_output_tree_stem_name( "ssnetsplit_wcfilter" );    
    _splithits_wcfilter.process( iolcv, ioll );    

    // PREP: ENFORCE UNIQUE PIXEL PREDICTION USING MAX SCORE FOR TRACK HITS
    _choosemaxhit.set_input_larflow3dhit_treename( "ssnetsplit_wcfilter_trackhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "maxtrackhit_wcfilter" );
    _choosemaxhit.set_verbosity( larcv::msg::kINFO );
    _choosemaxhit.process( iolcv, ioll );

    // PREP: SPLIT SHOWER/TRACK FOR FULL HIT SET
    _splithits_full.set_larmatch_tree_name( "larmatch" );
    _splithits_full.set_output_tree_stem_name( "ssnetsplit_full" );
    _splithits_full.process( iolcv, ioll );
    
    // Make keypoints
    recoKeypoints( iolcv, ioll );

    if ( true ) {
      // for debug
      _ana_run = ev_adc->run();
      _ana_subrun = ev_adc->subrun();
      _ana_event  = ev_adc->event();
      _ana_tree->Fill();
      return;
    }
    
    // FILTER KEYPOINTS: To be on clusters larger than X hits
    // _kpfilter.set_verbosity( larcv::msg::kDEBUG );
    // _kpfilter.set_input_keypoint_tree_name( "taggerfilterkeypoint" );
    // _kpfilter.set_input_larflowhits_tree_name( "taggerfilterhit" );
    // _kpfilter.process( iolcv, ioll );

    
  
    // PARTICLE RECO
    recoParticles( iolcv, ioll );
    
    // INTERACTION RECO

    // Cosmic reco: tracks at end points + deltas and michels

    // Multi-prong internal reco
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
    

    // Fill Ana Tree
    _ana_run = ev_adc->run();
    _ana_subrun = ev_adc->subrun();
    _ana_event  = ev_adc->event();
    _ana_tree->Fill();
    
  }

  /**
   * make keypoints for use to help make particle track and nu interaction candidates
   *
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

  void KPSRecoManager::recoParticles( larcv::IOManager& iolcv,
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

    // SPLIT TRACK CLUSTERS
    _projsplitter.set_verbosity( larcv::msg::kDEBUG );
    _projsplitter.set_input_larmatchhit_tree_name( "maxtrackhit" );
    _projsplitter.set_output_tree_name("trackprojsplit");
    _projsplitter.process( iolcv, ioll );

    // PCA-TRACKER
    // _pcatracker.set_verbosity( larcv::msg::kDEBUG );
    // _pcatracker.process( iolcv, ioll );

    // CLEAN UP SHOWER HITS BY CHOOSING MAX FROM Y-plane
    _choosemaxhit.set_input_larflow3dhit_treename( "showerhit" );
    _choosemaxhit.set_output_larflow3dhit_treename( "maxshowerhit" );
    _choosemaxhit.set_verbosity( larcv::msg::kINFO );
    _choosemaxhit.process( iolcv, ioll );

    // SHOWER 1-KP RECO: make shower using clusters and single keypoint
    _showerkp.set_ssnet_lfhit_tree_name( "maxshowerhit" );
    _showerkp.set_verbosity( larcv::msg::kDEBUG );
    _showerkp.process( iolcv, ioll );

    // TRACK CLUSTER-ONLY RECO: make tracks without use of keypoints

    // SHOWER CLUSTER-ONLY RECO: make showers without use of keypoints
    
  }

  void KPSRecoManager::multiProngReco( larcv::IOManager& iolcv,
                                       larlite::storage_manager& ioll )
  {

    _nuvertexmaker.set_verbosity( larcv::msg::kDEBUG );
    _nuvertexmaker.clear();
    //_nuvertexmaker.add_keypoint_producer( "keypoint_bigcluster" );
    //_nuvertexmaker.add_keypoint_producer( "keypoint_smallcluster" );
    _nuvertexmaker.add_keypoint_producer( "keypoint" );
    _nuvertexmaker.add_cluster_producer("trackprojsplit", NuVertexCandidate::kTrack );
    _nuvertexmaker.add_cluster_producer("showerkp", NuVertexCandidate::kShowerKP );
    _nuvertexmaker.add_cluster_producer("showergoodhit", NuVertexCandidate::kShower );
    _nuvertexmaker.process( iolcv, ioll );
    
  }

  /**
   * create ana file and define output tree
   *
   */
  void KPSRecoManager::make_ana_file()
  {
    _ana_file = new TFile(_ana_input_file.c_str(), "recreate");
    _ana_tree = new TTree("KPSRecoManagerTree","Ana Output of KPSRecoManager algorithms");
    _ana_tree->Branch("run",&_ana_run,"run/I");
    _ana_tree->Branch("subrun",&_ana_subrun,"subrun/I");
    _ana_tree->Branch("event",&_ana_event,"event/I");    
  }
  
}
}
