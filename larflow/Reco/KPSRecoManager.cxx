#include "KPSRecoManager.h"

#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventImage2D.h"


namespace larflow {
namespace reco {

  KPSRecoManager::KPSRecoManager()
    : larcv::larcv_base("KPSRecoManager")
  {    
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

    // PREP: form clusters for both track and shower
    // _cluster_track.set_input_larflowhit_tree_name(  "trackhit" );
    // _cluster_track.set_output_larflowhit_tree_name( "lmtrack" );    
    // _cluster_track.process( iolcv, ioll );
    // _cluster_shower.set_input_larflowhit_tree_name(  "showerhit" );
    // _cluster_shower.set_output_larflowhit_tree_name( "lmshower" );
    // _cluster_shower.process( iolcv, ioll );

    // KEYPOINT RECO: make keypoint candidates
    //  * larflow3dhit_larmatch_tree: output of KPS larmatch network
    // output:
    //  * _kpreco.output_pt_v: container of KPCluster objects
    _kpreco.set_min_cluster_size(   50.0, 0 );
    _kpreco.set_keypoint_threshold( 0.5, 0 );
    _kpreco.set_min_cluster_size(   20.0, 1 );    
    _kpreco.set_keypoint_threshold( 0.5, 1 );    
    _kpreco.set_larmatch_threshold( 0.5 );
    _kpreco.process( ioll );

    // FILTER LARMATCH POINTS USING TAGGER
    _wcfilter.set_verbosity( larcv::msg::kINFO );
    _wcfilter.set_input_larmatch_tree_name( "larmatch" );
    _wcfilter.process( iolcv, ioll );
    
    // FILTER KEYPOINTS: To be on clusters larger than X hits
    _kpfilter.set_verbosity( larcv::msg::kDEBUG );
    _kpfilter.set_input_keypoint_tree_name( "taggerfilterkeypoint" );
    _kpfilter.set_input_larflowhits_tree_name( "taggerfilterhit" );
    _kpfilter.process( iolcv, ioll );

    
    // PREP: split hits into shower and track hits
    // input:
    //  * image2d_ubspurn_planeX: ssnet (track,shower) scores
    //  * larflow3dhit_larmatch_tree: output of KPS larmatch network
    // output:
    //  * larflow3dhit_showerhit_tree
    //  * larflow3dhit_trackhit_tree
    _splithits.set_larmatch_tree_name( "taggerfilterhit" );
    _splithits.process( iolcv, ioll );
  
    // PARTICLE RECO
    recoParticles( iolcv, ioll );
    
    // INTERACTION RECO

    // Cosmic reco: tracks at end points + deltas and michels

    // Multi-prong internal reco

    // Single particle interactions
    
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

    // SHOWER 1-KP RECO: make shower using clusters and single keypoint
    _showerkp.set_ssnet_lfhit_tree_name( "showerhit" );
    _showerkp.set_verbosity( larcv::msg::kDEBUG );
    _showerkp.process( iolcv, ioll );

    // TRACK CLUSTER-ONLY RECO: make tracks without use of keypoints

    // SHOWER CLUSTER-ONLY RECO: make showers without use of keypoints
    
  }
  
}
}
