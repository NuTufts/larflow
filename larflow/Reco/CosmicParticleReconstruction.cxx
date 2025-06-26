#include "CosmicParticleReconstruction.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "ublarcvapp/Reco3D/TrackReverser.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "TrackdQdx.h"
#include "SplitHitsBySSNet.h"
#include "KeypointFilterByWCTagger.h"
#include "ChooseMaxLArFlowHit.h"
#include "KeypointReco.h"

namespace larflow {
namespace reco {

  void CosmicParticleReconstruction::set_default_param_values()
  {
    _flash_producer   = "simpleFlashCosmic";
    _wireimg_producer = "wire";
    _outoftime_tagged_pixels_producer = "thrumu";
    _larmatch_hit_producer   = "larmatch";
  }

  void CosmicParticleReconstruction::clear()
  {
    _cosmic_candidates_v.clear();
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
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "ssnetsplit_offtrigger_trackhit" ); /// track-like and out of time
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "keypointcosmic" ); /// cosmic keypoints
    ioll.set_data_to_write( larlite::data::kCRTTrack, "crttrack");

    // Stages

    // PrepSpacepoints: isolate out-of-time spacepoints using the out-of-time tagger using 
    // passing spacepoints are stored in the larlite storage_manager with the treename 'cosmicreco'
    prepSpacepoints( iolcv, ioll );

    // Reconstruct Track-Start and Track-End Keypoints using the larmatch info in the spacepoints
    recoKeypoints( iolcv, ioll );

    // // isolate track-like spacepoints and reconstruct into line-like segments
    // buildTrackFragments();

    // // use the CosmicTrackBuilder to make muon candidates
    // buildCosmicTracks();

    // // make flash predictions and make possible matches
    // makeFlashPredictionAndMatches();

    // // make CRT connections
    // makeCRTConnections();

    

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
    //_splithits_wcfilter.set_larmatch_tree_name( _spacepoint_input_container_name ); //< why by-pass cosmic removal?
    _splithits_wcfilter.set_larmatch_tree_name( "taggerfilterhit"  );
    _splithits_wcfilter.set_output_tree_stem_name( "ssnetsplit_wcfilter" );
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
    larflow::reco::KeypointReco  _kpreco_trackend;   ///< reconstruct keypoints from network scores for track class
      
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
    }

    larlite::event_larflow3dhit* ev_kpall
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint_all" );
    LARCV_INFO() << "Number of total track-start + track-end keypoints reconstructed: " << ev_kpall->size() << std::endl;

    
    // // filter out keypoints by in-time and cosmic
    // larlite::event_larflow3dhit* ev_kpintime = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint" );
    // larlite::event_pcaxis* ev_kp_pca = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "keypoint" );    
    // larlite::event_larflow3dhit* ev_kpcosmic = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypointcosmic" );
    // larlite::event_pcaxis* ev_kp_pca_cosmic = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "keypointcosmic" );

    // larcv::EventImage2D* ev_image2d_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "thrumu" );
    // int nplanes = ev_image2d_v->as_vector().size();

    // std::vector< larflow::reco::KeypointReco* > kpreco_v
    //   = { &_kpreco_trackstart,
	  //       &_kpreco_trackend };
    
    // // loop over algos for each keypoint class
    // int intime_cluster_index = 0;
    // int cosmic_cluster_index = 0;
    // for ( auto& pkpreco : kpreco_v ) {
    //   // loop over reco keypoints
    //   for ( auto const& kpc : pkpreco->output_pt_v ) {
	  //     // cut on max value keypoint score
	  //     if ( kpc.max_score < 0.5 ) // 0.7 too strong?
	  //       continue;
	      
	  //     // get if near a cosmic-tagged pixel
	  //     float thrumu_pixsum_allplanes = 0.;
	  //     std::vector<float> thrumu_pixsum(nplanes,0);
	  //     for (int p=0; p<3; p++) {
	  //       thrumu_pixsum[p] = _pt_image_projection.getPixelSumAroundProjPoint( kpc.max_pt_v, ev_image2d_v->as_vector().at(p), 2, 10.0 );
	  //       thrumu_pixsum_allplanes += thrumu_pixsum[p];
	  //     }
      
	  //     /// make larflow3dhit version and add thrumu projection info.
	  //     larlite::larflow3dhit kphit = kpc.as_larflow_hit();
	  //     kphit.push_back( thrumu_pixsum_allplanes );	
	  //     for (int p=0; p<3; p++)
	  //       kphit.push_back( thrumu_pixsum[p] );
      
	  //     if ( thrumu_pixsum_allplanes < 50.0 ) {
	  //       // then ok to pass on as potential nu candidate
	  //       ev_kpintime->push_back( kphit );
	  //       ev_kp_pca->push_back( kpc.get_pcaxis( intime_cluster_index ) );
	  //       intime_cluster_index++;
	  //     }
	  //     else {
	  //       // assign as comics
	  //       ev_kpcosmic->push_back( kphit );
	  //       ev_kp_pca_cosmic->push_back( kpc.get_pcaxis( cosmic_cluster_index ) );
	  //       cosmic_cluster_index++;
	  //     }
    //   }
    // }

    // // filter duplicates for intime
    // std::vector<int> intime_kp_status( ev_kpintime->size(), 1 );
    
    // for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {

    //   auto const& hit = ev_kpintime->at(ikp);
    //   int kp_type = int(hit[3]);
      
    //   // recursive check with those before
    //   for (int jkp=0; jkp<ikp; jkp++) {
	  //     if ( intime_kp_status[jkp]==0 ) {
	  //       // already filtered. skip.
	  //       continue;
	  //     }
	  //     auto const& past_hit = ev_kpintime->at(jkp);
	  //     int past_type = int(past_hit[3]);
	      
	  //     // if the same type, don't do the duplicate removal test
	  //     if ( kp_type==past_type ) {
	  //       continue;
	  //     }
      
	  //     float dist = 0.;
	  //     float dx = 0.;
	  //     for (int i=0; i<3; i++) {
	  //       dx = (past_hit[i]-hit[i]);
	  //       dist += dx*dx;
	  //     }
	  //     dist = sqrt(dist);
      
	  //     if ( dist>3.0 ) {
	  //       // no overlap
	  //       continue;
	  //     }
      
	  //     if ( past_type==0 && kp_type!=0 ) {
	  //       // past type is nu vertex. we de-activate in favor of that vertex
	  //       intime_kp_status[ikp] = 0;
	  //       break;
	  //     }
	  //     else if ( kp_type==0 && past_type!=0 ) {
	  //       // current keypoint is nu-type. deactivate past vertex
	  //       intime_kp_status[jkp] = 0;
	  //       // keep going
	  //     }
	  //     else if ( (kp_type==1 && past_type==2 )
	  //     	  || (kp_type==2 && past_type==1 ) ) {
	  //       // comparison between track start and track end
	  //       // if we're really close, then go with start label. will use to seed neutrino.
	  //       if ( dist<0.7 ) {
	  //         if ( kp_type==2 ) {
	  //           intime_kp_status[ikp] = 0;
	  //           break; // current kp has been deactivated. stop.
	  //         }
	  //         else if (past_type==2) {
	  //           intime_kp_status[jkp] = 0;
	  //           // keep going
	  //         }
	  //       }
	  //     }
	  //     else if ( (kp_type==3 && (past_type==1 || past_type==2))
	  //     	  || (past_type==3 && (kp_type==1 || kp_type==2)) ) {
    //       // shower keypoints override and remove track end and track start keypoints
	  //       if ( dist<0.7 ) {
	  //         if ( kp_type!=3 ) {
	  //           intime_kp_status[ikp] = 0;
	  //           break; // current kp has been deactivated. stop.
	  //         }
	  //         else if ( past_type!=3 ) {
	  //           intime_kp_status[jkp] = 0;
	  //           // keep-going
	  //         }
	  //       }
	  //     }//end of case overlap loop
    //   }
    // }
    
    // //std::vector<int> intime_kp_status( ev_kpintime->size(), 1 );
    // int num_deactivated = 0;
    // for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {
    //   if ( intime_kp_status[ikp]==0 ) {
	  //     num_deactivated++;
	  //     break;
    //   }
    // }

    // if ( num_deactivated>0 ) {
      
    //   std::vector< larlite::larflow3dhit > passing_keypoints;
    //   std::vector< larlite::pcaxis > passing_pcaxis;
    //   for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {
	  //      if ( intime_kp_status[ikp]==1 ) {
	  //        passing_keypoints.push_back( ev_kpintime->at(ikp) );
	  //        passing_pcaxis.push_back( ev_kp_pca->at(ikp) );
	  //      }
    //   }

    //   ev_kpintime->clear();
    //   ev_kp_pca->clear();
    //   for (int ikp=0; ikp<(int)passing_keypoints.size(); ikp++) {
	  //     ev_kpintime->push_back( passing_keypoints.at(ikp) );
	  //     ev_kp_pca->push_back( passing_pcaxis.at(ikp) );
    //   }
    //   LARCV_NORMAL() << "After cross-type duplicate filter. Number of intime keypoints: " << ev_kpintime->size() << std::endl;
    // }
    
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
      

    // LARCV_NORMAL() << "Num nu vertex candidates [keypoint]: " << ev_kpintime->size() << std::endl;
    // LARCV_NORMAL() << "Num cosmic vertex candidates [keypoint]: " << ev_kpcosmic->size() << std::endl;
    // for (int ikp=0; ikp<(int)ev_kpintime->size(); ikp++ ) {
    //   auto const& kphit = ev_kpintime->at(ikp);
    //   int kptype = -1;
    //   float maxkpscore = -1;
    //   float thrumupixsum = -1;
    //   if ( kphit.size()>=4 ) {
    //     kptype = kphit.at(3);
	  //     maxkpscore = kphit.at(4);
	  //     thrumupixsum = kphit.at(5);
    //   }
    //   LARCV_NORMAL() << " [" << ikp << "] type=" << kptype << " maxscore=" << maxkpscore << " cosmicpixsum=" << thrumupixsum << std::endl;
    // }
    
    // if ( _save_keypoints_in_anafile ) {
    //   for ( auto& pkprecotype : kpreco_v ) {
	  //     for ( auto& kpc : pkprecotype->output_pt_v ) {
	  //       if ( kpc._cluster_type==0 )
	  //         _event_kpc_nu_v.push_back( kpc );
	  //       else if ( kpc._cluster_type==1 || kpc._cluster_type==2 )
	  //         _event_kpc_track_v.push_back( kpc );
	  //       else if ( kpc._cluster_type>=3 )
	  //         _event_kpc_shower_v.push_back( kpc );
	  //     }
    //     // for ( auto& kpc : _kpreco_track_cosmic.output_pt_v  )
    //     //_event_kpc_cosmic_v.push_back( kpc );
    //   }
    // }
    
  }

}
}
