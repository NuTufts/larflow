#include "MRCNNCosmicReco.h"

#include "larflow/Reco/SplitHitsByParticleSSNet.h"

namespace larflow {
namespace mrcnnreco {


  void MRCNNCosmicReco::process(larcv::IOManager& iolcv,
				larlite::storage_manager& ioll) 
  {
    // we follow the gen2 reco path
    //make_ana_file();
    std::cout << "ana: " << _ana_output_file << " " << _ana_file << " " << _ana_tree << std::endl;

    // PREP: make bad channel image
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();
    
    larcv::EventChStatus* ev_chstatus =
      (larcv::EventChStatus*)iolcv.get_data(larcv::kProductChStatus, "wire");

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

    //_ana_file->cd();
    
    // PREP SETS OF HITS
    // ------------------
    prepSpacepoints( iolcv, ioll  ); // from KPSRecoManager
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

    // Write the cosmic containers
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "maxshowerhit" );         ///< final set of in-time shower hits      
    ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "offtrigger_maxtrackhit" );     ///< final set of out-of-time track hits
    
    // COSMIC RECO
    // This is what will be new for us
    //cosmicTrackReco( iolcv, ioll );
    
  }
  
}
}
