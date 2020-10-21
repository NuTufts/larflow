#ifndef __LARFLOW_CRTMATCH_CRTMATCHMANAGER_H__
#define __LARFLOW_CRTMATCH_CRTMATCHMANAGER_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflowcluster.h"

// algorithms
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"
#include "ublarcvapp/MCTools/LArbysMC.h"
#include "larflow/Reco/SplitHitsBySSNet.h"
#include "larflow/Reco/ChooseMaxLArFlowHit.h"
#include "larflow/Reco/KeypointReco.h"
#include "larflow/Reco/ProjectionDefectSplitter.h"
#include "larflow/Reco/TrackTruthRecoAna.h"

#include "CRTTrackMatch.h"
#include "CRTHitMatch.h"

namespace larflow {
namespace crtmatch {

  /**
   * @ingroup CRTMatchManager
   * @class CRTMatchManager
   * @brief Perform track matching to CRT tracks and CRT hits
   *
   * Class that runs both CRTTrackMatch and CRTHitMatch.
   *
   */
  class CRTMatchManager : public larcv::larcv_base {

  public:
    
    CRTMatchManager( std::string inputfile );
    virtual ~CRTMatchManager();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll);
    void store_output( larcv::IOManager& outlcv, larlite::storage_manager& outll, bool remove_if_no_flash=true );

    // sub-functions
    void recoKeypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void recoParticles( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void multiProngReco( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void truthAna( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void saveEventMCinfo(bool savemc);
    

    // output file configuration
    bool _save_event_mc_info; ///< if true, save event-level mc info to ana tree
    TFile* _ana_file; ///< output file for non-larlite and non-larcv reco products
    TTree* _ana_tree; ///< tree to store non-larlite and non-larcv reco products
    std::string _ana_output_file; ///< name of the ana file to create
    int _ana_run; ///< run number for tree entry
    int _ana_subrun; ///< subrun number for tree entry
    int _ana_event; ///< event number for tree entry
    void make_ana_file();    
    
    // output data products
    std::vector< larcv::Image2D > untagged_v;    ///< image where matched pixels are removed
    std::vector< larcv::Image2D > track_index_v; ///< image where crt track index labels image, so we can match larflow clusters to it
    std::vector< larlite::larflowcluster > _unmatched_clusters_v; ///< clusters not matched to crthit or crttracks


    // algorithms from the Reco module
    ublarcvapp::EmptyChannelAlgo _badchmaker; ///< bad channel image maker. also finds empty channels.    
    larflow::reco::SplitHitsBySSNet _splithits_ssnet;      ///< splits shower space points from track spacepoints
    larflow::reco::ChooseMaxLArFlowHit _choosemaxhit;      ///< reduce cosmic-track hits using max larmatch score
    larflow::reco::KeypointReco     _kpreco_track_cosmic;  ///< reconstruct keypoints from network scores for track class on wirecell cosmic-tagged spacepoints    
    larflow::reco::ProjectionDefectSplitter _projsplitter; ///< split wirecell filtered track clusters into straight clusters

    // mc algorithms
    ublarcvapp::mctools::LArbysMC _event_mcinfo_maker;       ///< extracts mc event info and saves info to tree    
    larflow::reco::TrackTruthRecoAna   _track_truthreco_ana; ///< match reco tracks to truth for performance studies
    
  };
  
}
}

#endif
