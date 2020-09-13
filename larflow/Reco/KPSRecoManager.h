#ifndef __KPS_RECO_MANAGER_H__
#define __KPS_RECO_MANAGER_H__

#include <string>

// ROOT
#include "TFile.h"
#include "TTree.h"

// larlite
#include "DataFormat/storage_manager.h"

// larcv
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

// ublarcvapp
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"
#include "ublarcvapp/MCTools/LArbysMC.h"

// larflow
#include "KeypointReco.h"
#include "KeypointFilterByClusterSize.h"
#include "KeypointFilterByWCTagger.h"
#include "SplitHitsBySSNet.h"
#include "ShowerRecoKeypoint.h"
#include "ProjectionDefectSplitter.h"
#include "ChooseMaxLArFlowHit.h"
#include "NuVertexMaker.h"
#include "CosmicTrackBuilder.h"
#include "NuTrackBuilder.h"
#include "NuShowerBuilder.h"
#include "NuVertexShowerReco.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class KPSRecoManager
   * @brief Uses all the different larflow::reco classes and executes event reconstruction
   *
   */
  class KPSRecoManager : public larcv::larcv_base {
  public:

    KPSRecoManager( std::string inputfile="outana_kpsrecomanager.root" );
    virtual ~KPSRecoManager();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    

    // image mods
    ublarcvapp::EmptyChannelAlgo _badchmaker; ///< bad channel image maker. also finds empty channels.

    // keypoint reconstruction
    KeypointReco     _kpreco_nu; ///< reconstruct keypoints from network scores for neutrino class
    KeypointReco     _kpreco_track; ///< reconstruct keypoints from network scores for track class
    KeypointReco     _kpreco_shower; ///< reconstruct keypoints from network scores for shower class
    KeypointReco     _kpreco_track_cosmic; ///< reconstruct keypoints from network scores for track class on wirecell cosmic-tagged spacepoints
    KeypointFilterByClusterSize _kpfilter; ///< filter out reconstructed keypoints on small clusters
    KeypointFilterByWCTagger _wcfilter; ///< filter out keypoints on wirecell cosmic-tagged pixes
    SplitHitsBySSNet _splithits_full; ///< splits shower space points from track spacepoints
    SplitHitsBySSNet _splithits_wcfilter; ///< splits shower spacepoints from track spacepoints for wc filtered hits
    ProjectionDefectSplitter _projsplitter; ///< split wirecell filtered track clusters into straight clusters
    ProjectionDefectSplitter _projsplitter_cosmic; ///< split cosmic-track clusters into straight clusters
    ShowerRecoKeypoint _showerkp; ///< reconstruct shower prongs using shower hits and shower keypoints
    ChooseMaxLArFlowHit _choosemaxhit; ///< reduce cosmic-track hits using max larmatch score
    NuVertexMaker       _nuvertexmaker; ///< make proto-vertices from prongs
    NuVertexShowerReco  _nuvertex_shower_reco; ///< make showers using neutrino vertex seed

    CosmicTrackBuilder  _cosmic_track_builder; ///< build tracks using cosmic clusters
    NuTrackBuilder      _nu_track_builder;  ///< build tracks for non-comic track clusters
    NuShowerBuilder     _nu_shower_builder; ///< build showers using those associated to vertex

    // MC event info
    ublarcvapp::mctools::LArbysMC _event_mcinfo_maker; ///< extracts mc event info and saves info to tree

    // Algorithms
    void recoKeypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void recoParticles( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void multiProngReco( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void saveEventMCinfo(bool savemc);

  protected:

    bool _save_event_mc_info; ///< if true, save event-level mc info to ana tree
    TFile* _ana_file; ///< output file for non-larlite and non-larcv reco products
    TTree* _ana_tree; ///< tree to store non-larlite and non-larcv reco products
    std::string _ana_output_file; ///< name of the ana file to create
    int _ana_run; ///< run number for tree entry
    int _ana_subrun; ///< subrun number for tree entry
    int _ana_event; ///< event number for tree entry
    void make_ana_file();
    
  public:

    /** @brief write the reco products to file */
    void write_ana_file() { _ana_file->cd(); _ana_tree->Write(); };
    
  };

}
}

#endif
