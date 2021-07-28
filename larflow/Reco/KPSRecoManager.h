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
#include "ShortProtonClusterReco.h"
#include "ChooseMaxLArFlowHit.h"
#include "NuVertexMaker.h"
#include "CosmicVertexBuilder.h"
#include "CosmicTrackBuilder.h"
#include "NuTrackBuilder.h"
#include "NuShowerBuilder.h"
#include "NuVertexShowerReco.h"
#include "NuVertexShowerTrunkCheck.h"
#include "NuVertexActivityReco.h"
#include "PerfectTruthNuReco.h"
#include "NuTrackKinematics.h"
#include "NuShowerKinematics.h"

#include "NuSelectionVariables.h"
#include "LikelihoodProtonMuon.h"
#include "CosmicProtonFinder.h"
#include "ShowerdQdx.h"

#include "NuSel1e1pEventSelection.h"
#include "NuSelProngVars.h"
#include "NuSelVertexVars.h"
#include "NuSelTruthOnNuPixel.h"
#include "NuSelShowerTrunkAna.h"
#include "NuSelWCTaggerOverlap.h"
#include "NuSelShowerGapAna2D.h"
#include "NuSelUnrecoCharge.h"
#include "NuSelCosmicTagger.h"
#include "TrackForwardBackwardLL.h"

// truth analysis
// #include "TrackTruthRecoAna.h"

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

    // larflow hit classification
    ChooseMaxLArFlowHit _choosemaxhit; ///< reduce cosmic-track hits using max larmatch score    
    SplitHitsBySSNet _splithits_full; ///< splits shower space points from track spacepoints
    SplitHitsBySSNet _splithits_wcfilter; ///< splits shower spacepoints from track spacepoints for wc filtered hits

    // clustering
    ProjectionDefectSplitter _projsplitter; ///< split wirecell filtered track clusters into straight clusters
    ProjectionDefectSplitter _projsplitter_cosmic; ///< split cosmic-track clusters into straight clusters
    ShortProtonClusterReco   _short_proton_reco;   ///< short proton reco using HIP space points only    
    ShowerRecoKeypoint _showerkp; ///< reconstruct shower prongs using shower hits and shower keypoints

    // mc-based reco
    PerfectTruthNuReco _perfect_reco; ///< make nuvertexcandidate using true trajectories and showers

    // Nu Vertex Seeds
    NuVertexMaker        _nuvertexmaker; ///< make proto-vertices from prongs
    NuVertexActivityReco _nuvertexactivity; ///< nu vertex activity
    NuVertexShowerReco   _nuvertex_shower_reco; ///< make showers using neutrino vertex seed
    NuVertexShowerTrunkCheck _nuvertex_shower_trunk_check; ///< repair shower trunk check

    CosmicTrackBuilder  _cosmic_track_builder; ///< build tracks using cosmic clusters
    CosmicVertexBuilder _cosmic_vertex_builder; ///< build stopmu vertices
    NuTrackBuilder      _nu_track_builder;  ///< build tracks for non-comic track clusters
    NuShowerBuilder     _nu_shower_builder; ///< build showers using those associated to vertex

    // Prong kinematics
    NuTrackKinematics   _nu_track_kine;  ///< calculate kinematics of tracks
    NuShowerKinematics  _nu_shower_kine; ///< calculate kinematics of showers
    
    // Edge-case handlers
    CosmicProtonFinder _cosmic_proton_finder; ///< identifies cosmic tracks that could proton-like, redirect to neutrino-pipeline

    //TrackTruthRecoAna   _track_truthreco_ana; ///< match reco tracks to truth for performance studies

    // Selection Variable Modules
    LikelihoodProtonMuon _sel_llpmu; ///< proton vs. muon likelihood ratio
    ShowerdQdx           _sel_showerdqdx; ///< shower dq/dx calculation

    NuSelProngVars prongvars;
    NuSelVertexVars vertexvars;
    NuSelShowerTrunkAna showertrunkvars;
    NuSelWCTaggerOverlap wcoverlapvars;
    NuSelShowerGapAna2D showergapana2d;
    NuSelUnrecoCharge   unrecocharge;
    NuSelCosmicTagger   cosmictagger;
    TrackForwardBackwardLL muvsproton;
    

    // Event Selection modules (only for development)
    NuSel1e1pEventSelection _eventsel_1e1p;

    // MC event info
    ublarcvapp::mctools::LArbysMC _event_mcinfo_maker; ///< extracts mc event info and saves info to tree
    

    // Stages
    void prepSpacepoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void recoKeypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void clusterSubparticleFragments( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void cosmicTrackReco( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void multiProngReco( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void makeNuCandidateSelectionVariables( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void runBasicKinematics( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void runBasicPID( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void runNuVtxSelection();

    void truthAna( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void saveEventMCinfo(bool savemc);
    void saveSelectedNuVerticesOnly( bool save_selected ) { _save_selected_only = save_selected; }; ///< if true, only store selected vertices

    void clear();

  protected:

    bool _save_event_mc_info; ///< if true, save event-level mc info to ana tree
    TFile* _ana_file; ///< output file for non-larlite and non-larcv reco products
    TTree* _ana_tree; ///< tree to store non-larlite and non-larcv reco products
    std::string _ana_output_file; ///< name of the ana file to create
    int _ana_run; ///< run number for tree entry
    int _ana_subrun; ///< subrun number for tree entry
    int _ana_event; ///< event number for tree entry
    float _t_event_elapsed; ///< runtime for event
    bool _save_selected_only; ///< if true, save only selected nu vertex candidates
    void make_ana_file();

    std::vector< larflow::reco::NuSelectionVariables > _nu_sel_v; ///< selection variables for nuvtx candidates
    std::vector< larflow::reco::NuVertexCandidate >    _nu_perfect_v; ///< store reco based on true trajectories

    bool _kMinize_outputfile_size;
    
  public:

    /** @brief write the reco products to file */
    void write_ana_file() { _ana_file->cd(); _ana_tree->Write(); };

    /** @brief Minimize the output file size by not saving intermediate vertex candidates */
    void minimze_output_size( bool domin=true ) { _kMinize_outputfile_size=domin; };
    
  };

}
}

#endif
