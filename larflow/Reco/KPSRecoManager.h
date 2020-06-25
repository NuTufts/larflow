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

// larflow
#include "KeypointReco.h"
#include "KeypointFilterByClusterSize.h"
#include "KeypointFilterByWCTagger.h"
#include "SplitHitsBySSNet.h"
#include "DBScanLArMatchHits.h"
#include "TrackReco2KP.h"
#include "ShowerRecoKeypoint.h"
#include "PCACluster.h"
#include "ProjectionDefectSplitter.h"
#include "PCATracker.h"
#include "ChooseMaxLArFlowHit.h"
#include "NuVertexMaker.h"

namespace larflow {
namespace reco {
    
  class KPSRecoManager : public larcv::larcv_base {
  public:

    KPSRecoManager( std::string inputfile="outana_kpsrecomanager.root" );
    virtual ~KPSRecoManager();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    

    // image mods
    ublarcvapp::EmptyChannelAlgo _badchmaker;

    // keypoint reconstruction
    KeypointReco     _kpreco_nu;
    KeypointReco     _kpreco_track;
    KeypointReco     _kpreco_shower;
    KeypointReco     _kpreco_track_cosmic;
    KeypointFilterByClusterSize _kpfilter;
    KeypointFilterByWCTagger _wcfilter;
    SplitHitsBySSNet _splithits_full;
    SplitHitsBySSNet _splithits_wcfilter;    
    TrackReco2KP     _tracker2kp;
    DBScanLArMatchHits _cluster_track;
    DBScanLArMatchHits _cluster_shower;
    PCACluster         _pcacluster;
    PCATracker         _pcatracker;
    ProjectionDefectSplitter _projsplitter;
    ProjectionDefectSplitter _projsplitter_cosmic;
    ShowerRecoKeypoint _showerkp;
    ChooseMaxLArFlowHit _choosemaxhit;
    NuVertexMaker       _nuvertexmaker;

    // Algorithms
    void recoKeypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void recoParticles( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void multiProngReco( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
                         

  protected:

    TFile* _ana_file;
    TTree* _ana_tree;
    std::string _ana_input_file;
    int _ana_run;
    int _ana_subrun;
    int _ana_event;
    void make_ana_file();
    
  public:
    void write_ana_file() { _ana_file->cd(); _ana_tree->Write(); };
    
  };

}
}

#endif
