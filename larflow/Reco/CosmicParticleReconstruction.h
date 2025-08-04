#ifndef __LARFLOW_RECO_COSMIC_PARTICLE_RECONSTRUCTION_H__
#define __LARFLOW_RECO_COSMIC_PARTICLE_RECONSTRUCTION_H__

#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/opflash.h"
#include "larlite/DataFormat/crttrack.h"
#include "larlite/DataFormat/crthit.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "larflow/Reco/CosmicParticleCandidate.h"
#include "larflow/Reco/KPCluster.h"

namespace larflow {
namespace reco {

  class CosmicParticleReconstruction : public larcv::larcv_base {
  public:

    CosmicParticleReconstruction();
    ~CosmicParticleReconstruction() {};
    
    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );   

    void clear();

    void set_default_param_values();

    void make_reco_output_file();

    /** @brief write the reco products to file */
    void write_output_file() { 
      if (!_ana_file) 
        return;
      _ana_file->cd();  
      _ana_file->Write(); 
    };

  protected:

    void prepSpacepoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void recoKeypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void buildTrackFragments( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void buildCosmicTracks( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void fillFlashMatchData( larlite::storage_manager& ioll );

    std::string _flash_producer;
    std::string _wireimg_producer;
    std::string _outoftime_tagged_pixels_producer;
    std::string _larmatch_hit_producer;

    std::vector< larflow::reco::CosmicParticleCandidate > _cosmic_candidates_v;

    // storage for keypoint clusters
    std::vector< larflow::reco::KPCluster > _event_kpc_track_start_v;
    std::vector< larflow::reco::KPCluster > _event_kpc_track_end_v;

    TFile* _ana_file; ///< output file for non-larlite and non-larcv reco products
    TTree* _ana_tree; ///< tree to store non-larlite and non-larcv reco products
    std::string _ana_output_file; ///< name of the ana file to create
    int _ana_run; ///< run number for tree entry
    int _ana_subrun; ///< subrun number for tree entry
    int _ana_event; ///< event number for tree entry
    float _t_event_elapsed; ///< runtime for event
    int _reco_status;

    bool _save_flashmatchdata_tree; ///< save additional TTree containing information for flash-match data preparation
    TTree* _flashmatchdata_tree;    ///< TTree storing a list of cosmic tracks, optical flashes, and CRT informationtr
    std::vector< larlite::track >         _flashmatchdata_track_v;      ///< list of tracks to fill in an event
    //std::vector< larlite::larflowcluster> _flashmatchdata_track_hits_v; ///< list of hits associated to each track
    std::vector< larlite::opflash >       _flashmatchdata_opflash_v;    ///< list of optical flashes in an event
    std::vector< larlite::crttrack >      _flashmatchdata_crttrack_v;   ///< list of crt tracks in an event
    std::vector< larlite::crthit >        _flashmatchdata_crthit_v;     ///< list of crt hits in an event
    std::vector< std::vector< std::vector<float> > >  _flashmatchdata_track_hits_v; ///< list of hits associated to each track


  };
  
}
}

#endif
