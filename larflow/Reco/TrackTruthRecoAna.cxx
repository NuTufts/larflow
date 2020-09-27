#include "TrackTruthRecoAna.h"

#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "ublarcvapp/ubdllee/dwall.h"
#include "larflow/Reco/geofuncs.h"

namespace larflow {
namespace reco {

  TrackTruthRecoAna::TrackTruthRecoAna()
    : larcv::larcv_base("TrackTruthRecoAna")
  {
    _psce  = new larutil::SpaceChargeMicroBooNE;
  }

  TrackTruthRecoAna::~TrackTruthRecoAna()
  {
    delete _psce;
    _psce = nullptr;
  }

  /**
   *
   *
   */
  void TrackTruthRecoAna::process( larcv::IOManager& iolcv,
                                   larlite::storage_manager& ioll,
                                   std::vector< larflow::reco::NuVertexCandidate >& nuvtx_v )
  {

    // build the pixel graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    
    mcpg.buildgraph( iolcv, ioll );
    mcpg.printGraph();
    
    // get track truth info
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack, "mcreco");

    // loop over vertices
    vtxinfo_v.clear();

    for ( int ivtx=0; ivtx<(int)nuvtx_v.size(); ivtx++ ) {

      larflow::reco::NuVertexCandidate& nuvtx = nuvtx_v.at(ivtx);
    
      // loop over the reco tracks
      VertexTrackTruthRecoInfo vtxinfo;
      vtxinfo.vtxid = ivtx;

      LARCV_INFO() << "=== Reco Nu-Vertex [" << ivtx << "]:"
                   << " ntracks=" << nuvtx.track_v.size()
                   << " nclusters=" << nuvtx.track_hitcluster_v.size()
                   << " ===" << std::endl;
      
      for ( int ireco=0; ireco<(int)nuvtx.track_v.size(); ireco++ ) {
        auto& recotrack   = nuvtx.track_v.at(ireco);
        auto& recocluster = nuvtx.track_hitcluster_v.at(ireco);
        TrackTruthRecoInfo trackinfo = _make_truthmatch_info( recotrack, recocluster, mcpg, *ev_mctrack );

        LARCV_INFO() << "  Reco track [" << ireco << "] results:" << std::endl;
        LARCV_INFO() << "    Matched Truth TrackID: " << trackinfo.matched_true_trackid << " pid=" << trackinfo.matched_true_pid << std::endl;
        LARCV_INFO() << "    Matched mean-squared error: " << trackinfo.matched_mse << std::endl;
        
        vtxinfo.trackinfo_v.emplace_back( std::move(trackinfo) );
      }

      vtxinfo_v.emplace_back(std::move(vtxinfo));
      
    }//end of vertex loop
    
  }
  
  /**
   * @brief Get SCE-corrected path for a true track trajectory
   *
   * @param[in] mct Given true track
   * @return vector of (x,y,z) 3D space points with x-relative to the event trigger
   *
   */
  std::vector< std::vector<float> >
  TrackTruthRecoAna::getSCEtrueTrackPath( const larlite::mctrack& mct,
                                          const larutil::SpaceChargeMicroBooNE* psce )
  {

    std::vector< std::vector<float> > sce_path_v;
    sce_path_v.reserve( mct.size() );
    
    for ( int istep=0; istep<(int)mct.size(); istep++ ) {

      auto const& step = mct.at(istep);
      std::vector<float> fpt = { (float)step.T(),
                                 (float)step.X(),
                                 (float)step.Y(),
                                 (float)step.Z() };
          
      float tick = ublarcvapp::mctools::CrossingPointsAnaMethods::getTick( fpt, 4050.0, psce );
      std::vector<double> offsets = psce->GetPosOffsets( fpt[1], fpt[2], fpt[3] );
          
      std::vector<float> pathpt(3,0);
      pathpt[0] = (tick-3200.0)*larutil::LArProperties::GetME()->DriftVelocity()*0.5;
      pathpt[1] = step.Y() + offsets[1];
      pathpt[2] = step.Z() + offsets[2];
                    
      sce_path_v.push_back( pathpt );
    }
    
    return sce_path_v;
  }

  /**
   * @brief get mean-squared error of cluster hits to track trajectory segments
   *
   * @param[in] truetrack_path_v  points along trajectory after space charge effects applied
   * @param[in] track_hit_cluster space points associated to reco track cluster
   * @return average squared distance to trajectory defined by truetrack_path_v 
   */
  float TrackTruthRecoAna::_get_mse_cluster_truetrack_v( const std::vector< std::vector<float> >& truetrack_path_v,
                                                         const larlite::larflowcluster& track_hit_cluster )
  {

    if ( truetrack_path_v.size()<2 || track_hit_cluster.size()==0 ) {
      // something wrong with the track
      return 1.0e9; // large mse as sentinal value
    }

    const float max_r = 3.0; //cm
    
    // for each hit, we need to find the path end point to attach it to
    size_t nhits = track_hit_cluster.size();

    float ave_mse = 0.;
    
    for ( size_t ihit=0; ihit<nhits; ihit++ ) {
      auto const& lfhit = track_hit_cluster[ihit];

      std::vector<float> testpt = { lfhit[0], lfhit[1], lfhit[2] };
      
      // search path to find best segment matched to
      
      bool foundseg = false; /// found a matched segment
      int best_seg = -1;     /// index of start point of segment
      int best_endpt = -1;   /// also check if close to an end point
      float min_endr2 = 1e9; /// min dist-squared to closest end point
      float min_segr2 = 1e9;  /// min dist-squared perpendicular to segment
      
      for (size_t ipt=0;ipt<truetrack_path_v.size()-1; ipt++) {

        const std::vector<float>& pt1 = truetrack_path_v[ipt];
        const std::vector<float>& pt2 = truetrack_path_v[ipt+1];

        float seglen = 0.;
        float r2_end = 0.;        
        std::vector<float> ptdir(3,0);
        for (int i=0; i<3; i++) {
          ptdir[i] = pt2[i]-pt1[i];
          seglen   += ptdir[i]*ptdir[i];
          r2_end   += (pt2[i]-testpt[i])*(pt2[i]-testpt[i]);
        }
        seglen = sqrt(seglen);
        if ( seglen>0 ) {
          for (int i=0; i<3; i++)
            ptdir[i] /= seglen;
        }

        if ( seglen==0.0 )
          continue;

        float r  = larflow::reco::pointLineDistance3f( pt1, pt2, testpt );
        float s  = larflow::reco::pointRayProjection3f( pt1, ptdir, testpt );
        float r2 = r*r;
        
        if (s>=0 && s<=seglen && r2<min_segr2) {
          best_seg = ipt;
          foundseg = true;
          min_segr2 = r2;
        }
        
        if ( r2_end<min_endr2 ) {
          best_endpt = ipt+1;
          min_endr2 = r2_end;
        }//end of if within max_r of line segment
        
      }//end of loop over track points
      
      float min_r2 = (min_segr2<min_endr2) ? min_segr2 : min_endr2;
      ave_mse += min_r2;
      
    }//end of hit loop
    
    ave_mse /= float(nhits);
    return ave_mse;
  }

  /**
   * @brief produce truth-match info for reco track cluster
   *
   * 
   *
   */
  TrackTruthRecoInfo TrackTruthRecoAna::_make_truthmatch_info( const larlite::track& track,
                                                               const larlite::larflowcluster& track_hit_cluster,
                                                               ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                               const larlite::event_mctrack& ev_mctrack )
  {

    TrackTruthRecoInfo info; // the class whose members we want to populate

    // first we need a list of tracks
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> nu_nodes = mcpg.getNeutrinoParticles();
    std::vector< const larlite::mctrack* > pmctrack_v;
    for ( auto& pnode : nu_nodes ) {
      if (pnode->type!=0)
        continue;
      
      try {
        pmctrack_v.push_back( &ev_mctrack.at(pnode->vidx) );
      }
      catch (std::exception& e) {
        LARCV_WARNING() << "could not get mctrack object for node. "
                        << "index=" << pnode->vidx
                        << " trackid=" << pnode->tid
                        << " pdg=" << pnode->pid << std::endl;
      }
    }

    // get ave mse for each track
    float min_mse = 1e9;
    int min_imctrack = -1;
    std::vector<float> mse_v( pmctrack_v.size(), -1 );
    for ( int imctrack=0; imctrack<(int)pmctrack_v.size(); imctrack++ ) {
      
      std::vector< std::vector<float> > true_det_path_v
        = getSCEtrueTrackPath( *pmctrack_v[imctrack], _psce );

      if ( true_det_path_v.size()<2 )
        continue;

      float mse = _get_mse_cluster_truetrack_v( true_det_path_v, track_hit_cluster );
      mse_v[imctrack] = mse;
      if ( mse<min_mse ) {
        min_mse = mse;
        min_imctrack = imctrack;
      }
    }

    if ( min_imctrack>=0 ) {
      info.matched_true_trackid = pmctrack_v[min_imctrack]->TrackID();
      info.matched_true_pid     = pmctrack_v[min_imctrack]->PdgCode();
    }
    else {
      info.matched_true_trackid = -1;
      info.matched_true_pid     = -1;
    }
    info.matched_mse          = min_mse;
    info.truetrack_completeness = 0;
    info.dist_to_trueend      = 0;

    return info;
    
  }

  /**
   * @brief add analysis info to TTree
   *
   * @param[inout] tree The ROOT tree to add the class member, vtxinfo_v.
   *
   */
  void TrackTruthRecoAna::bindAnaVariables( TTree* tree ) {
    tree->Branch("track_truthreco_vtxinfo_v", &vtxinfo_v);
  }


}
}
