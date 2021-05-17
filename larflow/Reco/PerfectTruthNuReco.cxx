#include "PerfectTruthNuReco.h"

#include "LArUtil/LArProperties.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/TruthTrackSCE.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "TrackdQdx.h"

#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  PerfectTruthNuReco::PerfectTruthNuReco()
    : larcv::larcv_base("PerfectTruthNuReco")
  {
    _psce = new larutil::SpaceChargeMicroBooNE;
  }

  PerfectTruthNuReco::~PerfectTruthNuReco()
  {
    delete _psce;
    _psce = nullptr;
  }
  
  NuVertexCandidate
  PerfectTruthNuReco::makeNuVertex( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();

    larlite::event_larflow3dhit* ev_lm
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "taggerfilterhit" );

    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data( larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );

    std::vector<int> used_v( ev_lm->size(), 0 );
    // get tracks
    NuVertexCandidate nuvtx;
    makeTracks( nuvtx, *ev_mctrack, *ev_lm, adc_v, used_v );
    makeShowers(  nuvtx, *ev_mcshower, *ev_lm, adc_v, used_v );

    nuvtx.keypoint_producer = "perfectreco";
    nuvtx.keypoint_index = 0;
    nuvtx.pos = ublarcvapp::mctools::NeutrinoVertex::getPos3DwSCE( ioll, _psce );
    nuvtx.tick = 3200 + nuvtx.pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    
    return nuvtx;
  }


  void PerfectTruthNuReco::makeTracks( NuVertexCandidate& nuvtx,
                                       const larlite::event_mctrack& ev_mctrack,
                                       const larlite::event_larflow3dhit& ev_lm,
                                       const std::vector<larcv::Image2D>& adc_v,
                                       std::vector<int>& used_v  )
  {

    LARCV_DEBUG() << "Make Tracks" << std::endl;    
    ublarcvapp::mctools::TruthTrackSCE track_convertor( _psce );
    TrackdQdx dqdxalgo;
    
    for ( auto const& track : ev_mctrack ) {

      if ( track.Origin()!=1 ) {
        //not neutrino
        continue;
      }
      
      larlite::track sce_track = track_convertor.applySCE( track );

      int npts = sce_track.NumberTrajectoryPoints();
      
      // now we assign hits
      larlite::larflowcluster track_cluster;
      track_cluster.reserve( ev_lm.size()/10+1 );
      
      for ( size_t idx=0; idx<ev_lm.size(); idx++ ) {
        if (used_v[idx]!=0 )
          continue;
        auto const& hit = ev_lm[idx];
        float dist = 1e9;
        int min_step = -1;
        std::vector<float> pt = { hit[0], hit[1], hit[2] };
        track_convertor.dist2track( pt, sce_track, dist, min_step );
        if ( dist<1.5 && (min_step>=0 && min_step<npts) ) {
          used_v[idx]=1;
          track_cluster.push_back( hit );
        }
      }//end of hit loop

      // dqdx calculation
      larlite::track dqdx_track = dqdxalgo.calculatedQdx( sce_track, track_cluster, adc_v );

      if ( track_cluster.size()>=5 ) {      
        nuvtx.track_v.emplace_back( std::move(dqdx_track) );
        nuvtx.track_hitcluster_v.emplace_back( std::move(track_cluster) );
      }
    }
    
  }

  void PerfectTruthNuReco::makeShowers( NuVertexCandidate& nuvtx,
                                       const larlite::event_mcshower& ev_mcshower,
                                       const larlite::event_larflow3dhit& ev_lm,
                                       const std::vector<larcv::Image2D>& adc_v,
                                       std::vector<int>& used_v  )
  {

    const float ar_molier_rad_cm = 9.04;
    const float ar_rad_length_cm = 14.3;
    
    TrackdQdx dqdxalgo;
    
    for ( auto const& shower : ev_mcshower ) {

      if ( shower.Origin()!=1 ) {
        //not neutrino
        continue;
      }
      
      auto const& profile = shower.DetProfile();
      //float pmom = profile.Momentum().Vect().Mag();
      //TVector3 dir = profile.Momentum().Vect();
      TVector3 dir = shower.Start().Momentum().Vect();
      float pmom = dir.Mag();
      TVector3 vstart = shower.Start().Position().Vect();


      if ( dir.Mag()<0.1 )
        continue;
      
      std::vector<float> fdir(3,0);
      std::vector<float> fstart(3,0);
      std::vector<float> fend(3,0);
      TVector3 vend;
      for (int i=0; i<3; i++) {
        dir[i] /= pmom;
        fdir[i] = (float)dir[i];
        fstart[i] = vstart[i];
        fend[i] = fstart[i] + 10.0*fdir[i];
        vend[i] = fend[i];
      }

      // space charge correction
      //_psce;
      std::vector<double> s_offset = _psce->GetPosOffsets(vstart[0],vstart[1],vstart[2]);
      vstart[0] = fstart[0] - s_offset[0] + 0.7;
      vstart[1] = fstart[1] + s_offset[1];
      vstart[2] = fstart[2] + s_offset[2];

      TVector3 v3end = { vend[0], vend[1], vend[2] };
      std::vector<double> e_offset = _psce->GetPosOffsets(v3end[0],v3end[1],v3end[2]);
      v3end[0] = vend[0] - e_offset[0] + 0.7;
      v3end[1] = vend[1] + e_offset[1];
      v3end[2] = vend[2] + e_offset[2];

      TVector3 sce_dir = v3end-vstart;
      float sce_dir_len = sce_dir.Mag();
      if ( sce_dir_len>0 ) {
        for (int i=0; i<3; i++)
          sce_dir[i] /= sce_dir_len;
      }      

      // make shower trunk object
      larlite::track trunk;
      trunk.reserve(2);
      trunk.add_vertex( vstart );
      trunk.add_vertex( v3end );
      trunk.add_direction( sce_dir );
      trunk.add_direction( sce_dir );      
      
      // make shower cluster object
      larlite::larflowcluster shower_cluster;
      for ( size_t idx=0; idx<ev_lm.size(); idx++ ) {
        if (used_v[idx]!=0 )
          continue;

        auto const& hit = ev_lm[idx];

        // should check if hit is on true nu-pixels
        
        std::vector<float> pos = { hit[0], hit[1], hit[2] };

        float s = larflow::reco::pointRayProjection3f( fstart, fdir, pos );
        float r = larflow::reco::pointLineDistance3f( fstart, fend, pos );

        float max_r = (s<0) ? 1.0 : s*ar_molier_rad_cm/ar_rad_length_cm;
        
        if ( s>-0.5 && r<max_r && s<50.0 ) {
          shower_cluster.push_back( hit );
          used_v[idx] = 1;
        }        
      }//end of hit loop

      if ( shower_cluster.size()>5 ) {

        larflow::reco::cluster_t clust = larflow::reco::cluster_from_larflowcluster( shower_cluster );
        larlite::pcaxis pca = larflow::reco::cluster_make_pcaxis( clust );
      
        //larlite::track dqdx_trunk = dqdxalgo.calculatedQdx( trunk, shower_cluster, adc_v );
        nuvtx.shower_v.emplace_back( std::move(shower_cluster) );
        nuvtx.shower_trunk_v.emplace_back( std::move(trunk) );
        nuvtx.shower_pcaxis_v.emplace_back( std::move(pca) );
        
      }
      
    }//end of mcshower loop
    
  }  
}
}
