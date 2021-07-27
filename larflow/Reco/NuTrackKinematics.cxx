#include "NuTrackKinematics.h"


namespace larflow {
namespace reco {

  NuTrackKinematics::NuTrackKinematics()
    : larcv::larcv_base("NuTrackKinematics")
  {
    _load_data();
  }

  NuTrackKinematics::~NuTrackKinematics()
  {
    delete _sMuonRange2T;
    delete _sProtonRange2T;
    _sMuonRange2T = _sProtonRange2T = nullptr;
    _splinefile_rootfile->Close();
  }

  void NuTrackKinematics::_load_data()
  {

    std::string larflowdir = std::getenv("LARFLOW_BASEDIR");
    std::string splinefile = larflowdir + "/larflow/Reco/data/Proton_Muon_Range_dEdx_LAr_TSplines.root";

    _splinefile_rootfile = new TFile( splinefile.c_str(), "open" );
    _sMuonRange2T   = (TSpline3*)_splinefile_rootfile->Get("sMuonRange2T");
    _sProtonRange2T = (TSpline3*)_splinefile_rootfile->Get("sProtonRange2T");  
    
  }

  void NuTrackKinematics::clear() {
    _track_mu_mom_v.clear();
    _track_p_mom_v.clear();
    _track_mu_ke_v.clear();
    _track_p_ke_v.clear();
  }
  
  /**
   * @brief calculate kinematic variables
   * using range to find KE
   * using average of track directions within 10 cm of vertex to find direction for momentum
   *
   */
  void NuTrackKinematics::analyze( larflow::reco::NuVertexCandidate& nuvtx )
  {

    _track_mu_mom_v.clear();
    _track_p_mom_v.clear();
    _track_mu_ke_v.clear();
    _track_p_ke_v.clear();
    
    // for each track fill kinematic information
    for (int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++) {

      auto const& track = nuvtx.track_v[itrack];
      
      // for each track, get track length and angle
      float tracklen = get_tracklen( track );

      // get KE in MeV
      float ke_mu = _sMuonRange2T->Eval( tracklen );
      float ke_p  = _sProtonRange2T->Eval( tracklen );

      // get direction of track
      TVector3 trunkdir = get_trackdir_radius( track, 10.0, nuvtx.pos );
      if ( trunkdir.Mag()<1.0e-1 ) {
        // bad direction for some reason
        trunkdir[0] = 0.;
        trunkdir[1] = 0.;
        trunkdir[2] = 1.0;
      }

      float Emu = 105.7 + ke_mu;
      float Ep  = 938.3 + ke_p;

      float pmu = sqrt(Emu*Emu - 105.7*105.7);
      float pp  = sqrt(Ep*Ep - 938.3*938.3);
      
      TLorentzVector v_mu;
      v_mu.SetPxPyPzE( pmu*trunkdir[0], pmu*trunkdir[1], pmu*trunkdir[2], Emu );
      TLorentzVector v_p;
      v_p.SetPxPyPzE( pp*trunkdir[0], pp*trunkdir[1], pp*trunkdir[2], Ep );

      LARCV_INFO() << "track[" << itrack << "] tracklen=" << tracklen << " cm; "
                   << "dir=(" << trunkdir[0] << "," << trunkdir[1] << "," << trunkdir[2] << "); "
                   << "KEmu=" << ke_mu << " KEp=" << ke_p
                   << std::endl;

      _track_length_v.push_back( tracklen );
      _track_mu_ke_v.push_back( ke_mu );
      _track_p_ke_v.push_back( ke_p );
      _track_mu_mom_v.push_back( v_mu );
      _track_p_mom_v.push_back( v_p );
      
    }
    
  }

  float NuTrackKinematics::get_tracklen( const larlite::track& track )
  {
    float tracklen = 0;
    int npts = (int)track.NumberTrajectoryPoints();
    for (int i=npts-1; i>0; i--) {
      auto const& step = track.LocationAtPoint(i);
      auto const& next = track.LocationAtPoint(i-1);

      float mag = (next-step).Mag();
      tracklen += mag;
    }

    return tracklen;
  }

  
  TVector3 NuTrackKinematics::get_trackdir_radius( const larlite::track& track,
                                                   const float radius,
                                                   const std::vector<float>& vtx )
  {

    TVector3 vvtx( vtx[0], vtx[1], vtx[2] );
    
    int npts = (int)track.NumberTrajectoryPoints();

    TVector3 start = track.LocationAtPoint(npts-1);
    
    TVector3 ave(0,0,0);
    int nvec = 0;
    for (int i=npts-1; i>0; i--) {
      auto const& step = track.LocationAtPoint(i);
      auto const& next = track.LocationAtPoint(i-1);

      float rad_dist = (step-start).Mag();

      if ( rad_dist>radius )
        continue;
      
      TVector3 dir = (next-step);
      float mag = (dir).Mag();
      if ( mag>0 ) {
        for (int v=0; v<3; v++)
          dir[v] /= mag;
        ave += dir;
        nvec ++;
      }
      
    }

    if ( nvec>0 ) {
      for (int i=0; i<3; i++) {
        ave[i] /= (float)nvec;
      }
      float avelen = ave.Mag();
      for (int i=0; i<3; i++)
        ave[i] /= avelen;
    }
    
    return ave;
  }
  
}
}
