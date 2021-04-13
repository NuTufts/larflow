#include "NuSelCosmicTagger.h"

#include "larflow/SCBoundary/SCBoundary.h"

namespace larflow {
namespace reco {


  /**
   * @brief check if candidate neutrino interaction is a showering cosmic muon
   *
   * check if two tracks are back-to-back.
   * check if one or more of the track ends are on the space-charge boundary.
   *
   */
  void NuSelCosmicTagger::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                   larflow::reco::NuSelectionVariables& nusel )
  {
    tagShoweringMuon2Track( nuvtx, nusel );
    tagShoweringMuon1Track( nuvtx, nusel );
    
  }
  
  /**
   * @brief check if candidate neutrino interaction is a showering cosmic muon
   *
   * check if two tracks are back-to-back.
   * check if one or more of the track ends are on the space-charge boundary.
   *
   */
  void NuSelCosmicTagger::tagShoweringMuon2Track( larflow::reco::NuVertexCandidate& nuvtx,
                                                  larflow::reco::NuSelectionVariables& nusel )
  {

    LARCV_DEBUG() << "2-track variables: ntracks = " << nuvtx.track_v.size() << std::endl;
    
    struct TrackPair_t
    {
      int idx1;
      int idx2;
      float cos_track;
      float dwall1;
      float dwall2;
      TrackPair_t(int i1, int i2)
        : idx1(i1), idx2(i2)
      {};
    };

    larflow::scb::SCBoundary scb; /// space charge boundary utility

    // the strategy only works for candidates with 2 or more tracks
    if ( nuvtx.track_v.size()<2 )
      return;

    int ntracks = nuvtx.track_v.size();

    // store combination of pair data
    std::vector< TrackPair_t > trackpair_v;
    trackpair_v.reserve( ntracks*(ntracks-1) );

    // we'll need the last point distance to the space-charge boundary    
    std::vector<double> track_dwall( ntracks, -1 );    
    for (int itrack=0; itrack<ntracks; itrack++) {
      int npts = nuvtx.track_v[itrack].NumberTrajectoryPoints();
      //auto const& lastpt = nuvtx.track_v[itrack].LocationAtPoint(npts-1);
      auto const& lastpt = nuvtx.track_v[itrack].LocationAtPoint(0);      
      int boundary_type = -1;
      track_dwall[itrack] = scb.dist2boundary( lastpt[0], lastpt[1], lastpt[2], boundary_type );
      LARCV_DEBUG() << "  track[" << itrack << "] "
                    << " point=(" << lastpt[0] << "," << lastpt[1] << "," << lastpt[2] << ")"
                    << " sdb-dwall=" << track_dwall[itrack]
                    << " boundary=" << boundary_type << std::endl;      
    }

    // double loop over pairs of tracks
    for (int itrack=0; itrack<ntracks-1; itrack++) {

      auto const& track_i = nuvtx.track_v[itrack];
      
      // define the direction.
      // take the average of the first steps
      TVector3 idir = _defineTrackDirection( nuvtx.track_v[itrack] );
      LARCV_DEBUG() << "track[" << itrack << "] dir=(" << idir[0] << "," << idir[1] << "," << idir[2] << ")" << std::endl;

      for (int jtrack=itrack+1; jtrack<ntracks; jtrack++) {

        TrackPair_t pairij(itrack,jtrack);
        
        TVector3 jdir = _defineTrackDirection( nuvtx.track_v[jtrack] );

        pairij.cos_track = 0;
        for (int v=0; v<3; v++)
          pairij.cos_track += idir[v]*jdir[v];

        LARCV_DEBUG() << "  pair-w-track[" << jtrack << "] dir=(" << jdir[0] << "," << jdir[1] << "," << jdir[2] << ")"
                      << " cos-track=" << pairij.cos_track
                      << std::endl;
        
        pairij.dwall1 = track_dwall[itrack];
        pairij.dwall2 = track_dwall[jtrack];

        trackpair_v.push_back( pairij );

      } // end of track pair loop (track j)
    }//end of track pair loop (track i)

    LARCV_DEBUG() << "number of track-pairs made: " << trackpair_v.size() << std::endl;
    
    // populate the selection variable
    // the dwall and costrack of the pair with track closest to the edge
    nusel.showercosmictag_mindwall_dwall = 1000;   // sentinal, should trigger replacement
    nusel.showercosmictag_mindwall_costrack = 1.0;

    // the dwall and costrack of the pair with most backtoback
    nusel.showercosmictag_maxbacktoback_dwall = 0;
    nusel.showercosmictag_maxbacktoback_costrack = 1.0; // sentinal, should trigger replacement
    
    for ( auto& trackpair : trackpair_v ) {
      // minimum length in tracks considered
      
      // at least one near the boundary
      float pair_mindwall = ( trackpair.dwall1 < trackpair.dwall2 ) ? trackpair.dwall1 : trackpair.dwall2;

      // save info on pair with straightest line
      if ( trackpair.cos_track < nusel.showercosmictag_maxbacktoback_costrack ) {
        nusel.showercosmictag_maxbacktoback_costrack = trackpair.cos_track;
        nusel.showercosmictag_maxbacktoback_dwall    = pair_mindwall;
      }

      // save info on pair with closest boundary
      
      if ( pair_mindwall < nusel.showercosmictag_mindwall_dwall ) {
        nusel.showercosmictag_mindwall_dwall    = pair_mindwall;
        nusel.showercosmictag_mindwall_costrack = trackpair.cos_track;
      }
      
    }

    LARCV_DEBUG() << "mindwall pair: dwall=" << nusel.showercosmictag_mindwall_dwall
                  << " costrack=" << nusel.showercosmictag_mindwall_costrack
                  << " ang-track=" << acos(nusel.showercosmictag_mindwall_costrack)*180/3.14159
                  << std::endl;
    LARCV_DEBUG() << "best cos-track pair: dwall=" << nusel.showercosmictag_maxbacktoback_dwall
                  << " costrack=" << nusel.showercosmictag_maxbacktoback_costrack
                  << " ang-track=" << acos(nusel.showercosmictag_maxbacktoback_costrack)*180/3.14159      
                  << std::endl;
    
  }

  /**
   * @brief check if candidate neutrino interaction is a showering cosmic muon
   *
   * variables for nu candidates with at least 1 track and 1 shower.
   * look for the most downward muon track that is near the boundary.
   *
   */
  void NuSelCosmicTagger::tagShoweringMuon1Track( larflow::reco::NuVertexCandidate& nuvtx,
                                                  larflow::reco::NuSelectionVariables& nusel )
  {

    LARCV_DEBUG() << "1-track/1-shower tagger: "
                  << " ntracks = " << nuvtx.track_v.size()
                  << " nshowers = " << nuvtx.shower_v.size()
                  << std::endl;
    
    larflow::scb::SCBoundary scb; /// space charge boundary utility

    nusel.showercosmictag_maxboundarytrack_length = 0;
    nusel.showercosmictag_maxboundarytrack_verticalcos = 0;
    nusel.showercosmictag_maxboundarytrack_showercos = -1.;
    
    
    // the strategy only works for candidates with 2 or more tracks
    if ( nuvtx.track_v.size()==0 )
      return;

    int ntracks  = nuvtx.track_v.size();
    int nshowers = nuvtx.shower_v.size();


    // get the longest boundary muon
    float maxlen_boundary_track = 0;
    int max_boundary_track_index = -1;
    float boundary_track_dwall = 1000.0;
    
    // we'll need the last point distance to the space-charge boundary    
    for (int itrack=0; itrack<ntracks; itrack++) {
      int npts = nuvtx.track_v[itrack].NumberTrajectoryPoints();
      //auto const& lastpt = nuvtx.track_v[itrack].LocationAtPoint(npts-1);
      auto const& lastpt = nuvtx.track_v[itrack].LocationAtPoint(0);      
      int boundary_type = -1;
      float track_dwall = scb.dist2boundary( lastpt[0], lastpt[1], lastpt[2], boundary_type );

      if ( track_dwall<5.0 ) {

        float tracklen = _getTrackLength( nuvtx.track_v[itrack] );
        
        if ( tracklen>maxlen_boundary_track ) {
          max_boundary_track_index = itrack;
          boundary_track_dwall = track_dwall;
          maxlen_boundary_track = tracklen;
        }
      }
    }

    LARCV_DEBUG() << "longest boundary track[" << max_boundary_track_index << "] "
                  << "length=" << maxlen_boundary_track
                  << std::endl;

    float max_boundary_track_verticalcos = 0.;
    float max_boundary_track_showercos = -1.0;
    
    if ( max_boundary_track_index>=0 ) {

      auto const& track_i = nuvtx.track_v[max_boundary_track_index];
      
      // define the direction.
      // take the average of the first steps
      TVector3 idir = _defineTrackDirection( track_i );

      // make it downward going
      if ( idir[1]>0 ) {
        idir *= -1.0;
      }

      // get projection along vertical axis (y-axis)
      max_boundary_track_verticalcos = idir[1];

      // get shower cosine with respect to this track direction

      for (int ishower=0; ishower<nshowers; ishower++) {

        auto const& shower_trunk = nuvtx.shower_trunk_v[ishower];

        TVector3 trunk_dir =
          shower_trunk.LocationAtPoint( shower_trunk.NumberTrajectoryPoints()-1 ) - shower_trunk.LocationAtPoint(0);

        float trunk_mag = trunk_dir.Mag();

        float showercos = 0.;
        for (int v=0; v<3; v++)
          showercos += idir[v]*trunk_dir[v]/trunk_mag;
        
        if ( showercos>max_boundary_track_showercos )
          max_boundary_track_showercos = showercos;
        
      } // end of shower loop

      nusel.showercosmictag_maxboundarytrack_length = maxlen_boundary_track;
      nusel.showercosmictag_maxboundarytrack_verticalcos = max_boundary_track_verticalcos;
      nusel.showercosmictag_maxboundarytrack_showercos = max_boundary_track_showercos;
            
      LARCV_DEBUG() << "boundary track found. index=" << max_boundary_track_index << std::endl;
      LARCV_DEBUG() << "  length=" << maxlen_boundary_track << " cm" << std::endl;
      LARCV_DEBUG() << "  dwall=" << boundary_track_dwall << " cm" << std::endl;
      LARCV_DEBUG() << "  max-shower-cos=" << max_boundary_track_showercos
                    << " ang=" << acos(max_boundary_track_showercos)*180/3.14159
                    << std::endl;
      LARCV_DEBUG() << "  vertical cosine=" << max_boundary_track_verticalcos << std::endl;

      
    }//end of if booundary muon found

       
  }


  TVector3 NuSelCosmicTagger::_defineTrackDirection( const larlite::track& track )
  {
    TVector3 avedir(0,0,0);

    int npts = track.NumberTrajectoryPoints();

    // return null direction (shouldnt happen)
    if (npts<2)
      return avedir;

    int nvecs = 0;
    float tracklen = 0.;
    //for (int i=0; i<npts-1; i++) {
    for (int i=npts-1; i>=1; i--) {
      const TVector3& start = track.LocationAtPoint(i);
      const TVector3& end   = track.LocationAtPoint(i-1);
      TVector3 stepdir = end-start;
      float mag = stepdir.Mag();

      tracklen += mag;
      if ( tracklen>10.0 ) {
        // don't average steps far away from vertex
        if ( nvecs==0 && mag>0 ) {
          //make sure there is at least one direction
          avedir = stepdir;
          nvecs++;
        }
        break;
      }
      
      if ( mag<=0 )
        continue;
      for (int v=0; v<3; v++)
        stepdir[v] /= mag;

      avedir += stepdir;
      nvecs++;
    }

    if ( nvecs>1 ) {
      for (int v=0; v<3; v++)
        avedir[v] /= (float)nvecs;
    }
    
    return avedir;
  }

  /**
   * @brief get track length from larlite track object
   *
   */
  float NuSelCosmicTagger::_getTrackLength( const larlite::track& track )
  {
    int npts = track.NumberTrajectoryPoints();

    // return null direction (shouldnt happen)
    if (npts<2)
      return 0;

    float tracklen = 0.;
    //for (int i=0; i<npts-1; i++) {
    for (int i=npts-1; i>=1; i--) {
      const TVector3& start = track.LocationAtPoint(i);
      const TVector3& end   = track.LocationAtPoint(i-1);
      TVector3 stepdir = end-start;
      float mag = stepdir.Mag();

      tracklen += mag;
    }
    
    return tracklen;
  }
  

  
  /**
   * @brief check if candidate neutrino interaction is a stopping muon
   *
   * check if y-most point is on the space-charge boundary. 
   * check dE/dx is consistant with decay muon
   *
   */
  void NuSelCosmicTagger::tagStoppingMuon( larflow::reco::NuVertexCandidate& nuvtx,
                                           larflow::reco::NuSelectionVariables& nusel )
  {
  }


}
}
