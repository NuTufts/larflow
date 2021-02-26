#include "NuSelProngVars.h"

namespace larflow {
namespace reco {

  void NuSelProngVars::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                larflow::reco::NuSelectionVariables& output )
  {

    output.ntracks = 0;
    output.nshowers = 0;
    output.max_shower_length = 0.;
    output.max_track_length = 0.;
    output.max_shower_nhits = 0;
    output.max_track_nhits = 0;

    for ( int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++ ) {
      auto const& lltrack = nuvtx.track_v[itrack];
      float tracklen = 0;
      int npts = lltrack.NumberTrajectoryPoints();
      for (int ipt=0; ipt<npts-1; ipt++) {
        float seglen = 0;
        float dx = 0;
        for (int i=0; i<3; i++) {
          dx = lltrack.LocationAtPoint(ipt)[i] - lltrack.LocationAtPoint(ipt+1)[i];
          seglen += dx*dx;
        }
        seglen = sqrt(seglen);
        tracklen += seglen;
      }

      if ( output.max_track_length<tracklen )
        output.max_track_length = tracklen;

      auto const& cluster = nuvtx.track_hitcluster_v[itrack];
      if ( cluster.size()>output.max_track_nhits )
        output.max_track_nhits = (int)cluster.size();

      if ( tracklen>_min_track_length && cluster.size()>10 )
        output.ntracks++;      
    }

    float min_shower_gap = 1e6;
    float max_shower_gap = 0;

    for ( int ishower=0; ishower<(int)nuvtx.shower_v.size(); ishower++ ) {
      auto const& cluster = nuvtx.shower_v[ishower];
      auto const& pca     = nuvtx.shower_pcaxis_v[ishower];
      auto const& trunk   = nuvtx.shower_trunk_v[ishower];
      float showerlen = 0.;
      float dx = 0.;
      float path_len_both[2] = { 0., 0.};
      for (int i=0; i<3; i++) {
        dx = pca.getEigenVectors()[3][i]-pca.getEigenVectors()[4][i];
        showerlen += dx*dx;
        path_len_both[0] += ( trunk.LocationAtPoint(0)[i]-nuvtx.pos[i] )*( trunk.LocationAtPoint(0)[i]-nuvtx.pos[i] );
        path_len_both[1] += ( trunk.LocationAtPoint(1)[i]-nuvtx.pos[i] )*( trunk.LocationAtPoint(1)[i]-nuvtx.pos[i] );
      }
      float path_len = ( path_len_both[0] < path_len_both[1] ) ? path_len_both[0] : path_len_both[1];
      showerlen = sqrt(showerlen);
      path_len = sqrt( path_len );
      if ( cluster.size()>output.max_shower_nhits )
        output.max_shower_nhits = (int)cluster.size();
      if ( showerlen>output.max_shower_length )
        output.max_shower_length = showerlen;
      if ( path_len<min_shower_gap )
        min_shower_gap = path_len;
      if ( path_len>max_shower_gap )
        max_shower_gap = path_len;

      LARCV_DEBUG() << " shower[" << ishower << "] pca-len=" << showerlen << std::endl;

      if ( showerlen>_min_shower_length && cluster.size()>10 )
        output.nshowers++;
      
    }//end of shower loop

    output.min_shower_gap = min_shower_gap;
    output.max_shower_gap = max_shower_gap;

    
    float min_track_gap = 1e6;
    float max_track_gap = 0;
        
    for ( int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++ ) {
      auto const& track   = nuvtx.track_v[itrack];      
      auto const& cluster = nuvtx.track_hitcluster_v[itrack];
      float tracklen[2] = {0.};
      int npts = (int)track.NumberTrajectoryPoints();
      for (int i=0; i<3; i++) {
        tracklen[0] += ( track.LocationAtPoint(0)[i] - nuvtx.pos[i] )*( track.LocationAtPoint(0)[i] - nuvtx.pos[i] );
        tracklen[1] += ( track.LocationAtPoint(npts-1)[i] - nuvtx.pos[i] )*( track.LocationAtPoint(npts-1)[i] - nuvtx.pos[i] );
      }

      float track_dist = ( tracklen[0]<tracklen[1] ) ? tracklen[0] : tracklen[1];

      if ( track_dist<min_track_gap )
        min_track_gap = track_dist;

      if ( track_dist>max_track_gap )
        max_track_gap = track_dist;
    }    

    output.min_track_gap = min_track_gap;
    output.max_track_gap = max_track_gap;
    
    LARCV_DEBUG() << "ntracks: " << output.ntracks << std::endl;
    LARCV_DEBUG() << "nshowers: " << output.nshowers << std::endl;
    LARCV_DEBUG() << "max_shower_length: " << output.max_shower_length << std::endl;
    LARCV_DEBUG() << "max_track_length: " << output.max_track_length << std::endl;
    LARCV_DEBUG() << "max_shower_nhits: " << output.max_shower_nhits << std::endl;
    LARCV_DEBUG() << "max_track_nhits: " << output.max_track_nhits << std::endl;
    LARCV_DEBUG() << "min_shower_gap: " << output.min_shower_gap << std::endl;
    LARCV_DEBUG() << "min_track_gap: " << output.min_track_gap << std::endl;    
    LARCV_DEBUG() << "max_shower_gap: " << output.max_shower_gap << std::endl;
    LARCV_DEBUG() << "max_track_gap: " << output.max_track_gap << std::endl;    

    
  }
  
}
}
