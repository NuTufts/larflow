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

      std::cout << " shower[" << ishower << "] pca-len=" << showerlen << std::endl;

      if ( showerlen>_min_shower_length && cluster.size()>10 )
        output.nshowers++;
      
    }//end of shower loop

    output.min_shower_gap = min_shower_gap;
    output.max_shower_gap = max_shower_gap;
    
    
  }
  
}
}
