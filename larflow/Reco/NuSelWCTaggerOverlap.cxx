#include "NuSelWCTaggerOverlap.h"

#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace reco {

  void NuSelWCTaggerOverlap::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                      larflow::reco::NuSelectionVariables& output,
                                      larcv::IOManager& iolcv )
  {

    int ntrack_pts_on_cosmic  = 0;
    int nshower_pts_on_cosmic = 0;
    int ntrack_pts = 0;
    int nshower_pts = 0;

    larcv::EventImage2D* ev_thrumu
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"thrumu");
    auto const& thrumu_v = ev_thrumu->as_vector();
    auto const& meta = thrumu_v.front().meta();

    std::vector< const larlite::larflowcluster* > cluster_v;
    std::vector<int> cluster_type_v;
    for (int ishower=0; ishower<(int)nuvtx.shower_v.size(); ishower++) {
      cluster_v.push_back( &nuvtx.shower_v[ishower] );
      cluster_type_v.push_back(0);
    }
    for (int itrack=0; itrack<(int)nuvtx.track_hitcluster_v.size(); itrack++)  {
      cluster_v.push_back( &nuvtx.track_hitcluster_v[itrack] );
      cluster_type_v.push_back(1);
    }

    for ( size_t ic=0; ic<cluster_v.size(); ic++) {
      
      const larlite::larflowcluster* pcluster = cluster_v[ic];
      int cluster_type = cluster_type_v[ic];

      for ( auto const& hit : *pcluster ) {

        if ( hit.tick <= meta.min_y() || hit.tick>=meta.max_y() )
          continue;

        if ( cluster_type==0 )
          nshower_pts++;
        else
          ntrack_pts++;
        
        int row = meta.row( hit.tick, __FILE__, __LINE__ );

        int nplanes_on_thrumu = 0;
        
        for (int p=0; p<(int)hit.targetwire.size(); p++) {


          bool pixfound = false;
          
          for ( int dr=-1; dr<=1; dr++ ) {
            if ( pixfound ) break;
            
            int r=row+dr;
            if ( r<0 || r>=(int)meta.rows() )
              continue;

            for (int dc=-1; dc<=1; dc++) {
              if ( pixfound ) break;
              
              int c = hit.targetwire[p]+dc;
              if ( c<0 || c>=(int)meta.cols() )
                continue;

              //float pixval    = adc_v[p].pixel(r,c);
              float thrumuval = thrumu_v[p].pixel(r,c,__FILE__,__LINE__);

              if ( thrumuval>10.0)
                pixfound = true;
              
            }//end of call loop

          }//end of row loop

          if (pixfound)
            nplanes_on_thrumu++;
        }//end of plane loop

        if ( nplanes_on_thrumu>0 ) {
          if ( cluster_type==0 )
            nshower_pts_on_cosmic++;
          else
            ntrack_pts_on_cosmic++;
        }
        
      }//end of hit loop
    }//end of cluster loop

    if ( nshower_pts>0 )
      output.frac_showerhits_on_cosmic = (float)nshower_pts_on_cosmic/(float)(nshower_pts);
    else
      output.frac_showerhits_on_cosmic = 0;
    
    if ( ntrack_pts>0 )
      output.frac_trackhits_on_cosmic = (float)ntrack_pts_on_cosmic/(float)(ntrack_pts);
    else
      output.frac_trackhits_on_cosmic = 0;
    
    if ( ntrack_pts+nshower_pts > 0 )
      output.frac_allhits_on_cosmic = (float)( nshower_pts_on_cosmic+ntrack_pts_on_cosmic )/(float)(ntrack_pts+nshower_pts);
    else
      output.frac_allhits_on_cosmic = 0;
    
    output.nshower_pts_on_cosmic =  nshower_pts_on_cosmic;
    output.ntrack_pts_on_cosmic  =  ntrack_pts_on_cosmic;

    LARCV_DEBUG() << "nshower_pts_on_cosmic: " << nshower_pts_on_cosmic << " frac=" << output.frac_showerhits_on_cosmic << std::endl;
    LARCV_DEBUG() << "ntrack_pts_on_cosmic: " << ntrack_pts_on_cosmic << " frac=" << output.frac_trackhits_on_cosmic << std::endl;
    LARCV_DEBUG() << "all_pts_on_cosmic: " << nshower_pts_on_cosmic+ntrack_pts_on_cosmic << " frac=" << output.frac_allhits_on_cosmic << std::endl;
    
  }
  

}
}
