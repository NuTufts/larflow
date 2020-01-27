#include "CRTTrackMatch.h"

#include "larlite/core/LArUtil/LArProperties.h"
#include "larlite/core/LArUtil/Geometry.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

#include "TH2D.h"
#include "TCanvas.h"
#include "TGraph.h"

namespace larflow {
namespace crtmatch {

  CRTTrackMatch::CRTTrackMatch() {
    _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Forward );
  }

  CRTTrackMatch::~CRTTrackMatch() {
    delete _sce;
  }
  
  void CRTTrackMatch::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    larlite::event_crttrack* ev_crttrack = (larlite::event_crttrack*)ioll.get_data( larlite::data::kCRTTrack, "crttrack" );

    
    // vizualize for debug
    std::vector<TH2D> h2d_v;
    h2d_v = larcv::rootutils::as_th2d_v( ev_adc->Image2DArray(), "crttrack_adc" );
    std::vector< TGraph* > graph_v[3];
    

    std::vector< crttrack_t > data_v;
    for ( size_t i=0; i<ev_crttrack->size(); i++ ) {
      crttrack_t data = _collect_chargepixels_for_track( ev_crttrack->at(i), ev_adc->Image2DArray() );
      
      std::cout << "[CRTTrackMatch::process] track collected "
                << "npts=" << data.pixelpos_vv.size() << " "
                << "plane pixels=("
                << data.pixellist_vv[0].size() << ","
                << data.pixellist_vv[1].size() << ","
                << data.pixellist_vv[2].size() << ")"
                << " length=" << data.len_intpc_sce
                << " totq=(" << data.totalq_v[0] << "," << data.totalq_v[1] << "," << data.totalq_v[2] << ")"
                << std::endl;
      if ( data.pixelpos_vv.size()>0 ) {
        for ( size_t p=0; p<3; p++ ) {
          if ( data.pixelcoord_vv[p].size()==0 ) continue;
          TGraph* g = new TGraph( data.pixelcoord_vv[p].size() );
          for ( size_t n=0; n<data.pixelcoord_vv.size(); n++ ) {
            g->SetPoint( n, data.pixelcoord_vv[n][p], data.pixelcoord_vv[n][3] );
          }
          graph_v[p].push_back( g );
        }
      }
    }


    TCanvas c("c","c",1200,600);
    c.Divide(1,3);
    for ( size_t p=0; p<3; p++ ) {
      c.cd(p+1);
      h2d_v[p].Draw("colz");
      std::cout << "graphs in plane[" << p << "]: " << graph_v[p].size() << std::endl;
      for ( auto& g : graph_v[p] ) {
        //std::cout << " " << g.N() << std::endl;
        g->SetLineWidth(1);
        g->SetLineColor(kMagenta);
        //g->SetMarkerStyle(20);
        g->Draw("Lsame");
      }
    }
    c.Update();
    
    char name[100];
    sprintf( name, "crttrackmatch_debug_run%d_subrun%d_event%d.png",
             (int)iolcv.event_id().run(),
             (int)iolcv.event_id().subrun(),
             (int)iolcv.event_id().event() );
    c.SaveAs(name);

    
  }
    
  CRTTrackMatch::crttrack_t
    CRTTrackMatch::_collect_chargepixels_for_track( const larlite::crttrack& crt,
                                                  const std::vector<larcv::Image2D>& adc_v ) {

    crttrack_t data( -1, &crt );

    float dir[3] = { crt.x2_pos-crt.x1_pos,
                     crt.y2_pos-crt.y1_pos,
                     crt.z2_pos-crt.z1_pos };
    
    float len = sqrt( (crt.x1_pos-crt.x2_pos)*(crt.x1_pos-crt.x2_pos)
                      + (crt.y1_pos-crt.y2_pos)*(crt.y1_pos-crt.y2_pos)
                      + (crt.z1_pos-crt.z2_pos)*(crt.z1_pos-crt.z2_pos) );

    if ( len==0 )
      return data;

    for ( size_t p=0; p<3; p++ ) dir[p] /= len;
    
    float stepsize = 0.3;
    int nsteps = len/stepsize+1;
    stepsize = len/float(nsteps);

    // need t0 offset
    float t0_usec = 0.5*(crt.ts2_ns_h1+crt.ts2_ns_h2)*0.001;
    
    std::cout << "[CRTTrackMatch::_collect_chargepixels_for_track]" << std::endl;
    std::cout << "  start=(" << crt.x1_pos << "," << crt.y1_pos << "," << crt.z1_pos << ")" << std::endl;
    std::cout << "  end=  (" << crt.x2_pos << "," << crt.y2_pos << "," << crt.z2_pos << ")" << std::endl;
    std::cout << "  len=" << len << " steps=" << nsteps << std::endl;
    std::cout << "  t0_usec=" << t0_usec << std::endl;

    std::vector< larcv::Image2D > pix_visited_v;
    for ( auto const& img : adc_v ) {
      larcv::Image2D visited(img.meta());
      visited.paint(0.0);
      pix_visited_v.emplace_back( std::move(visited) );
    }

    std::vector<double> last_pos;

    for (int istep=0; istep<nsteps; istep++) {

      // step position
      std::vector<double> pos = { crt.x1_pos+istep*stepsize*dir[0],
                                  crt.y1_pos+istep*stepsize*dir[1],
                                  crt.z1_pos+istep*stepsize*dir[2] };

      
      if ( pos[0]<1.0 || pos[0]>255.0 ) continue;
      if ( pos[1]<-116.0 || pos[1]>116.0  ) continue;
      if ( pos[2]<0.5 || pos[2]>1035.0 ) continue;

      //std::cout << " [" << istep << "] pos=(" << pos[0] << "," << pos[1] << "," << pos[2] << ")" << std::endl;      
      
      std::vector<double> offset = _sce->GetPosOffsets( pos[0], pos[1], pos[2] );
      pos[0] = pos[0] - offset[0] + 0.6;
      pos[1] = pos[1] + offset[1];
      pos[2] = pos[2] + offset[2];
      
      // space-charge correction
      bool inimage = true;
      std::vector<int> imgcoord_v(4);
      for (size_t p=0; p<adc_v.size(); p++ ) {
        imgcoord_v[p] = (int)(larutil::Geometry::GetME()->WireCoordinate( pos, (UInt_t)p )+0.5);
        if ( imgcoord_v[p]<0 || imgcoord_v[p]>=larutil::Geometry::GetME()->Nwires(p) ) {
          inimage = false;
        }
      }
      imgcoord_v[3] = 3200 + ( pos[0]/larutil::LArProperties::GetME()->DriftVelocity() + t0_usec )/0.5;
      //imgcoord_v[3] = 3200 + ( pos[0]/larutil::LArProperties::GetME()->DriftVelocity() )/0.5;

      //std::cout << " [" << istep << "] imgcoord=(" << imgcoord_v[0] << "," << imgcoord_v[1] << "," << imgcoord_v[2] << ", tick=" << imgcoord_v[3] << ")" << std::endl;      
      
      if ( !inimage ) continue;

      data.pixelcoord_vv.push_back( imgcoord_v );
      data.pixelpos_vv.push_back( pos );      


      if ( imgcoord_v[3]<=adc_v[0].meta().min_y() || imgcoord_v[3]>=adc_v[0].meta().max_y() ) continue;
      
      int row = adc_v[0].meta().row( imgcoord_v[3], __FILE__, __LINE__  );


      // we move through a neighborhood
      for (int dr=0; dr<=0; dr++ ) {
        int r = row+dr;
        if ( r<0 || r>=(int)adc_v[0].meta().rows() ) continue;

        for (int p=0; p<3; p++ ) {
          
          int col = adc_v[p].meta().col( imgcoord_v[p], __FILE__, __LINE__ );  
          for (int dc=-5; dc<=5; dc++) {
            int c = col+dc;
            if ( c<0 || c>=(int)adc_v[p].meta().cols() ) continue;
      
            if ( pix_visited_v[p].pixel( r, c )>0 ) continue;
            float pixval = adc_v[p].pixel( r, c );
            if ( pixval<10.0 ) continue;

            // store pixel
            std::vector<int> pix = { r, c};
            float rad = sqrt( dr*dr + dc*dc );

            data.pixellist_vv[p].push_back( pix );
            data.pixelrad_vv[p].push_back( rad );
            data.totalq_v[p] += pixval;

            pix_visited_v[p].set_pixel( r, c, 10.0 );

          }//end of dc loop
        }//end of plane loop
      }//end of dr neighborhood

      if ( last_pos.size()>0 ) {

        double step_len = 0.;
        for (int i=0; i<3; i++ )
          step_len += (last_pos[i]-pos[i])*(last_pos[i]-pos[i]);
        step_len = sqrt(step_len);
        data.len_intpc_sce += step_len;
      }
      last_pos = pos;
      
    }//end of step loop

    return data;
  }
  
}
}
