#include "NuSelShowerGapAna2D.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace reco {

  void NuSelShowerGapAna2D::analyze( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll,
                                     larflow::reco::NuVertexCandidate& nuvtx,
                                     larflow::reco::NuSelectionVariables& output )
  {

    // steps
    // we crop around the vertex.
    // we then make simply connected clusters in 2D.
    // we match the 3D shower prongs to the clusters
    // we match the vertex to a cluster
    // we then determine if the shower clusters are connected to the vertex cluster

    const int NPASSES = 3;
    const int CROP_WIDTH=20;
    
    // retrieve the images
    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();
    auto const& meta =  adc_v.front().meta();

    output.plane_connected_on_pass.clear();
    output.plane_connected_on_pass.resize(adc_v.size(),0);
    output.nplanes_connected = 0;

    if ( nuvtx.shower_v.size()==0)
      return; // do nothing
       

    // crop around the vertex, store in the following container
    std::vector<cv::Mat> crop_v;
    
    // define row range, which is inclusive [minrow,max_row]
    int vtxRow = meta.row( nuvtx.tick, __FILE__, __LINE__ );
    int min_row = vtxRow-CROP_WIDTH;
    int max_row = vtxRow+CROP_WIDTH;

    // check bounds
    if (min_row<0) min_row = 0;
    if ( max_row>=(int)meta.rows() )
      max_row = (int)meta.rows()-1;

    float min_tick = meta.pos_y(min_row);
    float max_tick = meta.pos_y(max_row);

    int nrows = max_row-min_row+1;
    int crop_vtx_row = vtxRow-min_row;

    std::vector<int> plane_common_contour_on_pass_v( adc_v.size(), -1 );

    // loop over wire planes
    for (int p=0; p<3; p++) {
      auto const& adc = adc_v[p];
      
      int min_col = nuvtx.col_v[p]-CROP_WIDTH;
      int max_col = nuvtx.col_v[p]+CROP_WIDTH;
      if ( min_col<0) min_col = 0;
      if ( max_col>=(int)meta.cols() ) max_col=(int)meta.cols()-1;
      int ncols = max_col-min_col+1;

      int crop_vtx_col = nuvtx.col_v[p]-min_col;
      
      cv::Mat cvcrop(nrows,ncols,CV_8UC1);
      unsigned char* px_ptr = (unsigned char*)cvcrop.data;
      for (int r=0; r<nrows; r++) {
        int orig_img_row = min_row+r;        
        for (int c=0; c<ncols; c++) {
          int orig_img_col = min_col+c;
          float q = adc.pixel(orig_img_row,orig_img_col,__FILE__,__LINE__);
          if ( q<0 ) q = 0; // threshold
          if ( q>=255 ) q = 255;
          px_ptr[ r*ncols + c ] = (uchar)((int)q);
        }
      }
      // threshold
      cv::threshold( cvcrop, cvcrop, 10, 255, cv::THRESH_BINARY );

      // we do the check over NPASSES, dilating each time.
      // we mark the dilation factor needed to merge the shower and vertex cluster
      for (int ipass=0; ipass<NPASSES; ipass++) {

        if ( ipass>0 )
          cv::dilate( cvcrop, cvcrop, cv::Mat(), cv::Point(-1,-1), 1, 1, 1 );
        
        // make contours
        std::vector<std::vector<cv::Point> >  contour_v;
        cv::findContours( cvcrop, contour_v, cv::RETR_LIST, cv::CHAIN_APPROX_NONE );

        //LARCV_DEBUG() << "Plane[" << p << "] pass[" << ipass << "] number of contours: " << contour_v.size() << std::endl;

        // find the vertex cluster, should be where?
        int vtx_ctr_index = -1;
        float best_dist = -1e6;
        int ictr = 0;
        cv::Point2f vtxpt( (float)crop_vtx_col, (float)crop_vtx_row );        
        for (auto& contour2d : contour_v) {
          double dist = cv::pointPolygonTest( contour2d, vtxpt, true );
          LARCV_DEBUG() << " ctr[" << ictr << "] test dist = " << dist << std::endl;
          if ( dist>-5 && dist>best_dist ) {
            vtx_ctr_index = ictr;
            best_dist = dist;
          }
          if ( dist>=0 ) {
            // inside, so stop
            break;
          }
          ictr++;
        }// end of contour loop

        if ( vtx_ctr_index<0 ) {
          // no contour found
          LARCV_DEBUG() << "No contour close enough to vtx: best dist = " << best_dist << std::endl;
          continue;
        }
        else {
          LARCV_DEBUG() << "Distance from vertex to closest cluster: best dist=" << best_dist << std::endl;
        }

        // now that the vtx contour is defined
        // we loop through each shower and each hit
        // we test for landing on/inside the contour
        auto& vertex_contour = contour_v[vtx_ctr_index];
        bool inside_hit_found = false;
        for ( size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++ ) {
          for ( auto& hit : nuvtx.shower_v[ishower] ) {

            if ( hit.tick<min_tick || hit.tick>max_tick ) continue;
            
            int hit_row = meta.row( hit.tick );
            int hit_col = hit.targetwire[p];

            // test if inside crop
            if ( hit_row>=min_row && hit_row<=max_row
                 && hit_col>=min_col && hit_col<=max_col ) {
              
              cv::Point2f shrpt( (float)hit_col-min_col, (float)hit_row-min_row );
              double hit_dist = cv::pointPolygonTest( vertex_contour, shrpt, true );
              //LARCV_DEBUG() << "  - hit-dist: " << hit_dist << std::endl;
              if (hit_dist>=0 )
                inside_hit_found = true;
            }//end of bounding box test
            
            if ( inside_hit_found ) {              
              break;
            }
            
          } //end of hit loop

          if (inside_hit_found)
            break;
        }//end of shower loop

        if ( inside_hit_found ) {
          plane_common_contour_on_pass_v[p] = ipass;
          break;
        }        
      }//end of PASS LOOP
      
    }// end of plane loop
    
    for (size_t p=0; p<plane_common_contour_on_pass_v.size(); p++ ) {
      LARCV_INFO() << "Plane[" << p << "] Found vertex and shower on common contour on PASS="
                   << plane_common_contour_on_pass_v[p] << std::endl;
      output.plane_connected_on_pass[p] = plane_common_contour_on_pass_v[p]+1;
      if ( output.plane_connected_on_pass[p]>=1 )
        output.nplanes_connected++;
    }
      
  }
  
}
}
