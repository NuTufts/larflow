#ifndef __CRSUITEFUNCTIONS_CXX__
#define __CRSUITEFUNCTIONS_CXX__

#include "CRSuiteFunctions.h"
// larutil
// #include "larlite/LArUtil/LArProperties.h"
// #include "larlite/LArUtil/Geometry.h"
// #include "larlite/LArUtil/ClockConstants.h"
// #include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
// #include "larlite/LArUtil/DetectorProperties.h"
// #include "larlite/LArUtil/TimeService.h"
// #include "ublarcvapp/dbscan/sDBScan.h"
//
// // larcv
// #include "larcv/core/DataFormat/IOManager.h"
// #include "larcv/core/DataFormat/EventClusterMask.h"
// #include "larcv/core/DataFormat/EventSparseImage.h"
// #include "larcv/core/DataFormat/EventImage2D.h"
// #include "larcv/core/DataFormat/EventPGraph.h"
// #include "larcv/core/Base/larcv_logger.h"
//
// // ROOT
// #include "TH2D.h"
// #include "TCanvas.h"
namespace larflow {
namespace cosmicremovalsuite {

Cosmic_Products::Cosmic_Products()
: meta(larcv::ImageMeta())
{
// Do Nothing, Meta Made
}


Cosmic_Products::Cosmic_Products(larcv::ImageMeta meta_in)
:  meta(meta_in)
{
// Do Nothing, Meta Made
}


void Cosmic_Products::get_maskrcnn_image(larcv::Image2D &img,larcv::Image2D &adc_im, std::vector< larcv::ClusterMask > cmask_v, bool flip_yaxis) {
    int num_masks = cmask_v.size();
    img.set_pixel(1,1, 100);
    // std::cout << "Inside Get Maskrcnn Image Function\n";
    for (int m_idx = 0; m_idx < num_masks; m_idx++){
      for (int pt_idx = 0; pt_idx < cmask_v[m_idx].points_v.size();pt_idx++){
        int pt_x = cmask_v[m_idx].box.min_x() +1+ cmask_v[m_idx].points_v[pt_idx].x;
        int pt_y = cmask_v[m_idx].box.min_y() +1+ cmask_v[m_idx].points_v[pt_idx].y;
        if (flip_yaxis) pt_y = 1007-pt_y;
        //
        if (pt_y > 1007) {
          // std::cout << pt_y << " PT_Y Failing \n";
          pt_y = 1007;
        }
        if (pt_x > 3455) {
          // std::cout << pt_x << " PT_X Failing \n";
          pt_x = 3455;
        }
        if (pt_y < 0) {
          // std::cout << pt_y << " PT_Y Failing \n";
          pt_y = 0;
        }
        if (pt_x < 0) {
          // std::cout << pt_x << " PT_X Failing \n";
          pt_x = 0;
        }
        if (adc_im.pixel(pt_y,pt_x) > 0){
          // img.set_pixel(pt_y,pt_x, 2*(m_idx+1)); // For different colored masks
          // img.set_pixel(pt_y,pt_x, 1); // for one color all pix
          if (cmask_v[m_idx].type == 1){
            img.set_pixel(pt_y,pt_x, cmask_v[m_idx].probability_of_class);
          }
          else if (cmask_v[m_idx].type == 5){
            img.set_pixel(pt_y,pt_x, -1*cmask_v[m_idx].probability_of_class);
          }
        }
      }
    }
  return ;
}





std::vector< std::vector<double> > get_list_nonzero_pts(larcv::Image2D &adc_im,double adc_thresh,int min_row,int max_row,int min_col, int max_col){
  //This function takes and adc image and returns a vector of
  //point2d of pixels with value >10 to compare to tagged images faster
  std::vector< std::vector<double> > adc_pts_vv;
  adc_pts_vv.reserve(40000);
  int cols = adc_im.meta().cols();
  int rows = adc_im.meta().rows();
  if ((max_col != -1) && (max_col < cols)){cols = max_col;}
  if ((max_row != -1) && (max_row < rows)){rows = max_row;}
  if (min_row < 0) {min_row = 0;}
  if (min_col < 0) {min_col = 0;}
  for (double col = min_col; col<cols;col++){
    for (double row = min_row; row<rows;row++){
      double val = adc_im.pixel(row,col);
      if (val > adc_thresh) {
        std::vector<double> this_pt_v = {col,row,val};
        adc_pts_vv.push_back(this_pt_v);
      }
    }
  }
  return adc_pts_vv;
}


void make_event_disp(larcv::Image2D &img, std::string name, double max_val){
  int cols = img.meta().cols();
  int rows = img.meta().rows();
  int run = 0; int subrun = 0; int event = 0;
  TH2D ev_disp = TH2D("ev_disp","ev_disp",cols,0,cols,rows,0,rows);
  for (int col = 0; col<cols;col++){
    for (int row = 0; row<rows;row++){
      double val_y = img.pixel(row,col);
      if (val_y > 100.) val_y = 100.0;
      ev_disp.SetBinContent(col,row,val_y);
    }
  }
  TCanvas can("can", "histograms ", 3456, 1008);
  can.cd();
  ev_disp.SetTitle(Form("Image Run: %d Subrun: %d Event: %d",run,subrun,event));
  ev_disp.SetXTitle("Column (Wire)");
  ev_disp.SetYTitle("Row (6 Ticks)");
  ev_disp.SetOption("COLZ");
  ev_disp.SetMaximum(max_val);
  ev_disp.Draw("");
  can.SaveAs(Form("%s_%d_%d_%d.png",name.c_str(),run,subrun,event));
}


double pixel_removal_fraction(std::vector< std::vector<double> > &adc_pts_vv, larcv::Image2D &tag_im, double adc_thresh, double tag_thresh){
  double fraction_removed = 0.0;
  double adc_npix =0;
  double tag_npix =0;
  for (int pt_idx =0;pt_idx<adc_pts_vv.size(); pt_idx++){
    double col = adc_pts_vv[pt_idx][0];
    double row = adc_pts_vv[pt_idx][1];
    double adc_val = adc_pts_vv[pt_idx][2];
    double tag_val = tag_im.pixel(row,col);
    if (adc_val > adc_thresh) {
      adc_npix++;
      if (tag_val > tag_thresh)  tag_npix++;
    }
  }

  fraction_removed = tag_npix/adc_npix;
  return fraction_removed;
}
double pixel_removal_fraction(std::vector< std::vector<double> > &adc_pts_vv, larcv::Image2D &tag_im1, larcv::Image2D &tag_im2,double adc_thresh, double tag1_thresh, double tag2_thresh){
  double fraction_removed = 0.0;
  double adc_npix =0;
  double tag_npix =0;
  for (int pt_idx =0;pt_idx<adc_pts_vv.size(); pt_idx++){
    double col = adc_pts_vv[pt_idx][0];
    double row = adc_pts_vv[pt_idx][1];
    double adc_val = adc_pts_vv[pt_idx][2];
    double tag1_val = tag_im1.pixel(row,col);
    double tag2_val = tag_im2.pixel(row,col);
    if (adc_val > adc_thresh) {
      adc_npix++;
      if ((tag1_val > tag1_thresh) || (tag2_val > tag2_thresh))  tag_npix++;
    }
  }
  fraction_removed = tag_npix/adc_npix;
  return fraction_removed;
}

double pixel_removal_fraction(larcv::Image2D &adc_im, larcv::Image2D &tag_im, double adc_thresh, double tag_thresh){
  double fraction_removed = 0.0;
  int cols = adc_im.meta().cols();
  int rows = adc_im.meta().rows();
  double adc_npix =0;
  double tag_npix =0;
  for (int col = 0; col<cols;col++){
    for (int row = 0; row<rows;row++){
      double adc_val = adc_im.pixel(row,col);
      double tag_val = tag_im.pixel(row,col);
      if (adc_val > adc_thresh) {
        adc_npix++;
        if (tag_val > tag_thresh)  tag_npix++;
      }
    }
  }
  fraction_removed = tag_npix/adc_npix;
  return fraction_removed;
}
double pixel_removal_fraction(larcv::Image2D &adc_im, larcv::Image2D &tag_im1, larcv::Image2D &tag_im2, double adc_thresh, double tag1_thresh, double tag2_thresh){
  double fraction_removed = 0.0;
  int cols = adc_im.meta().cols();
  int rows = adc_im.meta().rows();
  double adc_npix =0;
  double tag_npix =0;
  for (int col = 0; col<cols;col++){
    for (int row = 0; row<rows;row++){
      double adc_val = adc_im.pixel(row,col);
      double tag1_val = tag_im1.pixel(row,col);
      double tag2_val = tag_im2.pixel(row,col);
      if (adc_val > adc_thresh) {
        adc_npix++;
        if ((tag1_val > tag1_thresh) || (tag2_val > tag2_thresh))  tag_npix++;
      }
    }
  }
  fraction_removed = tag_npix/adc_npix;
  return fraction_removed;
}

std::vector<int> SSNET_Thresh_Counts(std::vector<double> threshes_v, larcv::Image2D const &ssnet_im, std::vector<std::vector<double> > hits_vv){
  std::vector<int>  counts_v(threshes_v.size());
  int cols = ssnet_im.meta().cols();
  int rows = ssnet_im.meta().rows();
  for (int hit_idx=0;hit_idx<hits_vv.size();hit_idx++){
    int row = hits_vv.at(hit_idx).at(1);
    int col = hits_vv.at(hit_idx).at(0);
    double ssnet_val = ssnet_im.pixel(row,col);
    for (int t_idx = 0; t_idx < counts_v.size(); t_idx++){
      double thresh = threshes_v.at(t_idx);
      if (ssnet_val >= thresh) {
        counts_v.at(t_idx)++;
      }
    }
  }
  return counts_v;
}

void cluster_sdbscan_spacepoints( const std::vector< std::vector<float> >& hit_v,
                                    std::vector< cluster_t >& cluster_v,
                                    const float maxdist, const int minsize, const int maxkd )
  {

    clock_t start = clock();

    // convert points into list of floats
    std::vector< std::vector<float> > points_v;
    points_v.reserve( hit_v.size() );

    for ( auto const& lfhit : hit_v ) {
      std::vector<float> hit = { lfhit[0], lfhit[1], lfhit[2] };
      points_v.push_back( hit );
    }

    auto sdbscan = ublarcvapp::dbscan::SDBSCAN< std::vector<float>, float >();
    sdbscan.Run( &points_v, 3, maxdist, minsize );

    auto noise = sdbscan.Noise;
    auto dbcluster_v = sdbscan.Clusters;

    for (int ic=0; ic<(int)dbcluster_v.size();ic++) {
      // skip the last cluster, which are noise points
      auto const& cluster = dbcluster_v[ic];
      cluster_t c;
      c.points_v.reserve(cluster.size());
      c.imgcoord_v.reserve(cluster.size());
      c.hitidx_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster ) {
        // store 3d position and 2D image coordinates
        c.points_v.push_back( points_v.at(hitidx) );
        c.hitidx_v.push_back(hitidx);
      }
      cluster_v.emplace_back(std::move(c));
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;

    std::cout << "[cluster_sdbscan_spacepoints] made clusters: " << dbcluster_v.size() << " elpased=" << elapsed << " secs" << std::endl;
  }

int IsInsideBoundaries(std::vector<double> const& point){
  if (
       (point[0] <    0.001) || (point[0] >  255.999)
    || (point[1] < -116.499) || (point[1] > 116.499)
    || (point[2] <    0.001) || (point[2] > 1036.999)
    ){
    return 0;
  }
  else{
    return 1;
  };
}

}
}
//end namespaces
  #endif
