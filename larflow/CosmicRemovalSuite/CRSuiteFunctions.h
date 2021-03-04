#ifndef __CRSUITEFUNCTIONS_H__
#define __CRSUITEFUNCTIONS_H__
// larutil
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/TimeService.h"
#include "ublarcvapp/dbscan/sDBScan.h"
// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/Base/larcv_logger.h"

// ROOT
#include "TH2D.h"
#include "TCanvas.h"
namespace larflow {
namespace cosmicremovalsuite {

class Cosmic_Products {
public:
  Cosmic_Products();
  Cosmic_Products(larcv::ImageMeta meta);
  ~Cosmic_Products() {}

  larcv::ImageMeta meta;

  void get_maskrcnn_image(larcv::Image2D &img, larcv::Image2D &adc_im, std::vector< larcv::ClusterMask > clustermask_v, bool flip_yaxis);


};

struct cluster_t {

  std::vector< std::vector<float> > points_v;        ///< vector of 3D space points in (x,y,z) coodinates
  std::vector< std::vector<int>   > imgcoord_v;      ///< vector of image coordinates (U,V,Y,tick)
  std::vector< int >                hitidx_v;        ///< vector of index of container this space point comes from
  std::vector< std::vector<float> > pca_axis_v;      ///< principle component axes
  std::vector<float>                pca_center;      ///< mean of the space points
  std::vector<float>                pca_eigenvalues; ///< eigenvalues of the principle components
  std::vector<int>                  ordered_idx_v;   ///< index of points_v, ordered by projected pos on 1st pca axis
  std::vector<float>                pca_proj_v;      ///< projection of point onto pca axis, follows ordered_idx_v
  std::vector<float>                pca_radius_v;    ///< distance of point from 1st pc axis, follows ordered_idx_v
  std::vector< std::vector<float> > pca_ends_v;      ///< points on 1st pca-line out to the maximum projection distance from center
  std::vector< std::vector<float> > bbox_v;          ///< axis-aligned bounding box. calculated along with pca
  float                             pca_max_r;       ///< maximum radius of points from the 1st PC axis
  float                             pca_ave_r2;      ///< average r2 of points from the first PC axis
  float                             pca_len;         ///< distance between min and max points along the first PC axis

};

std::vector<int> SSNET_Thresh_Counts(std::vector<double> threshes_v, larcv::Image2D const &ssnet_im, std::vector<std::vector<double> > hits_vv);

double pixel_removal_fraction(larcv::Image2D &adc_im, larcv::Image2D &tag_im1, larcv::Image2D &tag_im2, double adc_thresh=10.0, double tag1_thresh=0.0,double tag2_thresh=0.0);

double pixel_removal_fraction(larcv::Image2D &adc_im, larcv::Image2D &tag_im, double adc_thresh=10.0, double tag_thresh=0.0);

double pixel_removal_fraction(std::vector< std::vector<double> > &adc_pts_vv, larcv::Image2D &tag_im1, larcv::Image2D &tag_im2,double adc_thresh=10.0, double tag1_thresh=0, double tag2_thresh=0);

double pixel_removal_fraction(std::vector< std::vector<double> > &adc_pts_vv, larcv::Image2D &tag_im, double adc_thresh=10.0, double tag_thresh=0.);

void make_event_disp(larcv::Image2D &img, std::string name="test",double max_val=100);

void cluster_sdbscan_spacepoints( const std::vector< std::vector<float> >& hit_v,
                                    std::vector< cluster_t >& cluster_v,
                                    const float maxdist, const int minsize, const int maxkd );

std::vector< std::vector<double> > get_list_nonzero_pts(larcv::Image2D &adc_im,double adc_thresh=10.0,int min_row=0,int max_row=-1,int min_col=0, int max_col=-1);

int IsInsideBoundaries(std::vector<double> const& point);


}
}//end namespaces
#endif
