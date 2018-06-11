#ifndef __CONTOUR_ASTAR_CLUSTER_H__
#define __CONTOUR_ASTAR_CLUSTER_H__

/* ---------------------------------------------------
 * ContourAStarCluster
 * 
 * This algorithm clusters contours using astar to build
 * 3D model. The model is used to extend the track into
 * other contours. Takes BMTCV, the split contours as input.
 *
 * ---------------------------------------------------*/

#include <vector>
#include <set>
#include "DataFormat/Image2D.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "ContourShapeMeta.h"

namespace larlitecv {

  class ContourAStarCluster {
    friend class ContourAStarClusterAlgo; // builds/manipulates these objects
  public:
    ContourAStarCluster() {
      makeDebugImage(false);
      m_verbosity=0;
    };
    ContourAStarCluster( const std::vector<larcv::Image2D>& img_v, bool make_debug_img=false ) {
      makeDebugImage(make_debug_img);      
      setImageMeta(img_v);
      m_verbosity=0;
    };
    virtual ~ContourAStarCluster();

    int numPlanes() { return m_nplanes; };
    const std::vector< std::vector<float> >& getPath() const { return m_path3d; };
    void setVerbosity(int verb ) { m_verbosity=verb; };
    
    //protected:
  public: // temporary for debug
    std::vector< std::set<int> > m_bmtcv_indices; //< store indices of contours we've used
    std::vector< std::vector< const ContourShapeMeta*> > m_plane_contours; //< contours we've added to the cluster
    std::vector< larcv::ImageMeta > m_meta_v;
    std::vector< cv::Mat > m_cvimg_v;  //< stores binary image of pixels that are a part of the cluster
    std::vector< cv::Mat > m_cvpath_v; //< stores binary image of pixels that are a part of the path
    cv::Mat m_cvimg_debug;

    std::vector< std::vector< ContourShapeMeta > > m_current_contours;
    std::vector< std::vector< ContourShapeMeta > > m_path_contours;

    std::vector< std::vector<float> > m_path3d;

    int m_nplanes;
    int m_current_min;
    int m_current_max;
    std::vector< std::vector<int> > m_plane_rowminmax;
    std::vector< std::vector<int> > m_plane_colminmax;    
    bool fMakeDebugImage;

    void setImageMeta( const std::vector<larcv::Image2D>& img_v ); // set the size of the containers which have storage for each plane
    void addContour( int plane, const larlitecv::ContourShapeMeta* ctr, int idx );
    void updateCVImage();
    void updateClusterContour();
    void makeDebugImage( bool make=true ) { fMakeDebugImage = make; };
    void resetDebugImage( const std::vector<larcv::Image2D>& img_v );    
    std::vector<int> getOverlappingRowRange();
    bool getCluster3DPointAtTimeTick( const int row, const std::vector<larcv::Image2D>& img_v,
				      const std::vector<larcv::Image2D>& badch_v, bool use_badch,
				      std::vector<int>& imgcoords, std::vector<float>& pos3d );

    int m_verbosity;
    
  };
  
  class ContourAStarClusterAlgo {
    
  public:
    ContourAStarClusterAlgo() {
      m_stage_times.resize( kNumStages, 0.0 );
      m_verbosity=0;
      fMakeDebugImage=false;
    };
    virtual ~ContourAStarClusterAlgo() {};

    ContourAStarCluster buildClusterFromSeed( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
					      const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
					      const float min_dist );
    
    ContourAStarCluster makeSeedClustersFrom3DPoint( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
						     const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
						     const float min_dist );

    ContourAStarCluster makeCluster( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
				     const std::vector<larcv::Image2D>& badch_v,
				     const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
				     const float maxdist2cluster, const int maxloopsteps=3 );
    
    void extendSeedCluster( const std::vector<float>& pos3d, const std::vector<larcv::Image2D>& img_v,
			    const std::vector<larcv::Image2D>& badch_v,
			    const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
			    const int maxloopsteps, ContourAStarCluster& cluster );
    
    std::vector< std::set<int> > extendClusterUsingAStarPath( ContourAStarCluster& cluster, std::vector< std::vector<float> >& path3d,
							      const std::vector<larcv::Image2D>& img_v,
							      const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
							      const float distfromend, const float distextended, const float stepsize );
    
    void fillInClusterImage( ContourAStarCluster& cluster, const std::vector< std::vector<float> >& path3d,
			     const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
			     const std::vector< std::set<int> >& cluster_indices, const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
			     const float maxstepsize, const float tag_qthreshold, const int neighborhood );

    void makeDebugImage( bool make=true ) { fMakeDebugImage = make; };
    void printStageTimes();
    void setVerbosity( int verbose ) { m_verbosity=verbose; };
    
  protected:
    bool fMakeDebugImage;
    
    enum { kContourLoop=0, kPointPolyTest, kImagePrep, kAddContours, kCreateCluster, kNumStages };
    std::vector<float> m_stage_times;

    int m_verbosity;

    
    
  };

}

#endif
