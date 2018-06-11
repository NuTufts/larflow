#ifndef __CACAEndPointFilter_h__
#define __CACAEndPointFilter_h__

/* -------------------------------------------------------------------------------
 * CACAEndPointFilter: ContourAStarClusterAlgo End Point Filter
 * 
 * Uses the ContourAStar algo to cluster a track around a proposed end point
 * We then use the information from the returned track to determine if
 * the end point is good or not
 *
 *
 * Initial author: Taritree Wongjirad (twongj01@tufts.edu)
 * History:
 *   2017/09/05 - initial writing
 * ------------------------------------------------------------------------------*/

// stdlib
#include <vector>
#include <set>

// larlite
#include "DataFormat/opflash.h"

// larcv
#include "DataFormat/Image2D.h"


// larlitecv
#include "TaggerTypes/BoundarySpacePoint.h"
#include "MCTruthTools/crossingPointsAnaMethods.h"

// dev
#include "ContourShapeMeta.h"
#include "ContourAStarClusterAlgo.h"

namespace larlitecv {

  class CACAEndPtFilter {
  public:
    CACAEndPtFilter();
    virtual ~CACAEndPtFilter() {};

    // PRIMARY METHODS CALLED BY USER

    // take a list of boundary points and defect-split contours to analyze which candidate points are good
    // also removes duplicate boundary points which sit on the same set of contours
    // internal variables that save past restuls are reset once this function is called.
    void evaluateEndPoints( const std::vector< const std::vector<larlitecv::BoundarySpacePoint>* >& sp_v, const std::vector< larlite::event_opflash* >& flash_v,
			    const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
			    const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,
			    const float max_dtick,
			    std::vector< std::vector<int> >& passes_filter );

    // given the input list of boundary points given to the above function and the results of the above,
    // return a vector of new boundary points filled with only those points that pass
    // also the 3D direction is stored and the position is moved more towards the boundary (assuming the type)
    std::vector< larlitecv::BoundarySpacePoint >  regenerateFitleredBoundaryPoints( const std::vector<larcv::Image2D>& img_v );

    // CORE INTERNAL METHODS

    // evaluates one boundary point
    bool isEndPointGood( const larlitecv::BoundarySpacePoint& pt, const larlite::opflash* associated_flash,
			 const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v,
			 const std::vector< std::vector<ContourShapeMeta> >& plane_contours_v,			 
			 const float max_dtick );

    bool wasLastEndPointDuplicate() { return m_last_was_duplicate; }; //< Analysis to determine if last cluster generated is a duplicate of past unique end points    

    // UTILITT/GET/SET METHODS
    larlitecv::ContourAStarClusterAlgo& getAlgo() { return m_caca; }; //< get the clustering algo driving much of the analysis

    larlitecv::ContourAStarCluster& getLastCluster() { return m_last_clusters.back(); }; // get last cluster produced by the algo
    void clearClusters() { m_last_clusters.clear(); }; //< clear contour clusters produced

    void setTruthInformation( const std::vector<larlitecv::TruthCrossingPointAna_t>& truthinfo,
			      const std::vector<larlitecv::RecoCrossingPointAna_t>& recoinfo ); //< provide truth-based analysis of true-crossing point locations
    
    void setVerbosity( int v ) { m_verbosity = v; }; //< 0=quiet; >0 dump text

    void makeDebugImage( bool makeit=true ) { fMakeDebugImage = makeit; }; //< while evaluateEndPoints(...) runs, generate two images showing the location of end points that pass/fail
    int numDebugImages() { return m_cvimg_rgbdebug.size(); }; //< info on debug image vector
    cv::Mat& getDebugImage( int index=0 ) { return m_cvimg_rgbdebug[index]; }; //< 0=good end points (or if no truth given, all end points); 1=bad end points
    
    void printStageTimes(); //< print to screen timing of certain stages in the analysis

    void setDebugSet( const std::vector<int>& ptindex );
    
  protected:
    
    larlitecv::ContourAStarClusterAlgo m_caca; //< instance of contour-based, 3D-testing clustering algorithm
    std::vector< larlitecv::ContourAStarCluster > m_last_clusters; //< list of clusters produced when testing to see if space point is actual boundary
    // instances below are generated in larlitecv/app/MCTruthTools/crossingPointsAnaMethods.h/.cxx
    const std::vector<larlitecv::TruthCrossingPointAna_t>* m_truthinfo_ptr_v; //< store truth information if given by user
    const std::vector<larlitecv::RecoCrossingPointAna_t>*  m_recoinfo_ptr_v;  //< store truth infomation analysis results if given by user
    bool fTruthInfoLoaded; //< did user provide truth info
    int m_verbosity; //< verbosity level
    bool m_last_was_duplicate; //< store result of wasLastEndPointDuplicate(...)

    std::vector<cv::Mat> m_cvimg_rgbdebug; //< debug image made using opencv, plotting candidate endpoints and indicating if pass (magenta) or failed (yellow)
    bool fMakeDebugImage; //< if flag set by user (using method above), then make this image
    std::set<int> m_debug_set;

    // Duplicate handling
    bool isDuplicateEndPoint( const larlitecv::ContourAStarCluster& seedcluster, const larlitecv::BoundarySpacePoint& sp ); ///< function scans existing clusters to determine if duplicate
    struct PastClusterInfo_t {
      // summarizes past contour cluster of end points
      std::vector< std::set<int> > plane_bmtcv_indices;
      PastClusterInfo_t( const larlitecv::ContourAStarCluster& cluster ) {
	plane_bmtcv_indices = cluster.m_bmtcv_indices;
	pos.resize(3,0);
	dir.resize(3,0);
	type = -1;
	vecindex = -1;
	flashindex = -1;
	passed = 0;
	truthmatched = -1;
	popflash = NULL;
      };
      std::vector<float> pos;
      std::vector<float> dir;
      int type;
      int vecindex;
      int flashindex;
      int passed;
      int truthmatched;
      const larlite::opflash* popflash;
    };
    std::vector< PastClusterInfo_t > m_past_info;

    // Profiling
    enum { kSeedMaking=0, kDuplicateEval, kClusterExtension, kDebugImages, kOverall, kNumStages };
    std::vector< float > m_stage_times;
    
  };

}


#endif
