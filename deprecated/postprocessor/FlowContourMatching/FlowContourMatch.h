#ifndef __FLOWCONTOURMATCH__
#define __FLOWCONTOURMATCH__

#include <vector>
#include <map>
#include <set>
#include <array>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "TH2D.h"

#include "DataFormat/hit.h"
#include "DataFormat/chstatus.h"
#include "DataFormat/mctrack.h"

#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"

#include "ContourShapeMeta.h"
#include "ContourCluster.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"

// larlite data product
#include "DataFormat/larflow3dhit.h"

namespace larflow {

  class FlowMatchData_t {
  public:
    FlowMatchData_t( int srcid, int tarid );
    
    virtual ~FlowMatchData_t() { matchingflow_v.clear(); }

    FlowMatchData_t( const FlowMatchData_t& s );          // copy constructor
    
    int src_ctr_id;
    int tar_ctr_id;
    float score;


    // flow results that predicted the match
    struct FlowPixel_t {
      int src_wire;
      int tar_wire;
      int tick;
      int row;
      float pred_miss;
    };
    std::vector< FlowPixel_t > matchingflow_v;

    
  };

  
  class FlowContourMatch {
  public:

    // internal data structures and types
    // ----------------------------------
    typedef enum { kY2U=0, kY2V, kNumFlowDirs } FlowDirection_t; // indicates flow pattern
    static const int kSourcePlane[2];// = { 2, 2 };
    static const int kTargetPlane[2];// = { 0, 1 };

    typedef std::array<int,2> SrcTarPair_t;        // pair of source and target contour indices    
    struct HitFlowData_t {
      // information about a hit's match from source to target plane
      // filled in _make3Dhits function
      HitFlowData_t() : hitidx(-1),
	maxamp(-1.0),
	srcwire(-1),
	targetwire(-1),
	pixtick(-1),
	matchquality(-1),
	dist2center(-1),
	src_ctr_idx(-1),
	tar_ctr_idx(-1),
	endpt_score(-1),
	track_score(-1),
	shower_score(-1),
	renormed_track_score(-1),
	renormed_shower_score(-1),
	trackid(-1),
	truthflag(0),
	dWall(0)
      {
	X_truth.resize(3,-1);
      };
      int hitidx;       // index of hit in event_hit vector
      float maxamp;     // maximum amplitude
      int srcwire;      // source image pixel: column
      int targetwire;   // target image pixel: column
      int pixtick;      // image pixel: row
      int matchquality; // match quality (1,2,3)
      int dist2center;  // distance of source pixel to center of y
      int dist2charge;  // distance in columns from target pixel to matched charge pixel
      int src_ctr_idx;  // this becomes outdated once image changes up
      int tar_ctr_idx;  // this becomes outdated once image changes up
      float endpt_score;
      float track_score;
      float shower_score;
      float renormed_track_score;
      float renormed_shower_score;
      bool src_infill; // source infill
      bool tar_infill; // target infill
      std::vector<float> X; // 3D coordinates from larlite::geo
      std::vector<float> X_truth; // from mctrack projection
      int trackid; // from mctrack projection
      int truthflag;
      float dWall; //min dist to detector boundary
    };
    struct ClosestContourPix_t {
      // stores info about the contours nearby to the point where
      // the flow prediction lands in the target plane.
      // we use this info to sort
      // and choose the contour to build a 3D hit with
      int ctridx;
      int dist;
      int col;
      float scorematch;
      float adc;
    };
    struct PlaneHitFlowData_t {
      std::vector<HitFlowData_t> Y2U; // hitflow vector Y2U
      std::vector<HitFlowData_t> Y2V; // hitflow vector Y2V
      std::vector<int> consistency3d; // 3D consistency estimator (1,2,3,no)
      std::vector<float> dy; // sqrt(y1-y0)^2
      std::vector<float> dz; // sqrt(z1-z0)^2            
      bool ranY2U;
      bool ranY2V;
      void clear() {
	Y2U.clear();
	Y2V.clear();
	consistency3d.clear();
	dy.clear();
	dz.clear();
	ranY2U = false;
	ranY2V = false;
      };
    };
    
    // ------------------------------------------------
  
    FlowContourMatch();
    virtual ~FlowContourMatch();
    void clear( bool clear2d=true, bool clear3d=true, int flowdir=-1 ); // clear2d and clear3d

    // algorithm functions for User
    // -----------------------------

    // use this to turn pixels into hits. can use output in next function.
    // (use at beginning of event)
    void makeHitsFromWholeImagePixels( const larcv::Image2D& src_adc, larlite::event_hit& evhit_v, const float threshold );

    //use this to stitch infill crops together
    void stitchInfill( const larcv::Image2D& infill_crop,
		       larcv::Image2D& trusted,
		       larcv::Image2D& infill_whole,
		       const larcv::EventChStatus& ev_chstatus);

    // use this to mask and threshold infill image pixels. can use output in next function.
    void maskInfill( const larcv::Image2D& infill, const larcv::EventChStatus& ev_chstatus,
		     const float score_thresh, larcv::Image2D& masked_infill );

    // use this to add thresholded infill images to adc. can use output in next function.
    void addInfill( const larcv::Image2D& masked_infill, const larcv::EventChStatus& ev_chstatus,
		    const float threshold, larcv::Image2D& img_fill_v );

    // update the information for making 3D hits
    // -----------------------------------------
    // call once per subimage
    void fillPlaneHitFlow( const larlitecv::ContourCluster& contour_data,
			   const larcv::Image2D& src_adc,
			   const std::vector<larcv::Image2D>& tar_adc,
			   const std::vector<larcv::Image2D>& flow_img,
			   const larlite::event_hit& hit_v,
			   const float threshold,
			   bool runY2U = true,
			   bool runY2V = false);

    // match mctrack truth info to hits
    // ----------------------------------
    // call once per end of event (after all subimages have been processed)
    void mctrack_match(const larlite::event_mctrack& evtrack,
		       const std::vector<larcv::Image2D>& img_v);

    void mctrack_match(PlaneHitFlowData_t& plhit2flowdata,
		       const larlite::event_mctrack& evtrack,
		       const std::vector<larcv::Image2D>& img_v,
		       ::larutil::SpaceChargeMicroBooNE* psce=NULL,
		       const ::larutil::TimeService* ptsv=NULL);

    void label_mcid_w_ancestor_img(const std::vector<larcv::Image2D>& ancestor_v,
				   const std::vector<larcv::Image2D>& adcimg_v);


    // Get final output: larflow3dhit
    // -------------------------------    
    // call once per end of event (after all subimages have been processed)
    std::vector< larlite::larflow3dhit > get3Dhits_1pl( FlowDirection_t flowdir, bool makehits_for_nonmatches=true  );
    std::vector< larlite::larflow3dhit > get3Dhits_1pl( const std::vector<HitFlowData_t>& hit2flowdata, bool makehits_for_nonmatches=true );
    std::vector< larlite::larflow3dhit > get3Dhits_2pl( bool makehits_for_nonmatches=true, bool require_3Dconsistency=false );
    std::vector< larlite::larflow3dhit > get3Dhits_2pl( const PlaneHitFlowData_t& plhit2flowdata, bool makehits_for_nonmatches=true, bool require_3Dconsistency=false );

    // integrate endpoint/ssnet output
    // (call before getting 3d hits)
    // -------------------------------
    void integrateSSNetEndpointOutput( const std::vector<larcv::Image2D>& track_scoreimgs,
				       const std::vector<larcv::Image2D>& shower_scoreimgs,
				       const std::vector<larcv::Image2D>& endpt_scoreimgs );
    
    // integrate infill output
    // (call before getting 3d hits)
    //------------------------------
    void labelInfillHits( const std::vector<larcv::Image2D>& masked_infill);

    // diagnostic image
    // make stitched flow image using same information
    // that forms the 3d hits
    // --------------------------------------
    std::vector<larcv::Image2D> makeStitchedFlowImages( const std::vector<larcv::Image2D>& img_v );
    
    // algorithm sub-functions
    // ------------------------
    
  protected:
    void _match( FlowDirection_t flowdir,
		 const larlitecv::ContourCluster& contour_data,
		 const larcv::Image2D& src_adc,
		 const larcv::Image2D& tar_adc,
		 const larcv::Image2D& flow_img,
		 const larlite::event_hit& hit_v,
		 const float threshold );
    void _createMatchData( const larlitecv::ContourCluster& contour_data,
			   const larcv::Image2D& flow_img,
			   const larcv::Image2D& src_adc,
			   const larcv::Image2D& tar_adc,
			   const FlowDirection_t kflowdir );
    float _scoreMatch( const FlowMatchData_t& matchdata );
    void _scoreMatches( const larlitecv::ContourCluster& contour_data, int src_planeid, int tar_planeid, const FlowDirection_t kflowdir );
    void _greedyMatch(const FlowDirection_t kflowdir);
    void _make3Dhits( const larlite::event_hit& hit_v,
		      const larcv::Image2D& srcimg_adc,
		      const larcv::Image2D& tar_adc,
		      const int src_plane,
		      const int tar_plane,
		      const float threshold,
		      std::vector<HitFlowData_t>& hit2flowdata,
		      const FlowDirection_t kflowdir );

    void _fill_consistency3d(std::vector<HitFlowData_t>& Y2U,
			     std::vector<HitFlowData_t>& Y2V,
			     std::vector<int>& consistency3d,
			     std::vector<float>& dy,
			     std::vector<float>& dz);

    int _calc_consistency3d(float& dy,
			    float& dz);

    void _calc_coord3d(HitFlowData_t& hit_y2u,
		       std::vector<float>& X,
		       FlowDirection_t flowdir);

    void _calc_dist3d(std::vector<float>& X0,
		      std::vector<float>& X1,
		      float& dy,
		      float& dz);
    
    void _mctrack_to_tyz(const larlite::mctrack& truthtrack,
			 std::vector<std::vector<float>>& tyz,
			 std::vector<unsigned int>& trackid,
			 std::vector<double>& E,
			 std::vector<float>& dWall,
			 ::larutil::SpaceChargeMicroBooNE* psce,
			 const ::larutil::TimeService* ptsv);

    void _tyz_to_pixels(const std::vector<std::vector<float>>& tyz,
			const std::vector<unsigned int>& trackid,
			const std::vector<double>& E,
			const std::vector<float>& dWall,
			const larcv::Image2D& adc,
			std::vector<larcv::Image2D>& trackimg);

    
  public:
    // debug/visualization
    // -------------------
    void dumpMatchData();
    TH2D& plotScoreMatrix(const FlowDirection_t kflowdir);

    // edited copy of UBWireTool::getProjectedImagePixels
    // --------------------------------------------------
    std::vector<int> getProjectedPixel( const std::vector<float>& pos3d,
					const larcv::ImageMeta& meta,
					const int nplanes,
					const float fracpixborder=1.5 );

  public:
    // algorithm parameters 
    // --------------------
    int kTargetChargeRadius; // number of pixels around flow prediction to look for charge pixel

    // internal data members
    // ----------------------
    std::map< SrcTarPair_t, FlowMatchData_t > m_flowdata[2]; //< for each source,target contour pair, data about their connects using flow info

    int m_src_ncontours;      //< number of contours on source image
    int m_tar_ncontours[2];   //< number of contours on target image
    double* m_score_matrix[2];   //< scores between source and target contours using flow information
    TH2D* m_plot_scorematrix[2]; //< histogram of score matrix for visualization

    struct TargetPix_t {
      //< information to store target pixel information. target comes from flow predictions.
      float row;
      float col;
      float srccol;
    };
    typedef std::vector<TargetPix_t> ContourTargets_t;   //< list of target pixel info
    std::map< int, ContourTargets_t > m_src_targets[2];  //< for each source contour, a list of pixels in the source+target views that have been matched (one for each flowdir)

    const larcv::Image2D*   m_src_img;     // pointer to input source image
    const larcv::ImageMeta* m_srcimg_meta; // pointer to source image meta    
    const larcv::Image2D*   m_flo_img[2];     // pointer to input flow prediction
    const larcv::Image2D*   m_tar_img[2];     // pointer to input target image
    const larcv::ImageMeta* m_tarimg_meta[2]; // pointer to target image meta
    int* m_src_img2ctrindex; //< array associating (row,col) to source contours
    int* m_tar_img2ctrindex[2]; //< array associating (row,col) to target contours

    // key datamember in algo
    // stores info tying hits on source plane to flow determined after contour matching
    // stores for both flowdirections, saving info to help decide which one
    // should be used to set 3D position
    PlaneHitFlowData_t m_plhit2flowdata;

    //larutil
    const ::larutil::TimeService* m_ptsv;
    ::larutil::SpaceChargeMicroBooNE* m_psce;
  };



}

#endif
