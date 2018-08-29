#ifndef __FLOWCONTOURMATCH__
#define __FLOWCONTOURMATCH__

#include <vector>
#include <map>
#include <set>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "TH2D.h"

#include "DataFormat/hit.h"

#include "ContourShapeMeta.h"
#include "ContourCluster.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "FlowMatchHit3D.h"

namespace larflow {

  class FlowMatchData_t {
  public:
    FlowMatchData_t( int srcid, int tarid );
    
    virtual ~FlowMatchData_t() { matchingflow_v.clear(); }

    FlowMatchData_t( const FlowMatchData_t& s );          // copy constructor
    //FlowMatchData_t& operator=(const FlowMatchData_t &s); // asignment operator
    
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
    typedef std::array<int,2> SrcTarPair_t;        // pair of source and target contour indices
    struct HitFlowData_t {
      // information about a hit's match from source to target plane
      // filled in _make3Dhits function
      HitFlowData_t() : hitidx(-1), maxamp(-1.0), srcwire(-1), targetwire(-1), pixtick(-1), matchquality(-1), dist2center(-1), src_ctr_idx(-1), tar_ctr_idx(-1) {};
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
    // ------------------------------------------------
  
    FlowContourMatch();
    virtual ~FlowContourMatch();
    void clear( bool clear2d=true, bool clear3d=true ); // clear2d and clear3d

    // algorithm functions

    // the unit that we match with the flow is the hit
    void match( FlowDirection_t flowdir,
		const larlitecv::ContourCluster& contour_data,
		const larcv::Image2D& src_adc,
		const larcv::Image2D& tar_adc,
		const larcv::Image2D& flow_img,
		const larlite::event_hit& hit_v,
		const float threshold );

    // the unit that we match with the flow is the pixel
    // we will use the source image to create hits, which we pass to the flow
    void matchPixels( FlowDirection_t flowdir,
		      const larlitecv::ContourCluster& contour_data,
		      const larcv::Image2D& src_adc, 
		      const larcv::Image2D& tar_adc,
		      const larcv::Image2D& flow_img,
		      const float threshold,
		      larlite::event_hit& hit_v );
    void makeHitsFromWholeImagePixels( const larcv::Image2D& src_adc, larlite::event_hit& evhit_v, const float threshold );
    std::vector< FlowMatchHit3D > get3Dhits( bool makehits_for_nonmatches=true );
    std::vector< FlowMatchHit3D > get3Dhits( const std::vector<HitFlowData_t>& hit2flowdata, bool makehits_for_nonmatches=true );
    
    // algorithm sub-functions
    void _createMatchData( const larlitecv::ContourCluster& contour_data,
			   const larcv::Image2D& flow_img,
			   const larcv::Image2D& src_adc,
			   const larcv::Image2D& tar_adc );
    float _scoreMatch( const FlowMatchData_t& matchdata );
    void _scoreMatches( const larlitecv::ContourCluster& contour_data, int src_planeid, int tar_planeid );
    void _greedyMatch();
    void _make3Dhits( const larlite::event_hit& hit_v,
		      const larcv::Image2D& srcimg_adc,
		      const larcv::Image2D& tar_adc,
		      const int src_plane,
		      const int tar_plane,
		      const float threshold,
		      std::vector<HitFlowData_t>& hit2flowdata );
    
    

    // debug/visualization
    void dumpMatchData();
    TH2D& plotScoreMatrix();


    // algorithm parameters
    // --------------------
    int kTargetChargeRadius; // number of pixels around flow prediction to look for charge pixel

    // internal data members
    // ----------------------
    std::map< SrcTarPair_t, FlowMatchData_t > m_flowdata; //< for each source,target contour pair, data about their connects using flow info

    int m_src_ncontours;      //< number of contours on source image
    int m_tar_ncontours;      //< number of contours on target image
    double* m_score_matrix;   //< scores between source and target contours using flow information
    TH2D* m_plot_scorematrix; //< histogram of score matrix for visualization

    struct TargetPix_t {
      //< information to store target pixel information. target comes from flow predictions.
      float row;
      float col;
      float srccol;
    }; 
    typedef std::vector<TargetPix_t> ContourTargets_t; //< list of target pixel info
    std::map< int, ContourTargets_t > m_src_targets;   //< for each source contour, a list of pixels in the source+target views that have been matched

    const larcv::Image2D*   m_src_img;     // pointer to input source image
    const larcv::Image2D*   m_flo_img;     // pointer to input flow prediction
    const larcv::ImageMeta* m_srcimg_meta; // pointer to source image meta
    const larcv::Image2D*   m_tar_img;     // pointer to input target image
    const larcv::ImageMeta* m_tarimg_meta; // pointer to target image meta
    int* m_src_img2ctrindex; //< array associating (row,col) to source contours
    int* m_tar_img2ctrindex; //< array associating (row,col) to target contours


    std::vector<HitFlowData_t> m_hit2flowdata;
    
  };



}

#endif
