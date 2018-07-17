#ifndef __FLOWCONTOURMATCH__
#define __FLOWCONTOURMATCH__

#include <vector>
#include <map>
#include <set>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "TH2D.h"

#include "ContourShapeMeta.h"
#include "ContourCluster.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {

  class FlowMatchData_t {
  public:
    FlowMatchData_t( int srcid, int tarid );
    
    virtual ~FlowMatchData_t() {}

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
    typedef std::array<int,2> SrcTarPair_t;
  
    FlowContourMatch();
    virtual ~FlowContourMatch();

    void createMatchData( const larlitecv::ContourCluster& contour_data, const larcv::Image2D& flow_img, const larcv::Image2D& src_adc, const larcv::Image2D& tar_adc );
    float scoreMatch( const FlowMatchData_t& matchdata );
    void scoreMatches( const larlitecv::ContourCluster& contour_data, int src_planeid, int tar_planeid );
    void dumpMatchData();
    void greedyMatch();
    TH2D& plotScoreMatrix();
    
    std::map< SrcTarPair_t, FlowMatchData_t > m_flowdata;
    //std::map< std::array<int,2>, int >  m_flowdata_map;
    //std::vector< FlowMatchData_t > m_flowdata_v;

    int m_src_ncontours;
    int m_tar_ncontours;
    double* m_score_matrix;
    TH2D* m_plot_scorematrix;

    struct TargetPix_t {
      float row;
      float col;
    };
    typedef std::vector<TargetPix_t> ContourTargets_t;
    std::map< int, ContourTargets_t > m_src_targets;

  };



}

#endif
