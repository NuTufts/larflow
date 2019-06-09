#ifndef __CONTOUR_FLOW_MATCH_t_H__
#define __CONTOUR_FLOW_MATCH_t_H__

#include <map>
#include <vector>

namespace larflow {

  /**
   * Represents flow between pair of contours on different planes.
   *
   * Pair of contours represented by index of contours in ContourClusterAlgo
   * class.  A vector is provided to store the individual flow predictions
   * connecting these contours.
   */
  class ContourFlowMatch_t {
    
  public:
    
    ContourFlowMatch_t( int srcid=-1, int tarid=-1 );
    
    virtual ~ContourFlowMatch_t() { matchingflow_v.clear(); }

    ContourFlowMatch_t( const ContourFlowMatch_t& s );          // copy constructor
    
    int src_ctr_id; // source contour id
    int tar_ctr_id; // target contour id
    float score;    // score for this connection

    bool operator < ( const ContourFlowMatch_t& rhs ) const {
      if ( this->score < rhs.score ) return true;
      return false;
    };

    // flow data that contributed to the match
    // i.e. the pixel or hit data within the contour
    struct FlowPixel_t {
      int src_wire; 
      int tar_wire;
      int tar_orig;
      int tick;
      int row;
      float pred_miss;
    };
    std::vector< FlowPixel_t > matchingflow_v;
    
  };

  typedef std::pair<int,int> SrcTarPair_t; //< pair of source and target contour indices

  typedef std::map<SrcTarPair_t,ContourFlowMatch_t> ContourFlowMatchDict_t; //< collection of (src,tar) connections

};

#endif
