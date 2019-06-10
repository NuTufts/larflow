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
   * connecting these contours. The source and target contours were intially intended
   * to be created using a full image (not crops)
   */
  class ContourFlowMatch_t {
    
  public:
    
    ContourFlowMatch_t( int srcid=-1, int tarid=-1 );
    
    virtual ~ContourFlowMatch_t() { matchingflow_map.clear(); }

    ContourFlowMatch_t( const ContourFlowMatch_t& s );          // copy constructor
    
    int src_ctr_id; // source contour id
    int tar_ctr_id; // target contour id
    float score;    // score for this connection

    bool operator < ( const ContourFlowMatch_t& rhs ) const {
      if ( this->score < rhs.score ) return true;
      return false;
    };

    /**
     * flow data that contributed to the match.
     */
    struct FlowPixel_t {
      int src_wire; //< src col in full image
      int tar_wire; //< tar col in full image moving to nearest contour
      int tar_orig; //< tar col in full image using full
      int tick; //< tick in full image
      int row;  //< row in full image
      float pred_miss; //< distance to nearest contour
      float dist2cropcenter; //< for source pixels with many predicts from overlapping crop, we trust those closest to center.
      FlowPixel_t()
      : src_wire(-1),
        tar_wire(-1),
        tar_orig(-1),
        tick(-1),
        row(-1),
        pred_miss(0.0),
        dist2cropcenter(0.0)
      {};
    };
    std::map<int, std::vector<FlowPixel_t> >  matchingflow_map;
    std::vector<FlowPixel_t>& getFlowPixelList( int src_index ) {
      auto it = matchingflow_map.find( src_index );
      if ( it==matchingflow_map.end() ) {
        matchingflow_map.insert( std::pair<int,std::vector<FlowPixel_t> >( src_index, std::vector<FlowPixel_t>() ) );
        it = matchingflow_map.find( src_index );
      }
      return it->second;
    };
    
  };

  typedef std::pair<int,int> SrcTarPair_t; //< pair of source and target contour indices

  class ContourFlowMatchDict_t : public std::map<SrcTarPair_t,ContourFlowMatch_t> {
  public:
    ContourFlowMatchDict_t()
      : index_map_initialized(false),
      src_ctr_pixel_v_initialized(false)
      {};
    virtual ~ContourFlowMatchDict_t() {};

    // map from image (row,col) position to contour index
    std::vector<int> src_ctr_index_map; ///< map from (r*ncols+c) -> source contour index (full image)
    std::vector<int> tar_ctr_index_map; ///< map from (r*ncols+c) -> source contour index (full image)
    bool index_map_initialized;

    // list of pixel indexes per source contour
    std::vector< std::vector<int> > src_ctr_pixel_v;
    bool src_ctr_pixel_v_initialized;
  };
  
};

#endif
