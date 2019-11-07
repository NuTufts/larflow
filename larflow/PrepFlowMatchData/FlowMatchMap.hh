#ifndef __FLOW_MATCH_MAP_H__
#define __FLOW_MATCH_MAP_H__

#include <map>
#include <vector>

namespace larflow {

  class FlowMatchMap {

  public:
    FlowMatchMap()
    {};
    virtual ~FlowMatchMap() {
    };

    void add_matchdata( int src_index,
                        const std::vector<int>& target_indices,
                        const std::vector<int>& truth_v );
    
    int nsourceIndices() const { return _target_map.size(); };

    // retrieve candidate matches to source pixel via index
    const std::vector<int>& getTargetIndices( int src_index ) const;
    const std::vector<int>& getTruthVector( int src_index )   const;

    // retrieve candidate matches to source image via target index
    //const std::vector<int>& getTargetIndicesFromSourcePixel( int col, int row ) const;
    //const std::vector<int>& getTruthVectorFromSourcePixel( int col, int row ) const;
    
  protected:

    std::map< int, std::vector<int> > _target_map;
    std::map< int, std::vector<int> > _truth_map;
      
  };
    
}

#endif
