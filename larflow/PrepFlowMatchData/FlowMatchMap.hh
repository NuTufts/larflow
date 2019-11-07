#ifndef __FLOW_MATCH_MAP_H__
#define __FLOW_MATCH_MAP_H__

#include <map>
#include <vector>

namespace larflow {

  class FlowMatchMap {

  public:
    
    FlowMatchMap();
    FlowMatchMap( int source_plane, int target_plane );
    virtual ~FlowMatchMap() {};

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

    int get_source_plane_index() { return _source_plane; };
    int get_target_plane_index() { return _target_plane; };
    int get_other_plane_index()  { return _other_plane; };
    
  protected:

    int _source_plane;
    int _target_plane;
    int _other_plane;
    
    std::map< int, std::vector<int> > _target_map;
    std::map< int, std::vector<int> > _truth_map;
      
  };
    
}

#endif
