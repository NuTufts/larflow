#include "FlowMatchMap.hh"

#include <sstream>

namespace larflow {

  // ---------------------------------------------------------
  // FLOW MATCH MAP CLASS

  FlowMatchMap::FlowMatchMap()
    : _source_plane(-1),
      _target_plane(-1)
  {
  }
  
  FlowMatchMap::FlowMatchMap( int source_plane_index, int target_plane_index )
    : _source_plane(source_plane_index),
      _target_plane(target_plane_index)
  {
  }
  
  void FlowMatchMap::add_matchdata( int src_pixel_index,
                                    const std::vector<int>& target_indices,
                                    const std::vector<int>& truth_v ) {

    if ( truth_v.size()!=target_indices.size() ) {
      throw std::runtime_error( "truth and target index vectors not the same size" );
    }
    
    _target_map[src_pixel_index] = target_indices;
    _truth_map[src_pixel_index]  = truth_v;
    
  }

  const std::vector<int>& FlowMatchMap::getTargetIndices( int src_index ) const {

    auto it = _target_map.find( src_index );
    if ( it==_target_map.end() ) {
      std::stringstream msg;
      msg << "did not find source index=" << src_index << ".";
      throw std::runtime_error( msg.str() );
    }

    return it->second;
  }

  const std::vector<int>& FlowMatchMap::getTruthVector( int src_index ) const {

    auto it = _truth_map.find( src_index );
    if ( it==_truth_map.end() ) {
      std::stringstream msg;
      msg << "did not find source index=" << src_index << ".";
      throw std::runtime_error( msg.str() );
    }
    
    return it->second;
  }

}
