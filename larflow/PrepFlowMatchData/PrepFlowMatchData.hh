#ifndef __PREP_FLOW_MATCH_DATA_H__
#define __PREP_FLOW_MATCH_DATA_H__

/**
 * 
 * \defgroup PrepFlowMatchData
 *
 *
 */

#include <map>
#include <vector>

#include "TTree.h"

#include "larcv/core/Processor/ProcessBase.h"

namespace larflow {

  class FlowMatchMap {

  public:
    FlowMatchMap() {};
    virtual ~FlowMatchMap() {};

    void add_matchdata( int src_index,
                        const std::vector<int>& target_indices,
                        const std::vector<int>& truth_v );
    
    const std::vector<int>& getTargetIndices( int src_index ) const;
    const std::vector<int>& getTruthVector( int src_index )   const;

  protected:

    std::map< int, std::vector<int> > _target_map;
    std::map< int, std::vector<int> > _truth_map;
      
  };
  
  class PrepFlowMatchData : public larcv::ProcessBase {

  public:

    PrepFlowMatchData( std::string instance_name )
      : larcv::ProcessBase(instance_name),
      _ana_tree(nullptr)
    {};
    virtual ~PrepFlowMatchData() {};

    void configure( const larcv::PSet& );
    void initialize();
    bool process( larcv::IOManager& mgr );
    void finalize();

  protected:

    TTree* _ana_tree;
    
  };


}

#endif
