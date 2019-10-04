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
#include "larcv/core/Processor/ProcessFactory.h"

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
      _ana_tree(nullptr),
      _matchdata_v(nullptr)
    {};
    virtual ~PrepFlowMatchData() {
      // if ( _ana_tree ) delete _ana_tree;
      // if ( _matchdata_v ) delete _matchdata_v;
    };

    void configure( const larcv::PSet& );
    void initialize();
    bool process( larcv::IOManager& mgr );
    void finalize();

  protected:

    std::string _input_adc_producername;
    std::string _input_trueflow_producername;
    
    TTree* _ana_tree;
    std::vector< FlowMatchMap >* _matchdata_v;
    void _setup_ana_tree();

    std::map< int, std::vector<int> > _wire_bounds[2];
    void _extract_wire_overlap_bounds();    
  };

  class PrepFlowMatchDataFactory : public larcv::ProcessFactoryBase {
  public:
    PrepFlowMatchDataFactory() { larcv::ProcessFactory::get().add_factory("PrepFlowMatchData",this); };
    ~PrepFlowMatchDataFactory() {};
    larcv::ProcessBase* create(const std::string instance_name) { return new PrepFlowMatchData(instance_name); };
  };
  
  
}

#endif
