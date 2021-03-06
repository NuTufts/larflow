#ifndef __PREP_FLOW_MATCH_DATA_H__
#define __PREP_FLOW_MATCH_DATA_H__

/**
 * 
 * \defgroup PrepFlowMatchData
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <map>
#include <vector>

#include "TTree.h"

#include "larcv/core/Processor/ProcessBase.h"
#include "larcv/core/Processor/ProcessFactory.h"

#include "FlowMatchMap.hh"

namespace larflow {
  
  class PrepFlowMatchData : public larcv::ProcessBase {

  public:

    typedef enum { kU2V=0, kU2Y, kV2U, kV2Y, kY2U, kY2V, kNumFlows } FlowDir_t;
    
    PrepFlowMatchData( std::string instance_name );
    virtual ~PrepFlowMatchData() {
      // if ( _ana_tree ) delete _ana_tree;
      // if ( _matchdata_v ) delete _matchdata_v;
    };

    void configure( const larcv::PSet& );
    void initialize();
    bool process( larcv::IOManager& mgr );
    void finalize();

    static std::string getFlowDirName( FlowDir_t flowdir );
    void setSourcePlaneIndex( int index )        { _source_plane=index; };
    void setADCproducer( std::string name )      { _input_adc_producername=name; };
    void setChStatusProducer( std::string name ) { _input_chstatus_producername=name; };
    void setLArFlowproducer( std::string name )  { _input_trueflow_producername=name; };
    void hasMCtruth( bool hasmc )  { _has_mctruth=hasmc; };
    void useSoftTruthVector( bool usesoft ) { _use_soft_truth = usesoft; };
    void useAnaTree( bool useana ) { _use_ana_tree=useana; };
    void useGapCh( bool use=true ) { _use_gapch=use; };

    const std::vector<FlowMatchMap>& getMatchData() const;

  protected:

    std::string _input_adc_producername;
    std::string _input_chstatus_producername;
    std::string _input_trueflow_producername;
    bool        _has_mctruth;
    bool        _use_ana_tree;
    bool        _use_soft_truth;
    int         _positive_example_distance;
    int         _source_plane;
    bool        _use_3plane_constraint;
    bool        _debug_detailed_output;
    bool        _use_gapch;
    
    TTree* _ana_tree;
    std::vector< FlowMatchMap >* _matchdata_v;
    int _nfalse_pairs[2];
    int _ntrue_pairs[2];
    void _setup_ana_tree();

    std::map< int, std::vector<int> > _wire_bounds[2];
    void _extract_wire_overlap_bounds();

    int target_plane[2];
    FlowDir_t _flowdirs[2];

    // each entry corresponds to FlowDir_t index
    const int _source_planes[6] = { 0, 0, 1, 1, 2, 2 };
    const int _target_planes[6] = { 1, 2, 0, 2, 0, 1 };
    const int _other_planes[6]  = { 2, 1, 2, 0, 1, 0 };


    // methods
    // -------

    void _makeMatchabilityImage( const larcv::Image2D& srcimg,
                                 const std::vector<const larcv::Image2D*>& tarimg_v,
                                 const std::vector<const larcv::Image2D*>& flowimg_v,
                                 std::vector<larcv::Image2D>& matchability_v );

    // bad channel image. allow way to set it.
  protected:
    std::vector<const larcv::Image2D*> _pbadch_v;
  public:
    void provideBadChannelImages( const std::vector<larcv::Image2D>& badch_v ) {
      _pbadch_v.clear();
      for ( auto const& img : badch_v ) {
        _pbadch_v.push_back( &img );
      }
    };
    
    
  };

  class PrepFlowMatchDataFactory : public larcv::ProcessFactoryBase {
  public:
    PrepFlowMatchDataFactory() { larcv::ProcessFactory::get().add_factory("PrepFlowMatchData",this); };
    ~PrepFlowMatchDataFactory() {};
    larcv::ProcessBase* create(const std::string instance_name) { return new PrepFlowMatchData(instance_name); };
  };
  
  
}

#endif
