#ifndef __LARFLOW_MATCH_TRIPLET_PROCESSOR_H__
#define __LARFLOW_MATCH_TRIPLET_PROCESSOR_H__

#include "larcv/core/Processor/ProcessBase.h"
#include "larcv/core/Processor/ProcessFactory.h"
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"
#include "PrepMatchTriplets.h"

namespace larflow {

  /**
   * class that implements a larcv Processor for generating possible (u,v,y) triplets using PrepMatchTriplets class 
   *
   */
  class MatchTripletProcessor : public larcv::ProcessBase {

  public:

  MatchTripletProcessor(std::string instance_name)
      : larcv::ProcessBase(instance_name),
      _p_matchdata_v(nullptr),      
      _ana_tree(nullptr),
      _has_mc(false),
      _adc_treename("wire"),
      _chstatus_treename("wire"),
      _check_intersection(false)
      {};
    virtual ~MatchTripletProcessor() {};

    void configure( const larcv::PSet& );
    void initialize();
    bool process( larcv::IOManager& mgr );
    void finalize();

    std::vector< larflow::PrepMatchTriplets >* _p_matchdata_v;
    TTree* _ana_tree;
    bool   _has_mc;
    std::string _adc_treename;
    std::string _chstatus_treename;
    bool   _check_intersection;

    ublarcvapp::EmptyChannelAlgo _badchmaker;
      
  };

  /**
   * Factory class that builds MatchTripletProcessor instances for the LArCV process driver framework
   *
   */
  class MatchTripletProcessorFactory : public larcv::ProcessFactoryBase {
  public:
    MatchTripletProcessorFactory() { larcv::ProcessFactory::get().add_factory("MatchTripletProcessor",this); };
    ~MatchTripletProcessorFactory() {};
    larcv::ProcessBase* create(const std::string instance_name) { return new MatchTripletProcessor(instance_name); };
  };

}

#endif
