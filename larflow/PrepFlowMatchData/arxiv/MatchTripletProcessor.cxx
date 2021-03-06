#include "MatchTripletProcessor.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"

namespace larflow {

  static MatchTripletProcessorFactory __global_MatchTripletProcessorFactory__;

  /**
   * configure algorithm using parameters from a configuration pset
   *
   */
  void MatchTripletProcessor::configure( const larcv::PSet& pset )
  {
    _has_mc   = pset.get<bool>("HasMC");
    _adc_treename      = pset.get<std::string>("ADCName");
    _chstatus_treename = pset.get<std::string>("ChStatusName");
    _check_intersection = pset.get<bool>("CheckIntersection");
  }

  /**
   * initialize class, allocating mem for triplets and instantiating output TTree
   *
   */
  void MatchTripletProcessor::initialize()
  {

    // allocate ouput vector
    _p_matchdata_v = new std::vector<larflow::PrepMatchTriplets>(1);
    
    // setup anatree
    char treename[50];
    sprintf(treename,"larmatchtriplet");
    
    _ana_tree = new TTree(treename,"Possible wire combinations");
    _ana_tree->Branch( "triplet_v",  _p_matchdata_v );
    
  }

  /**
   * process data for one event, gathering input from the given larcv::IOManager
   *
   */
  bool MatchTripletProcessor::process( larcv::IOManager& mgr )
  {

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, _adc_treename );
    auto const& adc_v  = ev_adc->Image2DArray();

    larcv::EventChStatus* ev_chstatus
      = (larcv::EventChStatus*)mgr.get_data( larcv::kProductChStatus, _chstatus_treename );

    larcv::EventImage2D* ev_larflow = nullptr;
    if (_has_mc ) {
      ev_larflow = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, "larflow" );
    }
    
    auto badch_v = _badchmaker.makeGapChannelImage( adc_v, *ev_chstatus,
                                                    4, 3, 2400, 1008*6, 3456, 6, 1,
                                                    1.0, 100, -1.0 );
    LARCV_INFO() << "made badch_v, size=" << badch_v.size() << std::endl;
    
    (*_p_matchdata_v)[0].process( adc_v, badch_v, 10.0, _check_intersection );
    if (_has_mc)
      (*_p_matchdata_v)[0].make_truth_vector( ev_larflow->Image2DArray() );
    
    int ntriples = (int)(*_p_matchdata_v)[0]._triplet_v.size();

    LARCV_NORMAL() << "produced " << ntriples << " triplets to test/train" << std::endl;
    _ana_tree->Fill();

    return true;
  }


  /**
   * Wrte the output TTree
   *
   */
  void MatchTripletProcessor::finalize()
  {
    _ana_tree->Write();
  }
    
  


}
