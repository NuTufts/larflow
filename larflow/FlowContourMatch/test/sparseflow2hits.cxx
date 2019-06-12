#include <iostream>
#include <string>

// ROOT
#include "TApplication.h"

// larcv
#include "larcv/core/Base/larcv_logger.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/storage_manager.h"

// larflow
#include "larflow/FlowContourMatch/FlowContourMatch.h"

int main( int nargs, char** argv ) {

  std::cout << "================================" << std::endl;
  std::cout << "sparseflow2hits" << std::endl;
  std::cout << "--------------------------------" << std::endl;

  TApplication app("app",&nargs,argv);

  larcv::logger log("sparseflow2hits");

  std::string cropcfg = "ubcrop.cfg";
  
  larcv::IOManager io_fullview(larcv::IOManager::kREAD,"FullView",larcv::IOManager::kTickBackward);
  io_fullview.add_in_file( "larcvtruth-Run000002-SubRun002000.root" );
  io_fullview.initialize();
  
  larcv::IOManager io_sparse(larcv::IOManager::kREAD,"SparseInput");
  io_sparse.add_in_file( "test_larflow_sparseout.root" );
  io_sparse.initialize();
  
  larlite::storage_manager out_larlite(larlite::storage_manager::kWRITE );
  out_larlite.set_out_filename( "output_sparse_larflowhits.root" );
  out_larlite.open();

  int nentries = io_sparse.get_n_entries();

  log.send(larcv::msg::kINFO) << "Number of entries: " << nentries << std::endl;
  nentries = 1;

  for ( size_t i=0; i<nentries; i++ ) {
    
    io_sparse.read_entry(i);
    io_fullview.read_entry(i);    

    // whole images
    larcv::EventImage2D* ev_wire
      = (larcv::EventImage2D*)io_fullview.get_data(larcv::kProductImage2D,"wiremc");
    auto const& adc_v = ev_wire->Image2DArray();

    // truth whole images
    larcv::EventImage2D* ev_trueflow
      = (larcv::EventImage2D*)io_fullview.get_data(larcv::kProductImage2D,"larflow");
    auto const& trueflow_v = ev_trueflow->Image2DArray();

    // event chstatus
    larcv::EventChStatus* ev_status
      = (larcv::EventChStatus*)io_fullview.get_data(larcv::kProductChStatus,"wiremc");
    
    // output of sparse dualflow deploy
    larcv::EventSparseImage* ev_crop_dualflow
      = (larcv::EventSparseImage*)io_sparse.get_data(larcv::kProductSparseImage,"cropdualflow");
    auto const& dualflow_v = ev_crop_dualflow->SparseImageArray();

    
    log.send(larcv::msg::kINFO) << "ENTRY[" << i << "] num dualflow crops: " << dualflow_v.size() << std::endl;


    log.send(larcv::msg::kINFO) << "Make Reco Flow Hits" << std::endl;
      
    // RECO FLOW HITS
    std::vector<larlite::larflow3dhit> flowhit
      = larflow::makeFlowHitsFromSparseCrops( adc_v,
                                              dualflow_v,
                                              10.0,
                                              cropcfg );

    log.send(larcv::msg::kINFO) << "Make True Flow Hits" << std::endl;    

    // TRUTH FLOW HITS
    std::vector<larlite::larflow3dhit> trueflow
      = larflow::makeTrueFlowHitsFromWholeImage( adc_v,
                                                 *ev_status,
                                                 trueflow_v,
                                                 10.0,
                                                 "ubcroptrueflow.cfg" );

    // OUTPUT
    log.send(larcv::msg::kINFO) << "Save the output." << std::endl;
    
    larlite::event_larflow3dhit* ev_hitout
      = (larlite::event_larflow3dhit*)out_larlite.get_data(larlite::data::kLArFlow3DHit,"flowhits");

    int idx = 0;    
    for ( auto& hit : flowhit ) {
      hit.idxhit = idx;
      ev_hitout->emplace_back( std::move(hit) );
      idx++;
    }
    
    larlite::event_larflow3dhit* ev_truthout
      = (larlite::event_larflow3dhit*)out_larlite.get_data(larlite::data::kLArFlow3DHit,"trueflowhits");
    
    idx = 0;
    for ( auto& hit : trueflow ) {
      hit.idxhit = idx;
      ev_truthout->emplace_back( std::move(hit) );
      idx++;
    }
        
    out_larlite.set_id( ev_wire->run(), ev_wire->subrun(), ev_wire->event() );
    out_larlite.next_event(true);
  }
  
  out_larlite.close();
  
  return 0;
}
