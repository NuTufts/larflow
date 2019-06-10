#include <iostream>
#include <string>

// larcv
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

  std::string cropcfg = "ubcrop.cfg";
  
  larcv::IOManager io(larcv::IOManager::kREAD,"SparseInput");
  io.add_in_file( "test_sparseout_stitched.root" );
  io.initialize();

  larlite::storage_manager out_larlite(larlite::storage_manager::kWRITE );
  out_larlite.set_out_filename( "output_sparse_larflowhits.root" );
  out_larlite.open();

  int nentries = io.get_n_entries();

  std::cout << "Number of entries: " << nentries << std::endl;


  for ( size_t i=0; i<nentries; i++ ) {
    io.read_entry(i);

    // whole images
    larcv::EventImage2D* ev_wire = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"wire");
    auto const& adc_v = ev_wire->Image2DArray();
    larcv::EventSparseImage* ev_crop_dualflow = (larcv::EventSparseImage*)io.get_data(larcv::kProductSparseImage,"cropdualflow");
    auto const& dualflow_v = ev_crop_dualflow->SparseImageArray();
    std::cout << "ENTRY[" << i << "] num dualflow crops: " << dualflow_v.size() << std::endl;
    // for ( size_t i=0; i< dualflow_v.size(); i++ ) {
    //   auto const& spimg = dualflow_v[i];
    //   std::cout << "  dualflowdata[" << i << "] nfeatures over metas:" << std::endl;
    //   for ( size_t x=0; x<spimg.meta_v().size(); x++ )
    //     std::cout << "     " << spimg.meta_v().at(x).dump() << std::endl;
    // }

    // eventual goal
    std::vector<larlite::larflow3dhit> flowhit = larflow::makeFlowHitsFromSparseCrops( adc_v,
                                                                                       dualflow_v,
                                                                                       10.0, cropcfg );

    larlite::event_larflow3dhit* ev_hitout
      = (larlite::event_larflow3dhit*)out_larlite.get_data(larlite::data::kLArFlow3DHit,"flowhits");
    int idx = 0;
    for ( auto& hit : flowhit ) {
      hit.idxhit = idx;
      ev_hitout->emplace_back( std::move(hit) );
      idx++;
    }

    out_larlite.set_id( ev_wire->run(), ev_wire->subrun(), ev_wire->event() );
    out_larlite.next_event(true);
    break;
  }

  out_larlite.close();
    
  return 0;
}
