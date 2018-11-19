#include <iostream>
#include <string>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"


int main( int nargs, char** argv ) {

  std::string input_flowhits = argv[1];
  std::string input_supera   = argv[2];

  float threshold = 10;
  
  larlite::storage_manager io_larlite( larlite::storage_manager::kREAD );
  io_larlite.add_in_filename( input_flowhits );
  io_larlite.open();

  larcv::IOManager io_larcv( larcv::IOManager::kREAD );
  io_larcv.add_in_file( input_supera );
  io_larcv.initialize();

  int nentries = io_larlite.get_entries();

  for (int ientry=0; ientry<nentries; ientry++) {
    io_larlite.go_to(ientry);
    io_larcv.read_entry(ientry);

    // measure coverage of good, truth-labeled hits

    // get hits
    auto ev_larflowhits = (larlite::event_larflow3dhit*)io_larlite.get_data(larlite::data::kLArFlow3DHit,"flowhits");
    
    // get adc and ancestor img
    auto ev_adcimg   = (larcv::EventImage2D*)io_larcv.get_data("image2d","wire");
    auto ev_ancestor = (larcv::EventImage2D*)io_larcv.get_data("image2d","ancestor");
    auto ancestor_v = ev_ancestor->as_vector();
    
    std::vector< float > frac_charge_w_label; // fraction of charge w/ mc trackid label, for the source plane
    auto const& srcimg = ev_adcimg->as_vector()[2];
    larcv::Image2D marker( srcimg.meta() );
    marker.paint(0);
    
    // loop over hits and mark image with locations where hit has trackid
    for ( auto const& hit : *ev_larflowhits ) {

      if ( hit[0]==-1 && hit[0]==hit[1] && hit[0]==hit[2] ) continue; // bad hit
      if ( !srcimg.meta().contains( hit.tick, hit.srcwire ) ) continue; // out range
      
      int col = srcimg.meta().col( hit.srcwire );
      int row = srcimg.meta().row( hit.tick );

      int label = 1;
      if ( hit.trackid>=0 )
	label = 2;
      marker.set_pixel( row, col, label );
      
    }
    
    // count pixel charge with ids along with total
    int n_abovethresh_whit = 0;
    int n_abovethresh_nohit = 0;
    int n_abovethresh_hitwmclabel = 0;
    int n_abovethresh = 0;
    float pixval_abovethresh_whit = 0;
    float pixval_abovethresh_nohit = 0;
    float pixval_abovethresh_hitwmclabel = 0;    
    float pixval_abovethresh_total = 0.;

    for ( size_t row=0; row<srcimg.meta().rows(); row++) {
      for ( size_t col=0; col<srcimg.meta().cols(); col++) {
	float pixval  = srcimg.pixel(row,col);
	int markerval = marker.pixel(row,col);
	int ancestor_label = ancestor_v[2].pixel(row,col);
	if ( pixval>threshold ) {
	  if ( markerval>0 ) {
	    n_abovethresh_whit++;
	    pixval_abovethresh_whit += pixval;
	  }
	  else {
	    n_abovethresh_nohit = 0;
	    pixval_abovethresh_nohit += pixval;
	  }

	  if ( markerval>1 ) {
	    n_abovethresh_hitwmclabel++;
	    pixval_abovethresh_hitwmclabel += pixval;
	  }

	  pixval_abovethresh_total += pixval;
	  n_abovethresh++;
	}
      }
    }

    std::cout << "Entry " << ientry << std::endl;
    std::cout << "  Hit fractions" << std::endl;
    std::cout << "  -------------" << std::endl;
    std::cout << "  Fraction of pixels above threshold w/ hit: " << float(n_abovethresh_whit)/float(n_abovethresh) << std::endl;
    std::cout << "  Fraction of charge above threshold w/ hit: " << float(pixval_abovethresh_whit)/float(pixval_abovethresh_total) << std::endl;
    std::cout << std::endl;
    std::cout << "  Hit fractions" << std::endl;
    std::cout << "  -------------" << std::endl;    
    std::cout << "  Fraction of pixels w/ mc truth label: " << float(n_abovethresh_hitwmclabel)/float(n_abovethresh) << std::endl;
    std::cout << "  Fraction of charge w/ mc truth label: " << float(pixval_abovethresh_hitwmclabel)/float(pixval_abovethresh_total) << std::endl;
    
    // calculate fraction
    
  }// end of event loop

  std::cout << "End. Cleanup" << std::endl;
  io_larlite.close();
  io_larcv.finalize();

  
  return 0;
}
