#include <iostream>
#include <string>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"


int main( int nargs, char** argv ) {

  std::string input_flowhits = argv[1];
  std::string input_mcreco   = argv[2];  
  std::string input_supera   = argv[3];
  std::string output_flowhitsana = argv[4];

  float threshold = 10;
  
  larlite::storage_manager io_larlite( larlite::storage_manager::kREAD );
  io_larlite.add_in_filename( input_flowhits );
  io_larlite.add_in_filename( input_mcreco );
  io_larlite.open();

  larcv::IOManager io_larcv( larcv::IOManager::kREAD );
  io_larcv.add_in_file( input_supera );
  io_larcv.initialize();

  int nentries = io_larlite.get_entries();

  TFile fout( output_flowhitsana.c_str(), "new" );
  TTree eventana("eventana","Event-level measures");
  TTree trackana("trackana","MC Track measures");
  int run, subrun, event;
  float fracpix_whit;
  float fracpix_wmclabel;
  float fracpixval_whit;
  float fracpixval_wmclabel;

  eventana.Branch( "run", &run, "run/I" );
  eventana.Branch( "subrun", &subrun, "subrun/I" );
  eventana.Branch( "event", &event, "event/I" );
  eventana.Branch( "fracpix_whit", &fracpix_whit, "fracpix_whit/F" );
  eventana.Branch( "fracpix_wmclabel", &fracpix_wmclabel, "fracpix_wmclabel/F" );
  eventana.Branch( "fracpixval_whit", &fracpixval_whit, "fracpixval_whit/F" );
  eventana.Branch( "fracpixval_wmclabel", &fracpixval_wmclabel, "fracpixval_wmclabel/F" );

  int origin;
  int ancestorid;
  int pdg;
  float MeV;
  float track_fracpixval_total;
  float track_fracpix_whit;  
  float track_fracpix_wmclabel;
  float track_fracpixval_whit;
  float track_fracpixval_wmclabel;

  trackana.Branch( "run", &run, "run/I" );
  trackana.Branch( "subrun", &subrun, "subrun/I" );
  trackana.Branch( "event", &event, "event/I" );
  trackana.Branch( "MeV",    &MeV,    "MeV/F" );
  trackana.Branch( "ancestorid", &ancestorid, "ancestorid/I");  
  trackana.Branch( "pdg", &pdg, "pdg/I" );
  trackana.Branch( "origin", &origin, "origin/I" );  
  trackana.Branch( "pixval_total",        &track_fracpixval_total,    "pixval_total/F" );  
  trackana.Branch( "fracpix_whit",        &track_fracpix_whit,        "fracpix_whit/F" );
  trackana.Branch( "fracpix_wmclabel",    &track_fracpix_wmclabel,    "fracpix_wmclabel/F" );
  trackana.Branch( "fracpixval_whit",     &track_fracpixval_whit,     "fracpixval_whit/F" );
  trackana.Branch( "fracpixval_wmclabel", &track_fracpixval_wmclabel, "fracpixval_wmclabel/F" );
  

  for (int ientry=0; ientry<nentries; ientry++) {
    io_larlite.go_to(ientry);
    io_larcv.read_entry(ientry);

    run = io_larlite.run_id();
    subrun = io_larlite.subrun_id();
    event = io_larlite.event_id();    
    
    // measure coverage of good, truth-labeled hits

    // get hits
    auto ev_larflowhits = (larlite::event_larflow3dhit*)io_larlite.get_data(larlite::data::kLArFlow3DHit,"flowhits");

    // get mc track/shower objects
    auto ev_mchits_track  = (larlite::event_mctrack*) io_larlite.get_data(larlite::data::kMCTrack, "mcreco");
    auto ev_mchits_shower = (larlite::event_mcshower*)io_larlite.get_data(larlite::data::kMCShower,"mcreco");    

    // collect ancestor info
    std::map< int, float > ancestor_mev;
    std::map< int, float > ancestor_pdg;
    std::map< int, int >   ancestor_origin;    
    for ( auto const& mctrack : *ev_mchits_track ) {

      if ( mctrack.Origin()==1 ) {
	ancestor_pdg[mctrack.TrackID()] = -1; // neutrino
	ancestor_origin[mctrack.TrackID()] = 1;
      }
      else if (mctrack.Origin()==2 && mctrack.AncestorTrackID()==mctrack.TrackID() ) {
	ancestor_pdg[mctrack.TrackID()] = mctrack.PdgCode();
	ancestor_origin[mctrack.TrackID()] = 2;	
      }

    }
    for ( auto const& mcshower : *ev_mchits_shower ) {

      if ( mcshower.Origin()==1 ) {
	ancestor_pdg[mcshower.TrackID()] = -1; // neutrino
      }
      else if (mcshower.Origin()==2 && mcshower.AncestorTrackID()==mcshower.TrackID() ) {
	ancestor_pdg[mcshower.TrackID()] = mcshower.PdgCode();
      }
      
    }
    
    
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

    // completion counts per ancestorid
    std::map< int, float > pixval_ancestorid_total;
    std::map< int, float > pixval_ancestorid_whit;
    std::map< int, float > pixval_ancestorid_wmclabel;

    // the neutrion bin
    pixval_ancestorid_total[-1] = 0;
    pixval_ancestorid_whit[-1] = 0;
    pixval_ancestorid_wmclabel[-1] = 0;    

    for ( size_t row=0; row<srcimg.meta().rows(); row++) {
      for ( size_t col=0; col<srcimg.meta().cols(); col++) {
	float pixval  = srcimg.pixel(row,col);
	int markerval = marker.pixel(row,col);
	int ancestor_label = ancestor_v[2].pixel(row,col);
	auto it_pdg = ancestor_pdg.find( ancestor_label );
	if ( it_pdg==ancestor_pdg.end() ) continue;
	if ( it_pdg->second==-1 )
	  ancestor_label = -1;
	
	// per image metrics
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


	  // per ancestorid metric
	  if ( pixval_ancestorid_total.find( ancestor_label )==pixval_ancestorid_total.end() ) {
	    // create an entry in all three maps
	    pixval_ancestorid_total[ ancestor_label ]    = 0;
	    pixval_ancestorid_whit[ ancestor_label ]     = 0;
	    pixval_ancestorid_wmclabel[ ancestor_label ] = 0;	  
	  }
	  
	  pixval_ancestorid_total[ ancestor_label ]    += pixval;
	  if ( markerval>0 )
	    pixval_ancestorid_whit[ ancestor_label ]     += pixval;
	  if ( markerval>1 )	
	    pixval_ancestorid_wmclabel[ ancestor_label ] += pixval;
	  
	}// if above thresh
      }// end of col loop
    }//end of row loop

    fracpix_whit     = float(n_abovethresh_whit)/float(n_abovethresh);
    fracpix_wmclabel = float(pixval_abovethresh_whit)/float(pixval_abovethresh_total);
    fracpixval_whit     = float(n_abovethresh_hitwmclabel)/float(n_abovethresh);
    fracpixval_wmclabel = float(pixval_abovethresh_hitwmclabel)/float(pixval_abovethresh_total);
    std::cout << "Entry " << ientry << std::endl;
    std::cout << "  Hit fractions" << std::endl;
    std::cout << "  -------------" << std::endl;
    std::cout << "  Fraction of pixels above threshold w/ hit: " << fracpix_whit << std::endl;
    std::cout << "  Fraction of charge above threshold w/ hit: " << fracpix_wmclabel << std::endl;
    std::cout << std::endl;
    std::cout << "  Hit fractions" << std::endl;
    std::cout << "  -------------" << std::endl;    
    std::cout << "  Fraction of pixels w/ mc truth label: " << fracpixval_whit << std::endl;
    std::cout << "  Fraction of charge w/ mc truth label: " << fracpixval_wmclabel << std::endl;

    eventana.Fill();

    float mean_complete_whit      = 0.;
    float mean_complete_wmclabel  = 0.;
    int ntracks = 0;

    for ( auto const& it : pixval_ancestorid_total ) {
      if ( it.second==0 )
	continue;
      
      track_fracpixval_whit     = pixval_ancestorid_whit[it.first]/it.second;
      track_fracpixval_wmclabel = pixval_ancestorid_wmclabel[it.first]/it.second;
      track_fracpixval_total    = it.second;

      // requires ancestorid lookup (later)
      MeV = 0;
      ancestorid = it.first;
      if ( ancestorid>=0 ) {
	origin = ancestor_origin[it.first];
	pdg = ancestor_origin[it.first];
      }
      else {
	// neutrino
	origin = 1; 
	pdg = 0;  
      }

      // simple mean for now
      mean_complete_whit     += track_fracpixval_whit;
      mean_complete_wmclabel += track_fracpixval_wmclabel;

      trackana.Fill();
      ntracks += 1;
    }

    if ( ntracks>0 ) {
      mean_complete_whit /= float(ntracks);
      mean_complete_wmclabel /= float(ntracks);
    }
    
    std::cout << "mean complete w/ hit: " << mean_complete_whit << std::endl;
    std::cout << "mean complete w/ mclabel: " << mean_complete_wmclabel << std::endl;
    
    // calculate fraction
    
  }// end of event loop

  fout.Write();
  
  std::cout << "End. Cleanup" << std::endl;
  io_larlite.close();
  io_larcv.finalize();

  
  return 0;
}
