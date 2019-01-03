/**
 * program to stitch dlcosmic crops for 
 *  1) infill
 *  2) ssnet (track/shower/endpt)
 *  x) larflow image produced by postprocessor, of course
 */

#include <iostream>
#include <string>
#include <vector>

// larcv2
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/pixelmask.h"

#include "SSNetStitcher.h"

void find_rse_entry( larcv::IOManager& io, int run, int subrun, int event, int& current_entry, std::string img2d_producer="wire" ) {
  // searches for run, subrun, entry in IOManager
  std::cout << "find_rse_entry[LARCV]: look for (" << run << "," << subrun << "," << event << ") current_entry=" << current_entry << std::endl;
  for (int ipass=0; ipass<2; ipass++) {
    bool found_match = false;
    for ( int ientry=current_entry; ientry<io.get_n_entries(); ientry++ ) {
      io.read_entry(ientry);
      auto evimg = (larcv::EventImage2D*)io.get_data("image2d",img2d_producer);
      if ( run==evimg->run() && subrun==evimg->subrun() && event==evimg->event() ) {
	found_match = true;
	current_entry = ientry;
	break;
      }
    }
    if ( !found_match ) current_entry=0;
    else break;
  }
  std::cout << "find_rse_entry[LARCV]: found (" << run << "," << subrun << "," << event << ") @ entry=" << current_entry << std::endl;
}

void save_output( larlite::storage_manager& out_larlite,
                  dlcosmictag::SSNetStitcher* stitch_shower,
                  dlcosmictag::SSNetStitcher* stitch_track,
                  dlcosmictag::SSNetStitcher* stitch_endpt,
                  dlcosmictag::SSNetStitcher* stitch_infill,
                  int runid, int subrunid, int eventid ) {
  
  larlite::event_pixelmask* evout_shower_mask
    = (larlite::event_pixelmask*)out_larlite.get_data( larlite::data::kPixelMask, "shower" );
  larlite::event_pixelmask* evout_track_mask
    = (larlite::event_pixelmask*)out_larlite.get_data( larlite::data::kPixelMask, "track" );
  larlite::event_pixelmask* evout_endpt_mask
    = (larlite::event_pixelmask*)out_larlite.get_data( larlite::data::kPixelMask, "endpt" );        
  larlite::event_pixelmask* evout_infill_mask
    = (larlite::event_pixelmask*)out_larlite.get_data( larlite::data::kPixelMask, "infill" );
        
  std::vector<larlite::pixelmask> stitched_shower_v = stitch_shower->as_pixel_mask( 1.0e-3 );
  std::vector<larlite::pixelmask> stitched_track_v  = stitch_track->as_pixel_mask( 1.0e-3 );
  std::vector<larlite::pixelmask> stitched_endpt_v  = stitch_endpt->as_pixel_mask( 1.0e-3 );
  std::vector<larlite::pixelmask> stitched_infill_v = stitch_infill->as_pixel_mask( 1.0e-3 );
        
  for ( size_t p=0; p<3; p++ ) {
    evout_shower_mask->emplace_back( std::move(stitched_shower_v[p]) );
    
    evout_track_mask->emplace_back( std::move(stitched_track_v[p]) );
    
    evout_endpt_mask->emplace_back( std::move(stitched_endpt_v[p]) );
    
    evout_infill_mask->emplace_back( std::move(stitched_infill_v[p]) );
  }

  out_larlite.set_id( runid, subrunid, eventid );
  out_larlite.next_event();
  
}


int main( int args, char** argv ) {

  std::string input_dlcosmictag_larcv2 = argv[1];
  std::string input_wholeview_larcv2   = argv[2];  
  std::string output_stitched_larlite  = argv[3];

  larcv::IOManager input_larcv2( larcv::IOManager::kREAD );
  input_larcv2.add_in_file( input_dlcosmictag_larcv2 );
  input_larcv2.initialize();

  larcv::IOManager input_wholeview( larcv::IOManager::kREAD );
  input_wholeview.add_in_file( input_wholeview_larcv2 );
  input_wholeview.initialize();
  
  larlite::storage_manager out_larlite( larlite::storage_manager::kWRITE );
  out_larlite.set_out_filename( output_stitched_larlite );
  out_larlite.open();

  int nentries = input_larcv2.get_n_entries();
  int current_runid = -1;
  int current_subrunid = -1;
  int current_eventid = -1;
  int current_wholeview_entry = 0;

  int first_entry = 0;
  int num_entries_completed = 0;

  // stitcher algos
  dlcosmictag::SSNetStitcher* stitch_shower = nullptr;
  dlcosmictag::SSNetStitcher* stitch_track  = nullptr;
  dlcosmictag::SSNetStitcher* stitch_endpt  = nullptr;
  dlcosmictag::SSNetStitcher* stitch_infill = nullptr;  

  // wholeview event_image2d
  larcv::EventImage2D* ev_wholeview = nullptr;
  
  for ( int ientry=0; ientry<nentries; ientry++ ) {

    input_larcv2.read_entry( ientry );

    // get data

    // ssnet
    auto ev_trackimg  = (larcv::EventImage2D*) input_larcv2.get_data("image2d", "ssnetCropped_track");
    auto ev_showerimg = (larcv::EventImage2D*) input_larcv2.get_data("image2d", "ssnetCropped_shower");
    auto ev_endptimg  = (larcv::EventImage2D*) input_larcv2.get_data("image2d", "ssnetCropped_endpt");

    // infill
    auto ev_infillimg = (larcv::EventImage2D*) input_larcv2.get_data("image2d", "infillCropped");

    // get RSE
    int runid    = input_larcv2.event_id().run();
    int subrunid = input_larcv2.event_id().subrun();
    int eventid  = input_larcv2.event_id().event();
    
    if ( runid!=current_runid || subrunid!=current_subrunid || eventid!=current_eventid ) {
      // event-change over

      // load the supera file
      find_rse_entry( input_wholeview, runid, subrunid, eventid, current_wholeview_entry, "wire" );
      ev_wholeview = (larcv::EventImage2D*)input_wholeview.get_data( "image2d", "wire" );
      
      if ( ientry==first_entry ) {
        // first entry setup for stitchers
        stitch_shower = new dlcosmictag::SSNetStitcher( ev_wholeview->as_vector(), dlcosmictag::SSNetStitcher::kCENTER );
        stitch_track  = new dlcosmictag::SSNetStitcher( ev_wholeview->as_vector(), dlcosmictag::SSNetStitcher::kCENTER );
        stitch_endpt  = new dlcosmictag::SSNetStitcher( ev_wholeview->as_vector(), dlcosmictag::SSNetStitcher::kCENTER );
        stitch_infill = new dlcosmictag::SSNetStitcher( ev_wholeview->as_vector(), dlcosmictag::SSNetStitcher::kCENTER );
      }
      else {

        save_output( out_larlite,
                     stitch_shower, stitch_track, stitch_endpt, stitch_infill,
                     runid, subrunid, eventid );
        
        stitch_shower->clear();
        stitch_track->clear();
        stitch_endpt->clear();
        stitch_infill->clear();

        stitch_shower->setWholeViewMeta( ev_wholeview->as_vector() );
        stitch_track->setWholeViewMeta( ev_wholeview->as_vector() );
        stitch_endpt->setWholeViewMeta( ev_wholeview->as_vector() );
        stitch_infill->setWholeViewMeta( ev_wholeview->as_vector() );        
      }

      current_runid    = runid;
      current_subrunid = subrunid;
      current_eventid  = eventid;
    }

    // stitch in subimages
    for ( size_t p=0; p<3; p++ ) {
      stitch_shower->addSubImage( ev_showerimg->at(p), (int)p, 0.0 );
      stitch_track->addSubImage(  ev_trackimg->at(p),  (int)p, 0.0 );
      stitch_endpt->addSubImage(  ev_endptimg->at(p),  (int)p, 0.0 );
      stitch_infill->addSubImage( ev_infillimg->at(p), (int)p, 0.0 );
    }
    
  }//end of entry loop

  // write out again
  save_output( out_larlite,
               stitch_shower, stitch_track, stitch_endpt, stitch_infill,
               current_runid, current_subrunid, current_eventid );
  
  input_larcv2.finalize();
  input_wholeview.finalize();
  out_larlite.close();
  
  return 0;
}


