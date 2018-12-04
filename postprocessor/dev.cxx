#include <iostream>
#include <string>
#include <sstream>

// ROOT
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include <TApplication.h>

// larlite
#include "DataFormat/hit.h"
#include "DataFormat/opflash.h"
#include "DataFormat/spacepoint.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

// larlitecv
#include "Base/DataCoordinator.h"

// #ifdef USE_OPENCV
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #endif

// ContourTools
#include "ContourTools/ContourCluster.h"

// FlowContourMatching
#include "FlowContourMatching/FlowContourMatch.h"

struct InputArgs_t {
  
  InputArgs_t()
    : outputset_larcv(false),
      outputset_larlite(false),
      use_combined_dlcosmictag(false),
      use_hits(false),
      use_truth(false),
      has_supera(false),      
      has_reco2d(false),
      has_opreco(false),
      has_mcreco(false),
      has_infill(false),
      has_ssnet(false),
      has_larcvtruth(false),
      use_ancestor_img(false),
      makehits_useunmatched(false),
      makehits_require_3dconsistency(false),
      process_num_events(-1),
      jobid(-1),
      kVISUALIZE(false),
      kINSPECT(false)
  {};

  // LArCV input (cropped)
  // ----------------------
  bool use_combined_dlcosmictag;
  // if use_combined_dlcosmictag=True
  std::string input_dlcosmictag; 
  // if use_combined_dlcosmictag=False
  std::string input_larflow_y2u;
  std::string input_larflow_y2v;
  std::string input_cropped_adc;
  std::string input_infill;
  std::string input_ssnet;

  // Event data
  // ----------
  std::string input_supera;
  std::string input_larcvtruth;
  std::string input_mcinfo;
  std::string input_reco2d;
  std::string input_opreco;

  // Output params
  // -------------
  bool outputset_larlite;
  bool outputset_larcv;  
  std::string output_larlite;
  std::string output_larcv;

  // other pars
  // ----------
  int jobid;
  bool use_hits;
  bool use_truth;
  bool has_supera;  
  bool has_reco2d;
  bool has_opreco;
  bool has_mcreco;
  bool has_larcvtruth;
  bool has_infill;
  bool has_ssnet;
  bool use_ancestor_img;
  bool makehits_useunmatched;
  bool makehits_require_3dconsistency;
  int process_num_events;
  bool kVISUALIZE;
  bool kINSPECT;

};

struct Arg_t {
  Arg_t( std::string f, std::string h ) 
    : flag(f),
      help(h)
  {};
  std::string flag;
  std::string help;
};

InputArgs_t parseArgs( int nargs, char** argv ) {

  std::vector< Arg_t > commands;
  commands.push_back( Arg_t("-c",  "combined (cropped) dlcosmictag file [required or provide individual files below]") );
  commands.push_back( Arg_t("-y2u","individual (cropped) y2u file [required if combined not provided]") );
  commands.push_back( Arg_t("-y2v","individual (cropped) y2v file [required if combined not provided]") );
  commands.push_back( Arg_t("-adc","individual (cropped) adc file [required if combined not provided]") );
  commands.push_back( Arg_t("-in", "individual (cropped) infill file") );
  commands.push_back( Arg_t("-ss", "individual (cropped) ssnet file") );
  commands.push_back( Arg_t("-su", "event supera larcv file [required]") );
  commands.push_back( Arg_t("-oll","output larflow larlite file [required]") );
  commands.push_back( Arg_t("-olc","output larflow larcv file [required]") );  
  commands.push_back( Arg_t("-mc", "event mcinfo larlite file") );
  commands.push_back( Arg_t("-lcvt", "event larcvtruth LArCV file (w/ ancestor image)") );  
  commands.push_back( Arg_t("-op", "event opreco larlite file") );
  commands.push_back( Arg_t("-re", "event reco2d larlite file [required if use-hits]") );
  commands.push_back( Arg_t("-j",  "jobid") );
  commands.push_back( Arg_t("-n",  "number of events to run") );
  commands.push_back( Arg_t("--use-truth",  "Use pixelflow truth") );
  commands.push_back( Arg_t("--use-hits",   "Use gaushits as seeds to 3d hits") );
  commands.push_back( Arg_t("--use-require-3dconsist", "use 3d consistency cut") );
  commands.push_back( Arg_t("--use-unmatched", "use unmatched hits") );
  commands.push_back( Arg_t("--use-ancestor-img","use ancestor image to make truth") );
  commands.push_back( Arg_t("--has-infill","use ancestor image to make truth") );
  commands.push_back( Arg_t("--vis", "visualize cropped image matching") );
  commands.push_back( Arg_t("--inspect", "inspect visualization") );

  bool printhelp = false;
  for ( int iarg=1; iarg<nargs; iarg++ )
    if ( std::string(argv[iarg])=="-h" ) printhelp = true;
  
  if ( nargs==1 || printhelp ) {
    std::cout << "-------------------------------" << std::endl;
    std::cout << "Dev arguments" << std::endl;
    for ( auto const& arg : commands ) {
      std::cout << "  " << arg.flag << "  :: " << arg.help << std::endl;
    }
    throw std::runtime_error("end of help");
  }

  InputArgs_t argconfig;
  argconfig.use_combined_dlcosmictag = false;
  
  for ( int iarg=1; iarg<nargs; iarg++ ) {

    if ( argv[iarg][0]!='-' )
      continue;

    bool commandmatched = false;
    for ( auto const& command : commands ) {
      
      if ( std::string(argv[iarg])==command.flag ) {
	commandmatched = true;
	std::string strflag = argv[iarg];
	
	if ( command.flag=="-c" ) {
	  argconfig.use_combined_dlcosmictag = true;
	  argconfig.input_dlcosmictag = argv[iarg+1];
	}
	else if ( command.flag=="-y2u" )
	  argconfig.input_larflow_y2u = argv[iarg+1];
	else if ( command.flag=="-y2v" )
	  argconfig.input_larflow_y2v = argv[iarg+1];
	else if ( command.flag=="-adc" ) 
	  argconfig.input_cropped_adc = argv[iarg+1];
	else if ( command.flag=="-in" ) {
	  argconfig.input_infill = argv[iarg+1];
	  argconfig.has_infill = true;
	}
	else if ( command.flag=="-ss" ) {
	  argconfig.input_ssnet = argv[iarg+1];
	  argconfig.has_ssnet = argv[iarg+1];
	}
	else if ( command.flag=="-oll" ) {
	  argconfig.output_larlite = argv[iarg+1];
	  argconfig.outputset_larlite = true;
	}
	else if ( command.flag=="-olc" ) {
	  argconfig.output_larcv = argv[iarg+1];
	  argconfig.outputset_larcv = true;
	}
	else if ( command.flag=="-op" ) {
	  argconfig.input_opreco = argv[iarg+1];
	  argconfig.has_opreco = true;
	}
	else if ( command.flag=="-su" ) {
	  argconfig.input_supera = argv[iarg+1];
	  argconfig.has_supera = true;
	}
	else if ( command.flag=="-mc" ) {
	  argconfig.input_mcinfo = argv[iarg+1];
	  argconfig.has_mcreco = true;	  
	}
	else if ( command.flag=="-lcvt" ) {
	  argconfig.input_larcvtruth = argv[iarg+1];
	  argconfig.has_larcvtruth = true;
	}
	else if ( command.flag=="-re" ) {
	  argconfig.input_reco2d = argv[iarg+1];
	  argconfig.has_reco2d = true;
	}
	else if ( command.flag=="-j" )
	  argconfig.jobid = std::atoi(argv[iarg+1]);
	else if ( command.flag=="--vis")
	  argconfig.kVISUALIZE = true;
	else if ( command.flag=="--inspect") {
	  argconfig.kVISUALIZE = true;
	  argconfig.kINSPECT = true;
	}
	else if ( command.flag=="--use-unmatched" )
	  argconfig.makehits_useunmatched = true;
	else if ( command.flag=="--use-require-3dconsist" )
	  argconfig.makehits_require_3dconsistency = true;
	else if ( command.flag=="--use-hits" )
	  argconfig.use_hits = true;
	else if ( command.flag=="--use-truth" )
	  argconfig.use_truth = true;
	else if ( command.flag=="--use-ancestor-img" ) {
	  argconfig.use_ancestor_img = true;
	}
	else if ( command.flag=="--has-infill" ) {
	  argconfig.has_infill = true;
	}
	else if ( command.flag=="-n" )
	  argconfig.process_num_events = std::atoi(argv[iarg+1]);

	break;
      }//end of if command flag matched
    }

    if ( !commandmatched ) {
      std::stringstream ss;
      ss << "unrecognized command: '" << argv[iarg]<< "'";
      throw std::runtime_error(ss.str());
    }

  }//end of loop over arguments

  // Configuration Check
  // -------------------

  if ( !argconfig.has_supera ) {
    throw std::runtime_error("must provide supera file");
  }
  
  if ( argconfig.use_hits && !argconfig.has_reco2d ) {
    throw std::runtime_error("set to use hits, but no reco2d file with gaushits");  
  }
  
  if ( !argconfig.use_combined_dlcosmictag ) {
    if ( argconfig.input_larflow_y2u.empty() 
	 || argconfig.input_larflow_y2v.empty()
	 || argconfig.input_cropped_adc.empty() ) {
      throw std::runtime_error("missing required cropped file: y2u, y2v, adc");
    }
  }

  if ( !argconfig.outputset_larlite || !argconfig.outputset_larcv ) {
    throw std::runtime_error("failed to set both output larcv and larlite filenames");
  }

  if ( argconfig.jobid<0 ) argconfig.jobid=0;
  
  return argconfig;
}

void event_changeout( larlite::storage_manager& dataco_output,
		      larcv::IOManager& larcv_output,
		      larcv::IOManager& dataco_wholelarcv,
		      larlite::storage_manager& dataco_larlite,
		      larflow::FlowContourMatch& matching_algo,
		      const int runid,
		      const int subrunid,
		      const int eventid,
		      const InputArgs_t& inputargs ) {
  
  std::cout << "event_changeout." << std::endl;
  // save the larflow3dhit vector
  larlite::event_larflow3dhit* ev_larflowhit = (larlite::event_larflow3dhit*)dataco_output.get_data(larlite::data::kLArFlow3DHit,"flowhits");
  auto evout_opflash_beam   = (larlite::event_opflash*)dataco_output.get_data(larlite::data::kOpFlash, "simpleFlashBeam" );
  auto evout_opflash_cosmic = (larlite::event_opflash*)dataco_output.get_data(larlite::data::kOpFlash, "simpleFlashCosmic" );
  auto evout_mctrack        = (larlite::event_mctrack*) dataco_output.get_data(larlite::data::kMCTrack,  "mcreco" );
  auto evout_mcshower       = (larlite::event_mcshower*)dataco_output.get_data(larlite::data::kMCShower, "mcreco" );
  auto evout_wire           = (larcv::EventImage2D*) larcv_output.get_data("image2d","wire");
  auto evout_chstatus       = (larcv::EventChStatus*)larcv_output.get_data("chstatus","wire");

  // get supera images
  larcv::EventImage2D* ev_wholeimg  = (larcv::EventImage2D*) dataco_wholelarcv.get_data("image2d","wire");
  larcv::EventChStatus* ev_chstatus = (larcv::EventChStatus*)dataco_wholelarcv.get_data("chstatus","wire");
  
  // mctruth
  const larlite::event_mctrack* ev_track = nullptr;
  if ( inputargs.has_mcreco ) {
    ev_track = (larlite::event_mctrack*)dataco_larlite.get_data(larlite::data::kMCTrack, "mcreco");
  }

  // mc-truth tagging of hits
  if ( inputargs.has_mcreco ) {
    std::cout << "[event changeout] provide mctrack info for truth matching" << std::endl;
    matching_algo.mctrack_match(*ev_track,ev_wholeimg->as_vector());        
  }
  if ( inputargs.use_ancestor_img ) {
    // use ancestor images
    std::cout << "[event changeout] provide mctrack info via ancestor image" << std::endl;
    auto ev_ancestor_img = (larcv::EventImage2D*) dataco_wholelarcv.get_data("image2d","ancestor");
    if  ( ev_ancestor_img->as_vector().size()!=ev_wholeimg->as_vector().size() ) {
      throw std::runtime_error("[event changeout] problem loading ancestor images as requested.");
    }
    matching_algo.label_mcid_w_ancestor_img( ev_ancestor_img->as_vector(), ev_wholeimg->as_vector() );
  }
  
  // transfer image and chstatus data to output file
  for ( auto const& img : ev_wholeimg->as_vector() ) {
    evout_wire->append( img );
    evout_chstatus->insert( ev_chstatus->status( img.meta().id() ) );
  }
    
  // get opreco to save into output file
  if ( inputargs.has_opreco ) {
    auto ev_opflash_beam   = (larlite::event_opflash*)dataco_larlite.get_data(larlite::data::kOpFlash, "simpleFlashBeam" );
    auto ev_opflash_cosmic = (larlite::event_opflash*)dataco_larlite.get_data(larlite::data::kOpFlash, "simpleFlashCosmic" );
    
    for ( auto& flash : *ev_opflash_beam )
      evout_opflash_beam->emplace_back( std::move(flash) );
    for ( auto& flash : *ev_opflash_cosmic )
      evout_opflash_cosmic->emplace_back( std::move(flash) );
  }

  // transfer mctrack/mcshower info to output file
  if ( inputargs.has_mcreco ) {
    auto ev_mctrack   = (larlite::event_mctrack*) dataco_larlite.get_data(larlite::data::kMCTrack,  "mcreco" );
    auto ev_mcshower  = (larlite::event_mcshower*)dataco_larlite.get_data(larlite::data::kMCShower, "mcreco" );
    
    for ( auto& mctrack :  *ev_mctrack )
      evout_mctrack->emplace_back(  std::move(mctrack) );
    for ( auto& mcshower : *ev_mcshower )
      evout_mcshower->emplace_back( std::move(mcshower) );
  }
  
  // get the final hits made from flow
  std::vector< larlite::larflow3dhit > whole_event_hits3d_v = matching_algo.get3Dhits_2pl( inputargs.makehits_useunmatched,
											   inputargs.makehits_require_3dconsistency );
  
  std::cout << "Number of 3D (2-flow) hits: " << whole_event_hits3d_v.size() << std::endl;
  for ( auto& flowhit : whole_event_hits3d_v ) {
    // form spacepoints
    ev_larflowhit->emplace_back( std::move(flowhit) );
  }
  
  dataco_output.set_id( runid, subrunid, eventid );
  dataco_output.next_event();
  larcv_output.set_id( runid, subrunid, eventid );
  larcv_output.save_entry();
  
  return;
}

void stitch_infill(larcv::IOManager& ioinfill,
		   const std::vector<larcv::Image2D>& adcwhole,
		   const larcv::EventChStatus* ev_chstatus,
		   std::vector< larcv::Image2D >& infill_whole_v,
		   larflow::FlowContourMatch& matching_algo,
		   int run, int subrun, int event, int entry){


  std::vector< larcv::Image2D > trusted_v;
  for ( auto const& img : adcwhole ) {
    larcv::Image2D trusted( img.meta() );
    trusted.paint(0);
    trusted_v.emplace_back( std::move( trusted ) );
  }
  if ( infill_whole_v.size()!=adcwhole.size() ) {
    // create new image set (whole size)
    infill_whole_v.clear();
    for ( auto const& img : adcwhole ) {
      std::cout << "[stich_infill] creating new image for infill still: " << img.meta().dump() << std::endl;
      larcv::Image2D infill_whole( img.meta() );
      infill_whole_v.emplace_back( std::move( infill_whole ) );
    }
  }
  
  // clear wholeview infill
  for ( int iimg=0; iimg<(int)adcwhole.size(); iimg++ ) {
    larcv::Image2D& infill = infill_whole_v[iimg];
    infill.paint(0);
  }

  std::cout << "stitchsearch (" << run << "," << subrun << "," << event << ") read_entry=" << entry << std::endl;
  int ientry = entry;
  ioinfill.read_entry( ientry );
  auto evinfill = (larcv::EventImage2D*)ioinfill.get_data( "image2d", "infillCropped" );
  int entryrun    = evinfill->run();
  int entrysubrun = evinfill->subrun();
  int entryevent  = evinfill->event();

  bool ok = true;
  int nstitched = 0;
  while ( entryrun==run && entrysubrun==subrun && entryevent==event && ok) {
    std::cout << "stitchsearch (" << run << "," << subrun << "," << event << ") ientry=" << ientry << std::endl;
    for (int p=0; p<3; p++) {
      matching_algo.stitchInfill(evinfill->as_vector()[p],trusted_v[p],infill_whole_v[p],*ev_chstatus);
    }
    nstitched++;
    ientry++;
    ok = ioinfill.read_entry(ientry);
    if ( ok ) {
      evinfill = (larcv::EventImage2D*)ioinfill.get_data( "image2d", "infillCropped" );
      entryrun    = evinfill->run();
      entrysubrun = evinfill->subrun();
      entryevent  = evinfill->event();
    }
  }

  std::cout << "stiched together: " << nstitched << std::endl;
}

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

void find_rse_entry( larlite::storage_manager& io, int run, int subrun, int event, int& current_entry ) {
  // searches for run, subrun, entry in larlite
  std::cout << "find_rse_entry[larlite]: look for (" << run << "," << subrun << "," << event << ") current_entry=" << current_entry << std::endl;  
  for (int ipass=0; ipass<2; ipass++) {
    bool found_match = false;
    for ( int ientry=current_entry; ientry<io.get_entries(); ientry++ ) {
      io.go_to(ientry);
      if ( run==io.run_id() && subrun==io.subrun_id() && event==io.event_id() ) {
	found_match = true;
	current_entry = ientry;
	break;
      }
    }
    if ( !found_match ) current_entry=0;
    else break;
  }
  std::cout << "find_rse_entry[larlite]: found (" << run << "," << subrun << "," << event << ") @ entry=" << current_entry << std::endl;  
}




int main( int nargs, char** argv ) {

  gStyle->SetOptStat(0);
 
  std::cout << "larflow post-processor dev" << std::endl;

  // expects cropped output (from deploy/run_larflow_wholeview.py)
  // -------------------------------------------------------------

  // arg parsing
  InputArgs_t inputargs = parseArgs( nargs, argv );
  
  TApplication app ("app",&nargs,argv);  

  std::cout << "===========================================" << std::endl;
  std::cout << " Dev LArFlow Post-Processor " << std::endl;
  std::cout << " -------------------------- " << std::endl;
  
  using flowdir = larflow::FlowContourMatch;

  // data from larflow output: sequence of cropped images
  larcv::IOManager dataco( larcv::IOManager::kREAD );
  if ( inputargs.use_combined_dlcosmictag )  {
    dataco.add_in_file( inputargs.input_dlcosmictag );
  }
  else {
    dataco.add_in_file( inputargs.input_larflow_y2u );
    dataco.add_in_file( inputargs.input_larflow_y2v );
    dataco.add_in_file( inputargs.input_cropped_adc );
    if ( inputargs.has_ssnet )
      dataco.add_in_file( inputargs.input_ssnet );
    if ( inputargs.has_infill )
      dataco.add_in_file( inputargs.input_infill );
  }
  dataco.initialize();

  // we read ahead to stitch the infill image, so it gets its own feed
  larcv::IOManager dataco_infill( larcv::IOManager::kREAD );
  if ( inputargs.has_infill ) {
    if ( inputargs.use_combined_dlcosmictag ) {
      std::cout << "Has infill. Using combined file: " << inputargs.input_dlcosmictag << std::endl;
      dataco_infill.add_in_file( inputargs.input_dlcosmictag );
    }
    else {
      std::cout << "Has infill. Using separate file." << std::endl;      
      dataco_infill.add_in_file( inputargs.input_infill );
    }
    std::cout << "dataco_infill initialed" << std::endl;
    dataco_infill.initialize();    
  }

  
  // data from whole-view image
  std::stringstream strjobid;
  strjobid << inputargs.jobid;

  larcv::IOManager dataco_wholelarcv(larcv::IOManager::kREAD);
  dataco_wholelarcv.add_in_file( inputargs.input_supera );
  if ( inputargs.has_larcvtruth )
    dataco_wholelarcv.add_in_file( inputargs.input_larcvtruth );
  dataco_wholelarcv.initialize();
  int iwholelarcv_index = 0;
  
  // hit (and mctruth) event data
  larlite::storage_manager dataco_hits( larlite::storage_manager::kREAD );
  int iwholelarlite_index = 0;
  if ( inputargs.has_reco2d )
    dataco_hits.add_in_filename( inputargs.input_reco2d );
  if ( inputargs.has_opreco )
    dataco_hits.add_in_filename( inputargs.input_opreco );
  if ( inputargs.has_mcreco )
    dataco_hits.add_in_filename( inputargs.input_mcinfo );
  if ( inputargs.has_opreco || inputargs.has_mcreco || inputargs.has_reco2d ) {
    std::cout << "dataco_hits: open file" << std::endl;
    dataco_hits.open();
  }

  // output: 3D track hits
  larlite::storage_manager dataco_output( larlite::storage_manager::kWRITE );
  dataco_output.set_out_filename( inputargs.output_larlite );
  dataco_output.open();

  // output: larcv/chstatus
  larcv::IOManager io_larcvout( larcv::IOManager::kWRITE );
  io_larcvout.set_out_file( inputargs.output_larcv );
  io_larcvout.initialize();
  
  // cluster algo
  larlitecv::ContourCluster cluster_algo;
  larflow::FlowContourMatch matching_algo;
  larlite::event_hit pixhits_v;
    
  int nentries = dataco.get_n_entries();
  std::cout << "Number of entries in cropped file: " << nentries << std::endl;
  
  int current_runid    = -1;
  int current_subrunid = -1;
  int current_eventid  = -1;
  int current_ientrystart = -1;
  int nevents = 0;
  int eventstart = 0;

  //whole image infill
  larcv::Image2D infill_whole_y;
  bool isnewevent = false;

  
  larcv::EventImage2D*  ev_wholeimg = nullptr;
  larlite::event_hit*   ev_hit      = nullptr;
  larcv::EventChStatus* ev_chstatus = nullptr;

  // images to be made for each event
  std::vector<larcv::Image2D> badch_v;
  std::vector<larcv::Image2D> infill_whole_v; // infill output stitched into wholeview
  std::vector<larcv::Image2D> img_fill_whole_v; // adc whole view + infill pixels marked in dead regions
  
  for (int ientry=eventstart*54; ientry<nentries; ientry++) {

    dataco.read_entry(ientry,"larcv");

    // larflow input data (assumed to be cropped subimages)
    larcv::EventImage2D* ev_wire      = (larcv::EventImage2D*) dataco.get_data("image2d", "adc");
    larcv::EventImage2D* ev_flow[larflow::FlowContourMatch::kNumFlowDirs] = {NULL};
    ev_flow[flowdir::kY2U] = (larcv::EventImage2D*) dataco.get_data("image2d", "larflow_y2u");
    ev_flow[flowdir::kY2V] = (larcv::EventImage2D*) dataco.get_data("image2d", "larflow_y2v");
    const std::vector<larcv::Image2D>& wire_v = ev_wire->image2d_array();    
    bool hasFlow[2] = { false, false };
    for (int i=0; i<2; i++)
      hasFlow[i] = ( ev_flow[i]->valid() ) ? true : false;
    
    int runid    = ev_wire->run();
    int subrunid = ev_wire->subrun();
    int eventid  = ev_wire->event();
    std::cout << "Loading entry: " << ientry << " (rse)=(" << runid << "," << subrunid << "," << eventid << ")" << std::endl;
    if ( ientry==eventstart*54 ) {
      // first entry, set the current_runid
      current_runid    = runid;
      current_subrunid = subrunid;
      current_eventid  = eventid;
      current_ientrystart = ientry;
      isnewevent = true;
    }

    if ( current_runid!=runid || current_subrunid!=subrunid || current_eventid!=eventid ) {

      // if we are breaking, we cut out now, using the event_changeout all at end of file
      std::cout << "new event: (" << runid << "," << subrunid << "," << eventid << ")" << std::endl;
      std::cout << "last event: (" << current_runid << "," << current_subrunid << "," << current_eventid << ")" << std::endl;
      nevents++;      
      if ( inputargs.process_num_events>0 && nevents>=inputargs.process_num_events )
	break;

      std::cout << "Event turn over" << std::endl;
      
      // save flow used across the image
      std::vector<larcv::Image2D> usedflow_v = matching_algo.makeStitchedFlowImages( ev_wholeimg->as_vector() );
      auto ev_usedflow = (larcv::EventImage2D*) io_larcvout.get_data("image2d","hitflow");
      ev_usedflow->emplace( std::move(usedflow_v) );
      
      event_changeout( dataco_output, io_larcvout, dataco_wholelarcv, dataco_hits, matching_algo,
		       current_runid, current_subrunid, current_eventid, inputargs );


      // clear the algo
      matching_algo.clear();
      pixhits_v.clear();
      
      // set the current rse
      current_runid    = runid;
      current_subrunid = subrunid;
      current_eventid  = eventid;
      current_ientrystart = ientry;
      isnewevent = true;

      //std::cout << "entry to continue" << std::endl;
      //std::cin.get();      
    }
    

    // update event information
    if ( isnewevent ) {

      // load up the whole-view images from the supera file
      find_rse_entry( dataco_wholelarcv, current_runid, current_subrunid, current_eventid, iwholelarcv_index, "wire" );
      ev_wholeimg  = (larcv::EventImage2D*) dataco_wholelarcv.get_data("image2d","wire");      

      // channel status
      ev_chstatus = (larcv::EventChStatus*) dataco_wholelarcv.get_data("chstatus","wire");
      std::cout << "chstatus size: " << ev_chstatus->chstatus_map().size() << std::endl;
      for ( auto const& itchstatus : ev_chstatus->chstatus_map() ) {
	std::cout << " projection: " << itchstatus.first << " entries=" << itchstatus.second.as_vector().size() << std::endl;
      }
      
      
      // sync up larlite data
      if ( inputargs.has_reco2d || inputargs.has_opreco || inputargs.has_mcreco ) {
	find_rse_entry( dataco_hits, current_runid, current_subrunid, current_eventid, iwholelarlite_index );
      }

      if ( inputargs.has_reco2d ) {
	ev_hit = ((larlite::event_hit*)dataco_hits.get_data(larlite::data::kHit, "gaushit"));
	std::cout << "Number of reco2d hits in New Event: " << ev_hit->size() << std::endl;
      }

      // create whole image + infill merger
      if ( badch_v.size()==0 ) {
	// fill for first time
	for ( auto const& img : ev_wholeimg->as_vector() ) {
	  larcv::Image2D badch( img.meta() );
	  badch.paint(0.0);
	  badch_v.emplace_back( std::move(badch) );
	}
      }
      if ( infill_whole_v.size()==0 ) {
	for ( auto const& img : ev_wholeimg->as_vector() ) {
	  larcv::Image2D infill( img.meta() );
	  infill.paint(0.0);
	  infill_whole_v.emplace_back( std::move(infill) );
	}
      }
      // add infill pixel markers to whole-view image
      img_fill_whole_v.clear();
      if ( img_fill_whole_v.size()==0 ) {
	for ( auto const& img : ev_wholeimg->as_vector() ) {
	  std::cout << "[dev] creating new image for infill stitch: " << img.meta().dump();
	  larcv::Image2D img_fill(img);
	  img_fill_whole_v.emplace_back( std::move(img_fill) );
	}
      }
      
      // create a whole-image infills
      if ( inputargs.has_infill ) {
	std::cout << "perform infill stitch" << std::endl;
	stitch_infill( dataco_infill, ev_wholeimg->as_vector(), ev_chstatus, infill_whole_v, matching_algo, current_runid, current_subrunid, current_eventid, current_ientrystart );

	//mask infill and add to adc
	for ( int p=0; p<3; p++) {
	  larcv::Image2D& infill   = infill_whole_v[p];
	  larcv::Image2D& img_fill = img_fill_whole_v[p];
	  larcv::Image2D infillmask( infill.meta() );
	  infillmask.paint(0);
	  matching_algo.maskInfill( infill, *ev_chstatus, 0.96, infillmask );
	  matching_algo.addInfill(  infillmask, *ev_chstatus, 20.0,  img_fill );
	}
      }
      
      // if we use pixels to make hits, we need to extract them from the whole view
      if ( !inputargs.use_hits ) {
	pixhits_v.clear();
	std::cout << "Make pixhits from whole view of Y-plane" << std::endl;
	std::cout << "  run=" << ev_wholeimg->run() << " subrun=" << ev_wholeimg->subrun() << " event=" << ev_wholeimg->event() << std::endl;
	matching_algo.makeHitsFromWholeImagePixels( img_fill_whole_v[2], pixhits_v, 10.0 );
      }

      std::cout << "========================================================================" << std::endl;
      std::cout << "[ START OF NEW EVENT: Number of pixhits_v=" << pixhits_v.size() << " ]" << std::endl;
      std::cout << "========================================================================" << std::endl;
    }//end of isnew to handle event-level images/quantities

    
    const std::vector<larcv::Image2D>& whole_v = ev_wholeimg->as_vector();
    isnewevent = false;
    
  
    // Cropped info (per entry)
    // -------------------------
    
    // endpt+segment info
    larcv::EventImage2D* ev_trackimg  = nullptr;
    larcv::EventImage2D* ev_showerimg = nullptr;
    larcv::EventImage2D* ev_endptimg  = nullptr;
    if ( inputargs.has_ssnet ) {
      ev_trackimg  = (larcv::EventImage2D*)  dataco.get_data("image2d", "ssnetCropped_track");
      ev_showerimg = (larcv::EventImage2D*)  dataco.get_data("image2d", "ssnetCropped_shower");
      ev_endptimg  = (larcv::EventImage2D*)  dataco.get_data("image2d", "ssnetCropped_endpt");
      if ( !ev_trackimg->valid() || !ev_showerimg->valid() || !ev_endptimg->valid() )
	inputargs.has_ssnet = false;
    }
    
    //infill prediction (unmasked)
    larcv::EventImage2D* ev_infill = nullptr;
    if ( inputargs.has_infill ) {
      ev_infill = (larcv::EventImage2D*) dataco.get_data("image2d", "infillCropped");
      if ( !ev_infill->valid() )
	inputargs.has_infill = false;
    }

    //mask subimage
    std::vector<larcv::Image2D> img_fill_v;
    for ( size_t p=0; p<wire_v.size(); p++ ) {
      const larcv::Image2D& adc_cropped = wire_v[p];
      larcv::Image2D imgfill( adc_cropped );
      larcv::Image2D infillmask( adc_cropped.meta() );
      infillmask.paint(0);
      
      if ( inputargs.has_infill ) {
	// mask and fill cropped images
	matching_algo.maskInfill( ev_infill->as_vector()[p], *ev_chstatus, 0.96, infillmask );
	matching_algo.addInfill( infillmask, *ev_chstatus, 10.0, imgfill );
      }
      img_fill_v.emplace_back( std::move(imgfill) );
    }
    
    // truth
    larcv::EventImage2D* ev_trueflow  = nullptr;
    const std::vector<larcv::Image2D>* true_v = nullptr;
    std::vector<larcv::Image2D> flow_v;    
    if ( inputargs.use_truth ) {
      ev_trueflow = (larcv::EventImage2D*) dataco.get_data("image2d", "pixflow");
      true_v = &(ev_trueflow->image2d_array());      
    }
    // merged flow predictions
    if ( hasFlow[flowdir::kY2U] )
      flow_v.emplace_back( std::move(ev_flow[flowdir::kY2U]->modimgat(0) ) );
    else
      flow_v.push_back( larcv::Image2D() ); // dummy image
    if ( hasFlow[flowdir::kY2V] )
      flow_v.emplace_back( std::move(ev_flow[flowdir::kY2V]->modimgat(0) ) );
    else
      flow_v.push_back( larcv::Image2D() ); // dummy image
    
    // get cluster atomics for cropped u,v,y ADC image (infill-filled if has_infill set)
    cluster_algo.clear();    
    cluster_algo.analyzeImages( img_fill_v, badch_v, 20.0, 3 );

    if ( inputargs.use_hits ) {      
      if ( !inputargs.use_truth ) {
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, flow_v, *ev_hit, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
      }
      else
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, *true_v, *ev_hit, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
    }
    else {
      
      if ( !inputargs.use_truth )
	matching_algo.fillPlaneHitFlow(  cluster_algo, img_fill_v[2], img_fill_v, flow_v, pixhits_v, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
      else
	matching_algo.fillPlaneHitFlow(  cluster_algo, img_fill_v[2], img_fill_v, *true_v, pixhits_v, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
    }

    // update ssnet info
    if ( inputargs.has_ssnet && ev_trackimg->valid() && ev_showerimg->valid() && ev_endptimg->valid() ) {
      matching_algo.integrateSSNetEndpointOutput( ev_trackimg->as_vector(), ev_showerimg->as_vector(), ev_endptimg->as_vector() );
    }
    else {
      std::cout << "Shower/track/endpt info not valid" << std::endl;
    }

    // update infill labels in PlaneHitFlow objects
    if ( inputargs.has_infill && ev_infill->valid() && ev_infill->as_vector().size()==3 ) {
      matching_algo.labelInfillHits( ev_infill->as_vector() );
      std::cout << "infill labels here applied to planehitflow objects" << std::endl;
    }
    else {
      std::cout << "Infill info not valid" << std::endl;
    }

    if ( inputargs.kVISUALIZE ) {

      std::vector< larlite::larflow3dhit > hits3d_v = matching_algo.get3Dhits_2pl();
      
      // flip through matches
      TCanvas c("c","matched clusters",1200,400);
      TH2D hsrc = larcv::as_th2d( wire_v[2], "hsource_y" );
      hsrc.SetTitle("Source: Y-plane;wires;ticks");
      hsrc.SetMinimum(10.0);
      hsrc.SetMaximum(255.0);    
      TH2D htar_u = larcv::as_th2d( wire_v[0], "htarget_u" );
      htar_u.SetTitle("Target u-plane;wires;ticks");
      htar_u.SetMinimum(10.0);
      htar_u.SetMaximum(255.0);    
      TH2D htar_v = larcv::as_th2d( wire_v[1], "htarget_v" );
      htar_v.SetTitle("Target v-plane;wires;ticks");
      htar_v.SetMinimum(10.0);
      htar_v.SetMaximum(255.0);    
      
      const larcv::ImageMeta& src_meta = wire_v[2].meta();
      const larcv::ImageMeta* tar_meta[2] = { &(wire_v[0].meta()), &(wire_v[1].meta()) };
      
      std::cout << "Source Matches: " << std::endl;
      for (int i=0; i<matching_algo.m_src_ncontours; i++) {

	// we make a list of tgraphs for each contour
	// thickness determined by score
	// >0.8 = 5 (red)
	// >0.6 = 4 (orange)
	// >0.4 = 3 (yellow)
	// >0.2 = 2 (blue)
	// <0.1 = 1 (black)
	
	const larlitecv::Contour_t& src_ctr = cluster_algo.m_plane_atomics_v[2][ i ];
	TGraph src_graph( src_ctr.size()+1 );
	for ( int n=0; n<src_ctr.size(); n++) {
	  float col = src_meta.pos_x( src_ctr[n].x );
	  float row = src_meta.pos_y( src_ctr[n].y );
	  src_graph.SetPoint(n,col,row);
	}
	src_graph.SetPoint(src_ctr.size(), src_meta.pos_x(src_ctr[0].x), src_meta.pos_y(src_ctr[0].y));
	src_graph.SetLineWidth(5);
	src_graph.SetLineColor(kRed);
	
	// graph the predicted and true flow locations
	TGraph src_flowpix[2];
	TGraph src_truthflow[2];
	int nflowpoints = 0;
	for (int iflow=0; iflow<2; iflow++) {
	  auto it_src_targets = matching_algo.m_src_targets[iflow].find( i );
	  if ( it_src_targets!=matching_algo.m_src_targets[iflow].end() ) {
	    larflow::FlowContourMatch::ContourTargets_t& targetlist = it_src_targets->second;
	    nflowpoints = targetlist.size();
	    src_flowpix[iflow].Set( targetlist.size() );
	    if ( inputargs.use_truth )
	      src_truthflow[iflow].Set( targetlist.size() );
	    int ipixt = 0;
	    for ( auto& pix_t : targetlist ) {
	      src_flowpix[iflow].SetPoint(ipixt, tar_meta[iflow]->pos_x( pix_t.col ), tar_meta[iflow]->pos_y(pix_t.row) );
	      if ( inputargs.use_truth ) {
		int truthflow = -10000;
		truthflow = (*true_v)[iflow].pixel( pix_t.row, pix_t.srccol );	  
		try {
		  //std::cout << "truth flow @ (" << pix_t.col << "," << pix_t.row << "): " << truthflow << std::endl;
		  src_truthflow[iflow].SetPoint(ipixt, tar_meta[iflow]->pos_x( pix_t.srccol+truthflow ), tar_meta[iflow]->pos_y(pix_t.row) );
		}
		catch (...) {
		  std::cout << "bad flow @ (" << pix_t.srccol << "," << pix_t.row << ") "
			    << "== " << truthflow << " ==>> (" << pix_t.srccol+truthflow << "," << pix_t.row << ")" << std::endl;
		}
	      }
	      ipixt++;
	    }//end of loop over src_targets
	  }//if src targets
	}//end of loop over flowdir
	std::cout << "SourceIDX[" << i << "]  ";
	// graph the target contours
	std::vector< TGraph > tar_graphs[2];
	for (int iflow=0; iflow<2; iflow++) {
	  for (int j=0; j<matching_algo.m_tar_ncontours[iflow]; j++) {
	    float score = matching_algo.m_score_matrix[iflow][ i*matching_algo.m_tar_ncontours[iflow] + j ];
	    if ( score<0.01 )
	      continue;
	    
	    std::cout << "[" << j << "]=" << score << " ";
	    
	    float width = 1;
	    int color = 0;
	    if ( score>0.8 ) {
	      width = 5;
	      color = kRed;
	    }
	    else if ( score>0.6 ) {
	      width = 4;
	      color = kRed-9;
	    }
	    else if ( score>0.3 ) {
	      width = 3;
	      color = kOrange+1;
	    }
	    else if ( score>0.1 ) {
	      width = 2;
	      color = kOrange-9;
	    }
	    else {
	      width = 1;
	      color = kBlack;
	    }
	    
	    const larlitecv::Contour_t& tar_ctr = cluster_algo.m_plane_atomics_v[iflow][ j ];
	    TGraph tar_graph( tar_ctr.size()+1 );
	    for ( int n=0; n<tar_ctr.size(); n++) {
	      float col = tar_meta[iflow]->pos_x( tar_ctr[n].x );
	      float row = tar_meta[iflow]->pos_y( tar_ctr[n].y );
	      tar_graph.SetPoint(n,col,row);
	    }
	    tar_graph.SetPoint(tar_ctr.size(), tar_meta[iflow]->pos_x(tar_ctr[iflow].x),tar_meta[iflow]->pos_y(tar_ctr[0].y));	
	    tar_graph.SetLineWidth(width);
	    tar_graph.SetLineColor(color);
	    
	    tar_graphs[iflow].emplace_back( std::move(tar_graph) );
	  }
	}
	std::cout << std::endl;
	

	// plot matched hits
	TGraph gsrchits( hits3d_v.size() );
	for ( int ihit=0; ihit<(int)hits3d_v.size(); ihit++ ) {
	  larlite::larflow3dhit& hit3d = hits3d_v[ihit];
	  float x = hit3d.srcwire;
	  float y = hit3d.tick;
	  //std::cout << "src hit[" << ihit << "] (r,c)=(" << y << "," << x << ")" << std::endl;
	  gsrchits.SetPoint( ihit, x, y );
	}
	gsrchits.SetMarkerSize(1);
	gsrchits.SetMarkerStyle(21);
	gsrchits.SetMarkerColor(kBlack);

	// plot matched hits
	TGraph gtarhits( hits3d_v.size() );
	for ( int ihit=0; ihit<(int)hits3d_v.size(); ihit++ ) {
	  larlite::larflow3dhit& hit3d = hits3d_v[ihit];

	  // bool goody2u = true;
	  // bool goody2v = true;
	  // if ( hit3d.targetwire[0]<0 || hit3d.targetwire[0]>=2399 )
	  //   goody2u = false;
	  // if ( hit3d.targetwire[1]<0 || hit3d.targetwire[1]>=2399 )
	  //   goody2v = false;
	  // if ( !goody2u && !goody2v )
	  //   continue;
	  // int useflow = (goody2u) ? 0 : 1;
	  	  
	  gtarhits.SetPoint( ihit, hit3d.targetwire[0], hit3d.tick );
	}
	gtarhits.SetMarkerSize(1);
	gtarhits.SetMarkerStyle(21);
	gtarhits.SetMarkerColor(kBlack);

	// SETUP CANVAS
	// -------------
	c.Clear();
	c.Divide(3,1);
	
	// source
	c.cd(1);
	hsrc.Draw("colz");
	//gsrchits.Draw("P");      	
	src_graph.Draw("L");

	
	// target (u)
	c.cd(2);
	htar_u.Draw("colz");
	for ( auto& g : tar_graphs[0] ) {
	  g.Draw("L");
	}
	//gtarhits.Draw("P");
	if ( inputargs.use_truth ) {
	  src_truthflow[0].SetMarkerStyle(25);      
	  src_truthflow[0].SetMarkerSize(0.2);
	  src_truthflow[0].SetMarkerColor(kMagenta);
	  src_truthflow[0].Draw("P");
	}
	src_flowpix[0].SetMarkerStyle(25);      
	src_flowpix[0].SetMarkerSize(0.2);
	src_flowpix[0].SetMarkerColor(kCyan);            
	src_flowpix[0].Draw("P");

	// target (v)
	c.cd(3);
	htar_v.Draw("colz");
	for ( auto& g : tar_graphs[1] ) {
	  g.Draw("L");
	}
	//gtarhits.Draw("P");
	if ( inputargs.use_truth ) {
	  src_truthflow[1].SetMarkerStyle(25);      
	  src_truthflow[1].SetMarkerSize(0.2);
	  src_truthflow[1].SetMarkerColor(kMagenta);
	  src_truthflow[1].Draw("P");
	}
	src_flowpix[1].SetMarkerStyle(25);      
	src_flowpix[1].SetMarkerSize(0.2);
	src_flowpix[1].SetMarkerColor(kCyan);            
	src_flowpix[1].Draw("P");
	
	
	c.Update();
	c.Draw();
	if ( inputargs.kINSPECT )	 {
	  std::cout << "[ENTER] for next contour." << std::endl;
	  std::cin.get();
	}
	else {
	  if ( nflowpoints>20 ) {
	    char pngname[100];
	    sprintf( pngname, "dumpdev/flow_entry%d_cluster%d.png",ientry,i);
	    c.SaveAs( pngname );
	  }
	}
      }
    }//end of visualize

    //break;
    //std::cout << "[ENTER] for next entry." << std::endl;
    //std::cin.get();

  }//end of entry loop

  // save flow used across the image
  std::vector<larcv::Image2D> usedflow_v = matching_algo.makeStitchedFlowImages( ev_wholeimg->as_vector() );
  auto ev_usedflow = (larcv::EventImage2D*) io_larcvout.get_data("image2d","hitflow");
  ev_usedflow->emplace( std::move(usedflow_v) );

  // save the data from the last event  
  event_changeout( dataco_output, io_larcvout, dataco_wholelarcv, dataco_hits, matching_algo,
		   current_runid, current_subrunid, current_eventid, inputargs );
  
  std::cout << "Finalize output." << std::endl;
  dataco_hits.close();
  dataco_wholelarcv.finalize();
  dataco_output.close();
  io_larcvout.finalize();
  
  return 0;

}
