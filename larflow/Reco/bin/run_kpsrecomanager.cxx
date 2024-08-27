#include <iostream>

#include "argparse/argparse.hpp"
#include "larflow/Reco/KPSRecoManager.h"

int main( int nargs, char** argv )
{

  argparse::ArgumentParser program("run_kpsrecomanger");
  program.add_argument("-oana", "--output-ana")
    .required()
    .help("set path of KPSRecoManager ana output file");
  program.add_argument("-oll", "--output-larlite")
    .required()
    .help("set path of larlite output file");
  program.add_argument("-olcv", "--output-larcv")
    .required()
    .help("set path of larcv output file");
  program.add_argument("-idl", "--input-dlmerged")
    .default_value("")
    .help("set path of input DL Merged file");
  program.add_argument("-ilf", "--input-larflow")
    .default_value("")
    .help("set path of input larflow larlite file");
  program.add_argument("--ismc")
    .help("if flag provided, input files are assumed to have MC information.")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("-s","--start-entry")
    .help("starting entry number")
    .default_value(0);
  program.add_argument("-n","--num-entries")
    .help("number of entries. if default, run to end of file.")
    .default_value(-1);
  
  
  try {
    program.parse_args( nargs, argv );
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  
  std::cout << "run_kpsrecomanager" << std::endl;
  larflow::reco::KPSRecoManager recoman( program.get<std::string>("--output-ana"), 2 );
  recoman.minimze_output_size(true);
  recoman.set_verbosity( larcv::msg::kINFO );
  if ( program["--ismc"] == true ) { 
    recoman.saveEventMCinfo( true );
  }
  else {
    recoman.saveEventMCinfo( false );
  }

  // recoman.set_verbosity(larcv.msg.kINFO)
  // recoman.minimze_output_size(True)
  
  // if args.ismc:
  //     recoman.saveEventMCinfo( args.ismc )
  //     if args.run_perfect_mcreco:
  //         recoman.runPerfectMCreco( True )
  // if args.event_filter:
  //     recoman.saveSelectedNuVerticesOnly( args.event_filter )
  // if args.stop_after_keypointreco:
  //     recoman.debug_stop_at_keypoint_reco( True )
  // if args.stop_after_spacepointprep:
  //     print("STOP AFTER SPACEPOINT PREP")
  //     recoman.debug_stop_at_spacepoint_prep( True )
  //     print("[enter] to start")
  //     input()
  // if args.stop_after_subclustering:
  //     print("STOP AFTER SUBCLUSTERING")
  //     recoman.debug_stop_at_subclustering( True )
  //     print("[enter] to start")
  //     input()
  // if args.stop_after_nutracker:
  //     print("STOP AFTER MULTIPRONG RECO/NUTRACKBUILDER")
  //     recoman.debug_stop_at_nutracker( True )
  //     print("[enter] to start")
  //     input()
  // if args.save_all_keypoints:
  //     print("Save all reconstructed keypoints. Usually for selection development.")
  //     recoman.saveEventKeypoints( True )
  

  // INPUT/OUTPUT SETTINGS
  larcv::IOManager iolcv( larcv::IOManager::kBOTH, "larcv", larcv::IOManager::kTickBackward );
  iolcv.add_in_file( program.get<std::string>("--input-dlmerged") );
  iolcv.specify_data_read( larcv::kProductImage2D, "wire" );
  iolcv.specify_data_read( larcv::kProductImage2D, "thrumu" );
  iolcv.specify_data_read( larcv::kProductImage2D, "ancestor" );
  iolcv.specify_data_read( larcv::kProductImage2D, "segment" );
  iolcv.specify_data_read( larcv::kProductImage2D, "instance" );
  iolcv.specify_data_read( larcv::kProductImage2D, "larflow" );
  iolcv.specify_data_read( larcv::kProductChStatus, "wire" );
  iolcv.specify_data_read( larcv::kProductImage2D, "ubspurn_plane0" );
  iolcv.specify_data_read( larcv::kProductImage2D, "ubspurn_plane1" );
  iolcv.specify_data_read( larcv::kProductImage2D, "ubspurn_plane2" );
  iolcv.specify_data_read( larcv::kProductSparseImage, "sparseuresnetout" );
  iolcv.reverse_all_products();
  iolcv.set_out_file( program.get<std::string>("--output-larcv") );

  larlite::storage_manager ioll( larlite::storage_manager::kBOTH );
  ioll.add_in_filename(  program.get<std::string>("--input-dlmerged") );
  ioll.add_in_filename(  program.get<std::string>("--input-larflow") );  
  ioll.set_out_filename( program.get<std::string>("--output-larlite") );
  
  // set input products
  ioll.set_data_to_read( larlite::data::kLArFlow3DHit, "larmatch" );
  ioll.set_data_to_read( larlite::data::kMCTrack,  "mcreco" );
  ioll.set_data_to_read( larlite::data::kMCShower, "mcreco" );
  ioll.set_data_to_read( larlite::data::kMCTruth,  "generator" );
  ioll.set_data_to_read( larlite::data::kOpFlash,  "simpleFlashBeam" );
  ioll.set_data_to_read( larlite::data::kOpFlash,  "simpleFlashCosmic" );
  // set output products
  ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "larmatch" );
  ioll.set_data_to_write( larlite::data::kMCTruth, "generator" );
  ioll.set_data_to_write( larlite::data::kMCShower, "mcreco" );
  ioll.set_data_to_write( larlite::data::kMCTrack,  "mcreco" );

  //"Save products that can be plotted in vis_kpreco.py"
    
  //# cosmic reco saved, since nuvertex data in ana file is does not save cosmic info
  ioll.set_data_to_write( larlite::data::kTrack, "boundarycosmic" );
  ioll.set_data_to_write( larlite::data::kTrack, "boundarycosmicnoshift" );
  ioll.set_data_to_write( larlite::data::kTrack, "containedcosmic" );
  ioll.set_data_to_write( larlite::data::kTrack, "nutrack_fitted" );  
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "cosmicproton" ); // out-of-time track clusters with dq/dx consistent with possible proton
  ioll.set_data_to_write( larlite::data::kPCAxis, "cosmicproton" ); // out-of-time track clusters with dq/dx consistent with possible proton

  // keypoint reco
  ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "keypoint" ); // # save reco keypoints, used to seed nu candidates
  ioll.set_data_to_write( larlite::data::kLArFlow3DHit, "keypointcosmic" ); // # save reco keypoints, used to seed cosmic candidates    

  // cosmic hit clusters:  trade space for time, since can use track paths to pick up hits again
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "boundarycosmicnoshift" );
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "containedcosmic" );

  // cluster reco
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "trackprojsplit_wcfilter" ); //# in-time track clusters
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "showerkp" ); //# in-time shower clusters, found using shower keypoints
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "showergoodhit" ); // # in-time shower clusters
  ioll.set_data_to_write( larlite::data::kLArFlowCluster, "hip" ); // # in-time proton tracks
  ioll.set_data_to_write( larlite::data::kPCAxis, "trackprojsplit_wcfilter" ); // # in-time track clusters
  ioll.set_data_to_write( larlite::data::kPCAxis, "showerkp" );     //# in-time shower clusters, found using shower keypoints
  ioll.set_data_to_write( larlite::data::kPCAxis, "showergoodhit" ); //# in-time shower clusters
  ioll.set_data_to_write( larlite::data::kPCAxis, "hip" );           //# in-time proton tracks
    
  // save flash
  ioll.set_data_to_write( larlite::data::kOpFlash, "simpleFlashBeam" );
  ioll.set_data_to_write( larlite::data::kOpFlash, "simpleFlashCosmic" );
    
  recoman.minimze_output_size(false);
  
    
// if args.products in ["rerun"]:
//     print("Save enough info to allow rerunning")
//     #larcv
//     iolcv.addto_storeonly_list( larcv.kProductImage2D,  "wire" )
//     iolcv.addto_storeonly_list( larcv.kProductImage2D,  "thrumu" )
//     iolcv.addto_storeonly_list( larcv.kProductChStatus, "wire" )          
//     for p in range(3):
//         iolcv.addto_storeonly_list( larcv.kProductImage2D, "ubspurn_plane%d"%(p) )
//     iolcv.addto_storeonly_list( larcv.kProductSparseImage, "sparseuresnetout" )
//     for truthproduct in ["instance","segment","ancestor","larflow"]:
//         iolcv.addto_storeonly_list( larcv.kProductImage2D, truthproduct )
         
//     #larlite
//     io.set_data_to_write( larlite.data.kLArFlow3DHit, "larmatch" )
//     io.set_data_to_write( larlite.data.kMCTruth, "generator" )
//     io.set_data_to_write( larlite.data.kMCShower, "mcreco" )
//     io.set_data_to_write( larlite.data.kMCTrack,  "mcreco" )

// if args.products in ["rerun","min"]:

//     print("Save minimal amount of data, enough to plot in vis_kpreco.py")
    
//     # cosmic reco saved, since nuvertex data in ana file is does not save cosmic info
//     io.set_data_to_write( larlite.data.kTrack, "boundarycosmic" )
//     io.set_data_to_write( larlite.data.kTrack, "boundarycosmicnoshift" )
//     io.set_data_to_write( larlite.data.kTrack, "containedcosmic" )
//     io.set_data_to_write( larlite.data.kTrack, "nutrack_fitted" )  
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton
//     io.set_data_to_write( larlite.data.kPCAxis, "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton

//     # keypoint reco
//     io.set_data_to_write( larlite.data.kLArFlow3DHit, "keypoint" ) # save reco keypoints, used to seed nu candidates
//     io.set_data_to_write( larlite.data.kLArFlow3DHit, "keypointcosmic" ) # save reco keypoints, used to seed cosmic candidates    

//     # cosmic hit clusters:  trade space for time, since can use track paths to pick up hits again
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "boundarycosmicnoshift" )
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "containedcosmic" )

//     # cluster reco
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "trackprojsplit_wcfilter" ) # in-time track clusters
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "showerkp" )      # in-time shower clusters, found using shower keypoints
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "showergoodhit" ) # in-time shower clusters
//     io.set_data_to_write( larlite.data.kLArFlowCluster, "hip" )           # in-time proton tracks
//     io.set_data_to_write( larlite.data.kPCAxis, "trackprojsplit_wcfilter" ) # in-time track clusters
//     io.set_data_to_write( larlite.data.kPCAxis, "showerkp" )      # in-time shower clusters, found using shower keypoints
//     io.set_data_to_write( larlite.data.kPCAxis, "showergoodhit" ) # in-time shower clusters
//     io.set_data_to_write( larlite.data.kPCAxis, "hip" )           # in-time proton tracks
    
//     # save flash
//     io.set_data_to_write( larlite.data.kOpFlash, "simpleFlashBeam" )
//     io.set_data_to_write( larlite.data.kOpFlash, "simpleFlashCosmic" )  
    
//     recoman.minimze_output_size(False)

// if args.products in ["debug"]:
//     print("Saving all products loaded and made by reconstruction code")
//     # no output specify, so io managers will default to saving everything

  ioll.open();
  iolcv.initialize();

  int lcv_nentries = iolcv.get_n_entries();
  int ll_nentries  = ioll.get_entries();
  std::cout << "Number of entries in the larcv file: " << lcv_nentries << std::endl;
  std::cout << "Number of entries in the larlite file: " << ll_nentries << std::endl;

  int max_nentries = ( lcv_nentries < ll_nentries ) ? lcv_nentries : ll_nentries;
  std::cout << "Running with nentries: " << max_nentries << std::endl;

  int start_entry = program.get<int>("--start-entry");
  if ( start_entry )
    start_entry = 0;
  
  int end_entry = max_nentries;
  if ( program.get<int>("--num-entries")>0 ) {
    end_entry = start_entry + program.get<int>("--num-entries");
    if ( end_entry>max_nentries ) {
      end_entry = max_nentries;
    }
  }
  std::cout << "Running between entries [" << start_entry << ", " << end_entry << ")" << std::endl;
  
  for (int ientry=start_entry; ientry<end_entry; ientry++ ) {

    std::cout << "[ENTRY " << ientry << "]" << std::endl;

    ioll.go_to( ientry );
    iolcv.read_entry( ientry );

    std::cout << "run reco, make nu candidates, calculate selection variables" << std::endl;
    recoman.process( iolcv, ioll );
    
    ioll.set_id( ioll.run_id(), ioll.subrun_id(), ioll.event_id() );
    ioll.next_event();
    iolcv.save_entry();
  }

  ioll.close();
  iolcv.finalize();
  recoman.write_ana_file();

  return 0;
  
};
