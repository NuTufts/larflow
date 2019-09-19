#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <TApplication.h>

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctrack.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// cluster
#include "cluster/TruthCluster.h"
#include "cluster/TruthClusterWithAncestorImage.h"

#include "nlohman_json/json.hpp"

using json = nlohmann::json;

struct InputArgs_t {
  
  InputArgs_t()
    : hasinput(false),
      hasmc_mcinfo(false),
      hasmc_ancestor(false),
      outputset(false)
  {};

  bool hasinput;       // marked true when input file with larflow3dhits argument given
  bool hasmc_mcinfo;   // marked true when input file with mcinfo larlite specified
  bool hasmc_ancestor; // marked true when input file with ancestor larcv image specified
  bool outputset;      // marked true when output specified
  std::string input_larflowhits; // larflow hit output
  std::string input_mcinfo;      // mctrack/mcshower info
  std::string input_ancestor;    // ancestor image
  std::string output_json;       // output file

  void set_input( std::string in )    { input_larflowhits=in; hasinput=true; };
  void set_mcinfo( std::string in )   { input_mcinfo=in; hasmc_mcinfo=true; };
  void set_ancestor( std::string in ) { input_ancestor=in; hasmc_ancestor=true; };
  void set_output( std::string out )  { output_json=out; outputset=true; };
  bool is_config_ready() { return hasinput && ( hasmc_mcinfo || hasmc_ancestor ) && outputset; };
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
  commands.push_back( Arg_t("-i", "input: larflow hits [required]") );
  commands.push_back( Arg_t("-mc","input: use mcinfo  [must choose this or ancestor]") );
  commands.push_back( Arg_t("-an","input: use ancstor [must choose this or ancestor]") );
  commands.push_back( Arg_t("-o", "json file name [required]") );  

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
  
  for ( int iarg=1; iarg<nargs; iarg++ ) {

    if ( argv[iarg][0]!='-' )
      continue;

    bool commandmatched = false;
    for ( auto const& command : commands ) {
      
      if ( std::string(argv[iarg])==command.flag ) {
	commandmatched = true;
	std::string strflag = argv[iarg];
	
	if ( command.flag=="-i" ) {
	  argconfig.set_input( argv[iarg+1] );
	}
	else if ( command.flag=="-mc" ) {
	  argconfig.set_mcinfo( argv[iarg+1] );
	}
	else if ( command.flag=="-an" ) {
	  argconfig.set_ancestor( argv[iarg+1]  );
	}
	else if ( command.flag=="-o" ) {
          argconfig.set_output( argv[iarg+1] );
	}
	// operation complete
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
  if ( !argconfig.is_config_ready() )
    throw std::runtime_error("missing input and/or output files");

  return argconfig;
}


int main( int nargs, char** argv ) {

  std::cout << "conversion from larflow hit larlite to json file" << std::endl;

  // expects cropped output (from deploy/run_larflow_wholeview.py)
  // -------------------------------------------------------------

  // arg parsing
  InputArgs_t inputargs = parseArgs( nargs, argv );
  
  TApplication app ("app",&nargs,argv);  

  std::cout << "===========================================" << std::endl;
  std::cout << " JSON conversion tool " << std::endl;
  std::cout << " -------------------------- " << std::endl;
  
  // hit (and mctruth) event data
  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( inputargs.input_larflowhits );
  if ( inputargs.hasmc_mcinfo )
    io.add_in_filename( inputargs.input_mcinfo );
  io.open();

  larcv::IOManager iolcv( larcv::IOManager::kREAD, "larcv", larcv::IOManager::kTickBackward );
  if ( inputargs.hasmc_ancestor )
    iolcv.add_in_file( inputargs.input_ancestor );
  iolcv.initialize();
    
  int nentries = io.get_entries();
  std::cout << "Number of entries in file: " << nentries << std::endl;

  larflow::TruthCluster clusteralgo;
  
  json jall; // empty structure
  
  for (int ientry=0; ientry<nentries; ientry++) {
    std::cout << "process entry[" << ientry << "]" << std::endl;
    io.go_to(ientry);

    // get larflow hits
    // auto ev_lfhits   = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "flowhits" );
    auto ev_lfhits   = (larlite::event_larflow3dhit*)io.get_data( larlite::data::kLArFlow3DHit, "trueflowhits" );
    

    // for truth, we favor using ancestor image
    larlite::event_mctrack*  ev_mctrack  = nullptr;
    larlite::event_mcshower* ev_mcshower = nullptr;
    larcv::EventImage2D*     ev_ancestor = nullptr;

    json jevent; // event object

    std::vector<float> x_v(ev_lfhits->size(),0);
    std::vector<float> y_v(ev_lfhits->size(),0);
    std::vector<float> z_v(ev_lfhits->size(),0);
    std::vector<int>   c_v(ev_lfhits->size(),-1);
    std::map<int,int>  ancestor2clusteridx;
    
    if ( inputargs.hasmc_ancestor ) {
      // TRUTH LABEL USING ANCESTOR IMAGE
      std::cout << "  cluster via ancester image" << std::endl;
      ev_ancestor = (larcv::EventImage2D*) iolcv.get_data( larcv::kProductImage2D, "ancestor" );

      // source plane ancestor image
      const larcv::Image2D& ancestor_img = ev_ancestor->Image2DArray().at(2);
      const larcv::ImageMeta& meta = ancestor_img.meta();
      
      // loop through hits, project into ancestor image
      int nclusters = 0;
      int ihit = -1;
      
      for ( auto const& hit : *ev_lfhits ) {
        ihit++;
        
        int tick = hit.tick;
        int wire = hit.srcwire;

        int row = meta.row( tick );
        int col = meta.col( wire );
        
        int ancestorid = ancestor_img.pixel( row, col );
        int cid = -1;
        if ( ancestorid>0 ) {
          auto it = ancestor2clusteridx.find( ancestorid );
          if ( it==ancestor2clusteridx.end() ) {
            // make new cluster
            nclusters++;
            ancestor2clusteridx[ ancestorid ] = nclusters;
            cid = nclusters;
          }
          else {
            cid = it->second;
          }
        }

        // fill hit info (x,y,z,cluster)
        x_v[ihit] = hit[0];
        y_v[ihit] = hit[1];
        z_v[ihit] = hit[2];
        c_v[ihit] = cid;
      }
      
    }
    else {
      // TRUTH LABEL USING CLUSTERING AROUND MCTRACK/MCSHOWER
      ev_mctrack  = (larlite::event_mctrack*) io.get_data( larlite::data::kMCTrack,  "mcreco" );
      ev_mcshower = (larlite::event_mcshower*)io.get_data( larlite::data::kMCShower, "mcreco" );

      // cluster hits using proximity to truth tracks
      std::vector< std::vector<const larlite::larflow3dhit*> > cluster_v = clusteralgo.clusterHits( *ev_lfhits, *ev_mctrack, *ev_mcshower, true, true );
    
      int ihit=0;
      int icluster=0;
      for ( auto const& hit_v : cluster_v ) {
        
        int cid=-1;
        if ( icluster+1<cluster_v.size() )
          cid = icluster;
      
        for ( auto const& phit : hit_v ) {
          float x=(*phit)[0];
          float y=(*phit)[1];
          float z=(*phit)[2];
          
          if ( x==-1 && x==y && x==z )
            continue;
          
          x_v[ihit] = x;
          y_v[ihit] = y;
          z_v[ihit] = z;
          c_v[ihit] = cid;
          
          ihit++;
        }
      
        icluster++;
      }
      
      x_v.resize(ihit);
      y_v.resize(ihit);
      z_v.resize(ihit);
      c_v.resize(ihit);
    }
    
    json jx(x_v);
    json jy(y_v);
    json jz(z_v);
    json jc(c_v);
    
    jevent["x"] = jx;
    jevent["y"] = jy;
    jevent["z"] = jz;
    jevent["c"] = jc;
    
    char zevent[200];
    sprintf(zevent,"event%d",ientry);
    jall[zevent] = jevent;
    
  }//end of entry loop
  
  std::ofstream out(inputargs.output_json.c_str());
  out << std::setw(4) << jall << std::endl;
  out.close();
  
  return 0;

}
