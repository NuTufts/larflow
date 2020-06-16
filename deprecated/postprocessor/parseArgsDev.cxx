#include "parseArgsDev.h"

namespace dlcosmictag {
  
  InputArgs_t parseArgs( int nargs, char** argv ) {

    std::vector< Arg_t > commands;
    commands.push_back( Arg_t("-c",  "combined (cropped) dlcosmictag file [required or provide individual files below]") );
    commands.push_back( Arg_t("-y2u","individual (cropped) y2u file [required if combined not provided]") );
    commands.push_back( Arg_t("-y2v","individual (cropped) y2v file [required if combined not provided]") );
    commands.push_back( Arg_t("-adc","individual (cropped) adc file [required if combined not provided]") );
    commands.push_back( Arg_t("-in", "individual (cropped) infill file") );
    commands.push_back( Arg_t("-ss", "individual (cropped) ssnet file") );
    commands.push_back( Arg_t("-wss","wholeview ssnet file") );  
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
            argconfig.ssnet_cropped = true;
          }
          else if ( command.flag=="-wss" ) {
            argconfig.input_ssnet = argv[iarg+1];
            argconfig.has_ssnet = argv[iarg+1];
            argconfig.ssnet_cropped = false;          
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
}
