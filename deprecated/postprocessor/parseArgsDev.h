#ifndef __PARSE_ARG_DEV_H__
#define __PARSE_ARG_DEV_H__

#include <vector>
#include <iostream>
#include <string>

// argument parser for dev.cxx, used as development code for dlcosmictag post-processor
namespace dlcosmictag  {

  /**
   * Struct used to hold input arguments
   *
   */ 
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
      ssnet_cropped(false),
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
    bool ssnet_cropped;
    bool use_ancestor_img;
    bool makehits_useunmatched;
    bool makehits_require_3dconsistency;
    int process_num_events;
    bool kVISUALIZE;
    bool kINSPECT;

  };

  /**
   * Struct used to hold command flags and help
   *
   */   
  struct Arg_t {
  Arg_t( std::string f, std::string h ) 
  : flag(f),
      help(h)
    {};
    std::string flag;
    std::string help;
  };

  InputArgs_t parseArgs( int nargs, char** argv );
  
}

#endif
