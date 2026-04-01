#ifndef __LARFLOW_DATAPREP_SIMCH_TRIPLET_LABEL_MAKER_H__
#define __LARFLOW_DATAPREP_SIMCH_TRIPLET_LABEL_MAKER_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

#include "ublarcvapp/MCTools/MCParticleGraph.h"
#include "ublarcvapp/MCTools/MCPixelLabelMaker.h"

#include <array>
#include <map>

#include "PrepMatchTriplets.h"
#include "MCKeypointMaker.h"
#include "EventTriplets_t.h"
#include "ShowerFragmentOriginMaker.h"

namespace HighFive {
  class File;
}

namespace larflow {
namespace prep {

class SimChTripletLabelMaker : public larcv::larcv_base {

public:

  SimChTripletLabelMaker();

  ~SimChTripletLabelMaker();

  void set_is_data() { _is_mc = false; };
  void set_is_mc() { _is_mc = true; };
  void process_mcc9_sim() { _is_mc = true; _process_ub_mcc9=true; };
  void set_adc_treename( std::string treename ) { _adc_treename=treename; };

  void process( larlite::storage_manager& ioll, 
                larcv::IOManager& iolcv );

  void export_as_hdf(std::string hdf_outfile);

  void open_hdf_file(std::string hdf_outfile );

  void close_hdf_file();

  void save_truth_tripletinfo( bool saveit ) { _save_truth_triplet_info=saveit; };

  void save_entry( std::string groupname_prefix );

  void save_entry_to_hdf( 
    HighFive::File& file, 
    std::string groupname_prefix );

  void save_entry_sparseimg( HighFive::File& file, std::string groupname_prefix );

  void save_entry_truetriplets(
    HighFive::File& file,
    std::string groupname_prefix );

  void save_mc_particle_tree( HighFive::File& file, std::string groupname_prefix );

  void make_truthlabels_fromsimch(
      larlite::storage_manager& ioll, 
      larcv::IOManager& iolcv,
      ublarcvapp::mctools::MCPixelPGraph& mcpg,
      larutil::SpaceChargeMicroBooNE* psce );

  void make_reco_triplets(larcv::IOManager& iolcv);

  void label_reco_triplets( 
    ublarcvapp::mctools::EventMCPixelLabels& pixel3d,
    larflow::prep::EventTriplets_t& triplets );

  void transfer_truth_to_reco( 
      ublarcvapp::mctools::MCPixelLabels& truth_trip, 
      TripletLabels_t& reco_trip );

  void adjust_keypoints( 
    const std::vector< larflow::prep::MCKeypoint >& mckeypoints,
    larflow::prep::EventTriplets_t& labeled_reco_triplets,
    ublarcvapp::mctools::MCParticleGraph& mcpg );

  void make_keypoint_labels(
    float kp_sigma, 
    float score_threshold );

  void make_ssnet_labels( ublarcvapp::mctools::MCParticleGraph& mcpg );

  int get_ssnet_class_label( int pid );

  int get_refined_shower_class_label( 
    ublarcvapp::mctools::MCParticleGraph& mcpg,
    larflow::prep::TripletLabels_t& triplet );

  void make_low_energy_deposit_labels( int npts_threshold );

  /// Helper function to write compressed HDF5 datasets
  template<typename T>
  void dump_compressed( HighFive::File& file,
                        const std::string& dataset_path,
                        const T& data,
                        int compression_level = 6,
                        size_t chunk_size = 10000 );


  // algorithms
  ublarcvapp::mctools::MCParticleGraph   _mcpgraph;     ///< organizes true particle information into graph form    
  ublarcvapp::mctools::MCPixelLabelMaker _mcpixelmaker; ///< makes pixel3d objects from simch 
  larflow::prep::PrepMatchTriplets       _tripletmaker; ///< makes pixel3d objects from wireplane images
  larflow::prep::MCKeypointMaker         _mckpmaker;    ///< makes keypoint
  larflow::prep::ShowerFragmentOriginMaker _shower_fragment_maker; ///< makes shower fragment training data

  larflow::prep::EventTriplets_t          _ev_reco_triplets;

  std::vector< larflow::prep::MCKeypoint > _final_keypoint_list;

  larutil::SpaceChargeMicroBooNE* _psce;
  HighFive::File* _hdf_file;
  bool _save_weights_to_hdf;
  bool _save_truth_triplet_info;

  // name of producer with wire plane images (default "wiremc")
  std::string _adc_treename;

  // flag for running on data
  bool _is_mc;

  // flag for getting truth from PrepMatchTriplets, needed for old MicroBooNE MCC9 files
  bool _process_ub_mcc9;

};

}
}


#endif