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

namespace larflow {
namespace prep {

class SimChTripletLabelMaker : public larcv::larcv_base {

public:

  SimChTripletLabelMaker()
  : larcv::larcv_base("SimChTripletLabelMaker")
  {};

  virtual ~SimChTripletLabelMaker() {};

  void process( larlite::storage_manager& ioll, 
                larcv::IOManager& iolcv );

  void export_as_hdf(std::string hdf_outfile);

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

  // algorithms
  ublarcvapp::mctools::MCParticleGraph   _mcpgraph;     ///< organizes true particle information into graph form    
  ublarcvapp::mctools::MCPixelLabelMaker _mcpixelmaker; ///< makes pixel3d objects from simch 
  larflow::prep::PrepMatchTriplets       _tripletmaker; ///< makes pixel3d objects from wireplane images
  larflow::prep::MCKeypointMaker         _mckpmaker;    ///< makes keypoint

  larflow::prep::EventTriplets_t          _ev_reco_triplets;

  std::vector< larflow::prep::MCKeypoint > _final_keypoint_list;

};

}
}


#endif