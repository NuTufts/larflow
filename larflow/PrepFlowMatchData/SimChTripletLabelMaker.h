#ifndef __LARFLOW_DATAPREP_SIMCH_TRIPLET_LABEL_MAKER_H__
#define __LARFLOW_DATAPREP_SIMCH_TRIPLET_LABEL_MAKER_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

#include "ublarcvapp/MCTools/MCPixelPGraph.h"

#include <array>
#include <map>

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

  void label_reco_triplets();

  void transfer_truth_to_reco( 
      TripletLabels_t& truth_trip, 
      TripletLabels_t& reco_trip );

  EventTriplets_t _ev_triplets;
  EventTriplets_t _ev_reco_triplets;

};

}
}


#endif