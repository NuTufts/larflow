#ifndef __LARFLOW_DATAPREP_SIMCH_TRIPLET_LABEL_MAKER_H__
#define __LARFLOW_DATAPREP_SIMCH_TRIPLET_LABEL_MAKER_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"

#include <array>
#include <map>

#include "TripletLabels_t.h"

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


  std::vector<TripletLabels_t> _triplets_v;
  std::map< std::array<int,4>, unsigned long > _imgcoord_to_tripindex;

};

}
}


#endif