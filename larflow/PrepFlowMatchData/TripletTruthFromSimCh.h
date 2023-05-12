#ifndef __PREPMATCHTRIPLETS_TRIPLET_TRUTH_FROM_SIMCH_H__
#define __PREPMATCHTRIPLETS_TRIPLET_TRUTH_FROM_SIMCH_H__

#include <map>
#include <string>
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/mctrack.h"
#include "larlite/DataFormat/mcshower.h"
#include "larcv/core/Base/larcv_base.h"
#include "ublarcvapp/MCTools/SimChannelVoxelizer.h"
#include "MatchTriplets.h"

namespace larflow {
namespace prep {

  class TripletTruthFromSimCh : public larcv::larcv_base {
  public:
    
    TripletTruthFromSimCh()
      : larcv::larcv_base("TripletTruthFromSimCh") {};
    virtual ~TripletTruthFromSimCh() {};

    void process_truth_labels( larlite::storage_manager& ioll,
			       larflow::prep::MatchTriplets& triplets,
			       std::string simch_producer );

  protected:
    
    void make_truth_labels( larlite::storage_manager& ioll,
			    larflow::prep::MatchTriplets& triplets,
			    std::string simch_producer );

    // map from shower daughter IDs to mother IDs
    std::map<unsigned long, unsigned long> _shower_daughter2mother;
    void fill_daughter2mother_map( const std::vector<larlite::mcshower>& shower_v );

    std::map<unsigned long, int> _instance2class_map;
    void fill_class_map( const std::vector<larlite::mctrack>&  track_v,
                         const std::vector<larlite::mcshower>& shower_v );

    ublarcvapp::mctools::SimChannelVoxelizer voxelizer;    
        
  };
  
}
}

#endif
