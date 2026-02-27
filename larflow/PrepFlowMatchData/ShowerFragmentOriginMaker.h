#ifndef __LARFLOW_PREP_SHOWER_FRAGMENT_ORIGIN_MAKER_H__
#define __LARFLOW_PREP_SHOWER_FRAGMENT_ORIGIN_MAKER_H__

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "ublarcvapp/MCTools/MCParticleGraph.h"
#include "ShowerFragmentOrigin.h"
#include "MCKeypoint.h"
#include "EventTriplets_t.h"

namespace HighFive {
  class File;
}

namespace larflow {
namespace prep {

class ShowerFragmentOriginMaker : public larcv::larcv_base {

public:

  ShowerFragmentOriginMaker()
  : larcv::larcv_base("ShowerFragmentOriginMaker") 
  {};

  ~ShowerFragmentOriginMaker() {};

  ShowerFragmentOrigin _fragment_data;

  void clear() { _fragment_data.clear(); };

  void build_shower_fragments(
    const std::vector<MCKeypoint>& keypoints,
    ublarcvapp::mctools::MCParticleGraph& mcpg,
    larflow::prep::EventTriplets_t& pixel3d,
    float edep_cluster_threshold,
    int edep_cluster_size_threshold,
    float edep_point_threshold );

  void save_entry_to_hdf( HighFive::File& file, std::string group_prefix_name );

};


}
}

#endif