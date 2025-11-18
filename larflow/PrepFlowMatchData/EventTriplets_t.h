#ifndef __LARFLOW_PREP_EVENTTRIPLETS_T_H__
#define __LARFLOW_PREP_EVENTTRIPLETS_T_H__

#include <vector>
#include <map>
#include <array>

#include "TripletLabels_t.h"

namespace larflow {
namespace prep {

class EventTriplets_t {

public:

  EventTriplets_t(){};
  ~EventTriplets_t(){};

  void clear();

  std::vector<TripletLabels_t> _triplets_v; //< container of triplet info
  std::map< std::array<int,4>, unsigned long > _imgcoord_to_tripindex; //< (u,v,y,row) to position in _triplets_v

};

}
}

#endif