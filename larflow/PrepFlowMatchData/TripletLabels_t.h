#ifndef __LARFLOW_PREP_TRIPLETLABELS_T_H__
#define __LARFLOW_PREP_TRIPLETLABELS_T_H__

#include <set>
#include <array>
#include <vector>

namespace larflow {
namespace prep {

class TripletLabels_t {

public:

  TripletLabels_t()
  : index(-1),edep(0.0)
  {};

  ~TripletLabels_t() {};

  long index;

  std::array<int,5>   imgcoord; // (u,v,y,row,tick)
  std::array<float,3> pos;
  std::array<float,3> pos_reco;

  double edep;
  std::array<float,3> pixval;

  std::set<long> trackids;
  std::set<long> aids;
  std::set<int> pids;
  std::set<int> origin;


};


}
}

#endif