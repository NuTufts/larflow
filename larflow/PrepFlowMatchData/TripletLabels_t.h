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
  : index(-1),
  hasmatch(0),
  edep({0.0,0.0,0.0}),
  pixval({0,0,0}),
  ssnetlabel(0),
  ssnetboundary(0),
  ssnet_classcount_weight(0.0)
  {};

  ~TripletLabels_t() {};

  long index;
  int hasmatch;

  std::array<int,5>   imgcoord; // (u,v,y,row,tick)
  std::array<float,3> pos;
  std::array<float,3> pos_reco;

  std::array<double,3> edep;
  std::array<float,3> pixval;

  std::set<long> trackids;
  std::set<long> aids;
  std::set<int> pids;
  std::set<int> origin;

  std::vector<float> kpdist;     ///< distance to closest keypoint
  std::vector<float> kpscores;   ///< keypoint score to predict based on kp distance
  std::vector<float> kpweight;   ///< for each kptype, we balance near-kp vs. non-kp

  int ssnetlabel;     ///< ssnet class label
  int ssnetboundary;  ///< number of neighbors with a different class label
  float ssnet_classcount_weight; ///< class weight


};


}
}

#endif