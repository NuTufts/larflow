#ifndef __LARFLOW_FLASHMATCH_TYPES_H__
#define __LARFLOW_FLASHMATCH_TYPES_H__

#include <vector>

namespace larflow {

  typedef enum { kUnlabeled=-1, kCore, kNonCore, kGapFill, kExt, kNumQTypes } QPointType_t;

  struct QPoint_t {
    QPoint_t() {
      xyz.resize(3,0);
      tick = 0;
      pixeladc = 0.0;
      fromplaneid = -1;
      type = kUnlabeled;
    };
    std::vector<float> xyz; // (x,y,z) coordinates
    float tick;
    float pixeladc;
    int   fromplaneid; // { 0:U, 1:V, 2:Y, 3:UV-ave }
    QPointType_t type; // -1=unspecifed, 0=from flow pred, 1=from gapfil, 2=from tpc extension
  };

  struct FlashData_t : public std::vector<float> {
    FlashData_t() { truthmatched_clusteridx=-1; mctrackid=-1; mctrackpdg=-1; isneutrino=false; isbeam=false; intime=false; };
    int idx;
    int tpc_tick;
    int tpc_trigx;
    bool isbeam;
    bool isneutrino;
    bool intime;
    float tot;
    int mctrackid;
    int mctrackpdg;
    int truthmatched_clusteridx;
    int maxch;
    float maxchposz;      
  };

  struct QCluster_t : public std::vector<QPoint_t> {
    QCluster_t() { truthmatched_flashidx=-1; mctrackid=-1; isneutrino=false; };
    int idx;
    float min_tyz[3];
    float max_tyz[3];
    int mctrackid;
    int truthmatched_flashidx;
    bool isneutrino;
  };
    
  struct FlashHypo_t : public std::vector<float> {
  FlashHypo_t() :
    clusteridx(-1),
      flashidx(-1),
      tot(0.0),
      tot_intpc(0.0),
      tot_outtpc(0.0)
	{};
    int clusteridx;
    int flashidx;
    float tot;
    float tot_intpc;
    float tot_outtpc;
  };
 
}// end of namespace

#endif
