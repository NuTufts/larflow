#ifndef __PREP_AFFINITY_FIELD_H__
#define __PREP_AFFINITY_FIELD_H__

/**
 * prepare training data for affinity field
 *
 */

#include <vector>

// ROOT
#include "TTree.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"

// larflow
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larflow {
namespace keypoints {

  class PrepAffinityField : public larcv::larcv_base {

  public:

    PrepAffinityField();
    virtual ~PrepAffinityField();

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  const larflow::prep::PrepMatchTriplets& match_proposals );


  protected:
    
    // Ana Tree    
    int _run;
    int _subrun;
    int _event;
    TTree* _label_tree; ///< tree for storing labels

  public:

    void defineAnaTree();
    void fillAnaTree();
    void writeAnaTree();

  protected:

    larutil::SpaceChargeMicroBooNE* psce;

  protected:

    std::vector< std::vector<float> > _match_labels_v;

    void _determine_triplet_labels( const std::vector< std::vector<int> >& pixlist_v,
                                    const std::vector<float>& spacepoint_v,
                                    const std::vector< larcv::Image2D >& instance_v,
                                    const larlite::event_mctrack& ev_mctrack_v,
                                    const larlite::event_mcshower& ev_mcshower_v,
                                    std::vector<float>& label_v,
                                    float& weight );

    bool _get_track_direction( const larlite::mctrack& track,
                               const std::vector<double>& pt,
                               const std::vector<larcv::Image2D>& img_v,
                               std::vector<float>& label_v,
                               float& weight );

    bool _get_shower_direction( const larlite::mcshower& shower,
                                const std::vector<double>& pt,
                                const std::vector<larcv::Image2D>& img_v,
                                std::vector<float>& label_v,
                                float& weight );
    

    
  };

}
}

#endif
