#ifndef __PREP_AFFINITY_FIELD_H__
#define __PREP_AFFINITY_FIELD_H__

#include <vector>

// ROOT
#include "TTree.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/mctrack.h"
#include "larlite/DataFormat/mcshower.h"

// larflow
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larflow {
namespace keypoints {

  /**
   * @ingroup Keypoints
   * @class PrepAffinityField
   * @brief prepare training data for affinity field
   *
   * This class creates the training data for the particle direction
   * task of the network.  This output is inspired by the
   * "Part Affinity Field" output of the OpenPose network.
   *
   * For each proposed space point, we want the network to 
   * produce the following output:
   * \verbatim embed:rst:leading-asterisk
   *   * For a true space point that was made by a track-like particle, we want the output 
   *     to be a 3D vector giving the direction of the particle at that point of its trajectory
   *   * For a true space point that was made by a shower-like particle, we want the output
   *     to be the initial direction of the shower. As a result, the direction would be the same
   *     for all space points that were a part of the same shower
   *   * For a ghost-point, we want the vector to be (0,0,0).
   * \endverbatim
   *
   * This class will store the data into a ROOT tree (inside a ROOT file),
   * name `AffinityFieldTree`. It will consist of a nested vector.
   * The outer vector is over all proposed space points made by larflow::prep::PrepMatchTriplets
   * for an event.  The inner vector is the 3D truth vector we will train the network to output.
   * 
   * Usage: (to do)
   * 
   */  
  class PrepAffinityField : public larcv::larcv_base {

  public:

    PrepAffinityField();
    virtual ~PrepAffinityField();

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  const larflow::prep::PrepMatchTriplets& match_proposals );


  protected:
    
    // Ana Tree    
    int _run; ///< run ID number for an event (saved in the output tree `AffinityFieldTree`)
    int _subrun; ///< subrun ID number for an event (saved in the output tree `AffinityFieldTree`)
    int _event; ///< event ID number for an event (saved in the output tree `AffinityFieldTree`)
    TTree* _label_tree; ///< the tree that will store the labels. Named `AffinityFieldTree` in the ROOT file.

  public:

    void defineAnaTree();
    void fillAnaTree();
    void writeAnaTree();

  protected:

    larutil::SpaceChargeMicroBooNE* psce; ///< pointer to a copy of the space charge calculation utility class

  protected:

    std::vector< std::vector<float> > _match_labels_v; ///< container holding the calculated direction labels for the current event

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
