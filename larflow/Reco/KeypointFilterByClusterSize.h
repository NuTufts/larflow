#ifndef __Keypoint_Filter_By_Cluster_Size_h__
#define __Keypoint_Filter_By_Cluster_Size_h__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "DataFormat/storage_manager.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class KeypointFilterByClusterSize
   * @brief Filter out keypoints if the cluster they are nearest to is small
   *
   * For each keypoint, determine the closest cluster they are on. 
   * If the cluster is too small, remove the keypoint.
   *
   */
  class KeypointFilterByClusterSize : public larcv::larcv_base {

  public:

    /** brief default constructor **/
    KeypointFilterByClusterSize()
      : larcv::larcv_base("KeypointFilterByClusterSize")
      {
        _input_keypoint_tree_name = "keypoint";
        _input_larflowhits_tree_name = "larmatch";
      };
    virtual ~KeypointFilterByClusterSize() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    

  protected:

    std::string _input_keypoint_tree_name;     ///< tree name containing keypoints to filter
    std::string _input_larflowhits_tree_name;  ///< tree name containing input larflow3dhit objects

  public:

    /** @brief set the keypoint input tree name */
    void set_input_keypoint_tree_name( std::string keypoint ) { _input_keypoint_tree_name=keypoint; };

    /** @brief set the larflow hit tree name */
    void set_input_larflowhits_tree_name( std::string hit ) { _input_larflowhits_tree_name=hit; };
    
  };

}
}


#endif
