#ifndef __Keypoint_Filter_By_Cluster_Size_h__
#define __Keypoint_Filter_By_Cluster_Size_h__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "DataFormat/storage_manager.h"

namespace larflow {
namespace reco {

  class KeypointFilterByClusterSize : public larcv::larcv_base {

  public:

    KeypointFilterByClusterSize()
      : larcv::larcv_base("KeypointFilterByClusterSize")
      {
        _input_keypoint_tree_name = "keypoint";
        _input_larflowhits_tree_name = "larmatch";
      };
    virtual ~KeypointFilterByClusterSize() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    

  protected:

    std::string _input_keypoint_tree_name;
    std::string _input_larflowhits_tree_name;

  public:

    void set_input_keypoint_tree_name( std::string keypoint ) { _input_keypoint_tree_name=keypoint; };
    void set_input_larflowhits_tree_name( std::string hit ) { _input_larflowhits_tree_name=hit; };
    
  };

}
}


#endif
