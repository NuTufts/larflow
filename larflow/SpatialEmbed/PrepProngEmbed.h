#ifndef __LARFLOW_SPATIALEMBED_PRONGEMBED_H__
#define __LARFLOW_SPATIALEMBED_PRONGEMBED_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larflow/KeyPoints/LoaderKeypointData.h"

namespace larflow {
namespace spatialembed {

  /** 
   * @class PrepProngEmbed
   * @ingroup SpatialEmbed
   * @brief Convert larmatch output into data for 3D SpatialEmbed clustering for both train and test
   *
   */  
  class PrepProngEmbed : public larcv::larcv_base {

  public:

    PrepProngEmbed()
      : larcv::larcv_base("PrepProngEmbed")
      {};
    PrepProngEmbed( const std::vector<std::string>& input_root_files ); 
    virtual ~PrepProngEmbed(); 

    // void set_adc_image_treename( std::string name ) { _adc_image_treename=name; };
    // void set_truth_image_treename( std::string name ) { _truth_image_treename=name; };    

    larflow::keypoints::LoaderKeypointData _triplet_loader;

    void make_subcluster_fragments();
  };
  
}
}

#endif
