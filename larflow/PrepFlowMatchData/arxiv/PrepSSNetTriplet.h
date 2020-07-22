#ifndef __PREP_SSNET_TRIPLET_H__
#define __PREP_SSNET_TRIPLET_H__

/**
 * class responsible for making track/shower and PID label for triplet
 *
 */

#include <vector>
#include "TTree.h"

#include "DataFormat/storage_manager.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "PrepMatchTriplets.h"

namespace larflow {
namespace prepflowmatchdata {

  class PrepSSNetTriplet {
  public:

    PrepSSNetTriplet()
      : _label_tree(nullptr)
    {};
    virtual ~PrepSSNetTriplet();

    void make_ssnet_labels( larcv::IOManager& iolcv,
                            larlite::storage_manager& ioll,                            
                            const larflow::PrepMatchTriplets& tripletmaker );

    void make_trackshower_labels( const std::vector<larcv::Image2D>& segment_v,
                                  const larflow::PrepMatchTriplets& tripletmaker,
                                  const std::vector<int>& vtx_imgcoord );
    
    TTree* _label_tree;
    int _run;
    int _subrun;
    int _event;
    std::vector< int >                _trackshower_label_v;  //< 0:bg, 1:track, 2:shower
    std::vector< float >              _trackshower_weight_v; //< triplet weights based on topology
    std::vector< float >              _trackshower_num_v;    //< number of each class
    std::vector< std::vector<int> >   _pid_label_v;          //< 0:bg, 1:muon,  2: proton, 3:pion, 4:electron, 5:gamma
    std::vector< float >              _pid_weight_v;         //< class balancing weight
    std::vector< std::vector<float> > _boundary_weight_v;    //< upweights at boundary, vertex

    void defineAnaTree();
    void writeAnaTree();
    
    
  };
  
}
}

#endif
