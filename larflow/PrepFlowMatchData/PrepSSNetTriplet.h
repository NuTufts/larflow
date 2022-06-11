#ifndef __PREP_SSNET_TRIPLET_H__
#define __PREP_SSNET_TRIPLET_H__

#include <vector>
#include "TTree.h"

#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "PrepMatchTriplets.h"
#include "SSNetLabelData.h"

namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData 
   * @class PrepSSNetTriplet
   * @brief Make track/shower and PID label for triplet
   *
   * @author Taritre Wongjirad (taritree.wongjirad@tuts.edu)
   * @date $Data 2020/07/22 17:00$
   *
   * Revision History
   * 2020/07/22: Added doxygen documentation
   *
   */

  class PrepSSNetTriplet {
  public:

    PrepSSNetTriplet()
      : _label_tree(nullptr)
    {};
    virtual ~PrepSSNetTriplet();

    void make_ssnet_labels( larcv::IOManager& iolcv,
                            larlite::storage_manager& ioll,                            
                            const larflow::prep::PrepMatchTriplets& tripletmaker );

    void make_trackshower_labels( const std::vector<larcv::Image2D>& segment_v,
                                  const larflow::prep::PrepMatchTriplets& tripletmaker,
                                  const std::vector<int>& vtx_imgcoord );

    void clear();

    enum { kBG=0, kElectron, kGamma, kMuon, kPion, kProton, kOther, kNumClasses };
    
    TTree* _label_tree;  ///< ROOT TTree used to store label data to output file
    int _run;     ///< run number of event used to make labels
    int _subrun;  ///< subrun of event used to make labels
    int _event;   ///< event number of event used to make labels
    // std::vector< int >                _ssnet_label_v;  ///< pixel topology label:: 0:bg, 1:track, 2:shower
    // std::vector< float >              _ssnet_weight_v; ///< triplet weights based on topology
    // std::vector< int >                _ssnet_num_v;    ///< number of each class
    // std::vector< std::vector<int> >   _pid_label_v;          ///< particle ID labe:: 0:bg, 1:muon,  2: proton, 3:pion, 4:electron, 5:gamma
    // std::vector< float >              _pid_weight_v;         ///< class balancing weight
    // std::vector< std::vector<float> > _boundary_weight_v;    ///< upweights at boundary, vertex
    std::vector< SSNetLabelData > _ssnet_labeldata_v;

    void defineAnaTree();

    /** @brief Write ROOT TTree for storing labels. */
    void writeAnaTree();

    static int larcv2class( int larcv_label ); ///< convert larcv particle type enum to network class label
    
    
  };
  
}
}

#endif
