#ifndef __LARFLOW_VOXELIZER_PREP3DSPATIALEMBED_H__
#define __LARFLOW_VOXELIZER_PREP3DSPATIALEMBED_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larflow/Voxelizer/VoxelizeTriplets.h"

namespace larflow {
namespace spatialembed {

  /** 
   * @class Prep3DSpatialEmbed
   * @ingroup SpatialEmbed
   * @brief Convert larmatch output into data for 3D SpatialEmbed clustering for both train and test
   *
   */  
  class Prep3DSpatialEmbed : public larcv::larcv_base {

  public:

    Prep3DSpatialEmbed()
      : larcv::larcv_base("Prep3DSpatialEmbed"),
      _filter_by_instance_image(false),
      _tree(nullptr),
      _kowner(false),
      _in_pvid_row(nullptr),
      _in_pvid_col(nullptr),
      _in_pvid_depth(nullptr),
      _in_pinstance_id(nullptr),      
      _in_pq_u(nullptr),
      _in_pq_v(nullptr),
      _in_pq_y(nullptr)
      {};
    Prep3DSpatialEmbed( const std::vector<std::string>& input_root_files ); 
    virtual ~Prep3DSpatialEmbed() { if (_kowner) delete (TChain*)_tree; };

    /**
     * @brief Data for each non-zero voxel
     *
     */
    struct VoxelData_t {
      std::vector<int> voxel_index; ///< voxel index (row,col,depth)
      std::vector<float> feature_v; ///< feature, possible values: (q_u, q_v, q_z, f_u, f_v, f_z)
      std::vector<float> ave_xyz_v; ///< weighted average (x,y,z) position
      std::vector<int> imgcoord_v;
      int npts;   ///< number of space points we've added to this voxel
      float totw; ///< total weight of points
      int truth_instance_index;
      int truth_realmatch;
    };

    typedef std::vector<VoxelData_t> VoxelDataList_t;

    VoxelDataList_t process( larcv::IOManager& iolcv,
                             larlite::storage_manager& ioll,
                             bool make_truth_if_available );

    VoxelDataList_t process_larmatch_hits( const larlite::event_larflow3dhit& ev_lfhit_v,
                                           const std::vector<larcv::Image2D>& adc_v,
                                           const float larmatch_threshold );

    PyObject* makeTrainingDataDict( const VoxelDataList_t& voxeldata ) const;
    PyObject* makePerfectNetOutput( const VoxelDataList_t& voxeldata,
                                    const std::vector<int>& nvoxels_dim ) const;

    PyObject* process_numpy_arrays( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll,
                                    bool make_truth_if_available ) const;

    void bindVariablesToTree( TTree* atree );
    void fillTree( const Prep3DSpatialEmbed::VoxelDataList_t& data );

    void loadTreeBranches( TTree* atree );
    VoxelDataList_t getTreeEntry(int entry);
    PyObject*       getTreeEntryDataAsArray( int entry );
    PyObject* getNextTreeEntryDataAsArray();
    unsigned long getCurrentEntry();
    
    void generateTruthLabels( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              Prep3DSpatialEmbed::VoxelDataList_t& voxel_v );

    /** @brief set flag determining if we filter the voxels by overlap with instance image pixels */
    void setFilterByInstanceImageFlag( bool filter ) { _filter_by_instance_image = filter; };

    VoxelDataList_t filterVoxelsByInstanceImage( const Prep3DSpatialEmbed::VoxelDataList_t& voxel_v,
                                                 const std::vector<larcv::Image2D>& instance_v );

    const larflow::voxelizer::VoxelizeTriplets& getVoxelizer() const { return _voxelizer; };

    const TTree& getTree() { return *_tree; };

  protected:

    // parameters affecting behavior
    bool _filter_by_instance_image; ///< if true, will remove voxels that doesn't project into a neutrino pixel
    
    larflow::voxelizer::VoxelizeTriplets _voxelizer;
    TTree* _tree;
    unsigned long _current_entry;
    bool _kowner; //< indicates if we own the tree
    std::vector< int > vid_row;
    std::vector< int > vid_col;
    std::vector< int > vid_depth;
    std::vector< int > instance_id;
    std::vector< float > q_u;
    std::vector< float > q_v;
    std::vector< float > q_y;

    // pointers used to load vector branches when reading a tree
    std::vector< int >* _in_pvid_row;
    std::vector< int >* _in_pvid_col;
    std::vector< int >* _in_pvid_depth;
    std::vector< int >* _in_pinstance_id;    
    std::vector< float >* _in_pq_u;
    std::vector< float >* _in_pq_v;
    std::vector< float >* _in_pq_y;
    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
