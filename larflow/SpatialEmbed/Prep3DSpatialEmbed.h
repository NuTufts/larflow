#ifndef __LARFLOW_VOXELIZER_PREP3DSPATIALEMBED_H__
#define __LARFLOW_VOXELIZER_PREP3DSPATIALEMBED_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "TRandom3.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/PrepFlowMatchData/TripletTruthFixer.h"
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
      _filter_out_non_nu_pixels(false),      
      _tree(nullptr),
      _current_entry(0),
      _num_entries(0),
      _shuffle(false),
      _kowner(false),
      _kMaxTripletPerVoxel(5),
      _adc_image_treename("wire"),
      _truth_image_treename("segment"),
      _in_pvid_row(nullptr),
      _in_pvid_col(nullptr),
      _in_pvid_depth(nullptr),
      _in_pinstance_id(nullptr),
      _in_pancestor_id(nullptr),      
      _in_pparticle_id(nullptr),      
      _in_pq_u(nullptr),
      _in_pq_v(nullptr),
      _in_pq_y(nullptr),
      _in_ptriplet_idx_v(nullptr),
      _in_psubcluster_id(nullptr),
      _in_psubinstance_id(nullptr),
      _rand(nullptr)
      {};
    Prep3DSpatialEmbed( const std::vector<std::string>& input_root_files ); 
    virtual ~Prep3DSpatialEmbed(); 

    void set_adc_image_treename( std::string name ) { _adc_image_treename=name; };
    void set_truth_image_treename( std::string name ) { _truth_image_treename=name; };    

    /**
     * @brief Data for each non-zero voxel
     *
     */
    struct VoxelData_t {
      std::vector<int> voxel_index;  ///< voxel index (row,col,depth)
      std::vector<float> feature_v;  ///< feature, possible values: (q_u, q_v, q_z, f_u, f_v, f_z)
      std::vector<float> ave_xyz_v;  ///< weighted average (x,y,z) position
      std::vector<int> imgcoord_v;   ///< 2D plane coordinages (col,col,col,row)
      std::vector<int> tripletidx_v; ///< indices of spacepoints contributing to voxel. indices are to PrepMatchTriplets._triplet_v.
      int npts;   ///< number of space points we've added to this voxel
      float totw; ///< total weight of points
      int truth_instance_index;
      int truth_ancestor_index; 
      int truth_realmatch;
      int truth_pid;
      int subclusterid; ///< reco-ish
      int truth_subinstance_index; ///< unique pairs of (truth_instance_index,subclusterid)
    };

    typedef std::vector<VoxelData_t> VoxelDataList_t;

    VoxelDataList_t process( larcv::IOManager& iolcv,
                             larlite::storage_manager& ioll,
                             bool make_truth_if_available );
    
    VoxelDataList_t
    process_from_truelarflowhits( larcv::IOManager& iolcv,
                                  larlite::storage_manager& ioll );
    

    VoxelDataList_t process_larmatch_hits( const larlite::event_larflow3dhit& ev_lfhit_v,
                                           const std::vector<larcv::Image2D>& adc_v,
                                           const float larmatch_threshold );

    PyObject* makeTrainingDataDict( const VoxelDataList_t& voxeldata ) const;
    PyObject* makeTrainingDataDict( const std::vector<VoxelDataList_t>& voxeldata_v ) const;    
    PyObject* makePerfectNetOutput( const VoxelDataList_t& voxeldata,
                                    const std::vector<int>& nvoxels_dim,
                                    const int nsigma,
                                    const float seed_sigma ) const;

    PyObject* process_numpy_arrays( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll,
                                    bool make_truth_if_available ) const;

    void bindVariablesToTree( TTree* atree );
    void fillTree( const Prep3DSpatialEmbed::VoxelDataList_t& data );

    void loadTreeBranches( TTree* atree );
    VoxelDataList_t getTreeEntry(int entry);
    PyObject*       getTreeEntryDataAsArray( int entry );
    PyObject*       getNextTreeEntryDataAsArray();
    PyObject*       getTrainingDataBatch(int batch_size);
    unsigned long   getCurrentEntry() { return _current_entry; };

    PyObject* makeSubclusterTrainingDataDict( const std::vector<VoxelDataList_t>& voxeldata ) const;
    PyObject* getSubclusterTrainingDataBatch(int nevents);
    
    void generateTruthLabels( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll,
                              larflow::prep::PrepMatchTriplets& triplet_maker,                              
                              Prep3DSpatialEmbed::VoxelDataList_t& voxel_v );
    
    /** @brief set flag determining if we filter the voxels by overlap with instance image pixels */
    void setFilterByInstanceImageFlag( bool filter ) { _filter_by_instance_image = filter; };

    /** @brief set flag determining if we should filter out non-neutrino pixels */
    void setFilterOutNonNuPixelsFlag( bool filter ) { _filter_out_non_nu_pixels = filter; };

    /** @brief set flag determining if we should read random entry */
    void setShuffle( bool shuffle ) { _shuffle=shuffle; };
    
    VoxelDataList_t filterVoxelsByInstanceImage( const Prep3DSpatialEmbed::VoxelDataList_t& voxel_v,
                                                 const std::vector<larcv::Image2D>& instance_v );

    /** @brief set maximum number of spacepoints (aka triplets) that get assigned to voxel */
    void setMaximumTripletPerVoxel(  int maxtriplets ) { _kMaxTripletPerVoxel = maxtriplets; };

    const larflow::voxelizer::VoxelizeTriplets& getVoxelizer() const { return _voxelizer; };

    const TTree& getTree() { return *_tree; };

  protected:

    // parameters affecting behavior
    bool _filter_by_instance_image; ///< if true, will remove voxels that doesn't project into a neutrino pixel
    bool _filter_out_non_nu_pixels; ///< if true, will remove voxels with non-neutrino origin

    larflow::prep::PrepMatchTriplets _triplet_maker; ///< class that makes triplets
    larflow::prep::TripletTruthFixer _triplet_truth_fixer; ///< class that fixes missing instance labels
    larflow::voxelizer::VoxelizeTriplets _voxelizer; ///< class that defines voxelization
    TTree* _tree;
    unsigned long _current_entry;
    unsigned long _num_entries;
    bool _shuffle; ///< if true, shuffle entry when loading batch
    bool _kowner;  ///< indicates if we own the tree (and must delete in destructor)
    int  _kMaxTripletPerVoxel; ///< maximum number of spacepoints contributing to voxel
    std::string _adc_image_treename;
    std::string _truth_image_treename;
    // voxel-indexed information
    std::vector< int > vid_row;      ///< voxel row
    std::vector< int > vid_col;      ///< voxel col
    std::vector< int > vid_depth;    ///< voxel depth
    std::vector< int > instance_id;  ///< instance label
    std::vector< int > ancestor_id;  ///< ancestor label (not implemented)
    std::vector< int > particle_id;  ///< particle id label
    std::vector< float > q_u;        ///< charge on u-plane
    std::vector< float > q_v;        ///< charge on v-plane
    std::vector< float > q_y;        ///< charge on y-plane
    std::vector< std::vector<int> > triplet_idx_v; ///< indices of triplets in _triplet_maker that contribute to voxel
    std::vector< int > subcluster_id;  ///< labels from dbscan clustering
    std::vector< int > subinstance_id; ///< labels from unique (instance,dbscan cluster) pairs

    // pointers used to load vector branches when reading a tree
    std::vector< int >* _in_pvid_row;     ///< voxel row
    std::vector< int >* _in_pvid_col;     ///< voxel col
    std::vector< int >* _in_pvid_depth;   ///< voxel depth
    std::vector< int >* _in_pinstance_id; ///< instance label
    std::vector< int >* _in_pancestor_id; ///< ancestor label
    std::vector< int >* _in_pparticle_id; ///< particle id label
    std::vector< float >* _in_pq_u;       ///< charge on u-plane
    std::vector< float >* _in_pq_v;       ///< charge on v-plane
    std::vector< float >* _in_pq_y;       ///< charge on y-plane
    std::vector< std::vector<int> >* _in_ptriplet_idx_v; ///< indices of triplets in _triplet_maker that contribute to voxel
    std::vector< int >* _in_psubcluster_id;  ///< labels from dbscan clustering
    std::vector< int >* _in_psubinstance_id; ///< labels from unique (instance,dbscan cluster) pairs

    TRandom3* _rand;

    void _reassignSmallTrackClusters( Prep3DSpatialEmbed::VoxelDataList_t& voxel_v,
                                      const std::vector< larcv::Image2D >& instanceimg_v,
                                      std::map<int,int>& track_instance_count,
                                      const float threshold );
    

    void _generate_subcluster_labels( Prep3DSpatialEmbed::VoxelDataList_t& data,
                                      larlite::storage_manager& ioll,
                                      bool use_only_true_voxels );
    
  private:

    static bool _setup_numpy;
    
  };
  
}
}

#endif
