#ifndef __LARFLOW_VOXELIZER_VOXELCHARGECALCULATOR_H__
#define __LARFLOW_VOXELIZER_VOXELCHARGECALCULATOR_H__

#include <vector>
#include <array>

#include "larcv/core/Base/larcv_base.h"
#include "larlite/DataFormat/larflowcluster.h"

#include "VoxelizeTriplets.h"


// forward declarations
namespace larutil {
    class SpaceChargeMicroBooNE;
}

namespace larflow {
namespace voxelizer {

class VoxelChargeCalculator : public larcv::larcv_base {

public:

    VoxelChargeCalculator();
    ~VoxelChargeCalculator();

    void configure_voxelizer( float voxel_len_cm );

    void clear();

    void add_larflow_hit_cluster( const larlite::larflowcluster& cluster_hits );

    void add_cluster_hitinfo( const std::vector< std::vector<float> >& hitpos_v ,
                              const std::vector< std::vector<float> >& imgcoord_v );

    void set_images( const std::vector< larcv::Image2D >& img_v );

    void calculate_voxel_charge( float t0=0.0 );


    struct ClusterInfo_t {
        std::vector< std::array<float,3> > hitpos_v;   // (x,y,z) position in the detector
        std::vector< std::array<float,4> > hitcoord_v; // (tick,u,v,y) projected image position
    };


    struct VoxelChargeInfo_t {
        float t0_assumed;
        int num_outside_tpc;
        std::vector< std::array<int,3> >   voxel_indices_vv;
        std::vector< std::array<float,3> > voxel_centers_vv;
        std::vector< std::array<float,3> > voxel_avepos_vv;
        std::vector< std::vector<float> >  voxel_planecharge_vv;
        VoxelChargeInfo_t()
        : t0_assumed(0.0),
        num_outside_tpc(0) 
        {};
    };

protected:

    larutil::SpaceChargeMicroBooNE*      _sce;             ///< Space Charge Utility: For correcting the space charge effect
    larflow::voxelizer::VoxelizeTriplets _voxelizer;       ///< helps us assign 3d position (x,y,z) to a voxel grid
    std::vector< ClusterInfo_t >         _cluster_info_v;  ///< stores 3d positions and pixel positions for particle clusters
    std::vector< larcv::Image2D* >       _images_v;        ///< pointers to wire plane images we extract charge from
    std::vector< VoxelChargeInfo_t>      _voxel_charges_v; ///< container for results

    // struct to keep track of how many hits project down into a wireplane pixel
    typedef std::array<int,3> vindex_t;
    struct Pixel_t {
        int row;
        int col;
        float pixval;
        int num_hits;
        vindex_t index;
        Pixel_t()
        : row(-1),col(-1),pixval(0.0),num_hits(0)
        {};
    };



};


}
}

#endif