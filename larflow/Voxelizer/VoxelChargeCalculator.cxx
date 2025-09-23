#include "VoxelChargeCalculator.h"

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

namespace larflow {
namespace voxelizer {

    VoxelChargeCalculator::VoxelChargeCalculator()
    : larcv::larcv_base("VoxelChargeCalculator"),
    _sce(nullptr)
    {
        _sce = new larutil::SpaceChargeMicroBooNE( larutil::SpaceChargeMicroBooNE::kMCC9_Backward );
        configure_voxelizer( 5.0 ); // set default to 5 cm voxel
    }

    VoxelChargeCalculator::~VoxelChargeCalculator()
    {
        if ( _sce )
            delete _sce;
        _sce = nullptr;
    }

    void VoxelChargeCalculator::configure_voxelizer( float voxel_len_cm ) 
    {

        _voxelizer.set_voxel_size_cm( voxel_len_cm );

    }

    void VoxelChargeCalculator::clear() 
    {

        _cluster_info_v.clear();
        _voxel_charges_v.clear();
        _images_v.clear();

    }

    void VoxelChargeCalculator::add_larflow_hit_cluster( const larlite::larflowcluster& cluster_hits )
    {

        ClusterInfo_t trackinfo;

        size_t nhits = cluster_hits.size();

        for ( size_t ihit=0; ihit<nhits; ihit++ ) {

            auto const& lfhit = cluster_hits.at(ihit);

            std::vector<float> pos      = { lfhit[0], lfhit[1], lfhit[2] };
            std::vector<float> hitcoord = { (float)lfhit.tick, (float)lfhit.targetwire[0], (float)lfhit.targetwire[1], (float)lfhit.targetwire[2] };

            trackinfo.hitpos_v.push_back( pos );
            trackinfo.hitcoord_v.push_back( hitcoord );

        }

        _cluster_info_v.emplace_back( std::move(trackinfo) );

    }

    void VoxelChargeCalculator::add_cluster_hitinfo( const std::vector< std::vector<float> >& hitpos_v ,
                                                     const std::vector< std::vector<float> >& imgcoord_v ) 
    {

        ClusterInfo_t trackinfo;

        size_t nhits = hitpos_v.size();
        for ( size_t ihit=0; ihit<nhits; ihit++ ) {
            auto const& hit = hitpos_v.at(ihit);
            auto const& imgcoord = imgcoord_v.at(ihit);
            std::vector<float> xpos = { hit[0], hit[1], hit[2] };
            std::vector<float> ximgcoord = { imgcoord[0], imgcoord[1], imgcoord[2], imgcoord[3] };
            trackinfo.hitpos_v.push_back( xpos );
            trackinfo.hitcoord_v.push_back( ximgcoord );
        }

        _cluster_info_v.emplace_back( std::move(trackinfo) );
    }

    void VoxelChargeCalculator::set_images( const std::vector< larcv::Image2D >& img_v ) {
        for (size_t iimg=0; iimg<img_v.size(); iimg++) {
            const larcv::Image2D& img = img_v.at(iimg);
            _images_v.push_back( &img );
        }
    }

    void VoxelChargeCalculator::calculate_voxel_charge( float t0 )
    {

        // we loop through the spacepoints associated with the track and create 
        // a unique list of voxels the spacepoints occupy
        // then we get the voxel centers (and position mean)
        // and get the charge sum of the voxels

        VoxelChargeInfo_t voxelinfo;
        voxelinfo.t0_assumed = t0;

        // clear the output containers
        voxelinfo.voxel_planecharge_vv.clear();
        voxelinfo.voxel_indices_vv.clear();
        voxelinfo.voxel_avepos_vv.clear();
        voxelinfo.voxel_centers_vv.clear();

        struct clusterhit_t {
            int clusterindex;
            int hitindex;
        };
        std::map< vindex_t, std::vector<clusterhit_t> > voxelindex_to_hitindex;
        std::map< vindex_t, std::vector<float> >        voxelindex_to_avepos;

        const float x_t0_offset = t0*larutil::LArProperties::GetME()->DriftVelocity();

        int num_outside_voxels_or_tpc = 0;

        for (int icluster=0; (int)_cluster_info_v.size(); icluster++ ) {

            auto const& cluster = _cluster_info_v.at(icluster);

            for (size_t hitidx=0; hitidx<cluster.hitpos_v.size(); hitidx++) {

                std::vector<float> hit = cluster.hitpos_v.at(hitidx);
                hit[0] -= x_t0_offset; //already removed x offset

                // correct for the space charge effect
                bool applied_sce = false;
                std::vector<double> hit_sce 
                    = _sce->ApplySpaceChargeEffect( hit[0], hit[1], hit[2], applied_sce);
                std::vector<float> fhit_sce = { (float)hit_sce[0], (float)hit_sce[1], (float)hit_sce[2] };

                // get the voxel index
                vindex_t voxelindex;
                try {
                    auto ivoxel_v = _voxelizer.get_voxel_indices( fhit_sce );
                    for (int i=0; i<3; i++)
                        voxelindex[i] = ivoxel_v[i];
                }
                catch (...) {
                    num_outside_voxels_or_tpc++;
                    continue;
                }

                // also track if point inside voxel grid, but outside the tpc
                if ( hit[0]<0.0 || hit[0]>256.0 )
                    num_outside_voxels_or_tpc++;

                // find the voxel in our maps
                auto it_voxel_hitlist = voxelindex_to_hitindex.find( voxelindex );
                if ( it_voxel_hitlist==voxelindex_to_hitindex.end() ) {
                    // voxel not yet registered, create containers
                    voxelindex_to_hitindex[voxelindex] = std::vector<clusterhit_t>();
                    voxelindex_to_avepos[voxelindex]   = std::vector<float>(3,0);
                    it_voxel_hitlist = voxelindex_to_hitindex.find( voxelindex );
                }

                // add hit to voxel
                clusterhit_t clusterhit;
                clusterhit.clusterindex = icluster;
                clusterhit.hitindex     = hitidx;
                it_voxel_hitlist->second.push_back( clusterhit );
                // add position to avepos (will divide by number of hits later)
                voxelindex_to_avepos[voxelindex][0] += fhit_sce[0];
                voxelindex_to_avepos[voxelindex][1] += fhit_sce[1];
                voxelindex_to_avepos[voxelindex][2] += fhit_sce[2];
            }//end of loop over cluster hits
        }//end of loop over clusters
        voxelinfo.num_outside_tpc = num_outside_voxels_or_tpc;

        LARCV_INFO() << "Made Occupied Voxel list and associated 3D points to the voxels" << std::endl;
        LARCV_INFO() << "  nvoxels: " << voxelindex_to_hitindex.size() << std::endl;
        LARCV_INFO() << "  hits outside voxelized volume: " << num_outside_voxels_or_tpc << std::endl;

        // finished hit-to-voxel assignment
        // now need to sum up position and charge values for each voxel.
        // this struct represents a pixel and is responsible for:
        //   1. storing the (row,col) position of the pixel and pixel value
        //   2. count the number of hits that project into a pixel. 
        //        will divide pixel value evenly across the hits

        // For each plane, calculate the charge values assigned to the voxel
        int nplanes = _images_v.size();
        std::map<vindex_t,std::vector<float> > voxelindex_to_chargevalues;

        for (int plane=0; plane<nplanes; plane++) {

            auto const& meta = _images_v.at(plane)->meta();
            auto const* img  = _images_v.at(plane);

            std::vector< Pixel_t > pixel_v; // store the pixels associated with the hits of this track
            std::map< std::pair<int,int>, int > pix_to_index; // key (row,col) -> value is index in pixel_v
            int pixcount = 0;
            std::map< vindex_t, std::vector<int> > voxelindex_to_pixindexlist; // voxel index to vector of indices to pixel_v

            // loop over the occupied voxels once in order to get which pixels we project into
            // we also count the number of times we project down
            for (auto it_voxel=voxelindex_to_hitindex.begin(); it_voxel!=voxelindex_to_hitindex.end();it_voxel++ ) {

                // loop over all hits assigned to this voxel
                auto const& hitindex_list = it_voxel->second;

                for (auto& clusterhitidx : hitindex_list ) {

                    int clusteridx = clusterhitidx.clusterindex;
                    int hitidx     = clusterhitidx.hitindex;
                    auto const& cluster = _cluster_info_v.at(clusteridx);
                    auto const& imgpos  = cluster.hitcoord_v.at(hitidx);
                    int row = meta.row( imgpos[0] ); // tick to row
                    int col = imgpos[plane+1];
                    std::pair<int,int> pix(row,col);

                    auto it_pixel = pix_to_index.find( pix );
                    if ( it_pixel==pix_to_index.end() ) {
                        // not in map. make pixel.
                        Pixel_t pixel;
                        pixel.row = row;
                        pixel.col = col;
                        pixel.index = it_voxel->first;
                        pixel.pixval = img->pixel(row,col);
                        pixel.num_hits = 0;
                        pix_to_index[pix] = pixcount;
                        pixel_v.emplace_back( std::move(pixel) );
                        pixcount++;
                    }

                    // increment counter for number of hits projecting into this pixel
                    int pixindex = pix_to_index[pix];
                    pixel_v.at(pixindex).num_hits++;

                    // provide a list of pixels whose hits fall within a voxel
                    auto it_vox2pix = voxelindex_to_pixindexlist.find( it_voxel->first );
                    if ( it_vox2pix==voxelindex_to_pixindexlist.end() ) {
                        voxelindex_to_pixindexlist[it_voxel->first] = std::vector<int>();
                    }
                    voxelindex_to_pixindexlist[it_voxel->first].push_back( pixindex );

                }//end of loop over hit indices assigned to voxel

            }// end of loop over voxels

            // loop over the voxels again, using the assignments to sum the pixel values for each voxel
            for (auto it_voxel=voxelindex_to_hitindex.begin(); it_voxel!=voxelindex_to_hitindex.end();it_voxel++ ) {

                auto it_voxel_charge = voxelindex_to_chargevalues.find( it_voxel->first );
                if ( it_voxel_charge==voxelindex_to_chargevalues.end() ) {
                    voxelindex_to_chargevalues[it_voxel->first] = std::vector<float>(3,0.0);
                }

                auto it_vox2pix = voxelindex_to_pixindexlist.find( it_voxel->first );

                // sum charge to set value for voxel
                float charge_sum = 0.0;

                for ( auto& pixindex : it_vox2pix->second ) {
                    auto const& pixdata = pixel_v.at(pixindex);
                    charge_sum += pixdata.pixval/float(pixdata.num_hits);
                }

                // assign to voxel
                voxelindex_to_chargevalues[it_voxel->first].at(plane) = charge_sum;

            }//end of loop over occupied voxels

        }//end of loop over planes

        // now loop over voxels one last time and collect the voxel charge and position data.
        // our goal is to fill the following data members
        //   voxelinfo.voxel_planecharge_vv
        //   voxelinfo.voxel_indices_vv
        //   voxelinfo.voxel_avepos_vv
        //   voxelinfo.voxel_centers_vv

        auto const& origin = _voxelizer.get_origin();
        auto const& dimlen = _voxelizer.get_dim_len();

        for (auto it_voxel=voxelindex_to_hitindex.begin(); it_voxel!=voxelindex_to_hitindex.end();it_voxel++ ) {

            std::vector<float> ave_pos(3,0.0);

            ave_pos = voxelindex_to_avepos[it_voxel->first];
            for (int i=0; i<3; i++) {
                ave_pos[i] /= float(it_voxel->second.size());
            }

            voxelinfo.voxel_avepos_vv.push_back( ave_pos );
            voxelinfo.voxel_planecharge_vv.push_back( voxelindex_to_chargevalues[it_voxel->first] );

            std::vector<int> voxel_axis_indices_v = _voxelizer.get_voxel_indices( ave_pos );
            std::vector<int>   voxel_axis_indices(3,0);
            std::vector<float> centerpos(3,0.0);
            for (int i=0; i<3; i++) {
                voxel_axis_indices[i] = voxel_axis_indices_v[i];
                centerpos[i] = (float(voxel_axis_indices[i])+0.5)*dimlen[i] + origin[i];
            }
            voxelinfo.voxel_centers_vv.push_back( centerpos );
            voxelinfo.voxel_indices_vv.push_back( voxel_axis_indices );
        }

        _voxel_charges_v.emplace_back( std::move(voxelinfo) );
       
    }

    const VoxelChargeCalculator::VoxelChargeInfo_t& VoxelChargeCalculator::get_voxel_charge_info() 
    {

        if ( _voxel_charges_v.size()==0 ) {
            throw std::runtime_error("VoxelChargeCalculator::get_voxel_charge_info - Results have not been created.");
        }

        return _voxel_charges_v.at(0);
    }

}
}