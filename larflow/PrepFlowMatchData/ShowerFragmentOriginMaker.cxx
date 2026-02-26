#include "ShowerFragmentOriginMaker.h"

#include "larflow/RecoUtils/cluster_functions.h"

namespace larflow {
namespace prep {

  /**
  * @brief build shower fragments and define keypoints using clustered true spacepoints
  * 
  * @param[in] keypoints  provides shower start and origin at the particle-level
  * @param[in] pixel3d    provides labeled spacepoints with track ID and pdg code
  * @return    void       fills class member _fragment_data
  */
  void ShowerFragmentOriginMaker::build_shower_fragments( 
      const std::vector<MCKeypoint>& keypoints, 
      ublarcvapp::mctools::MCParticleGraph& mcpg,
      larflow::prep::EventTriplets_t& pixel3d,
      float edep_cluster_threshold,
      int edep_cluster_size_threshold,
      float edep_point_threshold )
  {

    _fragment_data.clear();

    // loop through mcparticlegraph particles, look for showers
    std::vector<long> shower_trackids;
    std::vector<int>  shower_pids;
    std::vector< std::vector<float> > shower_momenta_dir;
    std::vector< std::vector<float> > shower_start_v;
    std::vector< std::vector<float> > shower_origin_v;

    for ( auto& node : mcpg.node_v ) {
      if ( abs(node.pid)==11 || node.pid==22 ) {
        
        if (node.first_edep_pos[0]==0
            && node.first_edep_pos[1]==0
            && node.first_edep_pos[2]==0){
          continue;
        }

        float pnorm = 0.;
        for (int i=1; i<4; i++) {
          pnorm += node.mom4[i]*node.mom4[i];
        }
        pnorm = sqrt(pnorm);
        if ( pnorm>0 ) {
          std::vector<float> showerdir(3,0);
          for (int i=0; i<3; i++) {
            showerdir[i] = node.mom4[1+i]/pnorm;

          }

          // only save shower info if we get a momentum
          shower_trackids.push_back( node.tid );
          shower_momenta_dir.push_back( showerdir );
          shower_start_v.push_back( node.first_edep_pos );
          shower_origin_v.push_back( node.start );
          shower_pids.push_back( node.pid );
        }
      }
    }

    // loop through the showers found
    for (size_t ishower=0; ishower<shower_trackids.size(); ishower++) {
      long showerid = shower_trackids.at(ishower);

      // do we have a keypoint for this trackid
      long kpd_index = -1;
      for (size_t ikp=0; ikp<keypoints.size(); ikp++ ) {
        if ( keypoints.at(ikp).trackid==showerid ) {
          kpd_index = (long)ikp;
          break;
        }
      }

      // we get the originpt (i.e. creation pt) and start pt (first visible ionization pt)
      // if we have a mckeypoint object for this shower, we retrieve these
      std::vector<float> shower_origin(3,0);
      std::vector<float> shower_start(3,0);
      std::vector<float> showerdir = shower_momenta_dir.at(ishower);
      if ( kpd_index!=-1 ) {
        auto const& kp = keypoints.at(kpd_index);
        shower_origin = kp.startpt_appear;
        shower_start  = kp.keypt_appear;
      }
      else {
        // when does this happen?
        shower_origin = shower_origin_v.at(ishower);
        shower_start  = shower_start_v.at(ishower);
      }

      // gather up spacepoints matching the trackid
      std::vector< std::vector<float> > points_v;
      std::vector< std::vector<float> > edep_vv;
      std::vector< std::vector<float> > pixval_vv;
      std::vector< long > index_v;
      for ( auto const& pix3d : pixel3d._triplets_v ) {
        if (pix3d.hasmatch==0)
          continue;
        auto it_id = pix3d.trackids.find( showerid );
        if ( it_id!=pix3d.trackids.end() ) {
          std::vector<float> pos(3,0);
          std::vector<float> edep_v(3,0);
          std::vector<float> pixval_v(3,0);
          int nedep_above_threshold = 0;
          for (int i=0; i<3; i++) {
            pos[i]    = pix3d.pos_reco[i];
            edep_v[i] = pix3d.edep[i];
            pixval_v[i] = pix3d.pixval[i];
            if ( edep_v[i]>edep_point_threshold )
              nedep_above_threshold++;
          }
          if ( nedep_above_threshold>0 ) {
            points_v.push_back( pos );
            edep_vv.push_back( edep_v );
            pixval_vv.push_back( pixval_v );
            index_v.push_back( pix3d.index );
          }
        }
      }

      // now we cluster these points using dbscan
      float maxdist = 3.0;
      float minsize = 4;
      int maxkd = 10;
      std::vector< larflow::recoutils::cluster_t > cluster_v;
      larflow::recoutils::cluster_sdbscan_spacepoints( points_v, cluster_v, maxdist, minsize, maxkd);
      int nclusters = cluster_v.size(); // skip the last cluster which are noise points

      LARCV_INFO() << " shower[" << showerid << "] "
        << " num points=" << points_v.size() 
        << " num clusters=" << nclusters
        << std::endl;
      LARCV_INFO() << "  origin=(" << shower_origin[0] << "," << shower_origin[1] << "," << shower_origin[2] << ")" << std::endl;
      LARCV_INFO() << "  start=(" << shower_start[0] << "," << shower_start[1] << "," << shower_start[2] << ")" << std::endl;
      
      // for each cluster, define the "start" as the closest point to the origin
      // we also enforce a minimum cluster size
      for ( int icluster=0; icluster<nclusters; icluster++ ){
        std::vector<float> edep_planesum(3,0.0);
        auto const& cluster = cluster_v.at(icluster);

        for (int ihit=0; ihit<(int)cluster.hitidx_v.size(); ihit++) {
          auto hitidx = cluster.hitidx_v.at(ihit);
          auto const& hitedep = edep_vv.at(hitidx);
          for (int i=0; i<3; i++) {
            edep_planesum[i] += hitedep[i];
          }
        }
        int nabove_threshold_planes = 0;
        if ( cluster.hitidx_v.size()>0 ) {
          for (int i=0; i<3; i++) {
            if ( edep_planesum[i]>edep_cluster_threshold) {
              nabove_threshold_planes++;
            }
          }
        }

        LARCV_INFO() << "  [shower id=" << showerid << "] "
          << "   cluster edep: " 
          <<  edep_planesum[0] << ", "
          <<  edep_planesum[1] << ", "
          <<  edep_planesum[2] << " MeV"
          << " nabove=" << nabove_threshold_planes
          << " numhits=" << cluster.hitidx_v.size()
          << std::endl;

        if ( cluster.hitidx_v.size()<edep_cluster_size_threshold )
          continue;

        if ( nabove_threshold_planes>=2 ) {
          // qualifying cluster, get most upstream position
          std::vector<float> most_upstream_pt(3,0);
          std::vector<float> most_upstream_edep(3,0);
          float min_s = 1e9;
          bool found_qualifying_pt = false;
          std::vector<float> shower_forward(3,0);
          for (int i=0; i<3; i++)
            shower_forward[i] = shower_origin[i] + 10.0*showerdir[i];

          for ( auto& testpt : cluster.points_v ) {
            float s = larflow::recoutils::pointRayProjection3f( shower_origin, showerdir, testpt );
            float s_origin = larflow::recoutils::pointRayProjection3f( shower_origin, showerdir, testpt );
            float r_origin = larflow::recoutils::pointLineDistance3f( shower_origin, shower_forward, testpt );
            if ( s < min_s && s_origin>-3.0 && r_origin/s_origin<0.5 ) {
              min_s = s;
              most_upstream_pt = testpt;
              found_qualifying_pt = true;
              most_upstream_edep = edep_planesum;
            }
          }


          LARCV_INFO() << "  [shower id=" << showerid << "] "
                       << "   most upstream pt: " 
                       <<  most_upstream_pt[0] << ", "
                       <<  most_upstream_pt[1] << ", "
                       <<  most_upstream_pt[2] << " MeV"
                       << std::endl;

          // save shower fragment info
          ShowerFragmentOrigin::shcluster_t fragment_pts_v;
          ShowerFragmentOrigin::shcluster_t fragment_pixval_v;
          fragment_pts_v.reserve( cluster.points_v.size() );
          fragment_pixval_v.reserve( cluster.points_v.size() );
          for (size_t ipt=0; ipt<cluster.points_v.size(); ipt++) {
            fragment_pts_v.push_back(cluster.points_v.at(ipt) );
            fragment_pixval_v.push_back( pixval_vv.at( cluster.hitidx_v.at(ipt) ) );
          }
          _fragment_data.shower_fragments_v.emplace_back( std::move(fragment_pts_v) );
          _fragment_data.shower_edep_v.emplace_back( std::move(fragment_pixval_v) );
          _fragment_data.shower_trackid_v.push_back( showerid );
          _fragment_data.shower_pid_v.push_back( shower_pids.at(ishower) );
          //_fragment_data.shower_istrunk_v; // TODO: determine this by which start pt is closest to original shower_start
          //_fragment_data.shower_type_v; // TODO: need logic for this
          _fragment_data.shower_startpt_v.push_back( most_upstream_pt );
          _fragment_data.shower_originpt_v.push_back( shower_origin ); /// TODO: requires some logic for when to choose start or origin

        }

      }

    }//end of loop over shower ids

    LARCV_NORMAL() << "Created " << _fragment_data.shower_fragments_v.size() << " Shower Fragments with start/origin" << std::endl;

  }


}
}