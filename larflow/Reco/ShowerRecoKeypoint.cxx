#include "ShowerRecoKeypoint.h"

#include <ctime>
#include <fstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "nlohmann/json.hpp"
#include <cilantro/principal_component_analysis.hpp>

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

#include "PCACluster.h"
#include "geofuncs.h"

namespace larflow {
namespace reco {

  void ShowerRecoKeypoint::process( larcv::IOManager& iolc, larlite::storage_manager& ioll ) {

    // we process shower clusters produced by PCA cluster algo
    // steps:
    // (1) find trunk candidates:
    //      - we contour the shower pixels and look for straight segments
    //      - we gather track clusters that are connected to the shower clusters
    // (2) we build a shower hypothesis from the trunks:
    //      - we add points along the pca-axis of the cluster
    //      - does one end of the trunk correspond to the end of the assembly? define as start point
    //      - shower envelope expands from start
    //      - trunk pca and assembly pca are aligned
    // (3) choose the best shower hypothesis that has been formed
    
    // get shower larflow hits (use SplitHitsBySSNet)
    larlite::event_larflow3dhit* shower_lfhit_v
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _ssnet_lfhit_tree_name );

    // make shower clusters
    float maxdist = 2.0;
    int minsize = 20;
    int maxkd = 10;
    std::vector<cluster_t> cluster_v;
    cluster_larflow3dhits( *shower_lfhit_v, cluster_v, maxdist, minsize, maxkd );
    

    std::clock_t begin_process = std::clock();
    LARCV_INFO() << "start" << std::endl;
    LARCV_INFO() << "num larflow hits from [" << _ssnet_lfhit_tree_name << "]: " << shower_lfhit_v->size() << std::endl;
    LARCV_INFO() << "num shower clusters:  " << cluster_v.size() << std::endl;    

    // now for each shower cluster, we find some trunk candidates.
    // can have any number of such candidates per shower cluster
    // we only analyze clusters with a first pc-axis length > 1.0 cm
    std::vector< const cluster_t* > trunk_candidates_v;

    int idx = -1;
    for ( auto& showercluster : cluster_v ) {
      idx++;

      cluster_pca( showercluster );
      
      // metrics to choose shower trunk candidates
      // length
      float len =  showercluster.pca_len;

      // pca eigenvalue [1]/[0] ratio -- to ensure straightness
      float eigenval_ratio = showercluster.pca_eigenvalues[1]/showercluster.pca_eigenvalues[0];
      
      LARCV_DEBUG() << "shower cluster[" << idx << "]"
                    << " pca-len=" << len << " cm,"
                    << " pca-eigenval-ratio=" << eigenval_ratio
                    << std::endl;
      
      if ( len<1.0 ) continue;
      //if ( eigenval_ratio<0.1 ) continue;

      trunk_candidates_v.push_back( &showercluster );
    }

    LARCV_INFO() << "num of trunk candidates: " << trunk_candidates_v.size() << std::endl;

    // GET KEYPOINT DATA
    std::vector< const larlite::larflow3dhit* > keypoint_v;

    larlite::event_larflow3dhit* evout_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"keypoint");
    for ( auto const& kp : *evout_keypoint )
      keypoint_v.push_back( &kp );

    // MAKE TRUNK CANDIDATES FOR EACH SHOWER
    _reconstructClusterTrunks( trunk_candidates_v, keypoint_v );
    

    std::clock_t end_process = std::clock();
    LARCV_INFO() << "[ShowerRecoKeypoint::process] end; elapsed = "
                 << float(end_process-begin_process)/CLOCKS_PER_SEC << " secs"      
                 << std::endl;

    larlite::event_larflowcluster* evout_shower_cluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "showerkp" );
    for ( auto const& pc : trunk_candidates_v ) {
      larlite::larflowcluster lfcluster;
      for (auto const& idx : pc->hitidx_v ) {
        lfcluster.push_back( shower_lfhit_v->at(idx) );
      }
      evout_shower_cluster_v->emplace_back( std::move(lfcluster) );
    }

    larlite::event_pcaxis* evout_shower_pca_v
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "showerkp" );
    int pcidx = 0;
    for ( auto const& pc : trunk_candidates_v ) {
      // we store the cluster's pca if we do not have trunk candidates, otherwise
      if ( _shower_cand_v[pcidx].trunk_candidates_v.size()>0 ) {

        // for now we choose the best one by the smallest eigenval ratio
        float min_ratio = 1e9;
        int plot_idx = -1;
        for (int itrunk=0; itrunk<(int)_shower_cand_v[pcidx].trunk_candidates_v.size(); itrunk++ ) {
          if ( min_ratio>_shower_cand_v[pcidx].trunk_candidates_v[itrunk].pca_eigenval_ratio ) {
            min_ratio = _shower_cand_v[pcidx].trunk_candidates_v[itrunk].pca_eigenval_ratio;
            plot_idx = itrunk;
          }
        }
        larlite::pcaxis::EigenVectors e_v;
        std::vector<double> axis_v(3,0);
        for (int v=0; v<3; v++) axis_v[v] = _shower_cand_v[pcidx].trunk_candidates_v[plot_idx].pcaxis_v[v];
        e_v.push_back( axis_v );
        e_v.push_back( axis_v );
        e_v.push_back( axis_v );
        std::vector<double> startpt(3,0);
        std::vector<double> endpt(3,0);
        for (int v=0; v<3; v++ ) {
          startpt[v] = _shower_cand_v[pcidx].trunk_candidates_v[plot_idx].keypoint->at(v);
          endpt[v] = startpt[v] + 10.0*axis_v[v];
        }
        e_v.push_back( startpt );
        e_v.push_back( endpt );
          
        double eigenval[3] = { 10, 0, 0 };
        double centroid[3] = { (double)_shower_cand_v[pcidx].trunk_candidates_v[plot_idx].center_v[0],
                               (double)_shower_cand_v[pcidx].trunk_candidates_v[plot_idx].center_v[1],
                               (double)_shower_cand_v[pcidx].trunk_candidates_v[plot_idx].center_v[2] };                               
        
        larlite::pcaxis pca( true,
                             _shower_cand_v[pcidx].trunk_candidates_v[plot_idx].npts,
                             eigenval,
                             e_v,
                             centroid,
                             0,
                             pcidx );
        evout_shower_pca_v->emplace_back( std::move(pca) );                             
      }
      else {          
        larlite::pcaxis pca = cluster_make_pcaxis( *pc, pcidx );
        evout_shower_pca_v->emplace_back( std::move(pca) );
      }

      pcidx++;
    }

  }

  /**
   * 
   * match keypoints to shower clusters, use to define the trunk
   *
   * we match keypoints to shower clusters
   * for each keypoint assigned to cluster, define 1,3,5 cm hit cluster around each keypoint
   * going from 5,3,1 cm clusters, accept pca-axis based on eigenvalue ratio
   * 
   * use log likelihood function to pick best key-point trunk
   * output is shower cluster, keypoint, and trunk cluster
   * 
   */
  void ShowerRecoKeypoint::_reconstructClusterTrunks( const std::vector<const cluster_t*>& showercluster_v,
                                                      const std::vector<const larlite::larflow3dhit*>& keypoint_v )
  {

    const float radii[3] = { 5, 3, 1 };
    
    for ( size_t ishower=0; ishower<showercluster_v.size(); ishower++ ) {

      // get shower
      const cluster_t* pshower = showercluster_v[ishower];

      ShowerCandidate_t shower_cand;
      shower_cand.cluster_idx = ishower;
      shower_cand.cluster     = pshower;

      // loop over keypoints, finding shower pixels within 5,3,1
      for ( size_t ikeypoint=0; ikeypoint<keypoint_v.size(); ikeypoint++ ) {

        const larlite::larflow3dhit& keypoint = *(keypoint_v.at(ikeypoint));
        std::vector< std::vector<float> > pcaxis_v(3);
        std::vector< std::vector<float> > pca_center_v(3);        
        std::vector< float > pca_eigenval_ratio(3,0.0);
        std::vector< float > impact_par_v(3,0.0);

        bool inbbox = true;
        for (int v=0; v<3; v++) {
          if ( keypoint[v]<pshower->bbox_v[v][0]-10.0 || keypoint[v]>pshower->bbox_v[v][1]+10.0 )
            inbbox = false;
          if ( !inbbox ) break;
        }

        if ( !inbbox )
          continue;
        
        std::vector< std::vector<Eigen::Vector3f> > trunk_vv(3);
        float mindist = 1e9;
        
        for ( size_t ihit=0; ihit<pshower->points_v.size(); ihit++ ) {

          float dist = 0.;
          for (size_t v=0; v<3; v++)
            dist += ( pshower->points_v[ihit][v]-keypoint[v] )*( pshower->points_v[ihit][v]-keypoint[v] );
          dist = sqrt(dist);

          for (size_t irad=0; irad<3; irad++) {
            if ( dist<radii[irad] ) {
              auto const& pt = pshower->points_v[ihit];
              trunk_vv[irad].push_back( Eigen::Vector3f( pt[0], pt[1], pt[2] ) );
            }
          }

          if ( dist<mindist )
            mindist = dist;
          
        }//end of hit loop

        // get pca for each length, must have >10 points
        // we pick the best trunk by
        //  (1) trunk must have pca-ratio<0.15
        //  (2) rank by smallest impact parameter
        
        // LARCV_DEBUG() << "[ shower cluster[" << ishower << "] keypoint[" << ikeypoint << "] ]" << std::endl;
        // LARCV_DEBUG() << "  gap-dist: " << mindist << " cm" << std::endl;

        int best_trunk = -1;
        float best_trunk_impactpar = 1e9;
        
        for (size_t irad=0; irad<3; irad++) {
          
          //LARCV_DEBUG() << "  radius[" << radii[irad] << "] num points: " << trunk_vv[irad].size() << std::endl;
          
          if ( trunk_vv[irad].size()<10 ) continue;
          
          cilantro::PrincipalComponentAnalysis3f pca( trunk_vv[irad] );
          float eratio = pca.getEigenValues()(1)/pca.getEigenValues()(0);
          std::vector<float> e_v = { pca.getEigenVectors()(0,0),
                                     pca.getEigenVectors()(1,0),
                                     pca.getEigenVectors()(2,0) };
          std::vector<float> center_v = { pca.getDataMean()(0),
                                          pca.getDataMean()(1),
                                          pca.getDataMean()(2) };
            
          pcaxis_v[irad] = e_v;
          pca_center_v[irad] = center_v;
          pca_eigenval_ratio[irad] = eratio;
          std::vector<float> kp = { keypoint[0], keypoint[1], keypoint[2] };
          std::vector<float> pt2(3,0);
          for (int v=0; v<3; v++) pt2[v] = center_v[v] + e_v[v];
          float impact = pointLineDistance( center_v, e_v, kp );
          impact_par_v[irad] = impact;

          // LARCV_DEBUG() << "  radius["<< radii[irad] << " cm]: "
          //               << " pca ratio=" << pca_eigenval_ratio[irad]
          //               << " pca-0=(" << pcaxis_v[irad][0] << "," << pcaxis_v[irad][1] << "," << pcaxis_v[irad][2] << ")"
          //               << " impactpar=" << impact
          //               << std::endl;

          if ( eratio<0.15 && impact<best_trunk_impactpar ) {
            best_trunk = irad;
            best_trunk_impactpar = impact;
          }
          
        }// loop over radius size, calculating radius

        // define the best trunk candidate
        if ( best_trunk!=-1 ) {
          ShowerTrunk_t trunk;
          trunk.idx_keypoint = ikeypoint;          
          trunk.keypoint = keypoint_v.at(ikeypoint);
          trunk.pcaxis_v = pcaxis_v[best_trunk];
          trunk.center_v = pca_center_v[best_trunk];
          trunk.pca_eigenval_ratio = pca_eigenval_ratio[best_trunk];
          trunk.npts = (int)trunk_vv[best_trunk].size();
          trunk.gapdist = mindist;
          trunk.impact_par = impact_par_v[best_trunk];
          shower_cand.trunk_candidates_v.emplace_back( std::move(trunk) );

          LARCV_DEBUG() << "define shower[" << ishower << "] keypoint[" << ikeypoint << "] trunk" << std::endl;
          LARCV_DEBUG() << "  gap-dist: " << trunk.gapdist << " cm" << std::endl;
          LARCV_DEBUG() << "  eigenval ratio: " << trunk.pca_eigenval_ratio << std::endl;
          LARCV_DEBUG() << "  npts: " << trunk.npts << std::endl;
          LARCV_DEBUG() << "  impact-par: " << trunk.impact_par << " cm" << std::endl;

        }
        
      }//end of keypoint

      LARCV_INFO() << "Saving shower candidate with " << shower_cand.trunk_candidates_v.size() << " trunk candidates" << std::endl;

      // we will pick the best trunk candidate later when we expand the shower candidates with nearby clusters
      
      _shower_cand_v.emplace_back( std::move(shower_cand) );
      
    }//end of shower cluster loop
    
  }
  
}
}
