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
    std::vector< const cluster_t* > showerhit_cluster_v;

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

      showerhit_cluster_v.push_back( &showercluster );
    }

    LARCV_INFO() << "num of trunk candidates: " << showerhit_cluster_v.size() << std::endl;

    // GET KEYPOINT DATA
    std::vector< const larlite::larflow3dhit* > keypoint_v;

    larlite::event_larflow3dhit* evout_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"keypoint");
    for ( auto const& kp : *evout_keypoint )
      keypoint_v.push_back( &kp );

    // MAKE TRUNK CANDIDATES FOR EACH SHOWER
    _reconstructClusterTrunks( showerhit_cluster_v, keypoint_v );

    // BUILD SHOWERS FROM CLUSTERS + TRUNK CANDIDATES
    _buildShowers( showerhit_cluster_v );

    std::clock_t end_process = std::clock();
    LARCV_INFO() << "[ShowerRecoKeypoint::process] end; elapsed = "
                 << float(end_process-begin_process)/CLOCKS_PER_SEC << " secs"      
                 << std::endl;

    larlite::event_larflowcluster* evout_shower_cluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "showerkp" );

    larlite::event_pcaxis* evout_shower_pca_v
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "showerkp" );

    for ( size_t ireco=0; ireco<_recod_shower_v.size(); ireco++ ) {

      auto const& recoshower = _recod_shower_v[ireco];
      
      // make larflow3dhit cluster
      larlite::larflowcluster lfcluster;      
      for ( auto const& idx : recoshower.hitidx_v ) {
        lfcluster.push_back( shower_lfhit_v->at(idx) );
      }
      evout_shower_cluster_v->emplace_back( std::move(lfcluster) );
            
      // we store the cluster's pca if we do not have trunk candidates, otherwise
      larlite::pcaxis::EigenVectors e_v;
      std::vector<double> axis_v(3,0);
      for (int v=0; v<3; v++) axis_v[v] = recoshower.trunk.pcaxis_v[v];
      e_v.push_back( axis_v );
      e_v.push_back( axis_v );
      e_v.push_back( axis_v );
      std::vector<double> startpt(3,0);
      std::vector<double> endpt(3,0);
      for (int v=0; v<3; v++ ) {
        startpt[v] = recoshower.trunk.keypoint->at(v);
        endpt[v] = startpt[v] + 10.0*axis_v[v];
      }
      e_v.push_back( startpt );
      e_v.push_back( endpt );
          
      double eigenval[3] = { 10, 0, 0 };
      double centroid[3] = { (double)recoshower.trunk.center_v[0],
                             (double)recoshower.trunk.center_v[1],
                             (double)recoshower.trunk.center_v[2] };                               
        
      larlite::pcaxis pca( true,
                           recoshower.trunk.npts,
                           eigenval,
                           e_v,
                           centroid,
                           0,
                           (int)ireco );
      evout_shower_pca_v->emplace_back( std::move(pca) );                             
    }//end of reco'd shower loop

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
   * note: might want to move keypoint based on pca end near keypoint.
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

          // // closest cluster end
          // int iend = 0;
          // float enddist[2][3] = { 0 };
          // for (int v=0; v<3; v++) {
          //   enddist[0][v] = ( kp[v]-

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

          // we make sure the pca-axis is pointing away from keypoint
          // and towards the centroid of the trunk cluster
          float coscenter = 0.;
          for (size_t v=0; v<3; v++) {
            coscenter += trunk.pcaxis_v[v]*( trunk.center_v[v]-trunk.keypoint->at(v) );
          }
          if ( coscenter<0 ) {
            // flip the axis dir
            for (size_t v=0; v<3; v++)              
              trunk.pcaxis_v[v] *= -1.0;
          }
          
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

  /**
   * for each shower cluster with trunk candidate, 
   * we absorb points clusters that are within some radius of the trunk axis
   *
   */
  void ShowerRecoKeypoint::_buildShowers( const std::vector< const cluster_t*>& showerhit_cluster_v )
  {

    int nbad_cands = 0;
    for ( auto const& shower_cand : _shower_cand_v ) {
      if ( shower_cand.trunk_candidates_v.size()==0 ) continue;
      Shower_t shower  = _buildShowerCandidate( shower_cand, showerhit_cluster_v );
      if (shower.points_v.size()>0 ) {
        _recod_shower_v.emplace_back( std::move(shower) );
      }
      else
        nbad_cands++;
    }

    LARCV_INFO() << "Number of reco'd shower candidates. "
      << "ngood=" << _recod_shower_v.size()
      << " nbad=" << nbad_cands
      << std::endl;
    
  }

  /**
   * we build a shower from a shower candidate, by 
   * building out the shower using each trunk candidate
   * and selecting the best shower somehow
   *
   */
  ShowerRecoKeypoint::Shower_t
  ShowerRecoKeypoint::_buildShowerCandidate( const ShowerCandidate_t& shower_cand,
                                             const std::vector< const cluster_t*>& showerhit_cluster_v )
  {

    std::vector< std::set<int> > trunk_cluster_idxset_v;

    LARCV_DEBUG() << "Begin" << std::endl;
    
    // for each trunk candidate in the shower candidate, we let it absorb hits
    for (auto const& trunk : shower_cand.trunk_candidates_v ) {
      std::set<int> clusters = _buildoutShowerTrunkCandidate( trunk, showerhit_cluster_v );      
      if ( clusters.size()>0 )
        trunk_cluster_idxset_v.push_back(clusters);
    }

    if ( trunk_cluster_idxset_v.size()==0 ) {
      LARCV_DEBUG() << "shower candidate returns empty shower" << std::endl;
      return Shower_t(); //empty
    }

    if ( trunk_cluster_idxset_v.size()==1 ) {
      // single trunk candidate case
      // build the shower
      LARCV_DEBUG() << "shower candidate with one trunk returns non-empty shower" << std::endl;
      return _fillShowerObject( shower_cand, trunk_cluster_idxset_v[0], 0,
                                showerhit_cluster_v );
    }

    // now we have to evaluate which shower trunk is best
    std::set<int> union_cluster_idx;
    for ( auto& idx_set : trunk_cluster_idxset_v ) {
      for (auto& idx : idx_set ) {
        union_cluster_idx.insert(idx);
      }
    }
    LARCV_DEBUG() << "union cluster list size=" << union_cluster_idx.size() << std::endl;

    // return least squares value for each shower over the entirety of the cluster set
    // lowest value is considered the "best" cluster.
    
    LARCV_DEBUG() << "returns best-fit trunk candidate" << std::endl;    
    return Shower_t(); // empty for now
  }

  /**
   * build out the shower using the trunk candidate
   *
   * we absorb shower hits within some radius of the axis.
   * we track the shower hit indexes as well.
   *
   */
  std::set<int> ShowerRecoKeypoint::_buildoutShowerTrunkCandidate( const ShowerTrunk_t& trunk_cand,
                                                                   const std::vector< const cluster_t*>& showerhit_cluster_v )
  {

    const float ar_molier_rad_cm = 9.04;
    
    std::set<int> clusteridx_v;

    float pcalen = 0.;
    for (int v=0; v<3; v++)
      pcalen += trunk_cand.pcaxis_v[v]*trunk_cand.pcaxis_v[v];
    pcalen = sqrt(pcalen);

    for ( size_t idx=0; idx<showerhit_cluster_v.size(); idx++) {

      auto const& cluster = *showerhit_cluster_v[idx];
      
      // make two points along axis
      std::vector<float> alongpca(3,0);
      for (int v=0; v<3; v++)
        alongpca[v] = trunk_cand.center_v[v] + 10.0*trunk_cand.pcaxis_v[v];
      
      // first test bounding box before going in and checking this      
      bool testcluster = false;

      // make bbox 3d points
      // 2^3=8 bounding box vertex points
      std::vector<float> bbox_pt(3,0);
      // make permutation vector
      int state[3] = { 0, 0, 0 };
      
      for (int i=0; i<8; i++) {

        // generate bounding box pt
        //LARCV_DEBUG() << "permutation-vector[" << i << ": (" << state[0] << "," << state[1] << "," << state[2] << ")" << std::endl;        
        for (int v=0; v<3; v++)
          bbox_pt[v] = cluster.bbox_v[v][state[v]];

        // test
        float dist = pointLineDistance( trunk_cand.center_v, alongpca, bbox_pt );
        float proj = 0.;
        for (int v=0; v<3; v++ ) {
          proj += (bbox_pt[v]-trunk_cand.keypoint->at(v))*trunk_cand.pcaxis_v[v];
        }
        proj /= pcalen;

        
        if ( dist<2.0*ar_molier_rad_cm && fabs(proj)<100.0 ) {
          testcluster = true;
          break;
        }
        
        // increment to next permutation
        for (int v=0; v<3; v++) {
          if ( state[v]!=1 ) {
            state[v]++;
            break;
          }
          else {
            state[v]=0;
          }
        }

      }//end of loop over bbox pt permutations

      // nothing close
      if ( !testcluster ) {
        LARCV_DEBUG() << "No bounding box points are close" << std::endl;
        continue;
      }

      // if bbox pt close, we test to see if we absorb cluster
      // defined as 10% of points inside molier radius (this needs tuning)
      int npts_inside = 0;
      int npts_outside = 0;
      int npts_threshold = int( 0.1*cluster.points_v.size() );
      if ( npts_threshold==0 )
        npts_threshold = 1;
      int npts_reject = (int)cluster.points_v.size()-npts_threshold;
      bool accept = false;
      
      for ( auto const& hit : cluster.points_v ) {
        float dist = pointLineDistance( trunk_cand.center_v, alongpca, hit );
        float proj = 0.;
        for (int v=0; v<3;v++)
          proj += ( hit[v]-trunk_cand.keypoint->at(v) )*trunk_cand.pcaxis_v[v];
        proj /= pcalen;
        
        if ( (proj>0.0 && dist<2.0*ar_molier_rad_cm && proj<100.0 )
             || (proj<=0.0 && proj>-3.0 && dist<3.0 ) ) {
          npts_inside++;
        }
        else {
          npts_outside++;
        }
        
        if ( npts_inside>=npts_threshold ) {
          accept = true;
          break;
        }
        else if ( npts_outside>npts_reject ) {
          break;
        }
      }

      LARCV_DEBUG() << "Eval cluster: npt_inside=" << npts_inside
                    << " npts_outside=" << npts_outside
                    << " npts_threshold=" << npts_threshold
                    << std::endl;
      
      if ( accept ) {
        clusteridx_v.insert( idx );
      }
      
    }
    
    return clusteridx_v;
  }

  // std::set<int> ShowerRecoKeypoint::_extendShower( const ShowerCandidate_t& shower_cand,
  //                                                  const int trunk_idx,
  //                                                  const std::vector< const cluster_t* >& showerhit_cluster_v,
  //                                                  const float extend_range_min_cm, const float extend_range_max_cm )
  // {
    
  // }

  ShowerRecoKeypoint::Shower_t
  ShowerRecoKeypoint::_fillShowerObject( const ShowerCandidate_t& shower_cand,
                                         const std::set<int>& cluster_idx_set,
                                         const int trunk_idx,
                                         const std::vector< const cluster_t* >& showerhit_cluster_v )
  {

    Shower_t recoshower;
    recoshower.trunk = shower_cand.trunk_candidates_v[trunk_idx];

    for ( auto const& idx : cluster_idx_set ) {
      auto const& cluster = *showerhit_cluster_v[idx];

      for (size_t ihit=0; ihit<cluster.points_v.size(); ihit++ ) {
        recoshower.points_v.push_back( cluster.points_v[ihit] );
        recoshower.hitidx_v.push_back( cluster.hitidx_v[ihit] );
      }
      
    }
    return recoshower;
  }
  
}
}
