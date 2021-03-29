#include "ShowerRecoKeypoint.h"

#include <ctime>
#include <fstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "nlohmann/json.hpp"
#include <cilantro/principal_component_analysis.hpp>

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief process shower clusters produced by PCA cluster algo
   * 
   * Steps:
   * @verbatim embed:rst:leading-asterisk
   *  * form shower (sub)clusters using dbscan
   *  * identify which clusters are potentially the shower trunk using output keypoint network
   *  * build showers from these trunk clusters by associating the subclusters to the trunk clusters
   *  * for clusters assigned to two shower candidates, resolve conflicts
   *  * make reco shower object
   * @endverbatim
   *
   * Expected inputs:
   * @verbatim embed:rst:leading-asterisk
   *   * shower-labeled larflow3dhit. The class larflow::reco::SplitHitsBySSNet produces these.
   *   * `keypoint`: container of larflow3dhit representing reconstructed keypoints. These are made by larflow::reco::KeypointReco. 
   *                 uses only the shower-labeled keypoints.
   * @endverbatim
   *
   * Outputs:
   * @verbatim embed:rst:leading-asterisk
   *   * `showergoodhit`: collection of larflowcluster holding reconstruced shower clusters
   *   * `showergoodhit`: collection of pcaxis holding principle component info for clusters
   *   * `showerkp`: larflowcluster representing showers built using keypoint-identified trunk clusters
   *   * `showerkp`: pcaxis containing principle component info for the `showerkp` clusters
   *   * `showerkp`: larflow3dhit representing the shower keypoints used
   * @endverbatim
   *
   *  To do: make assignment of shower (sub)clusters to a trunk cluster via
   *  a shower likelihood of some kind, instead of just distance and proximinty to trunk direction.
   *
   * @param[in] iolcv LArCV IOManager, nothing retrieved from here.
   * @param[in] ioll  larlite storage_manager
   */
  void ShowerRecoKeypoint::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {
    // clear member containers
    _shower_cand_v.clear();
    _recod_shower_v.clear();
    
    // get shower larflow hits (use SplitHitsBySSNet)
    larlite::event_larflow3dhit* shower_lfhit_v
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _ssnet_lfhit_tree_name );

    std::clock_t begin_process = std::clock();
    LARCV_INFO() << "start" << std::endl;
    LARCV_INFO() << "num larflow hits from [" << _ssnet_lfhit_tree_name << "]: " << shower_lfhit_v->size() << std::endl;
    
    // filter out shower pixels by larmatch score
    larlite::event_larflow3dhit shower_goodhit_v;
    for (auto const& hit : *shower_lfhit_v ) {
      if ( hit.track_score>_larmatch_score_threshold ) {
        shower_goodhit_v.push_back(hit);
      }
    }
    LARCV_INFO() << "number of above-threshold hits: " << shower_goodhit_v.size() << " of " << shower_lfhit_v->size() << std::endl;
    
    // make shower clusters
    float maxdist = 5.0;
    int minsize = 20;
    int maxkd = 20;
    std::vector<cluster_t> cluster_v;
    cluster_larflow3dhits( shower_goodhit_v, cluster_v, maxdist, minsize, maxkd );
    LARCV_INFO() << "num shower clusters:  " << cluster_v.size() << std::endl;
    

    // now for each shower cluster, we find some trunk candidates.
    // can have any number of such candidates per shower cluster
    // we only analyze clusters with a first pc-axis length > 1.0 cm
    std::vector< const cluster_t* > showerhit_cluster_v;
    std::vector<int>                cluster_used_v( cluster_v.size(), 0 );

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

      cluster_used_v[idx] = 1;
      showerhit_cluster_v.push_back( &showercluster );
    }


    // save shower cluster pca's
    larlite::event_larflowcluster* evout_goodhit_cluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "showergoodhit" );
    larlite::event_pcaxis* evout_goodhit_pca_v
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "showergoodhit" );
    int pcidx = 0;
    for ( auto const& cluster : cluster_v ) {
      larlite::larflowcluster lfc;
      for (auto const& idx : cluster.hitidx_v ) {
        lfc.push_back( shower_goodhit_v[idx] );
      }
      larlite::pcaxis pca = cluster_make_pcaxis( cluster, pcidx );
      pcidx++;
      evout_goodhit_cluster_v->emplace_back( std::move(lfc) );
      evout_goodhit_pca_v->emplace_back( std::move(pca) );
    }
    
    LARCV_INFO() << "num of trunk candidates: " << showerhit_cluster_v.size() << std::endl;

    // GET KEYPOINT DATA
    std::vector< const larlite::larflow3dhit* > keypoint_v;

    larlite::event_larflow3dhit* evout_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"keypoint");
    for ( auto const& kp : *evout_keypoint ) {
      if (kp.at(3)==(int)larflow::kShowerStart ) {
        keypoint_v.push_back( &kp );
      }
    }
    LARCV_INFO() << "number of shower keypoints: " << keypoint_v.size() << " of " << evout_keypoint->size() << std::endl;

    // MAKE TRUNK CANDIDATES FOR EACH SHOWER
    _reconstructClusterTrunks( showerhit_cluster_v, keypoint_v );

    // BUILD SHOWERS FROM CLUSTERS + TRUNK CANDIDATES
    _buildShowers( showerhit_cluster_v );

    larlite::event_larflowcluster* evout_shower_cluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "showerkp" );

    larlite::event_pcaxis* evout_shower_pca_v
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "showerkp" );

    larlite::event_larflow3dhit* evout_shower_keypoint_v
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "showerkp" );

    std::vector<int> used_v( shower_goodhit_v.size(), 0 );
    for ( size_t ireco=0; ireco<_recod_shower_v.size(); ireco++ ) {

      auto const& recoshower = _recod_shower_v[ireco];
      
      // make larflow3dhit cluster
      larlite::larflowcluster lfcluster;      
      for ( auto const& idx : recoshower.hitidx_v ) {
        lfcluster.push_back( shower_goodhit_v.at(idx) );
        used_v[idx]++;
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
        startpt[v] = recoshower.trunk.start_v[v];
        endpt[v] = startpt[v] + 50.0*axis_v[v];
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

      larlite::larflow3dhit shower_keypoint;
      shower_keypoint.resize(3,0);
      for (int v=0; v<3; v++) {
        shower_keypoint[v] = recoshower.trunk.keypoint->at(v);
      }
      evout_shower_keypoint_v->push_back( shower_keypoint );
      
    }//end of reco'd shower loop

    // save unused shower points
    larlite::event_larflow3dhit* evout_unused_hit_v
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "showerkpunused" );

    for (size_t idx=0; idx<shower_goodhit_v.size(); idx++ ) {
      if ( used_v[idx]==0 )
        evout_unused_hit_v->push_back( shower_goodhit_v.at(idx) );
    }

    LARCV_INFO() << "Unused hits: " << evout_unused_hit_v->size() << std::endl;
    std::clock_t end_process = std::clock();
    float elapsed = (end_process-begin_process)/CLOCKS_PER_SEC;
    LARCV_INFO() << "end: elasped=" << elapsed  << " secs" << std::endl;
  }

  /**
   * 
   * @brief match keypoints to shower clusters, use to define the trunk
   *
   * we match keypoints to shower clusters.
   * for each keypoint assigned to cluster, define 1,3,5 cm hit cluster around each keypoint.
   * going from 5,3,1 cm clusters, accept pca-axis based on eigenvalue ratio between first and second principle component.
   * 
   * use log likelihood function to pick best key-point trunk.
   * output is shower cluster, keypoint, and trunk cluster.
   * 
   * note: might want to move keypoint based on pca end near keypoint.
   *
   * This method populates the _shower_cand_v data member.
   *
   * @param[in] showercluster_v Clusters made from shower-labeled spacepoints
   * @param[in] keypoint_v  Keypoints from the keypoint network
   */
  void ShowerRecoKeypoint::_reconstructClusterTrunks( const std::vector<const cluster_t*>& showercluster_v,
                                                      const std::vector<const larlite::larflow3dhit*>& keypoint_v )
  {

    const float radii[3] = { 10, 5, 3 };
    
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

        std::vector< cluster_t > trunk_v(3);
        float mindist = 1e9;
        
        for ( size_t ihit=0; ihit<pshower->points_v.size(); ihit++ ) {

          float dist = 0.;
          for (size_t v=0; v<3; v++)
            dist += ( pshower->points_v[ihit][v]-keypoint[v] )*( pshower->points_v[ihit][v]-keypoint[v] );
          dist = sqrt(dist);

          for (size_t irad=0; irad<3; irad++) {
            if ( dist<radii[irad] ) {
              auto const& pt = pshower->points_v[ihit];
              trunk_v[irad].points_v.push_back( std::vector<float>( {pt[0], pt[1], pt[2]} ) );
              trunk_v[irad].hitidx_v.push_back( ihit );
            }
          }

          if ( dist<mindist )
            mindist = dist;
          
        }//end of hit loop

        // get pca for each length, must have >10 points
        // we pick the best trunk by
        //  (1) trunk must have pca-ratio<0.15
        //  (2) rank by smallest impact parameter
        
        LARCV_DEBUG() << "[ shower cluster[" << ishower << "] keypoint[" << ikeypoint << "] ]" << std::endl;
        LARCV_DEBUG() << "  keypoint pos: (" << keypoint[0] << "," << keypoint[1] << "," << keypoint[2] << ")" << std::endl;
        LARCV_DEBUG() << "  gap-dist: " << mindist << " cm" << std::endl;

        if ( mindist>1.0 ) continue;

        int best_trunk = -1;
        float best_metric = -1;
        float best_trunk_impactpar = 1e9;
        float best_trunk_eratio    = 1e9;        

        std::vector<int> trunk_best_pca_end_v(3,-1);
        std::vector<int> trunk_best_shower_end_v(3,-1);        
        
        for (size_t irad=0; irad<3; irad++) {

          cluster_t& cluster = trunk_v[irad];
          if ( cluster.points_v.size()<10 ) continue;
          
          cluster_pca( cluster );

          //LARCV_DEBUG() << "  radius[" << radii[irad] << "] num points: " << trunk_v[irad].size() << std::endl;
          
          float eratio = cluster.pca_eigenvalues[1]/cluster.pca_eigenvalues[0];
          std::vector<float> e_v = cluster.pca_axis_v[0];
          std::vector<float> center_v = cluster.pca_center;
            
          pcaxis_v[irad] = e_v;
          pca_center_v[irad] = center_v;
          pca_eigenval_ratio[irad] = eratio;

          // find which trunk end is furthest from cluster center
          float max_end_dist = 0;
          int best_end = 0;
          for (int iend=0; iend<2; iend++ ) {
            float distend = 0.;
            for (int v=0; v<3; v++ ){
              distend += ( cluster.pca_ends_v[iend][v]-pshower->pca_center[v] )*(  cluster.pca_ends_v[iend][v]-pshower->pca_center[v] );
            }
            if ( distend>max_end_dist ) {
              best_end = iend;
              max_end_dist = distend;
            }
          }
          trunk_best_pca_end_v[irad] = best_end;

          // we check if the trunk-pca and the cluster pca vary a lot
          // we default to the shower pca if they do, treating the trunk pca as a refinement
          // we find this is necessary as the KPS network's larmatch predictions are pretty noisy right now
          // maybe if we get this to larmatch v1 performance (which is good), then we can go
          // back to using the trunk pca.
          // this is just a law of large numbers effect i think, i.e.
          //  since the shower cluster is more points, there are more right 3d point predictions
          //  still, shower clusters can have very weird shapes ...
          float cluster_trunk_cos = 0.;
          for (int v=0; v<3; v++) {
            cluster_trunk_cos += pcaxis_v[irad][v]*pshower->pca_axis_v[0][v];
          }
          cluster_trunk_cos = fabs(cluster_trunk_cos);

          // find closest shower cluster end
          float min_shower_end = 1e9;
          int best_shower_end = 0;
          for (int iend=0; iend<2; iend++) {
            float distend=0;
            for (int v=0; v<3; v++) {
              distend += ( pshower->pca_ends_v[iend][v]-keypoint[v] )*( pshower->pca_ends_v[iend][v]-keypoint[v] );
            }
            if (distend<min_shower_end ) {
              min_shower_end = distend;
              best_shower_end = iend;
            }
          }
          trunk_best_shower_end_v[irad] = best_shower_end;

          // distance from trunk axis to keypoint
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

          LARCV_DEBUG() << "  radius["<< radii[irad] << " cm]: "
                        << " pca ratio=" << pca_eigenval_ratio[irad]
                        << " pca-0=(" << pcaxis_v[irad][0] << "," << pcaxis_v[irad][1] << "," << pcaxis_v[irad][2] << ")"
                        << " impactpar=" << impact
                        << " |trunk.shower|=" << cluster_trunk_cos
                        << std::endl;

          // [note] e-ratio an important parameter that needs configuration
          if ( eratio<0.3 && impact<5.0 && ( cluster_trunk_cos>best_metric || best_trunk==-1 ) ) {
            best_trunk = irad;
            best_trunk_impactpar = impact;
            best_trunk_eratio = eratio;
            best_metric = cluster_trunk_cos;
          }
          
        }// loop over radius size, calculating radius

        // define the best trunk candidate
        if ( best_trunk!=-1 ) {
          ShowerTrunk_t trunk;
          trunk.idx_keypoint = ikeypoint;          
          trunk.keypoint = &keypoint;
          trunk.pcaxis_v = pcaxis_v[best_trunk];
          trunk.center_v = pca_center_v[best_trunk];
          trunk.start_v  = trunk_v[best_trunk].pca_ends_v[ trunk_best_pca_end_v[best_trunk] ];
          //trunk.start_v  = pshower->pca_ends_v[ trunk_best_shower_end_v[best_trunk] ];
          trunk.pca_eigenval_ratio = pca_eigenval_ratio[best_trunk];
          trunk.npts = (int)trunk_v[best_trunk].points_v.size();
          trunk.gapdist = mindist;
          trunk.impact_par = impact_par_v[best_trunk];

          // we make sure the pca-axis is pointing to the rest of the cluster
          // we use the distance of the trunk center to the whole shower cluster center
          float coscenter = 0.;
          for (size_t v=0; v<3; v++) {
            coscenter += trunk.pcaxis_v[v]*( pshower->pca_center[v]-trunk.start_v[v] );
          }
          if ( coscenter<0 ) {
            // flip the axis dir
            for (size_t v=0; v<3; v++)              
              trunk.pcaxis_v[v] *= -1.0;
          }
          
          LARCV_DEBUG() << "define shower[" << ishower << "] keypoint[" << ikeypoint << "] trunk" << std::endl;
          LARCV_DEBUG() << "  gap-dist: " << trunk.gapdist << " cm" << std::endl;
          LARCV_DEBUG() << "  eigenval ratio: " << trunk.pca_eigenval_ratio << std::endl;
          LARCV_DEBUG() << "  npts: " << trunk.npts << std::endl;
          LARCV_DEBUG() << "  impact-par: " << trunk.impact_par << " cm" << std::endl;

          shower_cand.trunk_candidates_v.emplace_back( std::move(trunk) );          

        }
        
      }//end of keypoint

      LARCV_DEBUG() << "Saving shower candidate with " << shower_cand.trunk_candidates_v.size() << " trunk candidates" << std::endl;

      // we will pick the best trunk candidate later when we expand the shower candidates with nearby clusters
      
      _shower_cand_v.emplace_back( std::move(shower_cand) );
      
    }//end of shower cluster loop
    
  }

  /**
   * @brief build out shower cluster from trunk candidates
   *
   * for each shower cluster with trunk candidate, 
   * we absorb points clusters that are within some radius of the trunk axis.
   *
   * we use the candidates held in _shower_cand_v that were made in _reconstructClusterTrunks().
   *
   * first step is to associate other clusters to the shower candidates based on pc axes using _buildShowerCandidate().
   *
   * for showers that try to claim the same cluster, we do some tests to resolve which is the real shower trunk
   *  using _chooseBestShowerForCluster().
   *
   * the real shower trunk gets to claim the conflicted clusters.
   *
   * once conflicts are resolved, the reco. showers get to claim additional unclaimed hits using _fillShowerObject().
   * 
   * This method fills _recod_shower_v, which contains our reconstructed showers.
   *
   * @param[in] showerhit_cluster_v The clusters made from shower-labeled spacepoints.
   *
   */
  void ShowerRecoKeypoint::_buildShowers( const std::vector< const cluster_t*>& showerhit_cluster_v )
  {

    int nbad_cands = 0;
    for ( auto const& shower_cand : _shower_cand_v ) {
      if ( shower_cand.trunk_candidates_v.size()==0 ) continue;
      Shower_t shower  = _buildShowerCandidate( shower_cand, showerhit_cluster_v );
      if (shower.cluster_idx.size()>0 ) {
        _recod_shower_v.emplace_back( std::move(shower) );
      }
      else
        nbad_cands++;
    }

    LARCV_INFO() << "Number of reco'd shower candidates. "
      << "ngood=" << _recod_shower_v.size()
      << " nbad=" << nbad_cands
      << std::endl;
    
    // now we deal with the clusters that were claimed by more than shower
    // first we look for these by making a map from cluster index to a set with
    // shower indices (following the _recod_shower_v container)
    std::map< int, std::set<int> > claiming_shower_idx;
    for ( int shwr_idx=0; shwr_idx<(int)_recod_shower_v.size(); shwr_idx++ ) {
      auto const& shower = _recod_shower_v[shwr_idx];
      for ( auto const& cluster_idx : shower.cluster_idx ) {
        if ( claiming_shower_idx.find(cluster_idx)==claiming_shower_idx.end() ) {
          // insert new index with empty set
          claiming_shower_idx[cluster_idx] = std::set<int>();
        }
        claiming_shower_idx[cluster_idx].insert( shwr_idx );
      }
    }

    // now we resolve the conflicts by pitting the showers together.
    // the one with the lowest average least squares to the axis keeps the cluster.
    for ( auto it=claiming_shower_idx.begin(); it!=claiming_shower_idx.end(); it++ ) {
      int cluster_idx = it->first;
      std::set<int>& shower_idx_v = it->second;
      int best_shower_idx = _chooseBestShowerForCluster( *showerhit_cluster_v[cluster_idx],
                                                         shower_idx_v,
                                                         showerhit_cluster_v );

      // remove cluster index from shower's list
      for ( auto& shwr_idx : shower_idx_v ) {
        if ( best_shower_idx==shwr_idx ) {
          LARCV_DEBUG() << "shower[" << shwr_idx << "] keeps cluster[" << cluster_idx << "]" << std::endl;
          continue; // let the best shower keep the cluster index
        }
        auto& shower = _recod_shower_v[shwr_idx];
        LARCV_DEBUG() << "shower[" << shwr_idx << "] removes cluster[" << cluster_idx << "]" << std::endl;        
        auto it_remove = shower.cluster_idx.find( cluster_idx );
        shower.cluster_idx.erase( it_remove );
      }
    }


    // fill the shower objects with hits
    int sidx=0; 
    for ( auto& shower : _recod_shower_v ) {
      _fillShowerObject( shower, showerhit_cluster_v );
      LARCV_DEBUG() << "made shower[" << sidx << "] with " << shower.points_v.size() << std::endl;
      sidx++;
    }

    
  }

  /**
   * @brief build out a shower starting with trunk candidates
   *
   * we build a shower from a shower candidate, by 
   * building out the shower using each trunk candidate
   * and selecting the best shower somehow
   *
   * @param[in] shower_cand Shower trunk candidate.
   * @param[in] showerhit_cluster_v All the shower clusters
   * @return A reconstructed shower candidate
   */
  ShowerRecoKeypoint::Shower_t
  ShowerRecoKeypoint::_buildShowerCandidate( const ShowerCandidate_t& shower_cand,
                                             const std::vector< const cluster_t*>& showerhit_cluster_v )
  {

    std::vector< std::set<int> > trunk_cluster_idxset_v;

    LARCV_DEBUG() << "Build shower candidate. showercluster[" << shower_cand.cluster_idx << "]" << std::endl;
    
    // for each trunk candidate in the shower candidate, we let it absorb hits
    for (auto const& trunk : shower_cand.trunk_candidates_v ) {
      //std::set<int> clusters = _buildoutShowerTrunkCandidate( trunk, showerhit_cluster_v );
      // just
      std::set<int> clusters;
      clusters.insert( shower_cand.cluster_idx );
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

      Shower_t shower;
      shower.trunk = shower_cand.trunk_candidates_v[0];
      shower.cluster_idx = _buildoutShowerTrunkCandidate( shower.trunk, showerhit_cluster_v ); 
      return shower;
    }

    // now we have to evaluate which shower trunk is best
    std::set<int> union_cluster_idx;
    for ( auto& idx_set : trunk_cluster_idxset_v ) {
      for (auto& idx : idx_set ) {
        union_cluster_idx.insert(idx);
      }
    }
    LARCV_DEBUG() << "showercluster[" << shower_cand.cluster_idx << "]"
                  << "union cluster list size=" << union_cluster_idx.size() << std::endl;

    // return least squares value for each shower over the entirety of the cluster set
    // lowest value is considered the "best" cluster.
    int best_trunk_idx = _chooseBestTrunk( shower_cand,
                                           union_cluster_idx,
                                           showerhit_cluster_v );
    
    LARCV_DEBUG() << "returns best-fit trunk candidate" << std::endl;
    Shower_t shower;
    shower.trunk = shower_cand.trunk_candidates_v[best_trunk_idx];

    // build out cluster after modifying direction to use pca-axis of cluster itself
    shower.cluster_idx = _buildoutShowerTrunkCandidate( shower.trunk, showerhit_cluster_v ); 
    
    return shower;

  }

  /**
   * @brief build out the shower using the trunk candidate
   *
   * we absorb shower hits within some radius of the axis.
   * we track the shower hit indexes as well.
   *
   * @param[in] trunk_cand Shower trunk candidate
   * @param[in] showerhit_cluster_v all the shower clusters
   *
   */
  std::set<int> ShowerRecoKeypoint::_buildoutShowerTrunkCandidate( const ShowerTrunk_t& trunk_cand,
                                                                   const std::vector< const cluster_t*>& showerhit_cluster_v )
  {

    const float ar_moliere_rad_cm = 9.04;
    
    
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

        
        if ( dist<ar_moliere_rad_cm && fabs(proj)<50.0 ) {
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
        //LARCV_DEBUG() << "No bounding box points are close" << std::endl;
        continue;
      }

      // if bbox pt close, we test to see if we absorb cluster
      // defined as 10% of points inside shower threshold (this needs tuning)
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
        
        if ( (proj>0.0 && dist<_shower_rad_threshold_cm && proj<50.0 )
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


  /**
   * @brief assemble reconstructd shower from verious ingredients
   *
   * @param[in] shower_cand Shower candidate
   * @param[in] cluster_idx_set Indices to showerhit_cluster_v that are assigned to this reco shower
   * @param[in] trunk_idx   Shower trunk used to build this shower
   * @param[in] showerhit_cluster_v All the shower clusters in the event
   * @return Reconstructed shower object
   */
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

  /**
   * @brief get hits of reconstructed shower from original shower clusters
   * 
   * @param[in] shower Reconstructed shower object
   * @param[in] showerhit_cluster_v Original shower clusters
   */
  void ShowerRecoKeypoint::_fillShowerObject( Shower_t& shower,
                                              const std::vector< const cluster_t* >& showerhit_cluster_v )
  {

    for ( auto const& idx : shower.cluster_idx ) {
      auto const& cluster = *showerhit_cluster_v[idx];

      for (size_t ihit=0; ihit<cluster.points_v.size(); ihit++ ) {
        shower.points_v.push_back( cluster.points_v[ihit] );
        shower.hitidx_v.push_back( cluster.hitidx_v[ihit] );
      }
      
    }
  }

  /**
   * @brief Choose which trunk candidates within a shower candidate is correct
   *
   * @param[in] shower_cand The shower candidate under consideration
   * @param[in] cluster_idx_v Indices of showerhit_cluster_v for clusters associated to the shower candidate
   * @param[in] showerhit_cluster_v all of the original shower clusters
   * @return index of the best trunk candidate
   */
  int ShowerRecoKeypoint::_chooseBestTrunk( const ShowerCandidate_t& shower_cand,
                                            const std::set<int>& cluster_idx_v,
                                            const std::vector< const cluster_t* >& showerhit_cluster_v )
  {

    float max_ll = -1e9;
    int best_trunk_idx = -1;

    for ( int trunk_idx=0; trunk_idx<(int)shower_cand.trunk_candidates_v.size(); trunk_idx++ ) {
      
      auto const& trunk_cand = shower_cand.trunk_candidates_v[trunk_idx];
     

      std::vector<float> alongpca(3,0);
      for (int v=0; v<3; v++)
        alongpca[v] = trunk_cand.center_v[v] + 10.0*trunk_cand.pcaxis_v[v];
      
      // we pick by maximizing log-likelihood
      // we build a likelihood via
      // P(s,r) = exp(-r/r_M)*exp(-s/X_0)
      //  where r_M is the molier radius
      //  where X_0 is the radiation length
      float ll = 0.;      
      int npoints = 0;
      float w_tot = 0.;
      for ( auto const& idx : cluster_idx_v ) {
        for (auto const& hit : showerhit_cluster_v[idx]->points_v ) {
          float dist = pointLineDistance( trunk_cand.center_v, alongpca, hit );
          float proj = 0.;
          for (int v=0; v<3; v++) {
            proj += (hit[v]-trunk_cand.start_v[v])*trunk_cand.pcaxis_v[v];
          }
          if ( proj<0 )
            proj = 0.;
          ll += -sqrt(dist)/9.0;
          if ( proj>0 )
            ll += -proj/14.0;
          else
            ll += proj;
          
          npoints++;
          w_tot += 1.0;
        }
      }
      if ( npoints>0 )
        ll /= w_tot;

      LARCV_DEBUG() << "trunk cand[" << trunk_idx << " of " << shower_cand.trunk_candidates_v.size() << "] "
                    << " kp=(" << trunk_cand.keypoint->at(0) << "," << trunk_cand.keypoint->at(1) << "," << trunk_cand.keypoint->at(2)  << ") "
                    << " log(L)=" << ll << " for npoints=" << npoints
                    << std::endl;
      
      if ( best_trunk_idx==-1 || ll>max_ll ) {
        max_ll = ll;
        best_trunk_idx = trunk_idx;
      }
      
    }

    LARCV_DEBUG() << "Choosing trunk index=" << best_trunk_idx << " with max_ll=" << max_ll << std::endl;
    
    return best_trunk_idx;
  }

  /**
   * @brief resolve which shower to assign cluster originally assigned to more than one
   * 
   * @param[in] cluster Shower cluster in question
   * @param[in] shower_idx_v Index of shower candidates that the cluster has been assigned to
   * @param[in] showerhit_cluster_v All the shower clusters
   * @return index of the best shower candidate to assign cluster
   */
  int ShowerRecoKeypoint::_chooseBestShowerForCluster( const cluster_t& cluster,
                                                       const std::set<int>& shower_idx_v,
                                                       const std::vector< const cluster_t* >& showerhit_cluster_v )
  {

    float min_least_sq = -1;
    int best_shower_idx = -1;

    for ( auto const& shower_idx : shower_idx_v ) {

      auto const& shower = _recod_shower_v.at(shower_idx);
      
      float ls = 0.;

      std::vector<float> alongpca(3,0);
      for (int v=0; v<3; v++)
        alongpca[v] = shower.trunk.center_v[v] + 10.0*shower.trunk.pcaxis_v[v];
      

      int npoints = 0;
      for ( auto const& hit : cluster.points_v ) {
        float dist = pointLineDistance( shower.trunk.center_v, alongpca, hit );
        ls += dist*dist;
        npoints++;
      }
      if ( npoints>0 )
        ls /= float(npoints);

      LARCV_DEBUG() << "reco shower cand[" << shower_idx << "] "
                    << " least-sq/npoints=" << ls << " for npoints=" << npoints
                    << std::endl;
                       
      if ( min_least_sq<0 || min_least_sq>ls ) {
        min_least_sq = ls;
        best_shower_idx = shower_idx;
      }
      
    }

    LARCV_DEBUG() << "Choosing shower with index=" << best_shower_idx << std::endl;
    
    return best_shower_idx;
  }
  
}
}
