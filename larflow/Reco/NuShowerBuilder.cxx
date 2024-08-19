#include "NuShowerBuilder.h"

#include "larlite/DataFormat/mctruth.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"
#include "larflow/Reco/cluster_functions.h"
#include "larflow/Reco/geofuncs.h"
#include "larflow/Reco/ProjectionDefectSplitter.h"

namespace larflow {
namespace reco {

  /**
   * @brief Run shower builder on neutrino candidate tracks
   *
   * Using NuVertexCandidate instances as a seed, 
   * build out showers using simple cone-based clustering.
   *
   * Steps of this algorithm:
   *  1) for each shower point cluster, 
   *     create the shower's "trunk" using closest hits 
   *     to the nu vertex position.
   *  2) for each shower cluster, the first principle
   *     component of the trunk defines it's direction.
   *     if num of points in the trunk is <4 or PCA analysis fails,
   *     then the shower cluster is skipped.
   *  3) The direction is refined using ProjectionDefectSplitter::fitLineSegmentToCluster
   *  4) the impact distance between the trunk dir and neutrino vertex is calculated
   *  5) after directions are calculated for all valid shower clusters, we sort by distance
   *     to the nu vertex, with closest first.
   *  6) Looping from closest prong to furthest, we start a shower seed
   *     if: the impact distance is < 20.0 cm and nhits>10
   *  7) when a seed is accepted, we absorb other shower clusters if 
   *     some fraction of points are <30 deg and 0.5 cm downstream of trunk start point
   * 
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   * @param[in] nu_candidate_v Neutrino proto-vertices produced by NuVertexMaker.
   */
  void NuShowerBuilder::process( larcv::IOManager& iolcv,
                                 larlite::storage_manager& ioll,
                                 std::vector<NuVertexCandidate>& nu_candidate_v,
				 std::vector<ClusterBookKeeper>& nu_cluster_book_v )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    // get shower clusters we want to merge into a meta cluster
    const int _min_shower_cluster_hits = 4;

    // this is the set of info we will compile for each reco prong+nu vertex combination
    // information in this struct is used to build shower prongs
    struct ShowerProngInfo_t {
      int vtxcluster_idx; // index in nuvtx.cluster_v container
      int segfit_idx; // index of fitted track in the segfit_v container
      float dist2vertex;
      float impactdist;
      int nhits;
      int used;
      std::vector<float> trunk_dir;
      std::vector<float> trunk_pt;
      std::vector<float> trunk_pt2;
      std::vector<float> trunk_pca;

      // default constructor
      ShowerProngInfo_t()
      : vtxcluster_idx(-1),
      segfit_idx(-1),
      dist2vertex(1.0e6),
      impactdist(1.0e6),
      nhits(0),
      used(0),
      trunk_dir( std::vector<float>{0,0,0} ),
      trunk_pt( std::vector<float>{0,0,0} ),
      trunk_pt2( std::vector<float>{0,0,0} ),
      trunk_pca( std::vector<float>{0,0,0} )
      {};
      
      bool operator<( const ShowerProngInfo_t& rhs) const {
        if ( dist2vertex < rhs.dist2vertex ) {
          return true;
        }
        return false;
      };
    };

    // wire plane images for getting dqdx later
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();

    std::vector<float> true_nu_vtx_pos(4,0);
    std::vector<float> sce_nu_vtx_pos(4,0);

    // reset mcanalysis variables
    _mcana_index_closest_recovtx = -1; // variable to indicate when to run shower mc analysis
    _map_prongindex_to_mcanainfo.clear();

    if ( _mc_analysis_mode ) {
      if ( _mcpg )
	      delete _mcpg;

      LARCV_DEBUG() << " INITIALIZE MC ANALYSIS FOR SHOWER RECO STUDY: build MCPixelPGraph" << std::endl;
      
      // we run the MCPixelPGraph to get truth information
      _mcpg = new ublarcvapp::mctools::MCPixelPGraph();
      _mcpg->buildgraph( iolcv, ioll );

      // get true position of neutrino
      larlite::event_mctruth* ev_mctruth =
	      (larlite::event_mctruth*)ioll.get_data(larlite::data::kMCTruth,"generator");
      const larlite::mctruth& mct = ev_mctruth->front();
      true_nu_vtx_pos[0] = mct.GetNeutrino().Nu().Trajectory().front().X();
      true_nu_vtx_pos[1] = mct.GetNeutrino().Nu().Trajectory().front().Y();
      true_nu_vtx_pos[2] = mct.GetNeutrino().Nu().Trajectory().front().Z();
      true_nu_vtx_pos[3] = mct.GetNeutrino().Nu().Trajectory().front().T();
      // convert to apparent position
      sce_nu_vtx_pos = ublarcvapp::mctools::MCPos2ImageUtils::Get()->to_imagepos( true_nu_vtx_pos[0],
										  true_nu_vtx_pos[1],
										  true_nu_vtx_pos[2],
										  true_nu_vtx_pos[3] );

      // find the closest vertex
      _mcana_index_closest_recovtx = -1;
      _mcana_closest_recovtx_dist = 1.0e9;
      for ( int ivtx=0; ivtx<(int)nu_candidate_v.size(); ivtx++ ) {
	      auto& nuvtx = nu_candidate_v.at(ivtx);
	      float dist = 0.;
      	for (int ii=0; ii<3; ii++) {
	        dist += ( nuvtx.pos[ii]-sce_nu_vtx_pos[ii] )*( nuvtx.pos[ii]-sce_nu_vtx_pos[ii] );
	      }
	      dist = sqrt(dist);
	      if ( dist < 3.0 && dist < _mcana_closest_recovtx_dist ) {
	        _mcana_index_closest_recovtx = ivtx;
	      }
      }
      LARCV_DEBUG() << "Closest reco neutrino vertex: "
		    << " index=" << _mcana_index_closest_recovtx
		    << " closest distance=" << _mcana_closest_recovtx_dist
		    << std::endl;
    }//end of mcanalysis mode: initialization, finding the vertex to evaluate
    

    // loop through the vertex candidates
    bool mcanalyze_this_nuvtx = false;
    for ( int inuvtx=0; inuvtx<(int)nu_candidate_v.size(); inuvtx++ ) {

      auto& nuvtx = nu_candidate_v.at(inuvtx);
      mcanalyze_this_nuvtx = false;
      if ( _mc_analysis_mode && inuvtx==_mcana_index_closest_recovtx )
        mcanalyze_this_nuvtx = true;

      // I should modularize below to handle code for single vertex
      LARCV_DEBUG() << "/////// [Vertex Start]: "
                    << "(" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")"
                    << "/////////////"
                    << std::endl;

      // loop over shower clusters, gathering info
      // containers for some quantities/metrics
      std::vector<larlite::track> segfit_v;
      std::vector<larflow::reco::cluster_t> pcacluster_v;

      // the container of prong information we will fill for shower clusters
      std::vector< ShowerProngInfo_t > pronginfo_v;

      // loop over clusters, for shower-types, fill in a ShowerProngInfo_t instance
      // note: where do we fill these clusters again?
      for ( int ivtx=0; ivtx<(int)nuvtx.cluster_v.size(); ivtx++) {
        
        auto const& vtxcluster = nuvtx.cluster_v[ivtx];

        // only deal with showers
        if ( vtxcluster.type!=NuVertexCandidate::kShower 
          && vtxcluster.type!=NuVertexCandidate::kShowerKP ) {
          continue;
        }

        // for each vertex cluster we make a shower by:
        // (1) fitting the region closest to the vertex with line segments
        // (2) look for track clusters directly along the path between shower and vertex
        // (3) define a cone using the mollier radius and absorb hits

        // for each cluster we want the following reco variables:
        // (1) trunk direction
        // (2) impact parameter (again)
        // (3) dq/dx at trunk
        // (4) distance to vertex
        // (5) number of hits
        // (6) total charge

        const larlite::larflowcluster& lfcluster =
          ((larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );

        // get distance of each hit to the vertex
	      // and find the minimum distance as well
        std::vector<float> dist2vtx(lfcluster.size(),0);
        float mindist = 1e9;
        
        for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
          dist2vtx[ihit] = 0.0;
          for (int i=0; i<3; i++) {
            dist2vtx[ihit] += ( lfcluster[ihit][i]-nuvtx.pos[i] )*( lfcluster[ihit][i]-nuvtx.pos[i] );
          }
          dist2vtx[ihit] = sqrt(dist2vtx[ihit]);
          if ( mindist>dist2vtx[ihit] ) {
            mindist = dist2vtx[ihit];
          }
        }

        // now collect hits
        larflow::reco::cluster_t prong;
        larlite::event_larflow3dhit lfhit_v;
        for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
          if ( dist2vtx[ihit]-mindist < 10.0 ) {
            std::vector<float> pos_and_weights(5,0);

            // get position
            for (int i=0; i<3; i++)
              pos_and_weights[i] = lfcluster[ihit][i];
            prong.hitidx_v.push_back( ihit );
            prong.points_v.push_back( pos_and_weights );

            lfhit_v.push_back( lfcluster[ihit] );

          }
        }

        RecoShowerInfo_t mcanainfo;
	      if ( _mc_analysis_mode && mcanalyze_this_nuvtx ) {
	        // When we run in  MC Analysis Mode,
	        // gather features about this shower fragment
          _gatherTruthShowerFeatures( prong, nuvtx, mcanainfo );
	      }

	      bool pass_showercluster_precuts = true;
	
        if ( prong.points_v.size()<_min_shower_cluster_hits ) {
	        pass_showercluster_precuts = false;
	      }

	      try {
	        larflow::reco::cluster_pca( prong );
	      }
	      catch (...) {
	        pass_showercluster_precuts = false;
	      }

	      if ( !pass_showercluster_precuts ) {
          // the prong has failed the precuts and will not be considered further
          // but if we are doing MC analysis for this vertex,
          // we want to record the truth info for this shower fragment 
          // (e.g. maybe we SHOULD have attached this vertex)
          if ( _mc_analysis_mode && mcanalyze_this_nuvtx ) {
	          mcanainfo._reco_outcome = kFailPreCuts;
            _map_prongindex_to_mcanainfo[ ivtx ] = mcanainfo;
          }
	        continue;
	      }

        // This prong will be analyzed. Get the quantities we will need
        // to decide if we should use the shower fragment as the start
        // of a shower prong.

        // fit line segment
        larlite::track segfit = larflow::reco::ProjectionDefectSplitter::fitLineSegmentToCluster( prong, lfhit_v, adc_v, 2.0 );
        int track_npts = segfit.NumberTrajectoryPoints();

        // which end is closest?
        float enddist[2] = {0,0};
        for (int i=0; i<3; i++) {
          enddist[0] += (nuvtx.pos[i]-segfit.LocationAtPoint(0)[i])*(nuvtx.pos[i]-segfit.LocationAtPoint(0)[i]);
          enddist[1] += (nuvtx.pos[i]-segfit.LocationAtPoint(track_npts-1)[i])*(nuvtx.pos[i]-segfit.LocationAtPoint(track_npts-1)[i]);
        }

        // Create and fill the struct for the shower fragment
        ShowerProngInfo_t info;
        info.vtxcluster_idx = ivtx;
        info.segfit_idx = (int)segfit_v.size();
        info.used = 0;
        info.nhits = (int)lfcluster.size();

        // direction from segment line fit
        info.trunk_dir.resize(3,0);
        info.trunk_pt.resize(3,0);
        info.trunk_pt2.resize(3,0);        
        if ( enddist[0]<enddist[1] ) {
          for (int i=0; i<3; i++) {
            info.trunk_dir[i] = segfit.DirectionAtPoint(0)[i];
            info.trunk_pt[i]  = segfit.LocationAtPoint(0)[i];
          }
          info.dist2vertex = sqrt(enddist[0]);          
        }
        else {
          for (int i=0; i<3; i++) {
            info.trunk_dir[i] = -1.0*segfit.DirectionAtPoint(track_npts-2)[i];
            info.trunk_pt[i]  = segfit.LocationAtPoint(track_npts-1)[i];
          }
          info.dist2vertex = sqrt(enddist[1]);          
        }
        for (int i=0; i<3; i++) {
          info.trunk_pt2[i] = info.trunk_pt[i] + 3.0*info.trunk_dir[i];
        }
        info.impactdist = larflow::reco::pointLineDistance( info.trunk_pt, info.trunk_pt2, nuvtx.pos );

        // pca direction
        float pca_s = larflow::reco::pointRayProjection3f( info.trunk_pt, prong.pca_axis_v[0], nuvtx.pos );
        info.trunk_pca.resize(3,0);        
        if ( pca_s<0 ) {
          for (int i=0; i<3; i++)
            info.trunk_pca[i] = prong.pca_axis_v[0][i];
        }
        else {
          for (int i=0; i<3; i++)
            info.trunk_pca[i] = -prong.pca_axis_v[0][i];
        }

        segfit_v.emplace_back( std::move(segfit) );
        pcacluster_v.emplace_back( std::move(prong) );

        pronginfo_v.emplace_back( std::move(info) );
        
      }//end of vtx cluster loop that fills pronginfo_v

      // sort shower prongs by distance to vertex
      std::sort( pronginfo_v.begin(), pronginfo_v.end() );

      // start to group pieces into showers
      for ( int iprong=0; iprong<(int)pronginfo_v.size(); iprong++ ) {
        auto& prong = pronginfo_v.at(iprong);
        //std::cout << "prong[" << iprong << "] dist=" << prong.dist2vertex << " nhits=" << prong.nhits << std::endl;
        if ( prong.used==1 )
          continue;

        // now decide what to do with this unused prong

        // fill in reco prong info if we're doing MC analysis on shower fragments
        if ( _mc_analysis_mode && mcanalyze_this_nuvtx ) {
          auto it_recoshowerinfo = _map_prongindex_to_mcanainfo.find( prong.vtxcluster_idx );
          if ( it_recoshowerinfo!=_map_prongindex_to_mcanainfo.end() ) {
            // found the corresponding mcana info
            auto& mcanainfo = it_recoshowerinfo->second;
            mcanainfo._vtx_impactpar = prong.impactdist;
            mcanainfo._reco_trunkdir.resize(3,0);
            for (int v=0; v<3; v++) {
              mcanainfo._reco_trunkdir[v] = prong.trunk_dir[v];
            }
            mcanainfo._recoshower_dist2vtx = prong.dist2vertex;
            mcanainfo._reco_outcome = kFailAttachment; // will change to accept if passes below
          }
        }

        // if trunk points back to vertex and is a large enough cluster
        if ( prong.impactdist<20.0 && prong.nhits>10 ) {
          //std::cout << "seed with prong[" << iprong << "]" << std::endl;

          // this prong is attached by the reco
          // indicate successful outcome
          if ( _mc_analysis_mode && mcanalyze_this_nuvtx ) {
            auto it_recoshowerinfo = _map_prongindex_to_mcanainfo.find( prong.vtxcluster_idx );
            // if we found the prong's mc ana info
            if ( it_recoshowerinfo!=_map_prongindex_to_mcanainfo.end() ) {
              auto& mcanainfo = it_recoshowerinfo->second;
              mcanainfo._reco_outcome = kAccept;
            }
          }

          // seed a new cluster          
          prong.used = 1;          
          larlite::larflowcluster shower;
          
          // copy hits
          auto const& vtxcluster = nuvtx.cluster_v[ prong.vtxcluster_idx ];          
          const larlite::larflowcluster& lfcluster =
            ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );

          for ( auto const& hit : lfcluster )
            shower.push_back( hit );

          // absorb other shower clusters if within 2 mollier radii (9 cm x 2 ) of trunk axis
          for (int jprong=iprong+1; jprong<(int)pronginfo_v.size(); jprong++) {

            // don't reuse cluster
            if ( pronginfo_v[jprong].used==1 )
              continue;

            std::vector<float> pt2(3,0);
            for (int i=0; i<3; i++)
              pt2[i] = prong.trunk_pt[i] + 3*prong.trunk_pca[i];
            
            float r = larflow::reco::pointLineDistance3f( prong.trunk_pt, pt2, pronginfo_v[jprong].trunk_pt );
            float s = larflow::reco::pointRayProjection3f( prong.trunk_pt, prong.trunk_pca, pronginfo_v[jprong].trunk_pt );
            float pt_ang = 0.;
            if ( s!=0.0 )
              pt_ang = atan(r/fabs(s))*180.0/3.14159;
            
            if ( pt_ang<30.0 && s>0.5 ) {

              auto& subprong = pronginfo_v[jprong];
              subprong.used = 1;
              
              auto const& jvtxcluster = nuvtx.cluster_v[ subprong.vtxcluster_idx];
              const larlite::larflowcluster& lfcl_absorb =
                ((larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, jvtxcluster.producer))->at( jvtxcluster.index );
              for ( auto const& hit : lfcl_absorb )
                shower.push_back( hit );
            }// if absorb cluster
          }//end of loop over potential subclusters

          larlite::track shower_trunk_dir;
          shower_trunk_dir.add_vertex( TVector3(prong.trunk_pt[0], prong.trunk_pt[1], prong.trunk_pt[2]) );
          shower_trunk_dir.add_vertex( TVector3(prong.trunk_pt[0]+20.0*prong.trunk_pca[0],
                                                prong.trunk_pt[1]+20.0*prong.trunk_pca[1],
                                                prong.trunk_pt[2]+20.0*prong.trunk_pca[2]) );
          shower_trunk_dir.add_direction( TVector3(prong.trunk_pca[0],prong.trunk_pca[1],prong.trunk_pca[2]) );
          shower_trunk_dir.add_direction( TVector3(prong.trunk_pca[0],prong.trunk_pca[1],prong.trunk_pca[2]) );
          
          // save shower to nuvtx candidate object
          nuvtx.shower_v.emplace_back( std::move(shower) );
          nuvtx.shower_trunk_v.emplace_back( std::move(shower_trunk_dir) );
          
        }//end of if valid seeding prong
      }//end of shower clusters @ vtx
    }//end of nu vertex candidates
    
    if ( _mc_analysis_mode ) {
      _fill_mcanalysis_tree();
    }

    LARCV_DEBUG() << "end" << std::endl;
  }

  /**
   * @brief here we record both truth-based and reco-based quantities to evaluate/tune 
   *        the shower to neutrino vertex attachment algorithm
   *
   */
  void NuShowerBuilder::_gatherTruthShowerFeatures( larflow::reco::cluster_t& prong,
						    larflow::reco::NuVertexCandidate& vtx,
                NuShowerBuilder::RecoShowerInfo_t& showerinfo )
  {
    // What we want to know?
    // Ultimately, should we attach the shower prong to the vertex, treating it like a trunk
    // right now the criteria is:
    // prong.impactdist<20.0 && prong.nhits>10
    // does this make sense? can we do something more sophisticated? (e.g. xgboost)
    
    // we call this before making pre-cuts on the shower
    // this is based on the number of hits essentially.
    // we ask: how well are reconstructing the size of the cluster?
    // does this reco prong correlate to a detectable trunk? a fragment? partially the trunk?

    // define a struct to hold the information we want to gather for each reco prong
    struct TrueShowerInfo_t {
      int geant_track_id;
      const std::vector< larcv::Image2D >* pix_mask_v; //< we are going to crop out the image per plane around the trunk
      std::vector< float > pix_sum_v; //< the pixel sum per plane
      TrueShowerInfo_t()
      : geant_track_id(-1),
      pix_mask_v(nullptr)
      {};
    };

    std::vector< TrueShowerInfo_t > trueprong_v;

    // first we loop througuh the true shower in the events
    bool exclude_neutrons = true;
    auto pnode_v = _mcpg->getNeutrinoParticles( exclude_neutrons );
    for (int inode=0; inode<(int)pnode_v.size(); inode++) {
      auto& pnode = pnode_v.at(inode);
      if ( pnode->pid==22 ) {
	      // photons
	      auto const& pointlist = _mcpg->getTruePhotonTrunk3DPoints( *pnode );
	      // check if this particle node has a photon trunk
	      if ( pointlist.size()==0 )
	        continue;

	      TrueShowerInfo_t prongtruth;
	      prongtruth.geant_track_id = pnode->tid;
	      prongtruth.pix_sum_v  = _mcpg->getTruePhotonTrunkPlanePixelSums( pnode->tid );
	      prongtruth.pix_mask_v = &(_mcpg->getTruePhotonTrunkPlaneImage2DMasks( pnode->tid ));
	      trueprong_v.emplace_back( std::move( prongtruth ) );
      } // if gamma node      
    }// end of loop over nodes of particles recorded by the simulation

    // we have to convert the 3d positions of the shower fragment to pixel locations
    typedef std::set< std::pair<int,int> > PixelSet_t;
    std::vector< PixelSet_t > plane_pixelsets_v(3);
    for (int ihit=0; ihit<(int)prong.points_v.size(); ihit++) {
      auto const& pt = prong.points_v.at(ihit);
      std::vector<float> imgpos =
	      ublarcvapp::mctools::MCPos2ImageUtils::Get()->to_imagepos( pt[0], pt[1], pt[2], 0.0 );
      // imagepos is (u,v,y,tick)
      if ( imgpos[3]<=0 ) {
	      // if the tick equals to 0000, then 3d point outside tpc
	      continue;
      }
      
      for (int p=0; p<3; p++) {
	      plane_pixelsets_v[p].insert( std::pair<int,int>( (int)imgpos[p], (int)imgpos[3] ) );
      }
    }//end of pixel gathering portion
    // Now we loop over the shower fragments and calculate variables for them
    float max_frac = 0;
    int max_frac_index = -1;
    int max_frac_trackid = -1;
    int max_frac_plane = -1;
    float max_frac_masksum = 0.;
    float max_frac_coverage = 0.;
    for (int iphoton=0; iphoton<(int)trueprong_v.size(); iphoton++) {
      auto const& trueprong = trueprong_v.at(iphoton);
      
      // get the plane with the most pixels
      int maxplane = -1;
      int maxplane_sum = 0;
      for (int p=0; p<(int)trueprong.pix_mask_v->size(); p++) {
	      const larcv::Image2D& maskcrop = trueprong.pix_mask_v->at(p);
	      float masksum = 0.;
	      for ( auto& mask : maskcrop.as_vector()  )
	        masksum += mask;
	      if (masksum>maxplane_sum) {
	        maxplane_sum = masksum;
	        maxplane = p;
	      }
      }

      // completely empty
      if ( maxplane==-1 || maxplane_sum<=0.0 )
	      continue;
      
      // ok now compare reco shower pixelset to ours in the larcv images
      // we loop over the reco prong's pixels
      int num_on_mask = 0;
      const larcv::Image2D& maxtruecrop = trueprong.pix_mask_v->at(maxplane);
      for ( auto& pix : plane_pixelsets_v[ maxplane ] ) {
	      float wire = (float)pix.first;
	      float tick = (float)pix.second;
	      if ( maxtruecrop.meta().contains( wire,tick ) ) {
	        if ( maxtruecrop.pixel( tick, wire )>0.5 )
	          num_on_mask++;
	      }
      }
      // calculate the fraction of reco fragment's pixels belong to the true prong
      float frac = float(num_on_mask)/float(plane_pixelsets_v[maxplane].size());
      if ( frac>0.0 && frac > max_frac ) {
	      max_frac = frac;
	      max_frac_index = iphoton;
	      max_frac_trackid = trueprong.geant_track_id;
	      max_frac_coverage = float(num_on_mask)/float(maxplane_sum);
	      max_frac_plane = maxplane;
      }
    }//end of loop over true photons

    if ( max_frac_index>=0 ) {
      LARCV_DEBUG() << "Shower Reco fragment matched to true photon trunk (trackid=" << max_frac_trackid << "): "
		    << " frac pixel overlap=" << max_frac
		    << std::endl;
      LARCV_DEBUG() << "True prong pixelsums: ("
		    << trueprong_v.at( max_frac_index ).pix_sum_v[0] << ", "
		    << trueprong_v.at( max_frac_index ).pix_sum_v[1] << ", "
		    << trueprong_v.at( max_frac_index ).pix_sum_v[2] << ")"
		    << std::endl;
    }
    else {
      LARCV_DEBUG() << "Shower Reco fragment unmatched to true photon trunks" << std::endl;
      // fill sentinal values and return
      showerinfo._trueprong_trackid   = -1;  ///< unmatched sentinal valuesthe true prong that had the highest pixel overlap with the reco fragment's pixels
      showerinfo._frac_truetrunk      = 0.; ///< the fraction of the true prong's pixels covered by the reco fragment
      showerinfo._frac_recopurity     = -1.0;          ///< how much of the reco fragment pixels overlap with the matched true prong
      showerinfo._cluster_pixsum_MeV  = 0.;
      showerinfo._true_trunkdir       = std::vector<float>{ 0, 0, 0 };
      showerinfo._trueprong_dist2vtx  = -1.0;
      showerinfo._recoshower_dist2vtx = -1.0;
      showerinfo._reco_trunkdir = std::vector<float>{ 0, 0, 0};
      showerinfo._vtx_impactpar = -1.0;
      // we do not match to any true photon trunks
      // so the true outcome for initial attachment should to not attach
      showerinfo._correct_outcome = 0;
      return;
    }

    // get the mcpg node, containing info about the true prong photon
    auto const& pnode_matched_trunk = _mcpg->findTrackID( max_frac_trackid );

    // get the true start point of the prong, in 3d
    std::vector<float> trueprong_first_edep_pos = pnode_matched_trunk->first_edep_pos;

    // we calculate the true trunk direction
    std::vector<float> trueprong_dir(3,0.0);
    auto const& ptlist = _mcpg->getTruePhotonTrunk3DPoints( max_frac_trackid );

    // we need to put the points into the cluster struct
    larflow::reco::cluster_t trueprong_cluster;
    trueprong_cluster.points_v.reserve( ptlist.size() );
    for (auto const& pt : ptlist ) {
      trueprong_cluster.points_v.push_back( pt );
    }

    try {
      // run pca code to calculate principle component of 3d points
      larflow::reco::cluster_pca( trueprong_cluster );

      // extract the ends of a line segment parallel to the 1st pc component that bounds the 3d points
      // nuvtx.pos is a reference point, meant to ensure that the pc axis line segment
      // has the closest point first.
      larlite::pcaxis clust_axis
	      = larflow::reco::cluster_make_pcaxis_wrt_point( trueprong_cluster, vtx.pos );
      float trueprong_mag = 0.;
      for (int v=0; v<3; v++) {
        trueprong_dir[v] = (clust_axis.getEigenVectors().at(4)[v]-clust_axis.getEigenVectors().at(3)[v]);
        trueprong_mag += trueprong_dir[v]*trueprong_dir[v];
      }
      trueprong_mag = sqrt(trueprong_mag);
      if ( trueprong_mag>0.0 ) {
      for (int v=0; v<3; v++)
        trueprong_dir[v] /= trueprong_mag;
      }
    }
    catch (...) {
      // problem running pca code. we define the trunk direction as the direction between the closest and first point from the trunk start
      float maxdist = 0.0;
      std::vector<float> maxpt(3,0.0);
      for ( auto const& pt : ptlist ) {
        float dist = 0.;
        for (int v=0; v<3; v++) {
          dist += (pt[v]-trueprong_first_edep_pos[v])*(pt[v]-trueprong_first_edep_pos[v]);
        }
        if ( dist > maxdist )  {
          maxpt = pt;
          maxdist = dist;
        }
      }
      maxdist = sqrt(maxdist);
      if ( maxdist>0.0 ) {
        for (int v=0; v<3; v++) {
          trueprong_dir[v] = (maxpt[v]-trueprong_first_edep_pos[v])/maxdist;
        }
      }
    }

    // calculate distance between true prong and reco vertex
    float dist2vtx = 0.;
    for (int v=0; v<3; v++) {
      float dd = trueprong_first_edep_pos[v]-vtx.pos[v];
      dist2vtx += dd*dd;
    }
    dist2vtx = sqrt(dist2vtx);

    // put info into the struct assigned to each reco fragment
    showerinfo._trueprong_trackid   = max_frac_trackid;  ///< the true prong that had the highest pixel overlap with the reco fragment's pixels
    showerinfo._frac_truetrunk      = max_frac_coverage; ///< the fraction of the true prong's pixels covered by the reco fragment
    showerinfo._frac_recopurity     = max_frac;          ///< how much of the reco fragment pixels overlap with the matched true prong
    showerinfo._cluster_pixsum_MeV  = trueprong_v.at( max_frac_index ).pix_sum_v[max_frac_plane]*0.0182;
    showerinfo._true_trunkdir       = trueprong_dir;
    showerinfo._trueprong_dist2vtx  = dist2vtx;
    showerinfo._recoshower_dist2vtx = -1;
    showerinfo._reco_trunkdir = std::vector<float>{ 0, 0, 0};
    showerinfo._vtx_impactpar = -1.0;
    // what is the "ground truth correct" outcome?
    // set detectable threshold
    if ( showerinfo._cluster_pixsum_MeV>10.0 )
      showerinfo._correct_outcome = 1;
    else
      showerinfo._correct_outcome = 0;
    
  }//end of _gatherTruthShowerFeatures
  
  void NuShowerBuilder::createMCAnalysisTree( TFile* outfile )
  {
    outfile->cd();
    _mcana_per_recoshower_tree = new TTree("nushowerbuilder_mcana_tree", "MC Analysis to evaluate and tune NuShowerBuilder Algorithm");

  }

  void NuShowerBuilder::_fill_mcanalysis_tree()
  {

    if ( !_mc_analysis_mode )
      return;

    // transfer variables for each reco shower fragment that was evaluated
    for ( auto it : _map_prongindex_to_mcanainfo ) {
      int nuvtx_icluster = it.first;
      auto& mcanainfo = it.second;

      _mcana_trueprong_pixsum_MeV   = mcanainfo._cluster_pixsum_MeV;
      _mcana_trueprong_efficiency   = mcanainfo._frac_truetrunk;
      _mcana_recofragment_purity    = mcanainfo._frac_recopurity;
      _mcana_trueprong_dist2vtx     = mcanainfo._trueprong_dist2vtx;
      _mcana_recofragment_dist2vtx  = mcanainfo._recoshower_dist2vtx;
      _mcana_recofragment_impactpar = mcanainfo._vtx_impactpar;
      _mcana_reco_outcome           = mcanainfo._reco_outcome;
      _mcana_groundtruth_outcome    = mcanainfo._correct_outcome;
      for (int v=0; v<3; v++) {
        _mcana_trueprong_trunkdir[v]    = mcanainfo._true_trunkdir[v];
        _mcana_recofragment_trunkdir[v] = mcanainfo._reco_trunkdir[v];
      }

      // save the values of the variables to the tree 
      _mcana_per_recoshower_tree->Fill();
    }
    
  }

}
}
