#include "NuShowerBuilder.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larflow/Reco/cluster_functions.h"
#include "larflow/Reco/geofuncs.h"
#include "larflow/Reco/ProjectionDefectSplitter.h"

namespace larflow {
namespace reco {

  /**
   * @brief Run shower builder on neutrino candidate tracks
   *
   * Using NuVertexCandidate instances as a seed, 
   * build out showers using simple cone-based clustering
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   * @param[in] nu_candidate_v Neutrino proto-vertices produced by NuVertexMaker.
   */
  void NuShowerBuilder::process( larcv::IOManager& iolcv,
                                 larlite::storage_manager& ioll,
                                 std::vector<NuVertexCandidate>& nu_candidate_v )
  {

    // get shower clusters we want to merge into a meta cluster

    // wire plane images for getting dqdx later
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();

    // loop through the vertex candidates
    for (auto& nuvtx : nu_candidate_v ) {

      LARCV_DEBUG() << "/////// [Vertex Start]: "
                    << "(" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")"
                    << "/////////////"
                    << std::endl;

      // loop over shower clusters, gathering info
      std::vector<larlite::track> segfit_v;
      std::vector<larflow::reco::cluster_t> pcacluster_v;
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
        
        bool operator<( const ShowerProngInfo_t& rhs) const {
          if ( dist2vertex < rhs.dist2vertex ) {
            return true;
          }
          return false;
        };
      };

      std::vector< ShowerProngInfo_t > pronginfo_v;

      int ivtx=0;
      for ( int ivtx=0; ivtx<(int)nuvtx.cluster_v.size(); ivtx++) {
        
        auto const& vtxcluster = nuvtx.cluster_v[ivtx];

        // only deal with tracks        
        if ( vtxcluster.type!=NuVertexCandidate::kShower && vtxcluster.type!=NuVertexCandidate::kShowerKP ) {
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
          ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );

        // get distance of each hit to the vertex
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

        if ( prong.points_v.size()<10 )
          continue;
        
        larflow::reco::cluster_pca( prong );

        // fit line segment
        larlite::track segfit = larflow::reco::ProjectionDefectSplitter::fitLineSegmentToCluster( prong, lfhit_v, adc_v, 2.0 );
        int track_npts = segfit.NumberTrajectoryPoints();

        // which end is closest?
        float enddist[2] = {0,0};
        for (int i=0; i<3; i++) {
          enddist[0] += (nuvtx.pos[i]-segfit.LocationAtPoint(0)[i])*(nuvtx.pos[i]-segfit.LocationAtPoint(0)[i]);
          enddist[1] += (nuvtx.pos[i]-segfit.LocationAtPoint(track_npts-1)[i])*(nuvtx.pos[i]-segfit.LocationAtPoint(track_npts-1)[i]);
        }

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

        // if trunk points back to vertex and is a large enough cluster
        if ( prong.impactdist<20.0 && prong.nhits>10 ) {
          //std::cout << "seed with prong[" << iprong << "]" << std::endl;
          
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
          shower_trunk_dir.add_direction( TVector3(0,0,0) );
          shower_trunk_dir.add_direction( TVector3(0,0,0) );          
          
          // save shower to nuvtx candidate object
          nuvtx.shower_v.emplace_back( std::move(shower) );
          nuvtx.shower_trunk_v.emplace_back( std::move(shower_trunk_dir) );
          
        }//end of if valid seeding prong
      }//end of shower clusters @ vtx
    }//end of nu vertex candidates
  }
  
}
}
