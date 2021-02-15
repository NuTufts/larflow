#include "NuVertexShowerTrunkCheck.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"

#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  void NuVertexShowerTrunkCheck::checkNuCandidateProngs( larflow::reco::NuVertexCandidate& nuvtx )
  {

    std::vector<int> trunk_absorbed( nuvtx.track_v.size(), 0 );
    int num_absorbed = 0;
    for (int i=0; i<(int)nuvtx.track_v.size(); i++) {
      auto& track = nuvtx.track_v[i];
      auto& track_hits = nuvtx.track_hitcluster_v[i];

      // get track length
      float tracklen = 0.;
      for (int ipt=0; ipt<(int)track.NumberTrajectoryPoints()-1;ipt++) {
        tracklen += ( track.LocationAtPoint(ipt)-track.LocationAtPoint(ipt+1) ).Mag();
      }

      if ( tracklen<7.0 )
        continue;

      for (int j=0; j<(int)nuvtx.shower_v.size(); j++) {

        if ( trunk_absorbed[i]==1 )
          continue;
        
        auto& shower = nuvtx.shower_v[j];
        auto& shower_trunk = nuvtx.shower_trunk_v[j];
        auto& shower_pca   = nuvtx.shower_pcaxis_v[j];
        float frac_path = 0.;
        float frac_core = 0.;
        bool add_to_shower = isTrackTrunkOfShower( nuvtx.pos,
                                                   track,
                                                   track_hits,
                                                   shower_trunk,
                                                   shower,
                                                   shower_pca,
                                                   frac_path,
                                                   frac_core );

        if ( add_to_shower ) {
          trunk_absorbed[i] = 1;
          num_absorbed++;
          if ( frac_path>0.8 ) {

            _addTrackAsNewTrunk( nuvtx.pos,
                                 track,
                                 track_hits,
                                 shower_trunk,
                                 shower,
                                 shower_pca );
          }
          else if ( frac_core>0.8 )
            _addTrackToCore( nuvtx.pos,
                             track,
                             track_hits,
                             shower_trunk,
                             shower,
                             shower_pca );
        }
      }//end of shower loop
    }//end of track loop

        
    if ( num_absorbed>0 ) {
      LARCV_DEBUG() << "Zero out " << num_absorbed << " tracks" << std::endl;
      // for (int ii=0; ii<(int)nuvtx.track_v.size(); ii ) {
      //   if ( trunk_absorbed[ii]==0 ) {
      //     keep_track_v.emplace_back( std::move(nuvtx.track_v[ii]) );
      //     keep_cluster_v.emplace_back( std::move(nuvtx.track_hitcluster_v[ii]) );
      //   }
      // }
      
      // resort track vector, removing absorbed tracks
      std::vector< larlite::track > keep_track_v;
      std::vector< larlite::larflowcluster > keep_cluster_v;
      keep_track_v.reserve(   (int)nuvtx.track_v.size() - num_absorbed );
      keep_cluster_v.reserve( (int)nuvtx.track_hitcluster_v.size() - num_absorbed );
      for (int ii=0; ii<(int)nuvtx.track_v.size(); ii++ ) {
        if ( trunk_absorbed[ii]==0 ) {
          keep_track_v.emplace_back( std::move(nuvtx.track_v[ii]) );
          keep_cluster_v.emplace_back( std::move(nuvtx.track_hitcluster_v[ii]) );
        }
      }
      std::swap( nuvtx.track_v, keep_track_v );
      std::swap( nuvtx.track_hitcluster_v, keep_cluster_v );
    }
    
  }

  bool NuVertexShowerTrunkCheck::isTrackTrunkOfShower( const std::vector<float>& vtxpos,
                                                       larlite::track& track,
                                                       larlite::larflowcluster& track_hitcluster,
                                                       larlite::track&          shower_trunk,
                                                       larlite::larflowcluster& shower_hitcluster,
                                                       larlite::pcaxis& shower_pcaxis,
                                                       float& frac_path, float& frac_core )
  {

    frac_path = 0.;
    frac_core = 0.;

    // we define a line segment between the vertex and shower
    TVector3 vstart; // vertex position
    for (int i=0; i<3; i++)
      vstart[i] = vtxpos[i];

    // closest point to the shower
    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      dist[iend] = (shower_trunk.LocationAtPoint(iend)-vstart).Mag();
    }

    std::vector<float> shrstart(3,0); // shower trunk start
    if ( dist[0]<dist[1] ) {
      for (int i=0; i<3; i++)
        shrstart[i] = shower_trunk.LocationAtPoint(0)[i];
    }
    else {
      for (int i=0; i<3; i++)
        shrstart[i] = shower_trunk.LocationAtPoint(1)[i];
    }


    std::vector<float> showerdir(3,0);
    float len = 0.;
    float pcalen = 0.;
    float dx = 0.;
    for (int i=0; i<3; i++) {
      showerdir[i] = shrstart[i]-vtxpos[i];
      len += showerdir[i]*showerdir[i];
      dx = (shower_pcaxis.getEigenVectors()[3][i]-shower_pcaxis.getEigenVectors()[4][i]);
      pcalen += dx*dx;
    }
    len = sqrt(len);
    for (int i=0; i<3; i++)
      showerdir[i] /= len;
    pcalen = sqrt(pcalen);

    int nhits_within_startpath = 0.;
    int nhits_within_shower = 0.;
    for (size_t ihit=0; ihit<track_hitcluster.size(); ihit++) {
      auto const& hit = track_hitcluster[ihit];
      float r = larflow::reco::pointLineDistance3f( vtxpos, shrstart, hit );
      float s_start = larflow::reco::pointRayProjection3f( vtxpos, showerdir, hit );
      if ( r<2.0 ) {
        if ( s_start>-2.0 && s_start<len )
          nhits_within_startpath++;
        else if ( s_start>len && s_start<len+pcalen+5.0 )
          nhits_within_shower++;
      }
    }

    frac_path = (float)nhits_within_startpath/(float)track_hitcluster.size();
    frac_core = (float)nhits_within_shower/(float)track_hitcluster.size();    
    LARCV_DEBUG() << " fraction of track hits within cylinder of vtx-to-shower start path: " << frac_path << std::endl;
    LARCV_DEBUG() << " fraction of track hits within cylinder of shower core: " << frac_core << std::endl;    

    if ( frac_path>0.8 || frac_core>0.8 )
      return true;
    
    return false;
  }

  larlite::larflowcluster
  NuVertexShowerTrunkCheck::makeMissingTrunkHits( const std::vector<float>& vtxpos,
                                                  const std::vector<larcv::Image2D>& adc_v,
                                                  larlite::track& shower_trunk,                                                  
                                                  larlite::larflowcluster& shower_hitcluster,
                                                  larlite::pcaxis& shower_pcaxis )
  {
    larlite::larflowcluster added_hit_v;

    // we define a line segment between the vertex and shower
    TVector3 vstart; // vertex position
    for (int i=0; i<3; i++)
      vstart[i] = vtxpos[i];

    // closest point to the shower
    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      dist[iend] = (shower_trunk.LocationAtPoint(iend)-vstart).Mag();
    }

    // define shower trunk start    
    std::vector<float> shrstart(3,0); 
    if ( dist[0]<dist[1] ) {
      for (int i=0; i<3; i++)
        shrstart[i] = shower_trunk.LocationAtPoint(0)[i];
    }
    else {
      for (int i=0; i<3; i++)
        shrstart[i] = shower_trunk.LocationAtPoint(1)[i];
    }

    // define shower dir
    std::vector<double> showerdir(3,0);
    float len = 0.;
    for (int i=0; i<3; i++) {
      showerdir[i] = shrstart[i]-vtxpos[i];
      len += showerdir[i]*showerdir[i];
    }
    len = sqrt(len);
    for (int i=0; i<3; i++)
      showerdir[i] /= len;

    // step through, save unique wire combinations
    const float max_stepsize = 0.3;
    int nsteps = len/max_stepsize+1;
    double stepsize = (double)len/(double)nsteps;

    std::set< std::vector<int> > past_hits;
    auto const& meta = adc_v.front().meta();

    for (int istep=0; istep<nsteps; istep++) {
      std::vector<double> pos(3,0);
      for (int i=0; i<3; i++)
        pos[i] = (double)vtxpos[i] + stepsize*showerdir[i];

      float tick = pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200.0;
      if ( tick<=meta.min_y() || tick>=meta.max_y() )
        continue;
      int row = meta.row(tick,__FILE__,__LINE__);

      int nplanes_w_charge = 0;
      std::vector<int> imgcoord(4,0);
      for (int p=0; p<3; p++) {
        imgcoord[p] = (int)larutil::Geometry::GetME()->WireCoordinate( pos, p );
        int npix = 0;
        for (int dc=-2; dc<=2; dc++) {
          if ( npix>0 )
            break;
          int col = imgcoord[p] + dc;
          if ( col<0 || col>=(int)adc_v[p].meta().cols() )
            continue;
          float pixval = adc_v[p].pixel( row, col, __FILE__, __LINE__ );
          if ( pixval>10.0 )
            npix++;
        }
        if (npix>0)
          nplanes_w_charge++;
      }//end of plane
      
      if ( nplanes_w_charge>=2 ) {
      
        imgcoord[3] = row;
        if ( past_hits.find( imgcoord )==past_hits.end() ) {
          // create a hit
          larlite::larflow3dhit lfhit;
          lfhit.resize(19,0);
          for (int v=0; v<3; v++)
            lfhit[v] = pos[v];
          lfhit.tick = tick;
          lfhit.targetwire.resize(2,0);          
          for (int p=0; p<3; p++)
            lfhit.targetwire[p] = imgcoord[p];
          // fake larmatch score
          lfhit[9] = 1.0; 
          lfhit.track_score = 1.0;
          // ssnet score
          lfhit.renormed_shower_score = 1.0;
          added_hit_v.emplace_back( std::move(lfhit) );
          past_hits.insert( imgcoord );
        }

      }
    }//end of step loop

    
    
    return added_hit_v;
  }

  void NuVertexShowerTrunkCheck::_addTrackAsNewTrunk( const std::vector<float>& vtxpos,
                                                      larlite::track& track,
                                                      larlite::larflowcluster& track_hitcluster,
                                                      larlite::track&          shower_trunk,
                                                      larlite::larflowcluster& shower_hitcluster,
                                                      larlite::pcaxis& shower_pcaxis )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    // we will need to make a cluster in order to update the pca
    // then replace the shower trunk with start and end of track

    larflow::reco::cluster_t cluster;
    cluster.points_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );
    cluster.hitidx_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );

    int iidx = 0;
    for ( auto& hit : shower_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back(iidx);
      iidx++;
    }
    iidx = 0;
    for ( auto& hit : track_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back( -1-iidx );
      iidx++;
    }

    LARCV_DEBUG() << "stored " << cluster.points_v.size() << " combined hits. run pca." << std::endl;
    larflow::reco::cluster_pca( cluster );

    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      for (int i=0; i<3; i++) {
        dist[i] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
      }
    }

    int istart=0;
    int iend = (int)cluster.points_v.size();
    int dii = 1;
    if ( dist[1]<dist[0] ) {
      istart = (int)cluster.points_v.size() - 1;
      iend = 0;
      dii = -1;
    }

    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    LARCV_DEBUG() << "start point transfer: start=" << istart << " iend=" << iend << " dii=" << dii << std::endl;
    
    for (int ii=istart; ii!=iend; ii += dii ) {
      int iidx = cluster.ordered_idx_v[ii];
      int ipt  = cluster.hitidx_v[iidx];      
      // LARCV_DEBUG() << "  [" << ii << "] iidx=" << iidx << " of " << cluster.hitidx_v.size()
      //               << " ipt="
      //               << ipt
      //               << " (of " << shower_hitcluster.size() << " or " << track_hitcluster.size() << ")" 
      //               << std::endl;
      if ( ipt>=0 ) {
        lfcluster.push_back( shower_hitcluster[ipt] );
      }
      else {
        ipt += 1; // remove offset
        ipt *= -1; // make positive
        lfcluster.push_back( track_hitcluster[ipt] );
      }
    }//end of loop over pca order

    LARCV_DEBUG() << "make new trunk" << std::endl;    
    larlite::track new_trunk;
    new_trunk.reserve( 2 );
    new_trunk.add_vertex( track.LocationAtPoint(0) );
    new_trunk.add_direction( track.DirectionAtPoint(0) );
    new_trunk.add_vertex( track.LocationAtPoint( (int)track.NumberTrajectoryPoints()-1 ) );
    new_trunk.add_direction( track.DirectionAtPoint( (int)track.NumberTrajectoryPoints()-1 ) );

    LARCV_DEBUG() << "make new pca" << std::endl;
    larlite::pcaxis new_pca = larflow::reco::cluster_make_pcaxis( cluster );

    LARCV_DEBUG() << "swap new shower objects" << std::endl;        
    std::swap( shower_hitcluster, lfcluster );
    std::swap( shower_trunk, new_trunk );
    std::swap( shower_pcaxis, new_pca );
    
  }

  void NuVertexShowerTrunkCheck::_addTrackToCore(  const std::vector<float>& vtxpos,
                                                   larlite::track& track,
                                                   larlite::larflowcluster& track_hitcluster,
                                                   larlite::track&          shower_trunk,
                                                   larlite::larflowcluster& shower_hitcluster,
                                                   larlite::pcaxis& shower_pcaxis )
  {

    LARCV_DEBUG() << "start" << std::endl;

    larflow::reco::cluster_t cluster;
    cluster.points_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );
    cluster.hitidx_v.reserve( track_hitcluster.size()+shower_hitcluster.size() );

    int iidx = 0;
    for ( auto& hit : shower_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back(iidx);
      iidx++;
    }
    iidx = 0;
    for ( auto& hit : track_hitcluster ) {
      std::vector<float> pt = { hit[0], hit[1], hit[2] };
      cluster.points_v.push_back( pt );
      cluster.hitidx_v.push_back( -1-iidx );
      iidx++;
    }

    larflow::reco::cluster_pca( cluster );

    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      for (int i=0; i<3; i++) {
        dist[i] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
      }
    }        

    int istart=0;
    int iend = (int)cluster.points_v.size();
    int dii = 1;
    if ( dist[1]<dist[0] ) {
      istart = (int)cluster.points_v.size() - 1;
      iend = 0;
      dii = -1;
    }
    
    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );
    
    for (int ii=istart; ii!=iend; ii += dii ) {
      int iidx = cluster.ordered_idx_v[ii];
      int ipt  = cluster.hitidx_v[iidx];
      // LARCV_DEBUG() << "  [" << ii << "] iidx=" << iidx << " of " << cluster.hitidx_v.size()
      //               << " ipt="
      //               << ipt
      //               << " (of " << shower_hitcluster.size() << " or " << track_hitcluster.size() << ")" 
      //               << std::endl;
      if ( ipt>=0 ) {
        lfcluster.push_back( shower_hitcluster[ipt] );
      }
      else {
        ipt++; // remove offset
        ipt *= -1; // make positive
        lfcluster.push_back( track_hitcluster[ipt] );
      }
    }//end of loop over pca order


    larlite::pcaxis new_pca = larflow::reco::cluster_make_pcaxis( cluster );

    std::swap( shower_hitcluster, lfcluster );
    std::swap( shower_pcaxis, new_pca );
    
  }
  
  
}
}
