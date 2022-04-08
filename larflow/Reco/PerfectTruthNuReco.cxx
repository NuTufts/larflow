#include "PerfectTruthNuReco.h"

#include "larlite/LArUtil/LArProperties.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/TruthTrackSCE.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "TrackdQdx.h"

#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  PerfectTruthNuReco::PerfectTruthNuReco()
    : larcv::larcv_base("PerfectTruthNuReco")
  {
    _psce = new larutil::SpaceChargeMicroBooNE;
  }

  PerfectTruthNuReco::~PerfectTruthNuReco()
  {
    delete _psce;
    _psce = nullptr;
  }
  
  NuVertexCandidate
  PerfectTruthNuReco::makeNuVertex( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();

    larcv::EventImage2D* ev_instance
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "instance" );

    larlite::event_larflow3dhit* ev_lm
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "taggerfilterhit" );

    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data( larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );



    
    std::vector<int> used_v( ev_lm->size(), 0 );
    // get tracks
    NuVertexCandidate nuvtx;
    makeTracks( nuvtx, *ev_mctrack, *ev_lm, adc_v, used_v );
    makeShowers(  nuvtx, *ev_mcshower, *ev_lm, adc_v, ev_instance->as_vector(), used_v );

    nuvtx.keypoint_producer = "perfectreco";
    nuvtx.keypoint_index = 0;
    nuvtx.pos = ublarcvapp::mctools::NeutrinoVertex::getPos3DwSCE( ioll, _psce );
    nuvtx.tick = 3200 + nuvtx.pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    
    return nuvtx;
  }


  void PerfectTruthNuReco::makeTracks( NuVertexCandidate& nuvtx,
                                       const larlite::event_mctrack& ev_mctrack,
                                       const larlite::event_larflow3dhit& ev_lm,
                                       const std::vector<larcv::Image2D>& adc_v,
                                       std::vector<int>& used_v  )
  {

    LARCV_DEBUG() << "Make Tracks" << std::endl;    
    ublarcvapp::mctools::TruthTrackSCE track_convertor( _psce );
    track_convertor.set_verbosity( larcv::msg::kDEBUG );
    TrackdQdx dqdxalgo;
    
    for ( auto const& track : ev_mctrack ) {

      if ( track.Origin()!=1 ) {
        //not neutrino
        continue;
      }
      
      larlite::track sce_track = track_convertor.applySCE( track );

      int npts = sce_track.NumberTrajectoryPoints();
      
      // now we assign hits
      larlite::larflowcluster track_cluster;
      track_cluster.reserve( ev_lm.size()/10+1 );
      
      for ( size_t idx=0; idx<ev_lm.size(); idx++ ) {
        if (used_v[idx]!=0 )
          continue;
        auto const& hit = ev_lm[idx];
        float dist = 1e9;
        int min_step = -1;
        std::vector<float> pt = { hit[0], hit[1], hit[2] };
        track_convertor.dist2track( pt, sce_track, dist, min_step );
        if ( dist<1.5 && (min_step>=0 && min_step<npts) ) {
          used_v[idx]=1;
          track_cluster.push_back( hit );
        }
      }//end of hit loop

      // dqdx calculation
      larlite::track dqdx_track = dqdxalgo.calculatedQdx( sce_track, track_cluster, adc_v );

      if ( track_cluster.size()>=5 ) {      
        nuvtx.track_v.emplace_back( std::move(dqdx_track) );
        nuvtx.track_hitcluster_v.emplace_back( std::move(track_cluster) );
      }
    }
    
  }

  /**
   * 
   * @brief constructs clusters of larmatch hits for showers using truth information
   *
   */
  void PerfectTruthNuReco::makeShowers( NuVertexCandidate& nuvtx,
                                        const larlite::event_mcshower& ev_mcshower,
                                        const larlite::event_larflow3dhit& ev_lm,
                                        const std::vector<larcv::Image2D>& adc_v,
                                        const std::vector<larcv::Image2D>& instance_v,
                                        std::vector<int>& used_v  )
  {

    const float ar_molier_rad_cm = 9.04;
    const float ar_rad_length_cm = 14.3;

    // algorithm for dq/dx analysis
    TrackdQdx dqdxalgo;

    // first make shower daughter 2 mother map, use mcpixelpgraph as help
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg._fill_shower_daughter2mother_map( ev_mcshower );


    std::map<int,int> id2index;
    std::vector< larlite::larflowcluster > shower_v; // container for constructed shower clusters
    std::vector< larlite::track > shower_trunk_v; // container for associated trunk for each shower
    std::vector< larlite::pcaxis > shower_pca_v;  // container for PCA info for each shower

    // loop over shower objects in the MC truth
    // we will construct the shower trunk objects in this loop
    // the trunk line segment will be space-charge corrected
    for ( auto const& shower : ev_mcshower ) {

      // we do not make objects for non-neutrino showers (e.g. cosmics)
      if ( shower.Origin()!=1 ) {
        //not neutrino
        continue;
      }

      // Get the shower det profile object:
      // it is a better guide for where shower actually started
      auto const& profile = shower.DetProfile();
      //float pmom = profile.Momentum().Vect().Mag();
      //TVector3 dir = profile.Momentum().Vect();
      TVector3 dir = shower.Start().Momentum().Vect();
      float pmom = dir.Mag();
      TVector3 vstart = shower.Start().Position().Vect();
      TVector3 pstart = shower.DetProfile().Position().Vect();

      LARCV_DEBUG() << "shower start, mcstep[0]: (" << vstart[0] << "," << vstart[1] << "," << vstart[2] << ")" << std::endl;
      LARCV_DEBUG() << "shower start, profile: pstart.Mag()=" << pstart.Mag()
		    << " (" << pstart[0] << "," << pstart[1] << "," << pstart[2] << ")"
		    << std::endl;

      // check if profile is invalid. means no energy deposited inside tpc
      if (pstart[0]>=larlite::data::kINVALID_DOUBLE
	  || pstart[1]>=larlite::data::kINVALID_DOUBLE
	  || pstart[2]>=larlite::data::kINVALID_DOUBLE ) {
	LARCV_DEBUG() << "invalid profile. skip this shower. " << std::endl;
	continue;
      }
      
      if ( shower.PdgCode()==22 ) {
        LARCV_DEBUG() << "For gamma, use profile" << std::endl;
        vstart = pstart;
        dir = profile.Momentum().Vect();
        pmom = dir.Mag();
      }

      // if shower momentum is too small, skip it
      if ( dir.Mag()<0.1 )
        continue;

      // we define the direction with a line segment from start + 5 cm in true direction
      // we will then apply space charge corrections to this line
      std::vector<float> fdir(3,0);
      std::vector<float> fstart(3,0);
      std::vector<float> fend(3,0);
      TVector3 vend;
      for (int i=0; i<3; i++) {
        dir[i] /= pmom;
        fdir[i] = (float)dir[i];
        fstart[i] = vstart[i];
        fend[i] = fstart[i] + 5.0*fdir[i];
        vend[i] = fend[i];
      }

      // space charge correction if not gamma
      //_psce;
      TVector3 v3end = { vend[0], vend[1], vend[2] };
      
      if ( shower.PdgCode()!=22 ) {
        std::vector<double> s_offset = _psce->GetPosOffsets(vstart[0],vstart[1],vstart[2]);
        vstart[0] = fstart[0] - s_offset[0] + 0.7;
        vstart[1] = fstart[1] + s_offset[1];
        vstart[2] = fstart[2] + s_offset[2];

        std::vector<double> e_offset = _psce->GetPosOffsets(v3end[0],v3end[1],v3end[2]);
        v3end[0] = vend[0] - e_offset[0] + 0.7;
        v3end[1] = vend[1] + e_offset[1];
        v3end[2] = vend[2] + e_offset[2];
      }

      TVector3 sce_dir = v3end-vstart;
      float sce_dir_len = sce_dir.Mag();
      if ( sce_dir_len>0 ) {
        for (int i=0; i<3; i++)
          sce_dir[i] /= sce_dir_len;
      }      

      // make shower trunk object
      larlite::track trunk;
      trunk.reserve(2);
      trunk.add_vertex( vstart );
      trunk.add_vertex( v3end );
      trunk.add_direction( sce_dir );
      trunk.add_direction( sce_dir );

      // make shower cluster object
      larlite::larflowcluster shower_cluster;

      // make empty pca 
      larlite::pcaxis pca;

      shower_v.emplace_back( std::move(shower_cluster) );
      shower_trunk_v.emplace_back( std::move(trunk) );
      shower_pca_v.emplace_back( std::move(pca) );
      id2index[ shower.TrackID() ] = (int)shower_v.size()-1;

    }//end of shower loop, making trunks

    // now we associate larmatch hits to the different true showers
    for ( size_t idx=0; idx<ev_lm.size(); idx++ ) {

      // check if we've already used this hit
      if (used_v[idx]!=0 )
        continue;
      
      auto const& hit = ev_lm[idx];

      // does hit fall within instance image
      if ( hit.tick<=instance_v[0].meta().min_y() || hit.tick>=instance_v[0].meta().max_y() )
        continue;
      int row = instance_v[0].meta().row(hit.tick);

      // assign hit by checking instance map
      std::map<int,int> id_votes;
      for (int p=0; p<(int)hit.targetwire.size(); p++) {
        auto const& meta = instance_v[p].meta();
        if ( hit.targetwire[p]<=meta.min_x() || hit.targetwire[p]>=meta.max_x() )
          continue;
        int col = meta.col( hit.targetwire[p] );
        float pixval = 0.0;
	int iid = 0;
	try {
	  adc_v[p].pixel( row, col, __FILE__, __LINE__ );
	  iid = instance_v[p].pixel( row, col, __FILE__, __LINE__ );
	}
	catch (...) {
	  continue;
	}
        auto it_vote = id_votes.find(iid);
        if ( it_vote==id_votes.end() ) {
          id_votes[iid] = 0;
          it_vote = id_votes.find(iid);
        }
        it_vote->second++;            
      }
      
      int maxvote_id = 0;
      int maxvote_tot = 0;
      for ( auto it=id_votes.begin(); it!=id_votes.end(); it++ ) {
        if ( it->second>maxvote_tot ) {
          maxvote_tot = it->second;
          maxvote_id  = it->first;
        }
      }
      
      if ( maxvote_tot>0 && maxvote_id>0 ) {
        auto it_showerid = mcpg._shower_daughter2mother.find( maxvote_id );
        if ( it_showerid!=mcpg._shower_daughter2mother.end() ) {
          int showerid = it_showerid->second;
          auto it_shower = id2index.find( showerid );
          if ( it_shower!=id2index.end() ) {              
            shower_v[ it_shower->second ].push_back( hit );
            used_v[idx] = 1;
            }
        }
      }        
    }//end of hit loop

    for (size_t ishower=0; ishower<shower_v.size(); ishower++ ) {
      auto& shower_cluster = shower_v[ishower];
    
      if ( shower_cluster.size()>5 ) {

        auto& trunk = shower_trunk_v.at(ishower);

        larflow::reco::cluster_t clust = larflow::reco::cluster_from_larflowcluster( shower_cluster );
        larlite::pcaxis pca = larflow::reco::cluster_make_pcaxis( clust );
      
        //larlite::track dqdx_trunk = dqdxalgo.calculatedQdx( trunk, shower_cluster, adc_v );
        nuvtx.shower_v.emplace_back( std::move(shower_cluster) );
        nuvtx.shower_trunk_v.emplace_back( std::move(trunk) );
        nuvtx.shower_pcaxis_v.emplace_back( std::move(pca) );
        
      }
      
    }//end of mcshower loop

  }
  
}
}
