#include "TripletTruthFromSimCh.h"
#include "larlite/DataFormat/simch.h"

namespace larflow {
namespace prep {

  void TripletTruthFromSimCh::process_truth_labels( larlite::storage_manager& ioll,
						    larflow::prep::MatchTriplets& triplets,
						    std::string simch_producer )
  {
    
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower, "mcreco" );
    fill_daughter2mother_map( *ev_mcshower );
    fill_class_map( *ev_mctrack, *ev_mcshower );
    
    make_truth_labels( ioll, triplets, simch_producer );

  }

  /**
   * @brief stores map between daughter geant4 track ids to mother ids
   *
   * Used in conjuction with the 2D instance image labels.
   *
   */
  void TripletTruthFromSimCh::fill_daughter2mother_map( const std::vector<larlite::mcshower>& shower_v )
  {

    _shower_daughter2mother.clear();
    
    for (auto const& shower : shower_v ) {
      long showerid = shower.TrackID();
      if ( showerid<0 ) showerid *= -1;
      const std::vector<unsigned int>& dlist = shower.DaughterTrackID();      
      for (auto const& daughterid : dlist ) {
        _shower_daughter2mother[(unsigned long)daughterid]= (unsigned long)showerid;
      }
    }
    
  }

  /**
   * @brief stores map between instance id and particle class
   *
   * Used in conjuction with the SSNet class labels
   *
   */
  void TripletTruthFromSimCh::fill_class_map( const std::vector<larlite::mctrack>&  track_v,
					      const std::vector<larlite::mcshower>& shower_v )
  {

    _instance2class_map.clear();
    
    for (auto const& shower : shower_v ) {
      long showerid = shower.TrackID();
      if ( showerid<0 ) showerid *= -1;
      int  pid = shower.PdgCode();
      _instance2class_map[(unsigned long)showerid] = pid;
      for ( auto const& daughterid : shower.DaughterTrackID() ) {
        long id = (daughterid<0) ? daughterid*-1 : daughterid;
        _instance2class_map[(unsigned long)id] = pid;
      }
    }
    
    for (auto const& track : track_v ) {
      long trackid = track.TrackID();
      if ( trackid<0 ) trackid *= -1;
      int  pid = track.PdgCode();
      _instance2class_map[(unsigned long)trackid] = pid;
    }

    for (auto it=_instance2class_map.begin(); it!=_instance2class_map.end(); it++ ) {
      int larcv_class = 0;
      switch ( it->second ) {
      case 11:
      case -11:
        larcv_class = (int)larcv::kROIEminus;
        break;
      case 13:
      case -13:
        larcv_class = (int)larcv::kROIMuminus;
        break;
      case 211:
      case -211:
        larcv_class = (int)larcv::kROIPiminus;
        break;
      case 2212:
      case 2112:
        larcv_class = (int)larcv::kROIProton;
        break;
      case 22:
        larcv_class = (int)larcv::kROIGamma;
        break;
      case 111:
        larcv_class = (int)larcv::kROIPizero;
        break;
      case 130:
      case 310:
      case 311:
      case 312:
        larcv_class = (int)larcv::kROIKminus;
        break;
      default:
        larcv_class = (int)larcv::kROIUnknown;
        break;
      }
      it->second = larcv_class;
    }//end of iterator loop
    
  }
  
  void TripletTruthFromSimCh::make_truth_labels( larlite::storage_manager& ioll,
						 larflow::prep::MatchTriplets& triplets,
						 std::string simch_producer )
  {

    voxelizer.clear();
    voxelizer.set_simch_treename( simch_producer );
    voxelizer.process( ioll );

    // we will use the voxelized truth to label the triplets
    ublarcvapp::mctools::SimChannelVoxelizer::TPCInfo& tpcvoxels
      = voxelizer.getTPCInfo( triplets._cryoid, triplets._tpcid );

    size_t ntriplets = triplets._triplet_v.size();
    size_t nlabeled = 0;

    // truth labels to fill
    triplets._truth_v.resize( ntriplets, 0 );
    triplets._instance_id_v.resize( ntriplets, 0 );
    triplets._ancestor_id_v.resize( ntriplets, 0 );
    triplets._pdg_v.resize( ntriplets, 0 );
    triplets._origin_v.resize( ntriplets, 0 );

    for ( size_t itriplet=0; itriplet<triplets._triplet_v.size(); itriplet++ ) {
      float tick = triplets._wirecoord_v[itriplet][3];
      auto const& pos = triplets._pos_v[itriplet];
      auto voxcoord = voxelizer.makeVoxelCoord( tick, pos[1], pos[2], (int)tick+4900, tpcvoxels );
      auto it_vox   = tpcvoxels._voxcoord_2_sparsearrayindex.find( voxcoord );

      if ( it_vox==tpcvoxels._voxcoord_2_sparsearrayindex.end() ) {
	// did not find the voxel
	triplets._truth_v[itriplet] = 0;
	triplets._instance_id_v[itriplet] = 0;
	triplets._ancestor_id_v[itriplet] = 0;
	triplets._pdg_v[itriplet] = 0;
	triplets._origin_v[itriplet] = 0;
      }
      else {
	// get the labels from the voxels
	unsigned long index = it_vox->second;
	std::cout << "voxcoord sparse array index: " << index << std::endl;
	triplets._truth_v[itriplet] = 1;
	triplets._pdg_v[itriplet] = tpcvoxels._pdg_v.data[index];
	triplets._instance_id_v[itriplet] = tpcvoxels._trackid_v.data[index];
	triplets._ancestor_id_v[itriplet] = tpcvoxels._ancestorid_v.data[index];
	//triplets._origin_v[itriplet] = tpcvoxel._origin_v.data[index]; // need to confirm sim filled this.
	triplets._origin_v[itriplet] = 0;
	nlabeled++;
      }
	
    }//end of loop over triplets

    std::cout << "labeled " << nlabeled << " triplets of " << ntriplets << std::endl;
    
    return;
  }
 
}
}
