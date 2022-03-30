#include "PrepProngEmbed.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "TRandom3.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/DataFormat/larflow3dhit.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace spatialembed {

  /**
   * @brief constructor where input files have been specified
   *
   */
  PrepProngEmbed::PrepProngEmbed( const std::vector<std::string>& input_root_files )
    : larcv::larcv_base("PrepProngEmbed")
  {
    for ( auto const& input_file : input_root_files ) {
      _triplet_loader.add_input_file( input_file );
    }
  }
  
  PrepProngEmbed::~PrepProngEmbed()
  {
  }

  void PrepProngEmbed::make_subcluster_fragments()
  {
    auto& tripletdata = _triplet_loader.triplet_v->at(0);

    // we isolate hits for each instance label
    // then we use dbscan to cluster.
    // for track-clusters, we remove relabel spacepoints as bad for small disconnected clusters
    // for hadronic clusters

    std::vector< std::vector<int> > indexlist_vv; // list of indices for each unique cluster
    std::map<int,int> track_to_indexlist_entry;

    int ntriplets = tripletdata._triplet_v.size();
    indexlist_vv.reserve( 200 );
    for (int itrip=0; itrip<ntriplets; itrip++) {
      int origin = tripletdata._origin_v[itrip];
      int truthtrip = tripletdata._truth_v[itrip];

      if ( truthtrip==0 )
	continue;
      if ( origin!=1 )
	continue;
      
      int trackid = tripletdata._instance_id_v[itrip];
      auto it_indexlist = track_to_indexlist_entry.find(trackid);
      int entry = -1;
      if ( it_indexlist==track_to_indexlist_entry.end() ) {
	int newentry = indexlist_vv.size();
	indexlist_vv.push_back( std::vector<int>() );
	track_to_indexlist_entry[trackid] = newentry;
	entry = newentry;
      }
      else {
	entry = it_indexlist->second;
      }
      indexlist_vv[ entry ].push_back( itrip );      
    }

    // now we cluster the spacepoints for each index
    for (auto it=track_to_indexlist_entry.begin(); it!=track_to_indexlist_entry.end(); it++) {
      int trackid = it->first;
      int entry   = it->second;

      std::vector<int>& triplet_idx_v = indexlist_vv.at(entry);

      // are we a track or shower
      std::map<int,int> pdg_count;
      for (auto& origindex : triplet_idx_v ) {
	int pdg = tripletdata._pdg_v[origindex];
	auto it = pdg_count.find(pdg);
	if ( it==pdg_count.end() ) {
	  pdg_count[pdg] = 0;
	}
	pdg_count[pdg]++;
      }
      LARCV_DEBUG() << "track[" << trackid << "] pdg counts ----" << std::endl;
      for ( auto it=pdg_count.begin(); it!=pdg_count.end(); it++ )  {
	LARCV_DEBUG() << " pdg[" << it->first << "] counts=" << it->second << std::endl;
      }
      
      int npoint_removed = 0;      
      if ( triplet_idx_v.size()<20 ) {
	for (auto& origindex : triplet_idx_v ) {
	  tripletdata._origin_v[origindex] = 0;
	  tripletdata._truth_v[origindex] = 0;
	  tripletdata._instance_id_v[origindex] = 0;
	  tripletdata._ancestor_id_v[origindex] = 0;
	  npoint_removed++;
	}
      }
      else {
	// cluster
	std::vector< std::vector<float> > points_v;
	std::vector< int > orig_index_v;
	points_v.reserve( triplet_idx_v.size() );
	orig_index_v.reserve( triplet_idx_v.size() );
	for (auto& idx : triplet_idx_v ) {
	  points_v.push_back( tripletdata._pos_v[idx] );
	  orig_index_v.push_back( idx );
	}
	
	float maxdist = 1.0;
	int minsize = 5;
	int maxkd = 100;
	std::vector< larflow::reco::cluster_t > cluster_v;
	larflow::reco::cluster_sdbscan_spacepoints( points_v, cluster_v, maxdist, minsize, maxkd );

	std::vector<int> keep_v( orig_index_v.size(), 0);
	
	for (int cidx=0; cidx<(int)cluster_v.size(); cidx++) {
	  auto& c = cluster_v[cidx];
	  if ( c.points_v.size()>=20 ) {
	    for ( auto& pidx : c.hitidx_v ) {	    
	      keep_v[ pidx ] = 1;
	    }
	  }
	}

	for ( int pidx=0; pidx<(int)orig_index_v.size(); pidx++) {
	  if ( keep_v[pidx]==0 ) {
	    // remove labels for clusters in the noise or small cluster
	    int origindex = orig_index_v[pidx];
	    tripletdata._origin_v[origindex] = 0;
	    tripletdata._truth_v[origindex] = 0;
	    tripletdata._instance_id_v[origindex] = 0;
	    tripletdata._ancestor_id_v[origindex] = 0;
	    npoint_removed++;
	  }
	}//end of cluster loop
      }
      
      LARCV_DEBUG() << "track[" << trackid << "] removed " << npoint_removed << " spacepoints in small clusters" << std::endl;
      
    }//end of track id's found
    
  }

  
  
}
}
