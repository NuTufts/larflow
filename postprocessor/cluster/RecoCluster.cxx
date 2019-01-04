#include "RecoCluster.h"

#include <map>

#include "TRandom3.h"

#include "TVector3.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"

namespace larflow {

  void RecoCluster::filter_hits(const std::vector<larlite::larflow3dhit>& hits,
				std::vector<larlite::larflow3dhit>& fhits,
				int min_nn,
				float nn_dist,
				float fraction_kept ){

    // we filter out hits that:
    //  (1) without reconstructed 3D position
    //  (2) does not have enough neighbors (noise removal)
    // we also apply random filter to thin out points

    // initialize
    fhits.clear();
    TRandom3 rand( 1234 );

    // kdtree for denoising neighbor search
    std::vector< Eigen::Vector3f > _points;
    _points.reserve( hits.size() );
    std::map<int,int> indexmap; //
    for ( int iorig=0; iorig<(int)hits.size(); iorig++) {
      const std::vector<float>& hit = hits[iorig];
      bool hitok = true;
      for (int i=0; i<3; i++) {
	if ( std::isnan( hit[i] ) ) hitok = false;         // bad value
	if ( hit[0]==hit[1]==hit[2]==-1.0 ) hitok = false; // uninitialized
      }
      if ( hitok ) {
	indexmap[_points.size()] = iorig;

	_points.push_back( Eigen::Vector3f( hit[0], hit[1], hit[2] ) );
      }
    }
    int npoints = _points.size();

    std::cout << "[RecoCluster::filter_hits][INFO] good hit filter accepted " << npoints << " of " << hits.size() << " hits" << std::endl;

    cilantro::KDTree3f tree( _points );

    int nnoise = 0;
    for (int ihit=0; ihit<npoints; ihit++) {

      cilantro::NeighborSet<float> nn;
      tree.radiusSearch( _points[ihit], nn_dist, nn );

      if ( nn.size()<min_nn ) {
	nnoise++;
	continue;
      }

      // subset of detector
      // if ( hit.at(0)>125 || hit.at(2)>300 )
      //        continue;

      if ( fraction_kept==1.0 || rand.Uniform()<fraction_kept ) {
	int iorig = indexmap[ihit];
	fhits.push_back( hits[iorig]  );
      }
    }

    std::cout << "[RecoCluster::filter_hits][INFO] noise and final filter accepted " <<  fhits.size() << " of " << _points.size() << " good hits. num noise hits=" << nnoise  << std::endl;
  }
  

  std::vector< std::vector<larlite::larflow3dhit> > RecoCluster::clusterHits( const std::vector<larlite::larflow3dhit>& hits, std::string algo, bool return_unassigned ){

    std::vector<RecoCluster::Cluster_t> cluster_info;
    if(algo.find("py") != std::string::npos){
      cluster_info = createClustersPy( hits, algo );
    }
    else{
      cluster_info = createClusters( hits, algo );
    }
    Cluster_t unmatched = cluster_info.back(); // last entry is the unmatched hits cluster (at least for DBSCAN)
    cluster_info.pop_back(); // remove last element from list

    std::cout << "Created " << cluster_info.size() << " clusters using " << algo << std::endl;
    std::cout << "Number of unassigned hits: " << unmatched.phits.size() << " of " << hits.size() << std::endl;

    Cluster_t unassigned = assignUnmatchedToClusters( unmatched.phits, cluster_info );
    std::cout << "after NN assignement of unmatched hits, remaining unassigned: " << unassigned.phits.size() << std::endl;
    
    // for the output
    int nclusters = cluster_info.size();
    if ( return_unassigned )
      nclusters++;

    std::vector< std::vector<larlite::larflow3dhit> > output(nclusters);
    
    int icluster = -1;
    for ( auto& cluster : cluster_info ) {
      icluster++;      
      for ( auto& phit : cluster.phits ) {
	output[icluster].push_back( phit );
      }
    }

    if ( return_unassigned ) {
      output.at(nclusters-1) = unassigned.phits;
    }
    
    return output;
  }

  std::vector<RecoCluster::Cluster_t> RecoCluster::createClusters( const std::vector<larlite::larflow3dhit>& hit_v, std::string algo ) {
    std::vector<RecoCluster::Cluster_t> output;
    output.reserve(50);
    Cluster_t noclusterhits;
    
    if(algo=="spectral"){
      larflow::CilantroSpectral sc( hit_v, CilantroSpectral::kMaxDist, _spectral_param.NC, _spectral_param.MaxNN, _spectral_param.MaxDist, _spectral_param.Sigma );
      //larflow::CilantroSpectral sc( hit_v,40,5 );
      std::vector<std::vector<long unsigned int> > cpi;
      std::vector<long unsigned int> idx_mat;
      sc.get_cluster_indeces(cpi,idx_mat); // cpi = < <hit idx in cluster 0> <hit idx in cluster 1> ... <hit idx in cluster cpi.size()-1> >
      for (auto const& cl : cpi ) {
	// create new cluster
	Cluster_t info;
	for(int id = 0; id<cl.size(); id++){
	  info.phits.push_back( hit_v.at(cl[id]) );
	}
	// put into vector
	output.emplace_back( std::move(info) );
      }
    }
    if(algo=="DBSCAN"){
      // translate hits into vector< vector<float> >
      std::vector< std::vector<float> > ptlist(hit_v.size());
      for ( int ihit=0; ihit<(int)hit_v.size(); ihit++ ) {
	auto const& hit = hit_v[ihit];
	ptlist[ihit].resize(3,0);
	for (int i=0; i<3; i++) ptlist[ihit][i] = hit[i];
      }
      larflow::DBSCAN db;
      std::vector< std::vector<int> > dbscan_clusters = db.makeCluster( _dbscan_param.maxdist, _dbscan_param.minhits, _dbscan_param.maxkdneighbors, ptlist );
      for(int i=0; i<dbscan_clusters.size()-1; i++){ //last one is unclustered hits
	// create new cluster
	Cluster_t info;
	for ( auto const& idx : dbscan_clusters[i] ) {
	  info.phits.push_back( hit_v.at(idx) ); // store address
	}
	// put into vector
	output.emplace_back( std::move(info) );
      }
      //store unclustered hits
      for ( auto const& idx : dbscan_clusters[dbscan_clusters.size()-1] ){
	noclusterhits.phits.push_back( hit_v.at(idx) );
      }
    }

    // calculate axis-aligned bounding boxes for clusters
    for ( auto& cluster : output ) {
      // set first location
      for (int i=0; i<3; i++)
	cluster.aabbox[i][0] = cluster.aabbox[i][1] = cluster.phits.front().at(i);

      for ( auto& phit : cluster.phits ) {
	for (int i=0; i<3; i++) {
	  if ( phit.at(i) < cluster.aabbox[i][0] )
	    cluster.aabbox[i][0] = phit.at(i);
	  if ( phit.at(i) > cluster.aabbox[i][1] )
	    cluster.aabbox[i][1] = phit.at(i);
	}
      }
    }

    // append notruthhits cluster to end of cluster vector
    output.emplace_back( std::move(noclusterhits) );
    
    return output;
  }

  
  std::vector<RecoCluster::Cluster_t> RecoCluster::createClustersPy( const std::vector<larlite::larflow3dhit>& hit_v, std::string algo ) {
    std::vector<RecoCluster::Cluster_t> output;
    output.reserve(50);
    
    Cluster_t noclusterhits;

    // ======================================
    // here we need to call python wrapper
    // it calls functions from apply_cluster_algo.py
    //
    //             TO BE DONE
    // ======================================

    // calculate axis-aligned bounding boxes for clusters
    for ( auto& cluster : output ) {
      // set first location
      for (int i=0; i<3; i++)
	cluster.aabbox[i][0] = cluster.aabbox[i][1] = cluster.phits.front().at(i);

      for ( auto& phit : cluster.phits ) {
	for (int i=0; i<3; i++) {
	  if ( phit.at(i) < cluster.aabbox[i][0] )
	    cluster.aabbox[i][0] = phit.at(i);
	  if ( phit.at(i) > cluster.aabbox[i][1] )
	    cluster.aabbox[i][1] = phit.at(i);
	}
      }
    }

    // append notruthhits cluster to end of cluster vector
    output.emplace_back( std::move(noclusterhits) );
    
    return output;
  }

  RecoCluster::Cluster_t RecoCluster::assignUnmatchedToClusters( const std::vector<larlite::larflow3dhit>& unmatchedhit_v, std::vector<Cluster_t>& cluster_v ) {

    Cluster_t unassigned;

    // N^2 algorithm ... hopefully we save some time axis aligned checking
    for ( auto& phit : unmatchedhit_v ) {

      Cluster_t* best_matching_cluster = nullptr;
      float cluster_min_dist = -1;

      for ( auto& cluster : cluster_v ) {

	// calc dist to cluster aabb
	bool checkcluster = false;
	float mindist = -1;
	for (int i=0; i<3; i++) {
	  float dist2minedge = fabs( phit[i] - cluster.aabbox[i][0] );
	  float dist2maxedge = fabs( phit[i] - cluster.aabbox[i][1] );

	  bool isinside = ( phit[i] > cluster.aabbox[i][0] && phit[i] < cluster.aabbox[i][1] );
	  if ( isinside || dist2minedge < 2.0 || dist2maxedge > 2.0 )
	    checkcluster = true;

	  if ( checkcluster )
	    break; // we already know we have to
	}

	if ( checkcluster ) {
	  // aabbox checks out, check the internal hits
	  for ( auto pclusterhit : cluster.phits ) {
	    float dist2 = 0.;
	    for ( int i=0; i<3; i++) {
	      dist2 += (pclusterhit[i] - phit[i])*(pclusterhit[i] - phit[i]);
	    }
	    dist2 = sqrt(dist2);

	    if ( cluster_min_dist<0 || cluster_min_dist > dist2 ) {
	      cluster_min_dist = dist2;
	      best_matching_cluster = &cluster;
	    }
	  }
	}
      }//end of cluster loop

      // do we add the hit somewhere?
      if ( cluster_min_dist>=0 && cluster_min_dist < 3.0 ) {
	best_matching_cluster->phits.push_back( phit );
      }
      else {
	unassigned.phits.push_back( phit );
      }
    }//end of loop over given umatched clusters

    return unassigned;
  }
  
  void RecoCluster::filterLineClusters(std::vector<std::vector<larlite::larflow3dhit> > flowclusters,
				       std::vector<larlite::pcaxis> pcainfos,
				       std::vector<std::vector<std::pair<float,larlite::larflow3dhit> > > PCAdist_sorted_flowhits,
				       std::vector<int> isline){

    // get larlite::GeometryHelper
    auto geo = larutil::GeometryHelper::GetME();
    
    //-----------------------//
    // this returns both a vector of pcainfo for all clusters,
    // and a vector of ints indicating whether the cluster
    // passes the "line cluster" condition or not
    //-----------------------//
    
    pcainfos.clear();
    PCAdist_sorted_flowhits.clear();
    isline.assign(flowclusters.size(),-1);
    int ct=-1;
    for(auto &flowcluster : flowclusters){
      ct ++;
      larflow::CilantroPCA pca( flowcluster );
      larlite::pcaxis pcainfo = pca.getpcaxis();
      pcainfos.push_back( pcainfo );

      const double* eigval = pcainfo.getEigenValues();
      const double*  meanpos = pcainfo.getAvePosition();
      larlite::pcaxis::EigenVectors eigvec = pcainfo.getEigenVectors();

      // meanpos is origin and eigenvec[x,y,z][] is direction
      TVector3 origin(meanpos[0],meanpos[1],meanpos[2]);
      TVector3 direction(eigvec[0][0],eigvec[1][0],eigvec[2][0]);

      //this holds original hit and dist to meanpos along pca axis line
      std::vector<std::pair<float,larlite::larflow3dhit> > pair;

      for(int h=0; h<flowcluster.size(); h++){
	TVector3 point(flowcluster.at(h)[0],flowcluster.at(h)[1],flowcluster.at(h)[2]);
	float pcadist = geo->DistanceAlongLine3D(origin, direction,point);
	pair.push_back(make_pair(pcadist,flowcluster.at(h)));
      }
      //sort by pcadist
      sort(pair.begin(),pair.end());
      //fill sorted larflow cluster
      PCAdist_sorted_flowhits.push_back(pair);
	
      // we look for clusters with primary axis > 1./0.5 secondary axis
      float line_thresh = 1./0.5;
      if( eigval[1]/eigval[1] >= line_thresh) isline[ct] = 1;

      //std::cout << eigval[0] <<" "<< eigval[1] <<" "<< eigval[2] << std::endl;
    }
  }

  void RecoCluster::checkPCAalignment(std::vector<std::vector<std::pair<float,larlite::larflow3dhit> > > sorted_flowclusters,
				      std::vector<larlite::pcaxis> pcainfos,
				      std::vector<int> isline){
    
    //-----------------------------------------------//
    // check if PCA axes of line clusters are aligned
    // first we check 3d dist b/n the two extreme hits of each cluster
    // to each PCA axis parametrized line
    // if similar, then parallel
    //----------------------------------------------//

    // get larlite::GeometryHelper
    auto geo = larutil::GeometryHelper::GetME();

    std::vector<std::vector<int> > parallel_idx(sorted_flowclusters.size(),std::vector<int>(0));
    for(int i=0; i<sorted_flowclusters.size(); i++){
      // skip if not line cluster
      if(isline[i] !=1) continue;
    
      TVector3 end1(sorted_flowclusters.at(i).front().second.at(0),
		    sorted_flowclusters.at(i).front().second.at(1),
		    sorted_flowclusters.at(i).front().second.at(2));
      TVector3 end2(sorted_flowclusters.at(i).back().second.at(0),
		    sorted_flowclusters.at(i).back().second.at(1),
		    sorted_flowclusters.at(i).back().second.at(2));
      

      //now we loop over all other cluster's pca axes
      for(int j=0; j<pcainfos.size(); j++){
	if(j==i) continue; //vectors in same cluster order
	const double*  meanpos = pcainfos.at(j).getAvePosition();
	larlite::pcaxis::EigenVectors eigvec = pcainfos.at(j).getEigenVectors();
	
	// meanpos is origin and eigenvec[x,y,z][] is direction
	TVector3 origin(meanpos[0],meanpos[1],meanpos[2]);
	TVector3 direction(eigvec[0][0],eigvec[1][0],eigvec[2][0]);
	float approach1 = geo->DistanceToLine3D(origin, direction, end1);
	float approach2 = geo->DistanceToLine3D(origin, direction, end2);

	//are we parallel within parallel_thresh?
	float parallel_thresh = 50.0; //in mm (i hope...)
	if(std::abs(approach1 - approach2) <= parallel_thresh)
	  parallel_idx[i].push_back(j);
      }
      
    }
    
  }
}
