#include "LassoFlashMatch.h"
#include <iostream>
#include <cstring>
#include <exception>
#include <cmath>

// ROOT
#include "TRandom3.h"

namespace larflow {

  LassoFlashMatch::LassoFlashMatch( int npmts, ScoreType_t scorer, Minimizer_t min_method, bool bundle_flashes )
    : _npmts(npmts),
      _scorer(scorer),
      _min_method(min_method),
      _bundle_flashes(bundle_flashes),
      _cluster_weight(10.0),
      _l1weight(0.1),
      _l2weight(10.0),
      _nmatches(0)
  {};
  
  LassoFlashMatch::~LassoFlashMatch() {
    clear();
  }

  void LassoFlashMatch::clear() {
    _nmatches = 0;
    
  }  

  // ===========================================================================
  // SETUP FLASH/CLUSTER PAIRS
  // --------------------------

  int  LassoFlashMatch::addMatchPair( int iflashidx, int iclustidx, const FlashData_t& flash, const FlashHypo_t& hypo ) {

    int imatchidx = _nmatches;
    if ( _clusterindices.find( iclustidx )==_clusterindices.end() ) {
      _clusterindices.insert( iclustidx );
      int clustergroupidx = _clustergroup_vv.size();
      _clusteridx2group_m[iclustidx] = clustergroupidx;
      _clustergroup_vv.push_back( std::vector<int>() );
    }
    
    if ( _flashindices.find( iflashidx )==_flashindices.end() ) {
      _flashindices.insert( iclustidx );
      int flashgroupidx = _flashgroup_vv.size();
      _flashidx2group_m[iflashidx] = flashgroupidx;
      _flashgroup_vv.push_back( std::vector<int>() );
    }

    // set maps
    _match2clusteridx_v.push_back( iclustidx );
    _clustergroup_vv[ _clusteridx2group_m[iclustidx] ].push_back( imatchidx );
		      
    _match2flashidx_v.push_back( iflashidx );
    _flashgroup_vv[ _flashidx2group_m[iflashidx] ].push_back( imatchidx );
    
    // set flash data
    if ( _flashidx2data_m.find( iflashidx ) == _flashidx2data_m.end() )  {
      int idata = _flashdata_vv.size();
      _flashdata_vv.push_back( flash );
      _flashidx2data_m[iflashidx] = idata;
    }

    // set hypo
    _match_hypo_vv.push_back( std::vector<float>(_npmts,0) );
    float norm = 0.;
    for (int ich=0; ich<_npmts; ich++) {
      _match_hypo_vv[imatchidx][ich] = hypo[ich];
      norm += hypo[ich];
    }
    if ( fabs(norm-1.0)<1.0e-3 ) {
      // is normed
      for (int ich=0; ich<_npmts; ich++) _match_hypo_vv[imatchidx][ich] *= flash.tot;
    }

    // add match parameter
    _fmatch_v.push_back(0.0);

    _nmatches++;
    
    // sanity checks
    if ( _fmatch_v.size()!=_nmatches
	 || _match_hypo_vv.size()!=_nmatches
	 || _match2clusteridx_v.size()!=_nmatches
	 || _match2flashidx_v.size()  !=_nmatches ) {
      std::cout << "fmatch_v: " << _nmatches << " vs " << _fmatch_v.size() << std::endl;
      throw std::runtime_error("[LassoFlashMatch::addMatchPair][ERROR] sanity check filed!!");
    }


    return imatchidx;
  }

  // ===========================================================================
  // SETUP FLASH/CLUSTER PAIRS
  // --------------------------

  void LassoFlashMatch::fitSGD( const int niters, const int niters_per_print, const bool use_sgd, const float matchfrac ) {

    TRandom3 rand(1234);

    float lr = 1.0e-3;

    std::cout << "[LassoFlashMatch::gitSGD][INFO] Start." << std::endl;
    if ( true ) {
      std::vector<int> fmask_v;
      float score  = getTotalScore(fmask_v,false);
      float clustconstraint = calcClusterConstraint(fmask_v);
      float l1norm = calcL1norm( fmask_v );
      float l2norm = calcL2bounds( fmask_v );
      float totscore = score + _cluster_weight*clustconstraint + _l1weight*l1norm + _l2weight*l2norm;
      float fmag = 0.;
      for (int imatch=0; imatch<_nmatches; imatch++)
	fmag += _fmatch_v[imatch];
      std::cout << "[LassoFlashMatch::fitSGD][INFO] Initial results"
		<< " Tot="    << totscore
		<< " || "
		<< " score="  << score
		<< " clust="  << clustconstraint
		<< " l1norm=" << l1norm
		<< " l2norm=" << l2norm
		<< " fmag=" << fmag 
		<< std::endl;
      std::cout << "v: ";
      for (int imatch=0; imatch<_nmatches; imatch++) {
	std::cout << "(" << _fmatch_v[imatch] << ")";
      }
      std::cout << std::endl;
    }
    std::cout << "[LassoFlashMatch::gitSGD][INFO] Run fit." << std::endl;
    
    for (int iiter=0; iiter<niters; iiter++) {
      std::vector<int> fmask_v;
      int nmasked = 0;
      if ( use_sgd ) {
	fmask_v.resize( _nmatches, 0 );
	for ( size_t imatch=0; imatch<_nmatches; imatch++ )  {
	  if ( rand.Uniform()<matchfrac ) fmask_v[imatch] = 1;
	  else nmasked++;
	}
      }

      bool printout = (iiter%niters_per_print==0) && niters_per_print>0;
      updateF_graddesc( lr, fmask_v, printout );
      
      if ( (niters_per_print>0 && printout) || iiter+1==niters ) {
	float score  = getTotalScore(fmask_v,false);
	float clustconstraint = calcClusterConstraint(fmask_v);
	float l1norm = calcL1norm( fmask_v );
	float l2norm = calcL2bounds( fmask_v );
	float totscore = score + _cluster_weight*clustconstraint + _l1weight*l1norm + _l2weight*l2norm;
	float fmag = 0.;
	for (int imatch=0; imatch<_nmatches; imatch++)
	  fmag += _fmatch_v[imatch];
	std::cout << "[LassoFlashMatch::fitSGD][INFO] "
		  << " Tot="    << totscore
		  << " || "
		  << " score="  << score
		  << " clust(" << _cluster_weight << ")="  << clustconstraint
		  << " l1norm(" << _l1weight << ")=" << l1norm
		  << " l2norm(" << _l2weight << ")=" << l2norm
		  << " nmasked=" << nmasked << " of " << _nmatches
		  << " fmag=" << fmag 
		  << std::endl;
	std::cout << "v: ";
	for (int imatch=0; imatch<_nmatches; imatch++) {
	  std::cout << "(" << _fmatch_v[imatch] << ")";	  
 	  // if ( _fmatch_v[imatch]>0.67 ) {
	  //   std::cout << "(" << 1 << ")";
	  // }
	  // else if ( _fmatch_v[imatch]<0.3 )
	  //   std::cout << "(" << 0 << ")";
	  // else 
	  //   std::cout << "(" << 0.5 << ")";
	}
	std::cout << std::endl;
      }
    }
  }

  // =========================================================================
  // Setting the match vector
  // -------------------------
  
  void LassoFlashMatch::setFMatch( int imatch, float fmatch ) {
    _fmatch_v.at(imatch) = fmatch;
  }

  void LassoFlashMatch::setFMatch( int iflashidx, int iclustidx, float fmatch ) {
    auto it = _pair2match_m.find( std::pair<int,int>(iflashidx,iclustidx) );
    if ( it==_pair2match_m.end() )
      throw std::runtime_error("[LassoFlashMatch::setFMatch(flashidx,clusteridx)][ERROR] Pair not registered");
    _fmatch_v[ it->second ] = fmatch;
  }
  
  void LassoFlashMatch::setFMatch( const std::vector<float>& fmatch_v ) {
    if ( fmatch_v.size()!=_nmatches ) {
      throw std::runtime_error( "[LassoFlashMatch::setFMatch][ERROR] number of matches provided does not match expected" );
    }
    memcpy( _fmatch_v.data(), fmatch_v.data(), sizeof(float)*_nmatches );
  }

  // =========================================================================
  // Score calculations
  // --------------------------

  // Total
  // -------
  float LassoFlashMatch::getTotalScore( const std::vector<int>& fmask, bool use_regs ) {
    float score = 0.;
    float clusterconstraint = 0.;
    float L1norm = 0.;
    float L2bounds = 0.;
    bool use_fmask = false;
    if ( fmask.size()==_fmatch_v.size() )
      use_fmask = true;
    else if ( fmask.size()==0 )
      use_fmask = false;
    else
      throw std::runtime_error("[LassoFlashMatch::getTotalScore][ERROR] fmask size must either be 0 (don't use) or number of matches");
      
    
    std::vector<float> score_values;
    
    if ( !_bundle_flashes ) {
      score_values.resize( _nmatches, 0);
      for (int imatch=0; imatch<_nmatches; imatch++) {
	float matchscore = 0.;
	if ( !use_fmask || (use_fmask && fmask[imatch]==1) )
	  matchscore = scoreMatch(imatch);	  
	score_values[imatch] = matchscore;
	score += matchscore*_fmatch_v[imatch];
      }
    }
    else {
      score_values.resize( _flashindices.size(), 0 );
      for (auto const& iflashidx : _flashindices ) {
	float flash_score = scoreFlashBundleMaxDist( iflashidx, fmask ); // need to generalize to different scores later
	score_values[iflashidx] = flash_score;
	score += flash_score;
      }
    }

    if ( use_regs ) {
      clusterconstraint = calcClusterConstraint( fmask );
      L1norm   = calcL1norm( fmask );
      L2bounds = calcL2bounds( fmask );
    }
    
    return score+_cluster_weight*clusterconstraint+_l1weight*L1norm+_l2weight*L2bounds;
  }

  // scores
  // ---------

  float LassoFlashMatch::scoreMatch(int imatch) {
    // we can calculate score without bundling
    switch( _scorer ) {
    case kMaxDist:
      scoreMatchMaxDist( imatch );
      break;
    default:
      throw std::runtime_error("[LassoFlashMatch::scoreMatch][ERROR] score type not defined.");
      break;
    }
    
  }

  float LassoFlashMatch::scoreMatchMaxDist( int imatch ) {
    float data_tot = 0;
    float hypo_tot = 0;
    std::vector<float> data_cdf(_npmts,0);
    std::vector<float> hypo_cdf(_npmts,0);

    int iflashdata = _flashidx2data_m[ _match2flashidx_v[imatch] ];
    const std::vector<float>& flashdata = _flashdata_vv[iflashdata];
    const std::vector<float>& hypo      = _match_hypo_vv[imatch];
    
    for (size_t ich=0; ich<_npmts; ich++) {
      data_tot += flashdata[ich];
      data_cdf[ich] = data_tot;
      hypo_tot += hypo[ich];
      hypo_cdf[ich] = hypo_tot;
    }

    if ( (hypo_tot==0 || data_tot==0) && hypo_tot!=data_tot )
      return 1.0;

    if ( hypo_tot==0 && data_tot==0 )
      return 0.;
    
    float maxdist = 0.;
    for (size_t ich=0; ich<_npmts; ich++) {
      float dist = fabs( data_cdf[ich]/data_tot - hypo_cdf[ich]/hypo_tot );
      if ( dist>maxdist )
	maxdist = dist;
    }
    
    return maxdist;
  }

  float LassoFlashMatch::scoreFlashBundleMaxDist( int flashidx, const std::vector<int>& fmask ) {

    // check if any non-masked imatch indices associated to the flash is being used. otherwise, we return zero
    int nonmasked = 0;
    if ( fmask.size()>0 ) {
      for ( auto const& imatch : _flashgroup_vv[ _flashidx2group_m[flashidx] ] ) {
	if ( fmask[imatch]==1 )
	  nonmasked ++;
      }
      if (nonmasked==0)
	return 0.;
    }
    
    float data_tot = 0;
    float hypo_tot = 0;
    std::vector<float> data_cdf(_npmts,0);
    std::vector<float> hypo_cdf(_npmts,0);

    int iflashdata = _flashidx2data_m[ flashidx ];
    const std::vector<float>& flashdata = _flashdata_vv[iflashdata];
    

    for (size_t ich=0; ich<_npmts; ich++) {
      data_tot += flashdata[ich];
      data_cdf[ich] = data_tot;
      float hypo_ch_pe = 0.;
      for (auto const& imatch : _flashgroup_vv[ _flashidx2group_m[flashidx] ] ) {
	if ( fmask.size()==0 || fmask[imatch]==1 ) {
	  const std::vector<float>& hypo = _match_hypo_vv[imatch];
	  hypo_ch_pe += hypo[ich]*_fmatch_v[imatch];
	}
      }
      hypo_tot += hypo_ch_pe;
      hypo_cdf[ich] = hypo_tot;
    }

    if ( (hypo_tot==0 || data_tot==0) && hypo_tot!=data_tot )
      return 1.0;

    if ( hypo_tot==0 && data_tot==0 )
      return 0.; // should never evaluate -- why would i have an empty flash?
    
    float maxdist = 0.;
    for (size_t ich=0; ich<_npmts; ich++) {
      float dist = fabs( data_cdf[ich]/data_tot - hypo_cdf[ich]/hypo_tot );
      if ( dist>maxdist )
	maxdist = dist;
    }
    
    return maxdist;
  }


  // ----------------
  // Regularizers
  // ----------------

  float LassoFlashMatch::calcClusterConstraint( const std::vector<int>& fmask ) {
    // get the gradient for match pair, imatchidx

    float clusterloss = 0.;

    // we sum over bundles
    for ( auto const& clustergroup : _clustergroup_vv ) {
      float clusternorm = 0.;
      for ( auto const& imatchidx : clustergroup ) {
	if ( fmask.size()==0 || fmask[imatchidx]==1 )
	  clusternorm += _fmatch_v[imatchidx];
      }
      clusternorm -= 1.0;
      clusterloss += clusternorm*clusternorm;
    }

    return clusterloss;

  }

  float LassoFlashMatch::calcL1norm( const std::vector<int>& fmask ) {
    // get the gradient for match pair, imatchidx

    float L1norm = 0.;

    // we sum over bundles
    for ( int imatch=0; imatch<_nmatches; imatch++ ) {
      L1norm += fabs(_fmatch_v[imatch]);
    }
    
    return L1norm;

  }
  
  float LassoFlashMatch::calcL2bounds( const std::vector<int>& fmask ) {
    float L2bounds = 0.;
    for ( size_t imatchidx=0; imatchidx<_fmatch_v.size(); imatchidx++ ) {
      if ( fmask.size()==0 || fmask[imatchidx]==1 ) {
	float f = _fmatch_v[imatchidx];
	if ( f<0 ) L2bounds += f*f;
	else if ( f>1.0 ) L2bounds += (f-1.0)*(f-1.0);
      }
    }
    return L2bounds;
  }
  
  // ---------
  // Gradients
  // ---------

  float LassoFlashMatch::gradCluster( int imatchidx, const std::vector<int>& fmask ) {
    
    // get the gradient for match pair, imatchidx
    int iflashidx   = _match2flashidx_v[imatchidx];
    int iclusteridx = _match2clusteridx_v[imatchidx];

    // need the cluster bundle for this match
    std::vector<int>& imatch_bundle_v = _clustergroup_vv[ _clusteridx2group_m[iclusteridx] ];

    // is this match a part of a cluster bundle with any non-masked entries
    int nonmasked = 0;
    if ( fmask.size()>0 ) {
      for ( auto const& imatch : imatch_bundle_v )
	nonmasked += fmask[imatch];
      // no L1norm to form, grad is zero
      if ( nonmasked==0 )
	return 0;
    }
    
    float cluster_sum = 0.;
    for (auto& bundleimatch : imatch_bundle_v ) {
      if ( fmask.size()==0 || fmask[bundleimatch]==1 )
	cluster_sum += _fmatch_v[bundleimatch];
    }
    cluster_sum -= 1.0;
    
    
    return 2.0*cluster_sum;
  }

  float LassoFlashMatch::gradL1( int imatchidx, const std::vector<int>& fmask ) {
    if ( fmask.size()==0 || fmask[imatchidx]==1 ) {
      if ( _fmatch_v[imatchidx]>0 )
	return 1.0;
      else if ( _fmatch_v[imatchidx]<0 )
	return -1.0;
      else
	return 0.;
    }
    else
      return 0;
  }
  
  std::vector<float> LassoFlashMatch::get_gradCluster_df( const std::vector<int>& fmask ) {
    std::vector<float> gradcluster_v( _fmatch_v.size(), 0 );
    for (size_t imatchidx=0; imatchidx<_fmatch_v.size(); imatchidx++) {
      gradcluster_v[imatchidx] = gradCluster( imatchidx, fmask );
    }
    return gradcluster_v;
  }

  std::vector<float> LassoFlashMatch::get_gradL1_df( const std::vector<int>& fmask ) {
    std::vector<float> gradL1_v( _fmatch_v.size(), 0 );
    for (size_t imatchidx=0; imatchidx<_fmatch_v.size(); imatchidx++) {
      gradL1_v[imatchidx] = gradL1( imatchidx, fmask );
    }
    return gradL1_v;
  }
  
  std::vector<float> LassoFlashMatch::get_gradL2_df( const std::vector<int>& fmask ) {
    std::vector<float> gradL2( _fmatch_v.size(), 0 );
    for (size_t imatchidx=0; imatchidx<_fmatch_v.size(); imatchidx++) {
      float f = _fmatch_v[imatchidx];
      if ( fmask.size()==0 || fmask[imatchidx]==1 ) {
	if ( f<0 )
	  gradL2[imatchidx] = 2.0*f;
	else if ( f>1.0 )
	  gradL2[imatchidx] = 2.0*(f-1.0);
	else
	  gradL2[imatchidx] = 0.0;
      }
    }
    return gradL2;
  }

  std::vector<float> LassoFlashMatch::getScoreGrad( const std::vector<int>& fmask ) {
    std::vector<float> scoregrad( _fmatch_v.size(), 0 );

    if ( !_bundle_flashes ) {
      for (int imatch=0; imatch<_nmatches; imatch++) {
	float matchscore = 0.;
	if ( fmask.size()==0 || fmask[imatch]==1 )
	  matchscore = scoreMatch(imatch);
	scoregrad[imatch] = matchscore;
      }
    }
    else {
      for (auto const& iflashidx : _flashindices ) {
	float flash_score = scoreFlashBundleMaxDist( iflashidx, fmask ); // need to generalize to different scores later
	scoregrad[iflashidx] = flash_score;
      }
    }
    return scoregrad;
  }

  // ==================================================================================
  // Update rule for gradient descent fit
  // -------------------------------------

  void LassoFlashMatch::updateF_graddesc( float lr, const std::vector<int>& fmask, bool print ) {
    std::vector<float> scoregrad = getScoreGrad(  fmask );
    std::vector<float> clustgrad = get_gradCluster_df( fmask );
    std::vector<float> l1grad    = get_gradL1_df( fmask );
    std::vector<float> l2grad    = get_gradL2_df( fmask );

    std::vector<float> df(_nmatches,0);
    float gradmag = 0.;
    for ( size_t imatch=0; imatch<_nmatches; imatch++ ) {
      df[imatch] = scoregrad[imatch] + _cluster_weight*clustgrad[imatch] + _l1weight*l1grad[imatch] + _l2weight*l2grad[imatch];
      gradmag += df[imatch]*df[imatch];
    }
    //std::cout << " |grad|=" << sqrt(gradmag) << std::endl;

    for ( size_t imatch=0; imatch<_nmatches; imatch++ ) {
      _fmatch_v[imatch] -= lr*df[imatch];
    }
  }
  
  

}
