#include "LassoFlashMatch.h"
#include <iostream>
#include <cstring>
#include <exception>
#include <cmath>
#include <assert.h>

// ROOT
#include "TRandom3.h"

namespace larflow {

  LassoFlashMatch::LassoFlashMatch( const int npmts, ScoreType_t scorer, Minimizer_t min_method, bool bundle_flashes, bool use_b_terms )
    : _npmts(npmts),
      _scorer(scorer),
      _min_method(min_method),
      _bundle_flashes(bundle_flashes),
      _use_b_terms(use_b_terms),
      _cluster_weight(1e2),
      _l1weight(1e2),
      _l2weight(1e4),
      _bweight(1e3),
      _nmatches(0)
  {}
  
  LassoFlashMatch::~LassoFlashMatch() {
    clear();
  }

  void LassoFlashMatch::clear() {
    _nmatches = 0;

    // book-keeping vars
    _clusterindices.clear();
    _clustergroup_vv.clear();
    _clusteridx2group_m.clear();
    _match2clusteridx_v.clear();
    _match_hypo_vv.clear();
    _flashindices.clear();
    _flashdata_vv.clear();
    _flashidx2data_m.clear();
    _match2flashidx_v.clear();
    _flashgroup_vv.clear();
    _flashidx2group_m.clear();
    _fmatch_v.clear();
    _bpmt_vv.clear();
    _pair2match_m.clear();

    // learning schedule vars
    _learning_v.clear();
    _iter2learning.clear();
    
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
      _flashindices.insert( iflashidx );
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
      std::vector< float > b_v( flash.size(), 0 );
      _bpmt_vv.emplace_back( std::move(b_v) );
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
	 || _match2flashidx_v.size()  !=_nmatches
	 || _flashdata_vv.size()!=_bpmt_vv.size() )
      {
	std::cout << "fmatch_v: " << _nmatches << " vs " << _fmatch_v.size() << std::endl;
	throw std::runtime_error("[LassoFlashMatch::addMatchPair][ERROR] sanity check filed!!");
      }


    return imatchidx;
  }

  // ===========================================================================
  // SETUP FLASH/CLUSTER PAIRS
  // --------------------------

  LassoFlashMatch::Result_t LassoFlashMatch::fitSGD( const int niters, const int niters_per_print, const bool use_sgd, const float matchfrac ) {

    TRandom3 rand(1234);

    float lr = 1.0e-3;

    std::cout << "[LassoFlashMatch::gitSGD][INFO] Start." << std::endl;
    if ( true ) {
      std::vector<int> fmask_v;
      Result_t initeval = eval( true );
      float fmag = 0.;
      for (int imatch=0; imatch<_nmatches; imatch++)
	fmag += _fmatch_v[imatch];
      std::cout << "[LassoFlashMatch::fitSGD][INFO] Initial results"
		<< " Tot="    << initeval.totscore
		<< " || "
		<< " score="  << initeval.score
		<< " clust="  << initeval.cluster_constraint
		<< " l1norm=" << initeval.L1norm
		<< " l2norm=" << initeval.L2bounds
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

      // retrieve config
      LearningConfig_t config;
      config.lr = lr; // default
      config.matchfrac = matchfrac; // default from arguments
      config.use_sgd = use_sgd;  // default from arguments
      if ( _learning_v.size()>0 ) {
	config = getLearningConfig( iiter );
      }
      
      std::vector<int> fmask_v;
      int nmasked = 0;
      if ( config.use_sgd ) {
	fmask_v.resize( _nmatches, 0 );
	for ( size_t imatch=0; imatch<_nmatches; imatch++ )  {
	  if ( rand.Uniform()<config.matchfrac ) fmask_v[imatch] = 1;
	  else nmasked++;
	}
      }

      bool printout = (iiter%niters_per_print==0) && niters_per_print>0;
      updateF_graddesc( config.lr, fmask_v, printout );
      
      if ( niters_per_print>0 && printout ) {
	Result_t out = eval(true);
	float fmag = 0.;
	for (int imatch=0; imatch<_nmatches; imatch++)
	  fmag += _fmatch_v[imatch];
	std::cout << "[LassoFlashMatch::fitSGD][INFO] "
		  << " Tot="    << out.totscore
		  << " || "
		  << " score="  << out.score
		  << " clust(" << _cluster_weight << ")="  << out.cluster_constraint
		  << " l1norm(" << _l1weight << ")=" << out.L1norm
		  << " l2norm(" << _l2weight << ")=" << out.L2bounds
		  << " nmasked=" << nmasked << " of " << _nmatches
		  << " lr=" << config.lr
		  << " use_sgd(" << config.matchfrac << ")=" << config.use_sgd
		  << " fmag=" << fmag 
		  << std::endl;
	std::cout << "v: ";
	for (int imatch=0; imatch<_nmatches; imatch++)
	  std::cout << "(" << _fmatch_v[imatch] << ")";	  
	std::cout << std::endl;
      }
    }//iter loop

    return eval(true);
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

  void LassoFlashMatch::setUseBterms( bool use_b_terms ) {
    _use_b_terms = use_b_terms;
    if ( use_b_terms ) {
      // make sure we setup the _bpmt_vv data member
      _bpmt_vv.resize( _flashdata_vv.size() );
      for ( size_t iflash=0; iflash<_flashdata_vv.size(); iflash++ ) {
	_bpmt_vv[iflash].resize( _flashdata_vv[iflash].size(), 0.0 );
      }
    }
  }

  // =========================================================================
  // Score calculations
  // --------------------------

  LassoFlashMatch::Result_t LassoFlashMatch::eval( bool isweighted ) {
    
    // for output purposes
    Result_t out;
    out.isweighted = isweighted;

    std::vector<int> fmask_v;
    out.totscore = getTotalScore( fmask_v, true,  isweighted );
    out.score    = getTotalScore( fmask_v, false, isweighted );
    out.cluster_constraint = calcClusterConstraint( fmask_v );
    out.L1norm   = calcL1norm( fmask_v );
    out.L2bounds = calcL2bounds( fmask_v );
    if ( _use_b_terms )
      out.b2loss   = calcBloss();
    else
      out.b2loss = 0;

    if ( isweighted ) {
      out.cluster_constraint *= _cluster_weight;
      out.L1norm   *= _l1weight;
      out.L2bounds *= _l2weight;
      if ( _use_b_terms )
	out.b2loss   *= _bweight;
    }

    return out;
    
  };

  // Total Score
  // ------------
    float LassoFlashMatch::getTotalScore( const std::vector<int>& fmask, bool use_regs, bool use_weights ) {
    float score = 0.;
    float clusterconstraint = 0.;
    float L1norm = 0.;
    float L2bounds = 0.;
    float b2loss = 0.;
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
	float flash_score = 0.;
	switch (_scorer) {
	case kMaxDist:
	  flash_score = scoreFlashBundleMaxDist( iflashidx, fmask );
	  break;
	case kNLL:
	  flash_score = scoreFlashBundlePoissonNLL( iflashidx, fmask );
	  break;
	default:
	  assert(false);
	  break;
	};
	score_values[iflashidx] = flash_score;
	score += flash_score;
      }
    }

    if ( use_regs ) {
      clusterconstraint = calcClusterConstraint( fmask );
      L1norm   = calcL1norm( fmask );
      L2bounds = calcL2bounds( fmask );
      b2loss = calcBloss();
    }

    if ( use_weights )
      return score+_cluster_weight*clusterconstraint+_l1weight*L1norm+_l2weight*L2bounds+_bweight*b2loss;
    else
      return score+clusterconstraint+L1norm+L2bounds+b2loss;
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


  float LassoFlashMatch::scoreFlashBundlePoissonNLL( int flashidx, const std::vector<int>& fmask ) {
    
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

    // get data flash
    int iflashdata = _flashidx2data_m[ flashidx ];
    const std::vector<float>& flashdata = _flashdata_vv[iflashdata];
    const std::vector<float>* pb_v = nullptr;
    if ( _use_b_terms ) {
      pb_v = &(_bpmt_vv.at(iflashdata));
    }
    
    // first need to form hypothesis total
    std::vector<float> hypo(_npmts,0);

    // and apply data correction from b-terms
    std::vector<float> data(_npmts,0);
    
    for (size_t ich=0; ich<_npmts; ich++) {

      // hypo
      float hypo_ch_pe = 0.;
      for (auto const& imatch : _flashgroup_vv[ _flashidx2group_m[flashidx] ] ) {
	if ( fmask.size()==0 || fmask[imatch]==1 ) {
	  const std::vector<float>& hypo = _match_hypo_vv[imatch];
	  hypo_ch_pe += hypo[ich]*_fmatch_v[imatch];
	}
      }
      hypo[ich] = hypo_ch_pe;

      // data
      data[ich] = flashdata[ich];
      if ( _use_b_terms ) {
	data[ich] *= (1-(*pb_v)[ich]);
	if ( data[ich]<0 ) data[ich] = 0;
      }
      
    }
    
    // now get neg log-likelihood from poisson ratio
    float nll = 0;
    for (size_t ich=0; ich<_npmts; ich++) {
      float obs  = data[ich];
      float pred = hypo[ich];
      if ( pred<1.0e-3 )
	pred = 1.0e-3;
      float nll_bin = (pred-obs);
      if ( obs>0 )
	nll_bin += obs*(log(obs)-log(pred));
      nll += 2.0*nll_bin;
    }
    
    return nll;
  }
  
  
  // ----------------
  // Regularizers
  // ----------------

  float LassoFlashMatch::calcClusterConstraint( const std::vector<int>& fmask ) {
    // constraint that cluster matches to only one flash

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
    // sparsity constraint

    float L1norm = 0.;

    // we sum over bundles
    for ( int imatch=0; imatch<_nmatches; imatch++ ) {
      L1norm += fabs(_fmatch_v[imatch]);
    }
    
    return L1norm;

  }
  
  float LassoFlashMatch::calcL2bounds( const std::vector<int>& fmask ) {
    // soft-ish bounds to keep values between (0,1)
    
    float L2bounds = 0.;
    for ( size_t imatchidx=0; imatchidx<_fmatch_v.size(); imatchidx++ ) {
      if ( fmask.size()==0 || fmask[imatchidx]==1 ) {
	float f = _fmatch_v[imatchidx];
	if ( f<0 ) L2bounds += f*f;
	else if ( f>1.0 ) L2bounds += (f-1.5)*(f-1.5);
      }
    }
    return L2bounds;
  }

  float LassoFlashMatch::calcBloss() {
    float l2loss = 0;

    for ( size_t idata=0; idata<_flashdata_vv.size(); idata++ ) {
      const std::vector<float>& b_v = _bpmt_vv[idata];
      for ( auto const& b : b_v )
	l2loss += b*b;
    }

    return l2loss;
  }
  
  // ---------
  // Gradients
  // ---------

  LassoFlashMatch::Grad_t LassoFlashMatch::evalGrad( bool isweighted ) {
    
    Grad_t out;
    
    out.isweighted = isweighted;
    std::vector<int> fmask_v;
    out.score    = getScoreGrad( fmask_v, out.score_gradb );
    out.cluster_constraint = get_gradCluster_df( fmask_v );
    out.L1norm   = get_gradL1_df( fmask_v );
    out.L2bounds = get_gradL2_df( fmask_v );
    out.b2loss   = gradBloss();

    // total grad for fmatch parameters
    out.totgrad.resize( _nmatches, 0 );
    for (size_t imatch=0; imatch<_nmatches; imatch++) {
      if ( isweighted )
	out.totgrad[imatch] = out.score[imatch] + _cluster_weight*out.cluster_constraint[imatch] + _l1weight*out.L1norm[imatch] + _l2weight*out.L2bounds[imatch];
      else
	out.totgrad[imatch] = out.score[imatch] + out.cluster_constraint[imatch] + out.L1norm[imatch] + out.L2bounds[imatch];	
    }
    
    return out;
  }

  std::vector<float> LassoFlashMatch::getScoreGrad( const std::vector<int>& fmask, std::vector< std::vector<float> >& gradb_vv ) {
    switch (_scorer) {
    case kMaxDist:
      return gradMaxDist( fmask );
      break;
    case kNLL:
      return gradFlashBundleNLL( fmask, gradb_vv );
      break;
    };
    // never gets here
    std::vector<float> dummy;
    return dummy;
  }
  
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

  std::vector<float> LassoFlashMatch::gradMaxDist( const std::vector<int>& fmask ) {
    std::vector<float> scoregrad( _fmatch_v.size(), 0 );

    if ( !_bundle_flashes ) {
      for (int imatch=0; imatch<_nmatches; imatch++) {
	float matchscore = 0.;
	if ( fmask.size()==0 || fmask[imatch]==1 )
	  matchscore = scoreMatch(imatch); // almost certainly wrong
	scoregrad[imatch] = matchscore;
      }
    }
    else {
      for (auto const& iflashidx : _flashindices ) {
	float flash_score = scoreFlashBundleMaxDist( iflashidx, fmask ); // definitely wrong
	scoregrad[iflashidx] = flash_score;
      }
    }
    return scoregrad;
  }

  std::vector<float> LassoFlashMatch::gradFlashBundleNLL( const std::vector<int>& fmask, std::vector< std::vector<float> >& gradb_vv ) {

    std::vector<float> grad(_nmatches,0);
    if ( _use_b_terms ) {
      gradb_vv.resize( _flashindices.size() );
    }
    
    // do this by flashbundle
    for ( auto const& iflashidx : _flashindices ) {
      std::vector<int>& flashgroup  = _flashgroup_vv[ _flashidx2group_m[iflashidx] ];

      // need the overall prediction for each 
      std::vector<float> hypotot(_npmts,0);
      for (auto& imatchidx : flashgroup ) {
	for (int ich=0; ich<_npmts; ich++) {
	  hypotot[ich] += _match_hypo_vv[imatchidx][ich]*_fmatch_v[imatchidx];
	  if ( hypotot[ich]<1.0e-3 ) hypotot[ich] = 1.0e-3;
	}
      }
      
      // also need data
      const std::vector<float>& flashdata = _flashdata_vv[ _flashidx2data_m[iflashidx] ];
      std::vector<float> databmod( flashdata );  // a copy

      // grab the output b-vector if requested
      std::vector<float>* pb_v = nullptr;
      std::vector<float>* pgradb = nullptr;
      if ( _use_b_terms ) {
	pgradb = &( gradb_vv.at( _flashidx2data_m[iflashidx] ) );
	pgradb->resize( flashdata.size(), 0.0 );
	pb_v = &_bpmt_vv.at( _flashidx2data_m[iflashidx] );
	for ( size_t ich=0; ich<flashdata.size(); ich++ ) {
	  databmod[ich] *= (1-(*pb_v)[ich]);
	  if ( databmod[ich]<0 ) databmod[ich] = 0.;
	}
      }

      // grad with respect to match parameter      
      for ( auto& imatchidx : flashgroup ) {

	float dnlldf = 0.;
	std::vector<float> hypo = _match_hypo_vv[imatchidx];
	for (int ich=0; ich<_npmts; ich++) {
	  float obs = databmod[ich];
	  if ( obs>0 ) {
	    float pred_i = hypo[ich];
	    float pred_tot = hypotot[ich];
	    if ( pred_i<1.0e-3 )
	      pred_i = 1.0e-3;
	    float dnll_bin = pred_i*(1-flashdata[ich]/hypotot[ich]);
	    dnlldf += dnll_bin;
	  }
	}//end of channel looop
	grad[imatchidx] = dnlldf;	
      }//end of imatch index group

      if ( _use_b_terms ) {
	for (int ich=0; ich<_npmts; ich++) {
	  float obs = flashdata[ich];
	  float obswb = obs*(1-(*pb_v)[ich]);
	  if ( obswb<1.0e-3 ) obswb = 1.0e-3;
	  if ( obs>0 )
	    (*pgradb)[ich] = obs*(log(obswb)-log(hypotot[ich]));
	  else
	    (*pgradb)[ich] = 0.;
	}
      }
    }//end of loop over flashes
    
    return grad;
  }

  std::vector< std::vector<float> > LassoFlashMatch::gradBloss() {
    
    std::vector< std::vector<float> > grad( _flashdata_vv.size() );

    for ( size_t idata=0; idata<_flashdata_vv.size(); idata++ ) {
      const std::vector<float>& b_v = _bpmt_vv.at(idata);
      grad.at(idata).resize( b_v.size(), 0.0 );
      for ( size_t ib=0; ib<b_v.size(); ib++ ) {
	grad[idata][ib] = 2.0*b_v.at(ib);
      }
    }
    
    return grad;
  }
  
  // ==================================================================================
  // Update rule for gradient descent fit
  // -------------------------------------

  void LassoFlashMatch::updateF_graddesc( float lr, const std::vector<int>& fmask, bool print ) {

    std::vector< std::vector<float> > gradb_vv;
    std::vector<float> scoregrad = getScoreGrad(  fmask, gradb_vv );
    std::vector<float> clustgrad = get_gradCluster_df( fmask );
    std::vector<float> l1grad    = get_gradL1_df( fmask );
    std::vector<float> l2grad    = get_gradL2_df( fmask );
    std::vector< std::vector<float> > b2loss    = gradBloss();
    
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
    
    if ( _use_b_terms ) {
      // update b-terms
      for ( size_t iflash=0; iflash<_bpmt_vv.size(); iflash++ ) {
	std::vector<float>& b_v = _bpmt_vv.at(iflash);
	for ( size_t ich=0; ich<b_v.size(); ich++ ) {
	  b_v[ich] -= lr*( gradb_vv[iflash][ich] + _bweight*b2loss[iflash][ich] );
	}
      }
    }
  }
  
  LassoFlashMatch::LearningConfig_t LassoFlashMatch::getLearningConfig( int iter ) {
    if ( _learning_v.size()==0 ) {
      // return a default setting
      LearningConfig_t config;
      config.iter_start = -1;
      config.iter_end   = -1;
      config.lr = 1.0e-3;
      config.use_sgd = false;
      config.matchfrac = 1.0;
      return config;
    }

    if ( _index_current_config<0 )
      _index_current_config = 0;

    int ilast=0; 
    for ( int i=_index_current_config; i<(int)_learning_v.size(); i++ ) {
      LearningConfig_t& config = _learning_v[i];
      ilast = i;
      if ( config.iter_start<=iter && iter<=config.iter_end ) {
	_index_current_config = ilast;
	return config;
      }
    }
    // didnt find one? return the last one
    return _learning_v.back();
  }

  void LassoFlashMatch::printState( bool printfmatch ) {
    
    Result_t out = eval(true);
    float fmag = 0.;
    for (int imatch=0; imatch<_nmatches; imatch++)
      fmag += _fmatch_v[imatch];
    std::cout << "[LassoFlashMatch::printState][INFO] "
	      << " Tot="    << out.totscore
	      << " || "
	      << " score="  << out.score
	      << " clust(" << _cluster_weight << ")="  << out.cluster_constraint
	      << " l1norm(" << _l1weight << ")=" << out.L1norm
	      << " l2norm(" << _l2weight << ")=" << out.L2bounds
	      << " b2norm(" << _bweight << ")=" << out.b2loss
	      << " fmag=" << fmag 
	      << std::endl;
    if ( printfmatch ) {
      std::cout << "v: ";
      for (int imatch=0; imatch<_nmatches; imatch++)
	std::cout << "(" << _fmatch_v[imatch] << ")";	  
      std::cout << std::endl;
    }
  }

  void LassoFlashMatch::printClusterGroups( bool printgrads ) {
    std::cout << "===============================================" << std::endl;
    std::cout << "[LassoFlashMatch::printClusterGroups]" << std::endl;
    Grad_t grad = evalGrad(true);
    for ( auto& clusteridx : _clusterindices ) {
      int igroup = _clusteridx2group_m[clusteridx];
      std::cout << " clustergroup[ clusteridx=" << clusteridx << ", groupid=" << igroup << "] ";
      std::vector<int>& group = _clustergroup_vv[igroup];
      for ( auto& idx : group ) {
	if ( printgrads ) 
	  std::cout << " " << idx << "(" << _fmatch_v[idx] << "|dLtot/dx=" << grad.totgrad[idx] << "|dLscore/dx=" << grad.score[idx] << ")";
	else
	  std::cout << " " << idx << "(" << _fmatch_v[idx] << ")";
      }
      std::cout << std::endl;
    }
    std::cout << "===============================================" << std::endl;
  }

  void LassoFlashMatch::printFlashBundles( bool printgrads ) {
    std::cout << "===============================================" << std::endl;
    std::cout << "[LassoFlashMatch::printFlashBundles]" << std::endl;
    Grad_t grad = evalGrad(true);
    std::vector<int> fmask;
    for ( auto& flashidx : _flashindices ) {
      int igroup = _flashidx2group_m[flashidx];
      std::vector<int>& group = _flashgroup_vv[igroup];
      int iflashdata = _flashidx2data_m[flashidx];
      std::vector<float>& flashdata = _flashdata_vv[ iflashdata ];
      
      // get score
      float flash_score = 0;
      switch (_scorer) {
      case kMaxDist:
	flash_score = scoreFlashBundleMaxDist( flashidx, fmask );
	break;
      case kNLL:
	flash_score = scoreFlashBundlePoissonNLL( flashidx, fmask );
	break;
      default:
	assert(false);
	break;
      };

      // get total pe
      std::vector<float> hypo(_npmts,0);
      float hypo_pe = 0.;
      float data_pe = 0.;
      float data_pe_bmod = 0.;
      float bmod = 0.;
      for (size_t ich=0; ich<_npmts; ich++) {
	float hypo_ch_pe = 0.;
	for (auto const& imatch : group ) {
	  const std::vector<float>& hypo = _match_hypo_vv[imatch];
	  hypo_ch_pe += hypo[ich]*_fmatch_v[imatch];
	}
	hypo[ich] = hypo_ch_pe;
	hypo_pe += hypo_ch_pe;
	data_pe += flashdata[ich];
	if ( _use_b_terms ) {
	  float data_chpe_bmod = flashdata[ich]*(1-_bpmt_vv.at( iflashdata )[ich]);
	  if ( data_chpe_bmod<1.0e-3 ) data_chpe_bmod = 0.;
	  data_pe_bmod += data_chpe_bmod;
	  bmod += _bpmt_vv[iflashdata][ich]*_bpmt_vv[iflashdata][ich];
	}
      }
      
      std::cout << " flashgroup[ flashidx=" << flashidx << ", groupid=" << igroup << ","
		<< " data=" << data_pe << ",";
      if ( _use_b_terms )
	std::cout << " bmoddata=" << data_pe_bmod << ",";
      std::cout << " hypo=" << hypo_pe 
		<< "] ";
      std::cout << " [score=" << flash_score;
      if ( _use_b_terms )
	std::cout << ", |b|=" << bmod << " ";
      std::cout << "]";

      for ( auto& idx : group ) {
	if ( printgrads )
	  std::cout << " " << idx << "(" << _fmatch_v[idx] << "|dLtot/dx=" << grad.totgrad[idx] << "|dLscore/dx=" << grad.score[idx] << ")";
	else
	  std::cout << " " << idx << "(" << _fmatch_v[idx] << ") ";
      }
      std::cout << std::endl;      
    }
    std::cout << "===============================================" << std::endl;
  }

  void LassoFlashMatch::printBterms() {
    std::cout << "===============================================" << std::endl;
    std::cout << "[LassoFlashMatch::printBterms]" << std::endl;
    if ( !_use_b_terms ) {
      std::cout << "Not using B-terms" << std::endl;
      return;
    }
    

    for ( auto& flashidx : _flashindices ) {
      int iflashdata = _flashidx2data_m[flashidx];
      std::vector<float>& b_v = _bpmt_vv[iflashdata];
      std::cout << "[flash #" << iflashdata << " idx=" << flashidx << "]: ";
      for ( auto& b : b_v )
	std::cout << b << " ";
      std::cout << std::endl;
    }
    
  }
}
