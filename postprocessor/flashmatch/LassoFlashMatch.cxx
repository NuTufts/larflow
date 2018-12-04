#include "LassoFlashMatch.h"
#include <iostream>
#include <cstring>
#include <exception>
#include <cmath>
#include <assert.h>

// ROOT
#include "TStyle.h"
#include "TRandom3.h"
#include "TH2F.h"
#include "TCanvas.h"

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
  {
    _rand = new TRandom3( 1234 );
  }
  
  LassoFlashMatch::~LassoFlashMatch() {
    clear();
    delete _rand;
  }

  void LassoFlashMatch::clear() {
    _nmatches = 0;

    // book-keeping vars
    _clusterindices.clear();
    _clustergroup_vv.clear();
    _clusteridx2group_m.clear();
    _match2clusteridx_v.clear();
    _match2clustgroup_v.clear();    
    _match_hypo_vv.clear();
    
    _flashindices.clear();
    _flashdata_vv.clear();
    _flashidx2data_m.clear();
    _flashdata2idx_m.clear();    
    _match2flashidx_v.clear();
    _match2flashgroup_v.clear();    
    _flashgroup_vv.clear();
    _flashidx2group_m.clear();
    _fmatch_v.clear();
    _flastgrad_v.clear();
    _bpmt_vv.clear();
    _pair2match_m.clear();

    // truth-matching for stdout
    _truthpair_flash2cluster_idx.clear();
    _truthpair_cluster2flash_idx.clear();

    // learning schedule vars
    _learning_v.clear();
    _iter2learning.clear();
    _index_current_config = -1;
  }  

  // ===========================================================================
  // USER INPUTS: SETUP FLASH/CLUSTER PAIRS
  // ---------------------------------------

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
    _match2clustgroup_v.push_back( _clusteridx2group_m[iclustidx] );    
    _clustergroup_vv[ _clusteridx2group_m[iclustidx] ].push_back( imatchidx );
		      
    _match2flashidx_v.push_back( iflashidx );
    _match2flashgroup_v.push_back( _flashidx2group_m[iflashidx] );
    _flashgroup_vv[ _flashidx2group_m[iflashidx] ].push_back( imatchidx );
    
    // set flash data
    if ( _flashidx2data_m.find( iflashidx ) == _flashidx2data_m.end() )  {
      int idata = _flashdata_vv.size();
      _flashdata_vv.push_back( flash );
      _flashidx2data_m[iflashidx] = idata;
      _flashdata2idx_m[idata] = iflashidx;
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
    //std::cout << "[matchpair iflash=" << iflashidx << " iclust=" << iclustidx << "] hypotot=" << norm << std::endl;

    // create match object
    Match_t amatch( imatchidx, iflashidx, iclustidx, _flashidx2group_m[iflashidx], _clusteridx2group_m[iclustidx],
		    _flashgroup_vv[ _flashidx2group_m[iflashidx] ],
		    _clustergroup_vv[ _clusteridx2group_m[iclustidx] ] );
    _matchinfo_v.emplace_back( std::move(amatch) );

    // add match parameter
    _fmatch_v.push_back(0.0);
    _flastgrad_v.push_back(0.0);    

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

  // --------------------------------------------
  // USER INPUTS: TRUTH PAIRS (FOR DEBUG+OUTPUT)
  // --------------------------------------------
  void LassoFlashMatch::provideTruthPair( int iflashidx, int iclustidx ) {
    _truthpair_flash2cluster_idx[iflashidx] = iclustidx;
    _truthpair_cluster2flash_idx[iclustidx] = iflashidx;
  }

  // ===========================================================================
  // SETUP FLASH/CLUSTER PAIRS
  // --------------------------

  LassoFlashMatch::Result_t LassoFlashMatch::fitSGD( const int niters, const int niters_per_print, const bool use_sgd, const float matchfrac ) {

    float lr = 1.0e-3;

    std::cout << "[LassoFlashMatch::gitSGD][INFO] Start." << std::endl;
    if ( true ) {
      std::vector<int> fmask_v;
      Result_t initeval = eval( true );
      float fmag = 0.;
      for (int imatch=0; imatch<_nmatches; imatch++)
	fmag += _fmatch_v[imatch];
      std::cout << "[LassoFlashMatch::fitSGD][INFO] Initial results"	
		<< " Tot="    << initeval.totloss
		<< " || "
		<< " model_loss ="  << initeval.totloss
		<< " betagroup_loss(" << _cluster_weight << ")="  << initeval.betagroup_l2loss
		<< " l1norm(" << _l1weight << ")=" << initeval.beta_l1loss
		<< " betabounds_loss(" << _l2weight << ")=" << initeval.betabounds_l2loss
		<< " b2norm(" << _bweight << ")=" << initeval.alpha_l2loss
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
	  if ( _rand->Uniform()<config.matchfrac ) fmask_v[imatch] = 1;
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
		  << " Tot="    << out.totloss
		  << " model_loss ="  << out.totloss
		  << " betagroup_loss(" << _cluster_weight << ")="  << out.betagroup_l2loss
		  << " l1norm(" << _l1weight << ")=" << out.beta_l1loss
		  << " betabounds_loss(" << _l2weight << ")=" << out.betabounds_l2loss
		  << " b2norm(" << _bweight << ")=" << out.alpha_l2loss
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

    std::vector<int> fmask_v;
    out.totloss           = getTotalScore( fmask_v, true,  isweighted );
    out.ls_loss           = getTotalScore( fmask_v, false, isweighted );
    out.betagroup_l2loss  = calcClusterConstraint( fmask_v );
    out.beta_l1loss       = calcL1norm( fmask_v );
    out.betabounds_l2loss = calcL2bounds( fmask_v );
    if ( _use_b_terms )
      out.alpha_l2loss = calcBloss();
    else
      out.alpha_l2loss = 0;

    // deprecated
    // if ( isweighted ) {
    //   out.cluster_constraint *= _cluster_weight;
    //   out.L1norm   *= _l1weight;
    //   out.L2bounds *= _l2weight;
    //   if ( _use_b_terms )
    // 	out.b2loss   *= _bweight;
    // }

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
      
    //std::vector<float> score_values;
    
    if ( !_bundle_flashes ) {
      //score_values.resize( _nmatches, 0);
      for (int imatch=0; imatch<_nmatches; imatch++) {
	float matchscore = 0.;
	if ( !use_fmask || (use_fmask && fmask[imatch]==1) )
	  matchscore = scoreMatch(imatch);	  
	//score_values[imatch] = matchscore;
	score += matchscore*_fmatch_v[imatch];
      }
    }
    else {
      //score_values.resize( _flashindices.size(), 0 );
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
	//score_values[iflashidx] = flash_score;
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

  // =======================================================================
  // --------
  // scores
  // ---------

  float LassoFlashMatch::scoreMatch(int imatch) {
    // we can calculate score without bundling
    switch( _scorer ) {
    case kMaxDist:
      scoreMatchMaxDist( imatch );
      break;
    default:
      throw std::runtime_error("[LassoFlashMatch::scoreMatch][ERROR] unbundled score type not defined.");
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

  // ======================================================================================================
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
      int iflashdata = _flashidx2data_m[iflashidx];

      // need the overall prediction for each 
      std::vector<float> hypotot(_npmts,0);
      for (auto& imatchidx : flashgroup ) {
	for (int ich=0; ich<_npmts; ich++) {
	  hypotot[ich] += _match_hypo_vv[imatchidx][ich]*_fmatch_v[imatchidx];
	  if ( hypotot[ich]<1.0e-3 ) hypotot[ich] = 1.0e-3;
	}
      }
      
      // also need data
      const std::vector<float>& flashdata = _flashdata_vv[ iflashdata ];
      std::vector<float> databmod( flashdata );  // a copy

      // grab the output b-vector if requested
      std::vector<float>* pb_v   = nullptr;
      std::vector<float>* pgradb = nullptr;
      if ( _use_b_terms ) {
	pgradb = &( gradb_vv.at( iflashdata ) );
	pgradb->resize( flashdata.size(), 0.0 );
	pb_v = &_bpmt_vv.at( iflashdata );
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
	  float obs   = flashdata[ich];
	  float obswb = databmod[ich];
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

  // std::vector<float> LassoFlashMatch::gradLeastSquares( const std::vector<int>& fmask ) {
  //   // gradient of L(f_{k}) = || y_{ij} - f_{k} x_{kj} ||^2 w.r.t f_{k}
  //   // is 2 L(f_{k}) ( - sum{j} x_{kj} )

  //   std::vector< float > gradf( nmatches(), 0 );
    
  //   // do this by flashbundle
  //   for ( auto const& iflashidx : _flashindices ) {
  //     // loop over 'u'
      
  //     std::vector<int>& flashgroup  = _flashgroup_vv[ _flashidx2group_m[iflashidx] ]; // 'F(m)'
  //     int iflashdata = _flashidx2data_m[iflashidx]; // y_{u}

  //     // need the overall prediction for each
  //     // loop over clusters (which provide hypo) that pertain to this flash
  //     // this is x_j = sum{k in F(k)} f_k x_{kj}
  //     std::vector<float> hypotot(_npmts,0);
  //     for (auto& imatchidx : flashgroup ) {
  // 	if ( fmask[imatchidx]==0 ) continue;
  // 	for (int ich=0; ich<_npmts; ich++) {
  // 	  hypotot[ich] += _match_hypo_vv[imatchidx][ich]*_fmatch_v[imatchidx];
  // 	  if ( hypotot[ich]<1.0e-3 ) hypotot[ich] = 1.0e-3;
  // 	}
  //     }
      
  //     // get data (y_{ij})
  //     const std::vector<float>& flashdata = _flashdata_vv[ iflashdata ];
  //     std::vector<float> databmod( flashdata );
      
  //     // calc residual (y_j - x_j)
  //     std::vector<float> res(_npmtss,0);
  //     float totres = 0.;
  //     for (size_t j=0; _npmts; j++) {
  // 	res[j] = flashdata[j] - hypotot[j];
  // 	totres += res[j];
  //     }

  //     // gradient
  //     for (auto& imatchidx : flashgroup ) {
  // 	// index is 'm' but direct map to 'k' (the cluster)
  // 	if ( fmask[imatchidx]==0 ) continue;
	
  // 	float tothypo = 0.;
  // 	for ( size_t j=0; j<_npmts; j++ )
  // 	  tothypo += _match_hypo_vv[imatchidx][ich]*_fmatch_v[imatchidx];
  // 	gradf[ imatchidx ] +=  -2.0*totres*tothypo;
  //     }
      
  //   }//end of loop over flashes

  //   return gradf;
  // }
  
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
    
    for ( size_t imatch=0; imatch<_nmatches; imatch++ ) {
      _flastgrad_v[imatch] = scoregrad[imatch] + _cluster_weight*clustgrad[imatch] + _l1weight*l1grad[imatch] + _l2weight*l2grad[imatch];
    }
    
    for ( size_t imatch=0; imatch<_nmatches; imatch++ ) {
      _fmatch_v[imatch] -= lr*_flastgrad_v[imatch];
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
	      << " Tot="    << out.totloss
	      << " || "
	      << " model_loss ="  << out.totloss
	      << " betagroup_loss(" << _cluster_weight << ")="  << out.betagroup_l2loss
	      << " l1norm(" << _l1weight << ")=" << out.beta_l1loss
	      << " betabounds_loss(" << _l2weight << ")=" << out.betabounds_l2loss
	      << " b2norm(" << _bweight << ")=" << out.alpha_l2loss
	      << " fmag=" << fmag 
	      << std::endl;
    if ( printfmatch ) {
      std::cout << "v: ";
      for (int imatch=0; imatch<_nmatches; imatch++)
	std::cout << "(" << _fmatch_v[imatch] << ")";	  
      std::cout << std::endl;
    }
  }

  void LassoFlashMatch::printFmatch() {
    std::cout << "===============================================" << std::endl;
    std::cout << "[LassoFlashMatch::printFMatch]" << std::endl;
    for (int imatch=0; imatch<_nmatches; imatch++)
      std::cout << "[" << imatch << "|"
		<< " flashidx=" << _match2flashidx_v[imatch]
		<< " clustidx=" << _match2clusteridx_v[imatch]
		<< "] "
		<< _fmatch_v[imatch]
		<< " grad=" << _flastgrad_v[imatch]
		<< std::endl;
    std::cout << "===============================================" << std::endl;    
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

  void LassoFlashMatch::printClusterGroupsEigen( const Eigen::VectorXf& beta ) {
    std::cout << "===============================================" << std::endl;
    std::cout << "[LassoFlashMatch::printClusterGroups]" << std::endl;

    for ( auto& clusteridx : _clusterindices ) {
      int igroup = _clusteridx2group_m[clusteridx];
      std::cout << " clustergroup[ clusteridx=" << clusteridx << ", groupid=" << igroup << "] ";
      std::vector<int>& group = _clustergroup_vv[igroup];
      float grouptot = 0.;
      for ( auto& idx : group ) {
	std::cout << " " << idx << "(" << beta(idx) << ")";
	grouptot += beta(idx);
      }
      std::cout << " || grouptot=" << grouptot;
      std::cout << std::endl;
    }
    std::cout << "===============================================" << std::endl;
  }
  
  void LassoFlashMatch::printFlashBundlesEigen( const Eigen::MatrixXf& X,    const Eigen::VectorXf& Y,
						const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha ) {
    
    std::cout << "===============================================" << std::endl;
    std::cout << "[LassoFlashMatch::printFlashBundles]" << std::endl;

    Eigen::VectorXf alpha1 = Eigen::VectorXf::Ones( Y.rows() ) + alpha;
    Eigen::VectorXf Yalpha = Y.cwiseProduct(alpha1);
    Eigen::VectorXf Pred   = X*beta;
    float nobs = Yalpha.rows();

    Eigen::VectorXf Rfull = Yalpha-Pred;
    
    float allflashres = 0.;
    for ( auto& flashidx : _flashindices ) {
      int igroup = _flashidx2group_m[flashidx];
      std::vector<int>& group = _flashgroup_vv[igroup];
      int iflashdata = _flashidx2data_m[flashidx];
      std::vector<float>& flashdata = _flashdata_vv[ iflashdata ];
      
      // get total pe
      float hypo_pe = 0.;
      float data_pe = 0.;
      float data_pe_chk  = 0.;      
      float data_pe_bmod = 0.;
      float bmod = 0.;
      float res2 = 0.;
      float totres = 0.;
      for (size_t ich=0; ich<_npmts; ich++) {
	hypo_pe      += Pred( _npmts*igroup+ich );
	data_pe      += Y( _npmts*igroup+ich );
	data_pe_chk  += flashdata[ich];
	data_pe_bmod += Yalpha( _npmts*igroup+ich );
	bmod         += alpha( _npmts*igroup+ich )*alpha( _npmts*igroup+ich );
	float res    = Yalpha( _npmts*igroup+ich ) - Pred( _npmts*igroup+ich );
	float err    = Y( _npmts*igroup+ich ) + 0.5*Pred( _npmts*igroup+ich );
	float ols    = res*res;
	if ( err>0 ) ols /= err;
	res2 += ols;
	totres += fabs(res);
      }
      bmod = sqrt(bmod);
      res2 /= _npmts;
      allflashres += totres;
      
      std::cout << " flashgroup[ flashidx=" << flashidx << ", groupid=" << igroup << ","
		<< " data=" << data_pe << ","
	        //<< " data(check)=" << data_pe_chk
		<< " data(alpha)=" << data_pe_bmod << ","
		<< " hypo=" << hypo_pe 
		<< "] ";
      std::cout << " [loss-OLS=" << res2
		<< ", sum|res(w/alpha)|=" << totres
		<< ", |alpha|=" << bmod << " "
		<< "]";

      for ( auto& idx : group ) {
	std::cout << " " << idx << "(" << beta(idx) << ")";
      }
      std::cout << std::endl;      
    }
    std::cout << "total residual (w/alpha): " << allflashres << std::endl;
    std::cout << "total resisual (w/alpha) from matrix: " << Rfull.cwiseAbs().sum() << std::endl;    
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

  // ======================================================================
  // LARS OPTIMIZATION
  // -------------------
  /*
  LassoFlashMatch::Result_t LassoFlashMatch::fitLARS( float learning_rate ) {
    // least angle regression method

    // set fmatch to zero
    for (size_t m=0; m<_fmatch_v.size(); m++) _fmatch_v[m] = 0;

    // start with all elements masked
    std::vector<int> fmask( nmatches(), 0 ); // indexed in 'm'
    std::vector<int> allon( nmatches(), 1 ); // keep around allon for convenience

    // all pull terms are zero
    
    // because of the natural group-structure around the flashes, we apply LARS flash-wise
    // 
    // for each step, we want to find the x_{k} (summed over j) most correlated to y_{i}
    // dL_i/dx_k = 2 L( f_{k} )( - sum{j}[ x_{kj} ] )
    // dL_i/dx_k = 2 L( f )( - f_{k} )  :  hmm, this is always zero!
    // it's the largest
    // We use the x_{k} with the largest dL/df_k

    // first setup, find largest correlate, and turn it on
    std::vector<float> gradLS = gradLeastSquares( allon );
    size_t imatch_strongest = 0;
    float largestgrad = 0;
    for (size_t m=0; m<gradLS.size(); m++) {
      if ( fabs(gradLS[m])>largestgrad ) {
	largestgrad = fabs(gradLS[m]);
	imatch_strongest = m;
      }
    }
    fmask[imatch_strongest] = 1;
    
    int num_on = 1;
    int nsteps = 0;

    while (num_on<nmatches()) {

      std::vector<float> gradLS = gradLeastSquares( allon );

      int numon_pre = num_on;

      // take step in direction for active components only
      for ( size_t m=0; m<nmatches(); m++ ) {
	if ( fmask[m]==1 )
	  _fmatch_v[m] = _fmatch_v[m] - learning_rate*gradLS[m];
      }
      
      // calculate the grad norm for active components
      float gradnorm = 0.;
      for ( size_t m=0; m<nmatches(); m++ ) {
	if ( fmask[m]==1 )
	  gradnorm += gradLS[m]*gradLS[m];
      }
      gradnorm = sqrt(gradnorm);

      // check if another component has as much gradient as joint gradnorm
      int next_strongest = -1;
      float largestgrad = 0;
      for (size_t m=0; m<gradLS.size(); m++) {
	float gradm = fabs(gradLS[m]);
	if ( fmask[m]!=1 && gradm>gradnorm && gradm>largestgrad ) {
	  largestgrad = gradm;
	  next_strongest = (int)m;
	}
      }

      // turn off zero'd components
      for ( size_t m=0; m<nmatches(); m++ ) {
	if ( _fmatch_v[m]<=0 ) {
	  _fmatch_v[m] = 0;
	  fmask[m] = 0;
	}
      }

      // have a step with as strong a residual
      if ( next_strongest>=0 )
	fmask[next_strongest] = 1;
      
      num_on = 0;
      for ( auto const& mask : fmask ) num_on += mask;

      std::cout << "[step " << nsteps << "]" << std::endl;
      std::cout << "  num on (pre-updated): " << numon_pre << std::endl;
      std::cout << "  gradnorm: " << gradnorm << std::endl;
      std::cout << "  activated channel: " << next_strongest << " grad=" <<  largestgrad << " (-1 means none)" << std::endl;
      
      nsteps++;
    }
  }

  // LassoFlashMatch::Result_t LassoFlashMatch::fitLARS( float learning_rate ) {
  //   // We take each cluster separately
  //   // Each cluster is a source of partial hypotheses for several flashes
  //   // 
  // }
  */  

  LassoFlashMatch::Result_t LassoFlashMatch::fitLASSO( const int maxiters,
						       const float match_l1, const float clustergroup_l2,
						       const float peadjust_l2, const float greediness ) {
    LassoConfig_t config;
    config.maxiterations = maxiters;
    config.match_l1 = match_l1;
    config.clustergroup_l2 = clustergroup_l2;
    config.adjustpe_l2 = peadjust_l2;
    config.greediness = greediness;

    return fitLASSO( config );
    
  }
  
  LassoFlashMatch::Result_t LassoFlashMatch::fitLASSO( const LassoConfig_t& config ) {

    // LASSO FIT
    // ---------
    // minimize: (Y-Xbeta)^2 + (lambda_clustergroup)*sum{cluster groups}( (theta^T*theta - 1.0)^2 ) + sum{i hypothesis} lamda_l1norm | beta_i |


    // key constants
    const int nclusters = _clusterindices.size();
    const int nflashes  = _flashindices.size();
    const int npmts     = 32;

    // first we need to build the entire setup (sparse matrices might be better for this...)
    Eigen::MatrixXf X(npmts*nflashes, nmatches()); // hypotheses coming from (flash,cluster) pairs (w/ 32 pmts)
    Eigen::VectorXf Y(npmts*nflashes );            // the data, the observation are each PMTs, but easier to index with flash 
    Eigen::VectorXf beta(nmatches());              // weights for hypotheses. we want groups within this vector to be sparse
    Eigen::VectorXf alpha(npmts*nflashes);         // pmt-by-pmt modification, 1+alpha to help fit, basically a cheater param.
    X.setZero();
    Y.setZero();
    beta.setZero();
    alpha.setZero();
    
    // fill in data and MC to the full system matrices
    for ( int m=0; m<nmatches(); m++ ) {
      
      int uflashidx = _match2flashidx_v[m];
      int vclustidx = _match2clusteridx_v[m];

      int k = _match2clustgroup_v[m];
      int i = _match2flashgroup_v[m];

      // fill in the data
      for ( size_t jch=0; jch<32; jch++ ) {
	Y( i*npmts + jch ) = _flashdata_vv[i][jch];      
      }

      // fill in hypotheses (predictors)
      for ( size_t jch=0; jch<npmts; jch++ ) {      
	X( i*npmts + jch, m) = _match_hypo_vv[m][jch];
      }
    }//end of match loop
    
    // // Need X normalized along hypothesis dimensions
    // Eigen::MatrixXf Xnorm( npmts*nflashes, nmatches() );
    // for ( size_t m=0; m<nmatches(); m++ ) {
    //   float mnorm = X.col(m).norm();
    //   if ( mnorm>1.0e-6 )
    // 	Xnorm.col(m) = X.col(m).normalized();
    // }

    for ( size_t m=0; m<nmatches(); m++ )
      beta(m) = _fmatch_v[m];
    

    float eps = 1.0e-4;
    float learningrate = 1.0e-3;
    float greediness   = 0.5; // how much between the old and new solution do we move?
    int   maxiters = 10000;
    float lambda_alpha_L2 = 1.0e-2; // cost balanced to cost of turning

    Result_t fitresult;
    bool debug = false;

    switch( config.minimizer ) {
    case kCoordDesc:
      // fast, but can be a little greedy, so should run several times to choose meta-parameters
      fitresult = solveCoordinateDescent( X, Y, beta, alpha,
					  _match2clustgroup_v, _clustergroup_vv,
					  config.match_l1, config.clustergroup_l2, config.adjustpe_l2,					  
					  config.greediness,
					  config.convergence_limit, config.maxiterations, config.cycle_by_cov, debug );
      break;
    case kCoordDescSubsample:
      // fast, but can be a little greedy, so should run several times to choose meta-parameters
      fitresult = solveCoordDescentWithSubsampleCrossValidation( X, Y, beta, alpha,
								 config.match_l1, config.clustergroup_l2, config.adjustpe_l2,					  
								 config.greediness, 0.8, 100,
								 config.convergence_limit, config.maxiterations, config.cycle_by_cov, debug );
      break;
    case kGradDesc:
      // slow and can be tricky to converge without small learning rate, but author understands how this should work more than CoordDescent
      fitresult = solveGradientDescent( X, Y, beta, alpha,
					config.match_l1, config.clustergroup_l2, config.adjustpe_l2,
					config.learning_rate, 0.0,
					config.convergence_limit, config.maxiterations );
      break;
    default:
      throw std::runtime_error("[LassoFlashMatch::fitLASSO] unrecognized optimization method");
      break;
    }

    // store beta terms
    for ( size_t m=0; m<nmatches(); m++ ) {
      _fmatch_v[m] = beta(m);
    }

    // store alpha terms
    _bpmt_vv.resize( _flashindices.size() );
    for ( auto& flashidx : _flashindices ) {
      int igroup = _flashidx2group_m[flashidx];
      _bpmt_vv[igroup].resize( _npmts, 0 );
      for ( size_t ich=0; ich<_npmts; ich++ )
	_bpmt_vv[igroup][ich] = alpha( igroup*_npmts + ich );
    }
    
    return fitresult;
  }

  LassoFlashMatch::Result_t LassoFlashMatch::solveCoordinateDescent( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
								     Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
								     const std::vector<int>& match2clustergroup,
								     const std::vector< std::vector<int> >& clustergroup_vv,
								     const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
								     const float greediness,
								     const float convergence_threshold, const size_t max_iters,
								     const bool cycle_by_covar, const bool debug ) {

    Result_t result;
    result.converged = false;

    const int nbeta     = beta.rows();
    const int nclusters = clustergroup_vv.size();
    const int nobs      = Y.rows();

    if ( greediness<0 || greediness>1.0 ) {
      char zmsg[200];
      sprintf( zmsg, "[LassoFlashMatch::solveCoordinateDescent] Error. Greediness param should be between (0,1]. Given %.3e", greediness );
      throw std::runtime_error( zmsg );
    }

    size_t num_iter = 0;
    float dbetanorm = convergence_threshold+1;
    
    std::cout << "[LassoFlashMatch::solveCoordinateDescent] start." << std::endl;
    beta.setZero();

    int update_match_idx = 0;
    
    Eigen::VectorXf Xnorm( nbeta );
    for ( size_t m=0; m<nbeta; m++ )
      Xnorm(m) = X.col(m).norm();

    Eigen::VectorXf last_beta(beta);

    // define struct for sorting parameter indices by largest covariance with the residuals
    struct covar_t {
      int idx;
      float val;
      bool operator<( const covar_t& rhs ) const {
	if ( val>rhs.val ) return true;
	return false;
      };
    };
    std::vector< covar_t > covar_v( nbeta );
    std::vector< covar_t > covar_alpha_v( Y.rows() );

    float last_loss = calculateChi2Loss( X, Y, beta, alpha, clustergroup_vv,
					 lambda_L1, lambda_L2, lambda_alpha_L2, &result );
    
    while ( !result.converged && num_iter<max_iters ) {

      // each loop we update one-parameter
      // check convergence after n-loops
      // if we flag, cycle_by_cov, we update based on which has the highest covariance with the residual
      
      // loop through cluster groups
      // for each variable in cluster group, perform minimization
      // after complete update, check convergence.
      // i guess it's that easy?

      Eigen::VectorXf next_beta(beta);

      // get current model vectors
      Eigen::VectorXf model  = Eigen::VectorXf::Zero( Y.rows() ); // prediction for all pmts
      Eigen::VectorXf Yalpha = Eigen::VectorXf::Zero( Y.rows() ); // observed pe
      Eigen::VectorXf Rfull  = Eigen::VectorXf::Zero( Y.rows() ); // residual between yalpha-model
      Eigen::VectorXf Rnorm  = Eigen::VectorXf::Zero( Y.rows() ); // normalization for each pmt for uncertainty
      buildLSvectors( X, Y, next_beta, alpha, 0.5, model, Yalpha, Rfull, Rnorm );

      Eigen::VectorXf Rnormed = Rfull.cwiseProduct( Rnorm );
            
      // determine order of updates
      Eigen::VectorXf covar = Rnormed.transpose()*X;
      for ( size_t m=0; m<nbeta; m++ ) {
	covar_v[m].idx = m;
	covar_v[m].val = fabs(covar(m)/Xnorm(m));
      }
      if ( cycle_by_covar )
	std::sort( covar_v.begin(), covar_v.end() );

      // perform updates on all beta parameters
      for ( auto const& covar : covar_v )  {
	
	update_match_idx = covar.idx;

	// need to get the norm of the matches cluster group
	// int clustgroupidx = _match2clustgroup_v[update_match_idx];
	// const std::vector<int>& match_indices = _clustergroup_vv[clustgroupidx];
	int clustergroupidx = match2clustergroup[update_match_idx];
	const std::vector<int>& match_indices = clustergroup_vv[clustergroupidx];
      
	Eigen::VectorXf antimask = Eigen::VectorXf::Ones( next_beta.rows() );
	antimask( update_match_idx ) = 0.;
	
	Eigen::VectorXf model_wo_par = X*next_beta.cwiseProduct(antimask);
	Eigen::VectorXf Res_wo_par   = Yalpha-model_wo_par; 

	// solving: 0       = -X_k^t*(Y - X(not-group)*beta(not-notgroup) - X_k*beta_k) + 2*lambda_L2*(sum{theta(nokk)} + beta_k - 1)
	// let      R(notk) = Y-X(notk)*beta(notk)
	// then:    0       = -X_k^t*R(notk) + X_k^t*X_k*beta_k + 2*lambda_L2*(sum{theta(notk)} - 1) + 2*lambda_L2*beta_k
	//          beta_k  = { X_k^t*R(notk) - 2*lamda_L2*(sum{theta(notk)}-1) }/{ X_k^t*X_k + 2*lambda_L2 }

	float theta_loss_wo_par = -1;
	for ( auto const& k : match_indices ) {
	  if ( k!=update_match_idx )
	    theta_loss_wo_par += next_beta(k);
	}
	Eigen::VectorXf Xk = X.col( update_match_idx );

	// beta estimate from LS + L2 (convex portion)
	// --------------------------------------------
	// first, get the solution to the convex-only function
	float numer = Xk.transpose()*(Res_wo_par.cwiseProduct(Rnorm)) - 2*lambda_L2*theta_loss_wo_par;
	float denom = Xk.transpose()*(Xk.cwiseProduct(Rnorm)) + 2*lambda_L2;
	float lambdal1_normed = lambda_L1/( Xnorm(update_match_idx)*Xnorm(update_match_idx) );
	
	float beta_convex = numer/denom;

	// soft-thresholded value, weakens pull 
	float beta_lasso = fabs(beta_convex) - lambdal1_normed;
	if ( beta_lasso<0 ) beta_lasso;
	else {
	  if ( beta_convex<0 )
	    beta_lasso *= -1.0;
	}

	// force positive only values
	if ( beta_lasso<0 )
	  beta_lasso = 0.;

	// update beta parameter
	if ( greediness>0 )
	  next_beta( update_match_idx ) = greediness*beta_lasso + (1.0-greediness)*beta( update_match_idx );
	else
	  next_beta( update_match_idx ) = beta_lasso;
	
	if ( debug ) {
	  // for debug
	  Eigen::VectorXf Res_update = Y-X*next_beta;
	  float beta_ols = Xk.transpose()*Res_wo_par.cwiseProduct(Rnorm);
	  beta_ols /= Xk.transpose()*Xk;
	  float theta_norm_loss = theta_loss_wo_par + beta_lasso;
	  std::cout << "[" << update_match_idx << "| f:" << _match2flashidx_v[ update_match_idx ] << " c:" << _match2clusteridx_v[ update_match_idx ] << "] "
		    << " covar=" << covar.val
		    << " lambdaL1_normed=" << lambdal1_normed
		    << " w-ols=" << beta_ols
		    << " w-convex=" << beta_convex
		    << " w-lasso=" << beta_lasso
		    << " Xk^t*Res(not-k)=" << numer
		    << " theta-norm-loss=" << theta_norm_loss
		    << " convex-denom=" << denom
		    << " Xk-sum=" << Xk.sum()
		    << " sum|res-w-update|=" << Res_update.cwiseAbs().sum()
		    << std::endl;
	  
	}
      }//end of covar loop
      
      // full beta update has been performed
      Eigen::VectorXf dbeta = next_beta-beta;
      dbetanorm = dbeta.norm();

      if ( std::isnan(dbetanorm ) ) {
	std::cout << "[LassoFlashMatch::solveCoordinateDescent] next_beta now nan. stopping" << std::endl;
	std::cout << "beta: " << beta << std::endl;
	std::cout << "next_beta: " << next_beta << std::endl;
	break;
      }

      // alpha terms
      // no need to rank, because the alpha-terms don't correlate with one another. we also simultaneously update
      Eigen::VectorXf PredUpdate = X*next_beta;
      Eigen::VectorXf next_alpha( alpha );
      for ( size_t r=0; r<alpha.rows(); r++ ) {
	float alpha_numer = Y(r)*( PredUpdate(r) - Y(r) )*Rnorm(r);
	float alpha_denom = Y(r)*Y(r)*Rnorm(r) + 2*lambda_alpha_L2;
	float alpha_soln = alpha_numer/alpha_denom;
	if ( greediness>0 )
	  next_alpha(r) = greediness*alpha_soln + (1.0-greediness)*alpha(r);
      }
      Eigen::VectorXf dalpha = next_alpha - alpha;
      float dalphanorm = dalpha.norm();

      // calculate current loss
      float loss = calculateChi2Loss( X, Y, next_beta, next_alpha, clustergroup_vv,
				      lambda_L1, lambda_L2, lambda_alpha_L2 );
      float dloss = loss - last_loss;

      // determine convergence
      // ----------------------
      float totnorm = sqrt( dalphanorm*dalphanorm + dbetanorm*dbetanorm );
	
      // if ( totnorm<convergence_threshold )
      // 	converged = true;
      if ( fabs(dloss)<convergence_threshold )
	result.converged = true;

      // for debug
      if ( debug ) {
	
	std::cout << "====================================================================" << std::endl;
	std::cout << "[LassoFlashMatch::solveCoordinateDescent] iter=" << num_iter
		  << " dbetanorm=" << dbetanorm
		  << " dalphanorm=" << dalphanorm
		  << " dtotnorm=" << totnorm
		  << std::endl;
	std::cout << " Loss(after update)=" << loss << " dloss=" << dloss << std::endl;
	  
	printClusterGroupsEigen(next_beta);
	printFlashBundlesEigen(X,Y,next_beta,next_alpha);

	std::cin.get();	
      }// end of full-cycle update

      // update values
      beta  = next_beta;
      alpha = next_alpha;
      last_loss = loss;
      num_iter++;

      // store results
      result.numiters   = num_iter;
      result.dloss      = dloss;
      result.dalphanorm = dalphanorm;
      result.dbetanorm  = dbetanorm;
      
    }//end of if converges loop
    
    return result;
  }
  
  LassoFlashMatch::Result_t LassoFlashMatch::solveGradientDescent( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
								   Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
								   const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
								   const float learning_rate, const float stocastic_prob,
								   const float convergence_threshold, const size_t max_iters ) {

    Result_t result;
    result.converged = false;
    
    // not as efficient it seems, but at least i know what to do here
    const int nflashes  = _flashindices.size();
    const int npmts     = 32;
    const int nclusters = _clusterindices.size();
    bool debug = false;

    size_t num_iter = 0;
    float dbetanorm  = convergence_threshold+1;
    float dalphanorm = convergence_threshold+1;

    std::cout << "[LassoFlashMatch::solveGradientDescent] start." << std::endl;

    // number of Y entries that are non-zero
    float nobs = Y.rows();

    // prefit loss
    float last_loss = calculateChi2Loss( X, Y, beta, alpha, _clustergroup_vv,
					 lambda_L1, lambda_L2, lambda_L2, &result );
    
    while ( !result.converged && num_iter<max_iters ) {

      // loop through cluster groups
      // for each variable in cluster group, perform minimization
      // after complete update, check convergence.
      // i guess it's that easy?

      // current full residual
      Eigen::VectorXf model     = X*beta;
      Eigen::VectorXf model_err = 0.5*model;

      Eigen::VectorXf alpha1 = Eigen::VectorXf::Ones(Y.rows()) + alpha;
      Eigen::VectorXf Yalpha = Y.cwiseProduct(alpha1);
      Eigen::VectorXf Rfull  = Yalpha-model;
      Eigen::VectorXf Rerr2  = Y+model_err;

      Eigen::VectorXf Rnorm( Rfull );
      for ( size_t r=0; r<Rfull.rows(); r++ ) {
	if ( Rerr2(r)>0 )
	  Rnorm(r) = Rfull(r)/(Rerr2(r)*nobs);
	else
	  Rnorm(r) = Rfull(r)/nobs;
      }

      // // Current Loss
      // // ------------
      // float current_ols_loss = Rfull.transpose()*Rnorm;
      
      // float current_l1_beta      = 0;
      // for ( size_t m=0; m<nmatches(); m++ )
      // 	current_l1_beta += lambda_L1*fabs( beta(m) );
      
      // float current_l2_betagroup = 0;
      std::vector<float> group_theta_loss( nclusters, 0);      
      for ( size_t l=0; l<nclusters; l++ ) {
      	const std::vector<int>& match_indices = _clustergroup_vv[l];
      	float theta_loss = 0.;
      	for ( auto const& m : match_indices ) theta_loss += beta(m);
      	group_theta_loss[l] = theta_loss-1;
      	//current_l2_betagroup += lambda_L2*group_theta_loss[l]*group_theta_loss[l];
      }
      
      // float current_l2_alpha = 0;
      // for ( size_t r=0; r<alpha.rows(); r++ )
      // 	current_l2_alpha += lambda_alpha_L2*alpha(r)*alpha(r);

      // float current_total_loss = current_ols_loss + current_l1_beta + current_l2_betagroup + current_l2_alpha;
      
      // gradient on beta
      // ----------------
      
      // OLS Gradient
      Eigen::VectorXf gradOLS  = -(X.transpose()*Rnorm);

      Eigen::VectorXf gradL2( nmatches() );
      gradL2.setZero();

      for ( size_t l=0; l<nclusters; l++ ) {
	
  	// Define subset matrices for the cluster group
  	// ---------------------------------------------

  	// subset of hypotheses using same physical cluster
  	const std::vector<int>& match_indices = _clustergroup_vv[l];
  	int nhypos = match_indices.size(); 
	
  	// the rest of the single parameters are to be updated as if it were a 1D problem
  	//float theta_norm_loss = (theta.transpose()*theta-1.0);
	float theta_norm_loss = group_theta_loss[l];
	
  	for ( size_t k=0; k<nhypos; k++ )
	  gradL2( match_indices[k] ) = 2*lambda_L2*theta_norm_loss*beta(k);

      }

      // beta grad on L1 norm on matches
      // -------------------------------
      Eigen::VectorXf gradL1( nmatches() );
      gradL1.setZero();      
      for ( size_t m=0; m<nmatches(); m++ ) {

	// L1 not differentialble: soft-threshold function for gradient
	// to do so, we need to calculate the OLS solution
	Eigen::VectorXf antimask( nmatches() );
	antimask.setOnes();
	antimask( m ) = 0.;

	Eigen::VectorXf model_wo_par   = X*(beta.cwiseProduct(antimask));
	Eigen::VectorXf Res_wo_par     = Yalpha - model_wo_par; // residual w/o par
	Eigen::VectorXf Xk = X.col( m );
	Eigen::VectorXf XkErrNorm( Xk );
	for ( size_t r=0; r<Y.rows(); r++ ) {
	  if ( Rerr2(r)>0 )
	    XkErrNorm(r) /= (Rerr2(r)*nobs);
	  else
	    XkErrNorm(r) /= nobs;
	}
	float Xknorm       = Xk.transpose()*XkErrNorm;
	Eigen::VectorXf ResNorm_wo_par( Res_wo_par );
	for ( size_t r=0; r<Res_wo_par.rows(); r++ ) {
	  if ( Rerr2(r)>0 )
	    ResNorm_wo_par(r) /= (Rerr2(r)*nobs);
	  else
	    ResNorm_wo_par(r) /= nobs;
	}

	// solution (with L2 group loss)
	float cluster_group_loss = group_theta_loss[ _match2clustgroup_v[m] ];
	float XkR          = Xk.transpose()*ResNorm_wo_par;
	float numer        = XkR + 2*lambda_L2*( cluster_group_loss - beta(m) ); // last term we remove current beta
	float beta_ols     = numer/(Xknorm - 2*lambda_L2);

	float lambdal1_normed = lambda_L1/Xknorm;
	if ( beta_ols<-lambdal1_normed )   gradL1(m) =  lambdal1_normed;
	else if (beta_ols>lambdal1_normed) gradL1(m) = -lambdal1_normed;
	else gradL1(m) = 0;
	
      } //end of parameter loop
      
      Eigen::VectorXf total_grad = gradOLS + gradL2 + gradL1;
      Eigen::VectorXf next_beta  = beta - learning_rate*total_grad;

      for ( size_t m=0; m<nmatches(); m++ ) {
	if ( next_beta(m)<0 ) next_beta(m) = 0;
	if ( next_beta(m)>3 ) next_beta(m) = 3;
      }

      // gradient on alpha
      // ------------------
      Eigen::VectorXf grad_alpha = Y.cwiseProduct( Rnorm ) + 2*lambda_alpha_L2*alpha;
      Eigen::VectorXf next_alpha = alpha - learning_rate*grad_alpha;
      // enforce hard limits
      for ( size_t r=0; r<alpha.rows(); r++ ) {
	if ( next_alpha(r)<-1 ) next_alpha(r) = -1;
      }
      
      // calculate change in beta
      // ------------------------
      float dbetanorm = (next_beta-beta).norm();

      // calculate change in alpha
      // -------------------------
      Eigen::VectorXf dalpha = next_alpha-alpha;
      dalphanorm = dalpha.norm();

      if ( std::isnan(dbetanorm) || std::isnan(dalphanorm) ) {
  	std::cout << "[LassoFlashMatch::solveCoordinateDescent] next_beta now nan. stopping" << std::endl;
  	std::cout << "beta: " << beta << std::endl;
  	std::cout << "next_beta: " << next_beta << std::endl;
  	break;
      }

      // total diff norm
      // ---------------
      float total_diff_norm = sqrt( dbetanorm*dbetanorm + dalphanorm*dalphanorm );

      // calculate loss
      // --------------
      float loss = calculateChi2Loss( X, Y, next_beta, next_alpha, _clustergroup_vv,
				      lambda_L1, lambda_L2, lambda_alpha_L2, &result );
      float dloss = loss - last_loss;


      // determine convergence
      // ---------------------
      if ( total_diff_norm<convergence_threshold )
  	result.converged = true;

      // for debug
      if ( debug ) {

	float current_tot_res     = 0.;      
	float current_tot_resnorm = 0.;
	Eigen::VectorXf Yalphaupdate  = Eigen::VectorXf::Zero( Y.rows() );
	Eigen::VectorXf modelupdate   = Eigen::VectorXf::Zero( Y.rows() );
	Eigen::VectorXf Rupdate       = Eigen::VectorXf::Zero( Y.rows() );
	Eigen::VectorXf Rnormupdate   = Eigen::VectorXf::Zero( Y.rows() );
	buildLSvectors( X, Y, next_beta, next_alpha, 0.5, Yalpha, modelupdate, Rupdate, Rnormupdate );
	for ( size_t r=0; r<Rnormupdate.rows(); r++ ) {
	  current_tot_resnorm += fabs(Rupdate(r)*Rnormupdate(r));
	  current_tot_res     += fabs(Rupdate(r));
	}
	
	std::cout << "====================================================================" << std::endl;
	std::cout << "[LassoFlashMatch::solveGradientDescent] iter=" << num_iter << " dbetanorm=" << dbetanorm << std::endl;

	std::cout << "beta grad" << std::endl;
	std::cout << "---------" << std::endl;
	std::cout << "gradLS: " << gradOLS.transpose() << std::endl;
	std::cout << "gradL2: " << gradL2.transpose() << std::endl;
	std::cout << "gradL1: " << gradL1.transpose() << std::endl;
	std::cout << "totgrad: " << total_grad.transpose() << std::endl;
	std::cout << "total-grad norm: "  << total_grad.norm() << std::endl;
	std::cout << "dbetanorm: " << dbetanorm << std::endl;
	std::cout << std::endl;
	std::cout << "alpha grad" << std::endl;
	std::cout << "----------" << std::endl;
	std::cout << "alpha-grad norm: " << grad_alpha.norm() << std::endl;
	std::cout << "dalpha: " << dalphanorm << std::endl;
	std::cout << std::endl;
	std::cout << "total norm: " << total_diff_norm << std::endl;
	std::cout << std::endl;
	std::cout << "Last Total Loss: " << result.totloss << " = "
		  << " ols(" << result.ls_loss << ")"
		  << " betal1(" << result.beta_l1loss << ")"
		  << " betagroupl2(" << result.betagroup_l2loss << ")"
		  << " alphal2(" << result.alpha_l2loss << ")"
		  << std::endl;
	std::cout << "Total |res(w/alpha)|=" << current_tot_res << std::endl;
	std::cout << "Total |res(w/alpha)/err|=" << current_tot_resnorm << std::endl;	

	// update so bundles are upgraded properly
  	printClusterGroupsEigen( next_beta );
  	printFlashBundlesEigen( X, Y, next_beta, alpha );
	std::cin.get();	
      }
      
      beta  = next_beta;
      alpha = next_alpha;
      num_iter++;
      
    }//end of if converges loop

    return result;
  }

  float LassoFlashMatch::calculateChi2Loss( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
					    const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha,
					    const std::vector< std::vector<int> >& clustergroup_vv,
					    const float lambda_beta_L1, const float lambda_betagroup_L2,
					    const float lambda_alpha_L2, Result_t* result ) {
    
    Eigen::VectorXf Yalpha  = Eigen::VectorXf::Zero( Y.rows() );
    Eigen::VectorXf model   = Eigen::VectorXf::Zero( Y.rows() );
    Eigen::VectorXf R       = Eigen::VectorXf::Zero( Y.rows() );
    Eigen::VectorXf Rnormed = Eigen::VectorXf::Zero( Y.rows() );
    buildLSvectors( X, Y, beta, alpha, 0.5, Yalpha, model, R, Rnormed );
    
    float loss_ols = R.transpose()*Rnormed;

    float beta_l1  = 0.;
    for ( size_t m=0; m<beta.rows(); m++ )
      beta_l1 += fabs(beta(m));
    beta_l1 *= lambda_beta_L1;
    
    float betagroup_l2 = 0.;
    size_t nclusters   = clustergroup_vv.size();
    for ( size_t l=0; l<nclusters; l++ ) {
      const std::vector<int>& match_indices = clustergroup_vv[l];
      float theta_loss = -1.0;
      for ( auto const& m : match_indices ) theta_loss += beta(m);
      betagroup_l2 += lambda_betagroup_L2*theta_loss*theta_loss;
    }
      
    float alpha_l2 = 0.;
    for ( size_t r=0; r<alpha.rows(); r++ ) {
      alpha_l2 += lambda_alpha_L2*alpha(r)*alpha(r);
    }
    
    float totalloss = loss_ols + beta_l1 + betagroup_l2 + alpha_l2;
    
    if ( result!=nullptr ) {
      result->totloss          = totalloss;
      result->ls_loss          = loss_ols;
      result->beta_l1loss      = beta_l1;
      result->betagroup_l2loss = betagroup_l2;
      result->alpha_l2loss     = alpha_l2;
    }
    
    return totalloss;
    
  }

  // ==============================================================================
  // Solve with subsampling
  
  LassoFlashMatch::Result_t LassoFlashMatch::solveCoordDescentWithSubsampleCrossValidation( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
											    Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
											    const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
											    const float greediness, const float subsample_frac, const int nsubsamples,
											    const float convergence_threshold, const size_t max_iters,
											    const bool cycle_by_covar, const bool debug ) {
    // we run [nsubsamples] fits, removing [subsample_frac] flashes per trial
    // this takes advantage of coordinate descent's speed to build for each match,
    // an outcome distribution
    //
    // for those in the subsample, we solve using coord desc, keeping the other
    //  matches fixed to zero.

    // std::vector< std::vector<float> > beta_values( beta.rows() );
    // for ( size_t m=0; m<beta.rows(); m++ )
    //   beta_values[m].resize( nsubsamples, -1 );

    Result_t result;
    result.subsamplebeta_mean.resize( beta.rows(), 0 );    
    result.subsamplebeta.resize( beta.rows() );
    for ( size_t m=0; m<beta.rows(); m++ )
      memset( result.subsamplebeta[m].data(), 0, sizeof(int)*100 );
    std::vector<int> subsamplebeta_nfills( beta.rows(), 0 );
    
    
    for ( size_t isample=0; isample<nsubsamples; isample++ ) {

      // first draw sample
      // we rebuild/map X,Y,beta,alpha to subsample system
      SubSystem_t subsys = generateFlashRandomSubsample( X, Y, beta, alpha, subsample_frac );

      Result_t subresult = solveCoordinateDescent( *subsys.X, *subsys.Y, *subsys.beta, *subsys.alpha,
						   subsys.match2clustgroup, subsys.clustgroup_vv,
						   lambda_L1, lambda_L2, lambda_alpha_L2,
						   greediness, convergence_threshold, max_iters, cycle_by_covar, false );
      if ( subresult.converged ) {
	for ( int isubm=0; isubm<subsys.nmatches; isubm++ ) {
	  int ifullm = subsys.sub2fullmatchidx[isubm];
	  int xbin = (*subsys.beta)(isubm)/100;
	  if ( xbin>=0 && xbin<100 ) {
	    result.subsamplebeta[ifullm][xbin]++;
	  }
	  result.subsamplebeta_mean[ifullm] += (*subsys.beta)(isubm);
	  subsamplebeta_nfills[ifullm] += 1;
	}
      }
      
      std::cout << "[LassoFlashMatch::subsampleCoordDescSolver] sample=" << isample << " converged=" << subresult.converged << std::endl;
      if ( debug ) {
	
	Eigen::VectorXf beta_sample  = Eigen::VectorXf::Zero( beta.rows() );
	for ( int isubm=0; isubm<subsys.nmatches; isubm++ ) {
	  int ifullm = subsys.sub2fullmatchidx[isubm];
	  beta_sample(ifullm) = (*subsys.beta)(isubm);
	}
	Eigen::VectorXf alpha_sample = Eigen::VectorXf::Zero( alpha.rows() );
	for ( int isubo=0; isubo<subsys.nobs; isubo++ ) {
	  int ifullo = subsys.sub2fullobsidx[ isubo ];
	  alpha_sample(ifullo) = (*subsys.alpha)(isubo);
	}
	printClusterGroupsEigen( beta_sample );
	printFlashBundlesEigen( X, Y, beta_sample, alpha_sample );
	//std::cin.get();
      }//end of if debug

    }//end of sample loop


    // finish mean calc
    for ( size_t b=0; b<beta.rows(); b++ ) {
      if (subsamplebeta_nfills[b]>0)
	result.subsamplebeta_mean[b] /= float(subsamplebeta_nfills[b]);
      beta(b) = result.subsamplebeta_mean[b];
    }

    if ( debug ) {
      // visualize score distribution
      gStyle->SetOptStat(0);
      TCanvas c("c","c",800,600);
      TH2F h("subsamplebeta","",100,0,100,beta.rows(),0,beta.rows());
      for ( size_t b=0; b<beta.rows(); b++ ) {
	if (subsamplebeta_nfills[b]>0)
	  // draw hist for debug
	  for ( size_t i=0; i<100; i++ ) {
	    if ( result.subsamplebeta[b][i]>0 ) {
	      h.SetBinContent( i+1, b+1, result.subsamplebeta[b][i] );
	    }
	  }
      }
      h.Draw("colz");
      c.SaveAs("subsamplebeta.png");
      c.Close();
    }      
    
    
    // to fill out loss and stuff, we run once more, initialized with the current values
    Result_t finalresult = solveCoordinateDescent( X, Y, beta, alpha,
						   _match2clustgroup_v, _clustergroup_vv,
						   lambda_L1, lambda_L2, lambda_alpha_L2,
						   greediness, convergence_threshold, max_iters, cycle_by_covar, false );
    
    result.totloss           = finalresult.totloss;
    result.ls_loss           = finalresult.ls_loss;
    result.beta_l1loss       = finalresult.beta_l1loss;
    result.betagroup_l2loss  = finalresult.betagroup_l2loss;
    result.alpha_l2loss      = finalresult.alpha_l2loss;
    result.betabounds_l2loss = finalresult.betabounds_l2loss;
    result.dbetanorm         = finalresult.dbetanorm;
    result.dalphanorm        = finalresult.dalphanorm;
    result.dloss             = finalresult.dloss;
    result.numiters          = finalresult.numiters;
    result.converged         = finalresult.converged;
    
    return result;
  }

  void LassoFlashMatch::shuffleFisherYates( std::vector<int>& vec ) {
    int n = vec.size();
    for (int i = n-1; i > 0; i--)  {
      // Pick a random index from 0 to i
      int j = _rand->Integer(i+1);
      // Swap arr[i] with the element at random index
      int temp = vec[i];
      vec[i] = vec[j];
      vec[j] = temp;
    }
  }

  // ===============================================================================
  // Common/repeated calculations: modularize here for convenience and consistency
  // ------------------------------------------------------------------------------

  void LassoFlashMatch::buildLSvectors( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
					const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha,
					const float model_frac_err,
					Eigen::VectorXf& Yalpha, Eigen::VectorXf& model,
					Eigen::VectorXf& R, Eigen::VectorXf& Rnormed ) {

    Eigen::VectorXf alpha1 = Eigen::VectorXf::Ones( Y.rows() ) + alpha;
    float fnobs = (float)Y.rows();
    
    Yalpha = Y.cwiseProduct( alpha1 );
    model = X*beta;
    R = Yalpha - model;

    Eigen::VectorXf Rerr2 = Y + model_frac_err*model;
    for ( size_t r=0; r<R.rows(); r++ ) {
      if ( Rerr2(r)>0 )
	Rnormed(r) = 1.0/(Rerr2(r)*fnobs);
      else
	Rnormed(r) = 0.;
    }
  }

  LassoFlashMatch::SubSystem_t LassoFlashMatch::generateFlashRandomSubsample( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
									      const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha,
									      const float subsample_frac ) {

    SubSystem_t subsystem;
    
    const int nflashes = _flashindices.size();
    const int nsubflashes = int(subsample_frac*nflashes);
    const int npmts = 32;
    std::vector< int > indices( nflashes, 0 );
    for ( size_t i=1; i<nflashes; i++ ) indices[i] = i;
    shuffleFisherYates( indices );

    indices.resize(nsubflashes); // truncate

    // we have to (1) count the number of matches that pertain to the subset of flashes
    // we also have to collect the cluster groups for the subset of matches
    int imatchidx = 0;
    int iclustgroupidx = 0;
    std::set<int> clusteridx_set;
    std::vector<int> clustergroup2idx_v( _clusterindices.size(), 0 );
    std::vector< std::vector<int> > subflashgroup_vv( nsubflashes );
    subsystem.sub2fullobsidx.resize( nsubflashes*npmts );
    subsystem.sub2fullmatchidx.resize( nmatches(), -1 );

    for ( int iflash=0; iflash<nsubflashes; iflash++ ) {

      int flashidx    = indices[iflash];
      int iflashgroup = _flashidx2data_m[flashidx];
      for ( int ich=0; ich<npmts; ich++ ) {
	subsystem.sub2fullobsidx[ iflash*npmts + ich ] = iflashgroup*npmts + ich;
      }
      subflashgroup_vv[iflash].resize( _flashgroup_vv[iflashgroup].size() );
      const std::vector< int >& full_match_indices = _flashgroup_vv[ iflashgroup ];
      for ( auto const& m : full_match_indices ) {
	const Match_t& minfo = _matchinfo_v[m];
	subsystem.sub2fullmatchidx[ imatchidx ] = m;
	subsystem.full2submatchidx_m[ m ] = imatchidx;
	
	// collect cluster group info
	// int clustidx   = minfo.clustidx;
	// int clustgroup = minfo.clustgroupidx;
	int clustidx    = _match2clusteridx_v[m];
	int clustgroup  = _match2clustgroup_v[m];
	if ( clusteridx_set.find(clustidx)==clusteridx_set.end() ) {
	  // new clustidx
	  clusteridx_set.insert( clustidx );
	  std::vector<int> clustgroup_v(1,imatchidx);
	  subsystem.clustgroup_vv.push_back( clustgroup_v );      // used by fitter
	  subsystem.match2clustgroup.push_back( iclustgroupidx ); //used by fitter
	  subsystem.clustidx2clustgroup[clustidx] = iclustgroupidx;
	  clustergroup2idx_v[iclustgroupidx] = clustidx;
	  iclustgroupidx++;
	}
	else {
	  // existing clustidx in subgroup. append match.
	  int isubclustgroup = subsystem.clustidx2clustgroup[ clustidx ];
	  subsystem.clustgroup_vv[ isubclustgroup ].push_back( imatchidx );
	  subsystem.match2clustgroup.push_back( isubclustgroup );
	  clustergroup2idx_v[ isubclustgroup ] = clustidx;
	}
	Match_t submatch( imatchidx,
			  minfo.flashidx, minfo.clustidx, iflash, subsystem.match2clustgroup[imatchidx],
			  _flashgroup_vv[iflashgroup],
			  subsystem.clustgroup_vv[ subsystem.match2clustgroup[imatchidx] ] );
	
	subsystem.matchinfo_v.emplace_back( std::move(submatch) );
	imatchidx++;
      }// for match indices associated to this flash
    }//end of flash loop
    
    int nobs     = nsubflashes*npmts;
    subsystem.nmatches  = imatchidx;
    subsystem.nflashes  = nsubflashes;
    subsystem.nclusters = subsystem.clustgroup_vv.size();
    subsystem.npmts = 32;
    subsystem.nobs  = nobs;
    subsystem.X = new Eigen::MatrixXf( nobs, subsystem.nmatches );
    subsystem.Y = new Eigen::VectorXf( nobs );
    subsystem.alpha = new Eigen::VectorXf( nobs );
    subsystem.beta  = new Eigen::VectorXf( subsystem.nmatches );
    subsystem.X->setZero();
    subsystem.Y->setZero();
    subsystem.alpha->setZero();
    subsystem.beta->setZero();
    subsystem.sub2fullmatchidx.resize( subsystem.nmatches );    

    // fill Y,alpha
    for ( int iflash=0; iflash<subsystem.nflashes; iflash++ ) {

      int flashidx    = indices[iflash];
      int iflashgroup = _flashidx2data_m[flashidx]; // (full group)

      for ( size_t ich=0; ich<npmts; ich++ ) {
	(*subsystem.Y)( iflash*npmts + ich )     = Y( iflashgroup*npmts + ich);
	(*subsystem.alpha)( iflash*npmts + ich ) = alpha( iflashgroup*npmts + ich);
      }
    }
    
    // fill X,beta
    for ( size_t isubm=0; isubm<subsystem.nmatches; isubm++ ) {
      
      int ifullm      = subsystem.sub2fullmatchidx[ isubm ];
      int iflashgroup = subsystem.matchinfo_v[isubm].flashgroupidx;
      for ( size_t ich=0; ich<npmts; ich++ )
	(*subsystem.X)( iflashgroup*npmts + ich, isubm ) = X( _matchinfo_v[ifullm].flashgroupidx*npmts + ich, ifullm );
      
      (*subsystem.beta)( isubm ) = beta( ifullm );
      
    }

    /*
    std::cout << "=====================================================" << std::endl;
    std::cout << "[LassoFlashMatch::generateFlashRansomSubsample]" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;    
    std::cout << "subsample flashes: " << std::endl;
    for ( size_t f=0; f<subsystem.nflashes; f++ ) {
      std::cout << "[flash " << f << ": idx=" << indices[f] << "] ";
      const std::vector<int>& fullmatch_indices = _flashgroup_vv[f];
      for ( auto const& ifullm : fullmatch_indices ) {
	int isubm = subsystem.full2submatchidx_m[ ifullm ];
	const Match_t& submatchinfo = subsystem.matchinfo_v[ isubm ];
	std::cout << " (m" << isubm << ":ci" << submatchinfo.clustgroupidx << ":cidx" << submatchinfo.clustidx << ": " << (*subsystem.beta)( isubm ) << ") ";
      }
      std::cout << std::endl;
    }
    std::cout << "subsample classes: " << std::endl;
    for ( size_t l=0; l<subsystem.clustgroup_vv.size(); l++ ) {
      auto const& matchindices = subsystem.clustgroup_vv[l];
      std::cout << "[cluster group " << l << ": idx=" << clustergroup2idx_v[l] << "] ";       
      for ( auto const& subm : matchindices ) {
	const Match_t& submatchinfo = subsystem.matchinfo_v[ subm ];
	std::cout << " (m" << subm << ":fi" << submatchinfo.flashgroupidx << ":fidx" << submatchinfo.flashidx << ": " << (*subsystem.beta)( subm ) << ") ";
      }
      std::cout << std::endl;
    }
    */
    
    return subsystem;
  }
  
	
}
