#ifndef __LASSO_FLASHMATCH__
#define __LASSO_FLASHMATCH__

#include <vector>
#include <array>
#include <set>
#include <map>

#include "TH2F.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "FlashMatchTypes.h"


// ----------------------------
// Many-to-Many Flash Match
// ----------------------------
// Solver for L1-regularized flash-match fit
// Several solvers available
// The code is an indexing nightmare so read below
//
// LASSO regression is least-squares fit with with L1 regularizer
// min || y_{ij} - f_{k}*x_{ikj} ||_2 w.r.t. f_{k} s.t. sum{G(k)} || f_{k} ||_1 -1 = 0
// i: flash index
// k: cluster index
// G(k): a cluster group for those f_{k} where k refers to the same physical charge cluster
// j: pmt index
// where I suppressed the sums over i,j,k in the least-squares term
// y_{ij} is flash data for i-th flash
// x_{ik} is flash hypothesis for i-th flash + k-th cluster

// Book-keeping notes
// -------------------
// Presumably, user will have her own indices for physical flashes (u) and clusters (v).
// Before the fit, the user will propose various matches between cluster and flashes.
// The same cluster can be proposed to match many different flashes
// This is where the cluster groups G(k) come from. G(k) are the 'k' which refer to the same 'v'
// While clusters can be proposed to be matched to different flashes. The solution ideally is a unique map from
//  from one cluster to one flash  (but note, not from flash -> cluster which can be one to many)
// This constraint is where the L1 term comes in:  sum{G(k)} || f_{k} ||_1 - 1 = 0
//
// We have to track all of the above
// We make maps between the user's indices, provided when she registers flash-cluster match candiates
//  to our 'k' and 'i' indices
// We also define a 'match' index which is a list of (i,k) pairs (and (u,v) pairs which follow from user-to-internal indice maps)
//  match index 'm'
// Additionally, we have a pull term for each y_{ij}, b_{ij}. This is to help with light made by charge outside the tpc that is not modeled.

class TRandom3;

namespace larflow {
  
  class LassoFlashMatch {

  public:
    
    typedef enum { kMaxDist=0, kMaxDistZbins, kNLL } ScoreType_t;
    typedef enum { kCoordDesc=0, kGradDesc, kCoordDescSubsample } Minimizer_t;    

    struct Result_t {
      float totloss;           // last total loss
      float ls_loss;           // loss from Least Squares
      float beta_l1loss;       // loss from l1 beta constraint
      float betagroup_l2loss;  // loss from l2 beta-group constraint
      float alpha_l2loss;      // loss from l2 alpha constraint
      float betabounds_l2loss; // loss from enforcing beta>0 and beta<1.0 (deprecated)
      float dbetanorm;         // norm of beta gradient at last step
      float dalphanorm;        // norm of alpha gradient from last step
      float dloss;             // change in loss in last step
      int   numiters;          // number of iterations taken (complete par set update)
      bool  converged;         // fit converged
      std::vector< std::array<int,100> > subsamplebeta; // [subsample coord desc-only] distribution of beta values in 100-bin histogram
      std::vector< float > subsamplebeta_mean;          // [subsample coord desc-only] mean of beta distribution
      Result_t() 
      : totloss(0),
	ls_loss(0),
	beta_l1loss(0),
	betagroup_l2loss(0),
	alpha_l2loss(0),
	dbetanorm(0),
	dalphanorm(0),
	dloss(0),
	numiters(0),
	converged(false)
      {};
    };

    struct LassoConfig_t {
      
      Minimizer_t minimizer;    // method to perform minimization
      float match_l1;          // L1 penalty on activating a match
      float clustergroup_l2;   // L2 penalty to constain solution to one activated match per physical cluster
      float adjustpe_l2;       // L2 penalty to adjust the pe value in a pmt (to account for out-of-tpc charge)
      int maxiterations;       // maximum number of iterations
      float convergence_limit; // minimum change in total loss required to stop in converged state
      float greediness;        // [Coordinate descent only] how much to move to new solution, value between (0,1.0]
      bool  cycle_by_cov;      // [Coordinate descent only] if true, cycle through match parameters using largest covariance with residual
      float learning_rate;     // [Gradient descent only] factor applied to gradient for parameter update

      // constructor with some suggested defaults
      LassoConfig_t()
      : minimizer(kCoordDesc),
	match_l1(10.0),
	clustergroup_l2(1.0),
	adjustpe_l2(1.0e-2),
	maxiterations(100000),
	convergence_limit(1.0e-4),
	greediness(0.5),
	cycle_by_cov(true),
	learning_rate(1.0e-3)
      {};
    };    

    struct Grad_t {
      // For NLL and MaxDist fits
      std::vector<float> score;
      std::vector< std::vector<float> > score_gradb;
      std::vector<float> cluster_constraint;
      std::vector<float> L1norm;
      std::vector<float> L2bounds;
      std::vector< std::vector<float> > b2loss;
      std::vector<float> totgrad;      
      bool isweighted;
      Grad_t() {
	clear();
      };
      ~Grad_t() {
	clear();
      };
      void clear() {
	totgrad.clear();
	score.clear();
	cluster_constraint.clear();
	L1norm.clear();
	L2bounds.clear();
	b2loss.clear();
	isweighted = false;
      };
    };

    struct LearningConfig_t {
      int iter_start;
      int iter_end;
      float lr;
      bool use_sgd;
      float matchfrac;
    };

    
    LassoFlashMatch( const int npmts, ScoreType_t score=kMaxDist, Minimizer_t min_method=kCoordDesc, bool bundle_flashes=true, bool use_b_terms=false );
    virtual ~LassoFlashMatch();

    int  addMatchPair(int iflashidx, int iclusteridx, const FlashData_t& flash, const FlashHypo_t& hypo );
    void provideTruthPair( int iflashidx, int iclustidx );
    void clear();

    Result_t fitLASSO( const LassoConfig_t& config );
    Result_t fitLASSO( const int maxiters,
		       const float lambda_match_l1, const float lambda_clustergroup_l2,
		       const float lambda_peadjust_l2, const float greediness );
    
    Result_t fitSGD( const int niters, const int niters_per_print=1000, const bool use_sgd=false, const float matchfrac=0.5 ); // [deprecated]
    
    Result_t eval( bool isweighted ); // [deprecated]
    Grad_t   evalGrad( bool isweighted ); // [deprecated]
    void setWeights( float cluster_constraint, float l1weight, float l2weight ) { _cluster_weight=cluster_constraint; _l1weight=l1weight; _l2weight=l2weight; };
    void addLearningScheduleConfig( LearningConfig_t config ) { _learning_v.push_back(config); };
    void printState( bool printfmatch );
    void printClusterGroups( bool printgrads=false );
    void printFlashBundles( bool printgrads=false );
    void printBterms();
    void printFmatch();    
    void setLossFunction( ScoreType_t scoretype ) { _scorer = scoretype; };
    void setUseBterms( bool use_b_terms );
    Result_t fitLASSO();
    size_t nmatches() const { return _fmatch_v.size(); };
    float fmatch( int iflashidx, int iclustidx ) const { return 0; };
    
    const int _npmts;
    ScoreType_t _scorer; // [deprecated]
    Minimizer_t _min_method;
    bool _bundle_flashes;
    bool _use_b_terms;
    float _cluster_weight;
    float _l1weight;
    float _l2weight;
    float _bweight;
    TRandom3* _rand;
    
    // book-keeping indices
    // ---------------------
    // i: flash index
    // k: cluster index
    // G(k): a cluster group for {k} which refers to same physical charge cluster (but matched to different flash 'i'), index with 'g'.
    // j: pmt index
    // u: physical flash index (user defined)
    // v: physical cluster index (user defined)
    // m: match index for candidate pair of (u,v) or equivalently (i,k)
    // F(i): flash group for {i} that points to same 'u' (but matched to different cluster 'v'), index with 'l'
    

    struct Match_t {
      
      int matchidx; // internal label for flash-cluster match candidate
      int flashidx; // user label for observed flash
      int clustidx; // user label for charge cluster that produces flash hypo for observed flash
      int flashgroupidx;   // internal label for flash      
      int clustgroupidx; // internal label for cluster
      const std::vector<int>* flashgroup_v; // match indices that relate to same physical flash (includes current index)
      const std::vector<int>* clustgroup_v; // match indices that relate to same physical cluster (includes current index)      
      
      Match_t( int midx,
	       int fidx, int cidx, int fgidx, int cgidx,
	       const std::vector<int>& flashgroup_v, const std::vector<int>& clustgroup_v )
      : matchidx(midx),
	flashidx(fidx),
	clustidx(cidx),
	flashgroupidx(fgidx),
	clustgroupidx(cgidx),
	flashgroup_v(&flashgroup_v),
	clustgroup_v(&clustgroup_v)
      {};
	
    };

    // eventually replace all this map BS with the above
    int _nmatches;
    std::vector< Match_t >            _matchinfo_v;
    std::set< int >                   _clusterindices;      // cluster indices given by user (v)
    std::vector< std::vector<int> >   _clustergroup_vv;     // each inner vector holds match indices (m) that apply to same cluster indices (v), i.e. a list of G(k)
    std::map<int,int>                 _clusteridx2group_m;  // map from clusteridx to clustergroup_vv index (user index 'v' to internal G(k) index 'g')
    std::vector< int >                _match2clusteridx_v;  // map from matchidx to index of clusteridx_vv  (m -> v)
    std::vector< int >                _match2clustgroup_v;  // map from matchidx to clustergroup_vv index (m to internal G(k) index 'g')        
    std::vector< std::vector<float> > _match_hypo_vv;       // for given match index, the cluster flash hypothesis: x_{kj} terms
    
    std::set< int >                   _flashindices;        // flashdata indices given by user (u)
    std::vector< std::vector<float> > _flashdata_vv;        // flashdata ( y_{ij} terms )
    std::map< int, int >              _flashidx2data_m;     // map from flashidx to flashdata_vv index ( u -> i )
    std::map< int, int >              _flashdata2idx_m;     // map from flashidx to flashdata_vv index ( i -> u )
    std::vector<int>                  _match2flashidx_v;    // map from matchindex to flash index      ( m -> u )
    std::vector<int>                  _match2flashgroup_v;  // map from matchidx to flashgroup index (m to internal F(m) index 'l')
    std::vector< std::vector<int> >   _flashgroup_vv;       // each inner vector holds match indices that apply to same flash: F(m) index 'l'
    std::map<int,int>                 _flashidx2group_m;    // map from flashidx to group index ( u -> 'l' )
    std::vector<bool>                 _flashalwaysfit_v;    // flag to always fit this flash, even when using subsampling coord desc or stochastic grad desc
    
    std::vector<float>                _fmatch_v;            // f_{m} terms (which can be mapped to f_{k} throw above)
    std::vector<float>                _flastgrad_v;         // gradient of objective function w.r.t to f_{m}

    std::vector< std::vector<float> > _bpmt_vv;             // pull terms to adjust data to account for light due to non-tpc: b_{ij} terms

    std::map< std::pair<int,int>, int > _pair2match_m;

    std::map<int,int> _truthpair_flash2cluster_idx;
    std::map<int,int> _truthpair_cluster2flash_idx;

    // Defines a model subsystem to solve
    struct SubSystem_t {

      int nmatches;
      int nflashes;
      int nclusters;
      int npmts;
      int nobs;
      
      Eigen::MatrixXf* X;
      Eigen::VectorXf* Y;
      Eigen::VectorXf* beta;      
      Eigen::VectorXf* alpha;

      std::vector< Match_t > matchinfo_v;
      std::vector<int> sub2fullmatchidx;
      std::map<int,int> full2submatchidx_m;
      std::vector<int> sub2fullobsidx;
      std::vector<int> match2clustgroup;
      std::map<int,int> clustidx2clustgroup;
      std::vector< std::vector<int> > clustgroup_vv;
      
      SubSystem_t()
      : nmatches(0),
	nflashes(0),
	nclusters(0),
	npmts(32),
	nobs(0),
	X(nullptr),
	Y(nullptr),
	alpha(nullptr),
	beta(nullptr)
      {};
      ~SubSystem_t() {
	if ( X ) delete X;
	if ( Y ) delete Y;
	if ( alpha ) delete alpha;
	if ( beta ) delete beta;
      };
      
    };


    void setFMatch( int imatch, float fmatch );
    void setFMatch( int iflash, int iclust, float fmatch );    
    void setFMatch( const std::vector<float>& fmatch_v );

    // scores
    float getTotalScore( const std::vector<int>& fmask, bool use_regs=true, bool use_weights=true );
    float scoreMatch(int match);

    // maxdist scores
    float scoreMatchMaxDist( int imatch );
    float scoreFlashBundleMaxDist( int flashidx, const std::vector<int>& fmask );

    // poisson likelihood ratio
    float scoreFlashBundlePoissonNLL( int flashidx, const std::vector<int>& fmask );

    // norms
    float calcClusterConstraint( const std::vector<int>& fmask );
    float calcL1norm(   const std::vector<int>& fmask );
    float calcL2bounds( const std::vector<int>& fmask );
    float calcBloss();

    // gradients
    float gradCluster( int imatchidx, const std::vector<int>& fmask );
    float gradL1( int imatchidx, const std::vector<int>& fmask );
    std::vector<float> get_gradCluster_df( const std::vector<int>& fmask );    
    std::vector<float> get_gradL1_df( const std::vector<int>& fmask );
    std::vector<float> get_gradL2_df( const std::vector<int>& fmask );
    std::vector<float> getScoreGrad(  const std::vector<int>& fmask, std::vector< std::vector<float> >& gradb_vv );
    std::vector<float> gradFlashBundleNLL( const std::vector<int>& fmask, std::vector< std::vector<float> >& gradb_vv );
    std::vector<float> gradMaxDist( const std::vector<int>& fmask );
    std::vector< std::vector<float> > gradBloss();
    
    // update parameters
    void updateF_graddesc( float lr,  const std::vector<int>& fmask, bool print=false );

    // learning configuration
    std::vector< LearningConfig_t > _learning_v;
    std::map< int, int >            _iter2learning;
    int _index_current_config;
    LearningConfig_t getLearningConfig( int iter );

    // optimization methods
    Result_t solveCoordinateDescent( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
				     Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
				     const std::vector<int>& match2clustergroup,
				     const std::vector< std::vector<int> >& clustergroup_vv,
				     const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
				     const float greediness, 
				     const float convergence_threshold, const size_t max_iters,
				     const bool cycle_by_covar, const bool debug ); 

    Result_t solveGradientDescent( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
				   Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
				   const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
				   const float learning_rate, const float stocastic_prob,
				   const float convergence_threshold, const size_t max_iters );
    
    Result_t solveCoordDescentWithSubsampleCrossValidation( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
							    Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
							    const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
							    const float greediness, const float subsample_frac, const int nsubsamples,
							    const float convergence_threshold, const size_t max_iters,
							    const bool cycle_by_covar, const bool debug );

    // tool functions for optimization methods
    void buildLSvectors( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
			 const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha,
			 const float model_frac_err,
			 Eigen::VectorXf& Yalpha, Eigen::VectorXf& model,
			 Eigen::VectorXf& R, Eigen::VectorXf& Rnormed );    
    float calculateChi2Loss( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
			     const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha,
			     const std::vector< std::vector<int> >& clustergroup_vv,			     
			     const float lambda_beta_L1, const float lambda_betagroup_L2,
			     const float lambda_alpha_L2, Result_t* result=nullptr );
    void printFlashBundlesEigen( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha );
    void printClusterGroupsEigen( const Eigen::VectorXf& beta );


    // Subsystem routines
    void shuffleFisherYates( std::vector<int>& vec );    
    SubSystem_t generateFlashRandomSubsample( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
					      const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha,
					      const float subsample_frac );
    

    
  };


}

#endif
