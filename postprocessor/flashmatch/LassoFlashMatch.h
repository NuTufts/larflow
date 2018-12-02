#ifndef __LASSO_FLASHMATCH__
#define __LASSO_FLASHMATCH__

#include <vector>
#include <set>
#include <map>

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


namespace larflow {

  class LassoFlashMatch {

  public:
    
    typedef enum { kMaxDist=0, kMaxDistZbins, kNLL } ScoreType_t;
    typedef enum { kMinuit=0, kStochGrad, kMCMC }    Minimizer_t;

    struct Result_t {
      float totscore;
      float score;
      float cluster_constraint;
      float L1norm;
      float L2bounds;
      float b2loss;
      bool isweighted;
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
    
    LassoFlashMatch( const int npmts, ScoreType_t score=kMaxDist, Minimizer_t min_method=kStochGrad, bool bundle_flashes=true, bool use_b_terms=false );
    virtual ~LassoFlashMatch();

    int addMatchPair(int iflashidx, int iclusteridx, const FlashData_t& flash, const FlashHypo_t& hypo );    
    void clear();
    Result_t fitSGD( const int niters, const int niters_per_print=1000, const bool use_sgd=false, const float matchfrac=0.5 );
    Result_t fitLASSO( const float lambda_clustergroup, const float lambda_l1norm );
    Result_t eval( bool isweighted );
    Grad_t   evalGrad( bool isweighted );
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
    ScoreType_t _scorer;
    Minimizer_t _min_method;
    bool _bundle_flashes;
    bool _use_b_terms;
    float _cluster_weight;
    float _l1weight;
    float _l2weight;
    float _bweight;

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
    
    
    int _nmatches;
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
    
    std::vector<float>                _fmatch_v;            // f_{m} terms (which can be mapped to f_{k} throw above)
    std::vector<float>                _flastgrad_v;         // gradient of objective function w.r.t to f_{m}

    std::vector< std::vector<float> > _bpmt_vv;             // pull terms to adjust data to account for light due to non-tpc: b_{ij} terms

    std::map< std::pair<int,int>, int > _pair2match_m;      


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
    bool solveCoordinateDescent( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, Eigen::VectorXf& beta,
				 const float lambda_L1, const float lambda_L2, const float learning_rate,
				 const float convergence_threshold, const size_t max_iters, bool cycle_by_covar );

    bool solveGradientDescent( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y,
			       Eigen::VectorXf& beta, Eigen::VectorXf& alpha,
			       const float lambda_L1, const float lambda_L2, const float lambda_alpha_L2,
			       const float learning_rate, const float stocastic_prob,
			       const float convergence_threshold, const size_t max_iters );
    void printFlashBundlesEigen( const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, const Eigen::VectorXf& beta, const Eigen::VectorXf& alpha );
    void printClusterGroupsEigen( const Eigen::VectorXf& beta );
    

    

    
  };


}

#endif
