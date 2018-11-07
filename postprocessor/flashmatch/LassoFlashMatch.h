#ifndef __LASSO_FLASHMATCH__
#define __LASSO_FLASHMATCH__

#include <vector>
#include <set>
#include <map>
#include "FlashMatchTypes.h"

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
    Result_t eval( bool isweighted );
    Grad_t   evalGrad( bool isweighted );
    void setWeights( float cluster_constraint, float l1weight, float l2weight ) { _cluster_weight=cluster_constraint; _l1weight=l1weight; _l2weight=l2weight; };
    void addLearningScheduleConfig( LearningConfig_t config ) { _learning_v.push_back(config); };
    void printState( bool printfmatch );
    void printClusterGroups( bool printgrads=false );
    void printFlashBundles( bool printgrads=false );
    void printBterms();    
    void setLossFunction( ScoreType_t scoretype ) { _scorer = scoretype; };
    void setUseBterms( bool use_b_terms );

    
    int _npmts;
    ScoreType_t _scorer;
    Minimizer_t _min_method;
    bool _bundle_flashes;
    bool _use_b_terms;
    float _cluster_weight;
    float _l1weight;
    float _l2weight;
    float _bweight;

    int _nmatches;
    std::set< int >                   _clusterindices;      // cluster indices given by user
    std::vector< std::vector<int> >   _clustergroup_vv;     // each inner vector holds match indices that apply to same data cluster    
    std::map<int,int>                 _clusteridx2group_m;  // map from clusteridx to clustergroup_vv index
    std::vector< int >                _match2clusteridx_v;  // map from matchidx to index of clusteridx_vv
    std::vector< std::vector<float> > _match_hypo_vv;       // for given match index, the cluster flash hypothesis
    
    std::set< int >                   _flashindices;        // flashdata indices given by user
    std::vector< std::vector<float> > _flashdata_vv;        // flashdata
    std::map< int, int >              _flashidx2data_m;     // map from flashidx to flashdata_vv index
    std::vector<int>                  _match2flashidx_v;    // map from matchindex to flash index        
    std::vector< std::vector<int> >   _flashgroup_vv;       // each inner vector holds match indices that apply to same flash
    std::map<int,int>                 _flashidx2group_m;    // map from flashidx to group index
    std::vector<float>                _fmatch_v;
    std::vector< std::vector<float> > _bpmt_vv;             // pull terms to adjust data to account for light due to non-tpc

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
    
  };


}

#endif
