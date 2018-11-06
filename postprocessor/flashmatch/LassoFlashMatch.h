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
    
    LassoFlashMatch( const int npmts, ScoreType_t score=kMaxDist, Minimizer_t min_method=kStochGrad, bool bundle_flashes=false );
    ~LassoFlashMatch();

    int addMatchPair(int iflashidx, int iclusteridx, const FlashData_t& flash, const FlashHypo_t& hypo );    
    void clear();
    void fitSGD( const int niters, const int niters_per_print=1000, const bool use_sgd=false, const float matchfrac=0.5 );
    
    int _npmts;
    ScoreType_t _scorer;
    Minimizer_t _min_method;
    bool _bundle_flashes;
    float _cluster_weight;
    float _l1weight;
    float _l2weight;

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

    std::map< std::pair<int,int>, int > _pair2match_m;


    void setFMatch( int imatch, float fmatch );
    void setFMatch( int iflash, int iclust, float fmatch );    
    void setFMatch( const std::vector<float>& fmatch_v );

    // scores
    float getTotalScore( const std::vector<int>& fmask, bool use_regs=true );
    float scoreMatch(int match);

    // maxdist scores
    float scoreMatchMaxDist( int imatch );
    float scoreFlashBundleMaxDist( int flashidx, const std::vector<int>& fmask );

    // norms
    float calcClusterConstraint( const std::vector<int>& fmask );
    float calcL1norm(   const std::vector<int>& fmask );
    float calcL2bounds( const std::vector<int>& fmask );

    // gradients
    float gradCluster( int imatchidx, const std::vector<int>& fmask );
    float gradL1( int imatchidx, const std::vector<int>& fmask );
    std::vector<float> get_gradCluster_df( const std::vector<int>& fmask );    
    std::vector<float> get_gradL1_df( const std::vector<int>& fmask );
    std::vector<float> get_gradL2_df( const std::vector<int>& fmask );
    std::vector<float> getScoreGrad(  const std::vector<int>& fmask );
    void updateF_graddesc( float lr,  const std::vector<int>& fmask, bool print=false );
    
  };


}

#endif
