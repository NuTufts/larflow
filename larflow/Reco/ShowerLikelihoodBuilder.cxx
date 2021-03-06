#include "ShowerLikelihoodBuilder.h"

#include "larcv/core/DataFormat/DataFormatTypes.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/mcshower.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @brief default constructor 
   */
  ShowerLikelihoodBuilder::ShowerLikelihoodBuilder()
  {
    _hll          = new TH2F("lfshower_ll","",2000, -10, 190, 1000, 0, 100 );
    _hll_weighted = new TH2F("lfshower_ll_weighted","",2000, -10, 190, 1000, 0, 100 );
    _psce = new larutil::SpaceChargeMicroBooNE();
  }

  ShowerLikelihoodBuilder::~ShowerLikelihoodBuilder()
  {
    // if ( _hll ) delete _hll;
    // _hll = nullptr;
  }

  /**
   * @brief process data for one event, retrieving data from larcv and larlite io managers
   *
   *
   * steps:
   *
   * first we need to assemble true triplet points of showers
   * @verbatim embed:rst:leading-asterisk
   *   * we start by masking out adc images by segment image, keeping shower pixels
   *   * then we pass the masked image into the triplet proposal algorithm, making true pixels
   *   * we filter out the proposals by true match + source pixel being on an electron (true ssnet label)
   * @endverbatim
   *
   * after ground truth points are made, we can build the calculations we want
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void ShowerLikelihoodBuilder::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {



    // get input data
    larcv::EventImage2D* ev_adc      =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wiremc" );
    larcv::EventImage2D* ev_seg      =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "segment" );
    larcv::EventImage2D* ev_trueflow =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "larflow" );

    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );

    std::vector<larcv::Image2D> masked_v;
    std::vector<larcv::Image2D> badch_v;
    for ( size_t p=0; p<3; p++ ) {
      auto const& adc = ev_adc->Image2DArray()[p];
      auto const& seg = ev_seg->Image2DArray()[p];

      // image with shower pixels only
      larcv::Image2D masked(adc.meta());
      masked.paint(0.0);

      // dummy badch, where every channel is a dummy channel
      larcv::Image2D badch(adc.meta());
      badch.paint(0.0);
      
      // find shower pixels
      for ( size_t r=0; r<adc.meta().rows(); r++) {
        for ( size_t c=0; c<adc.meta().cols(); c++) {

          int pid = (int)seg.pixel(r,c);
          if (pid>0) badch.set_pixel(r,c,50.0);
          
          if ( pid !=larcv::kROIEminus && pid!=larcv::kROIGamma )
            continue;
          
          // copy pixel
          if ( adc.pixel(r,c)>10 )
            masked.set_pixel( r, c, adc.pixel(r,c) );
          else
            masked.set_pixel( r, c, 15.0 );
        }
      }
      masked_v.emplace_back( std::move(masked) );
      badch_v.emplace_back( std::move(badch) );
    }

    // images made, now we make the triplets
    tripletalgo.process( masked_v, badch_v, 1.0, true );
    tripletalgo.make_truth_vector( ev_trueflow->Image2DArray() );

    // save the true triplets as larflow3dhits    
    std::vector<larlite::larflow3dhit> truehit_v;
    
    for ( size_t i=0; i<tripletalgo._triplet_v.size(); i++ ) {

      if ( tripletalgo._truth_v[i]!=1 )
        continue;
      
      larlite::larflow3dhit hit;
      // ==========================================
      // DEFINITION OF DATA STORED IN LARFLOW3DHIT
      // * [0-2]:   x,y,z
      // * [3-9]:   6 flow direction scores + 1 max scire (deprecated based on 2-flow paradigm. for triplet, [8] is the only score stored 
      // * [10-12]: 3 ssnet scores, (bg,track,shower)
      // * [13]:    1 keypoint label score
      // ==========================================

      hit.resize( 14, 0 );
      for (size_t v=0; v<3; v++ ) hit[v] = tripletalgo._pos_v[i][v];
      hit[12] = 1.0;
      hit[13] = 0.0;
      hit.tick = tripletalgo._triplet_v[i][3];
      hit.srcwire = tripletalgo._triplet_v[i][2];
      hit.targetwire.resize(3);
      
      
      for (size_t p=0; p<3; p++)
        hit.targetwire[p] = tripletalgo._sparseimg_vv[p][ tripletalgo._triplet_v[i][p] ].col;
      hit.idxhit = i;

      truehit_v.emplace_back( std::move(hit) );
        
    }// end of triplet loop
    std::cout << "Stored " << truehit_v.size() << " true shower hits" << std::endl;

    // cluster the truehits
    _make_truehit_clusters( truehit_v );

    // now we parse the truth, by building the particle graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.set_adc_treename( "wire" );
    mcpg.buildgraph( iolcv, ioll );

    // Get neutrino particles
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> nodes_v
      = mcpg.getNeutrinoParticles();

    // we loop over the different MC showers
    // we try to associate a truehit cluster to the mc shower object
    struct ShowerInfo_t {
      int idx;
      int pid;
      int origin;
      int priority;
      int matched_cluster;
      std::vector<float> shower_dir;
      std::vector<float> shower_dir_sce;
      std::vector<float> shower_vtx;
      std::vector<float> shower_vtx_sce;      
      bool operator<(const ShowerInfo_t& rhs ) {
        if ( priority<rhs.priority ) return true;
        else if ( priority==rhs.priority && idx<rhs.idx ) return true;
        return false;
      };
    };
    std::vector<ShowerInfo_t> shower_info_v;
      
    for ( size_t idx=0; idx<ev_mcshower->size(); idx++ ) {
      auto const& mcsh = ev_mcshower->at( idx );

      ShowerInfo_t info;
      info.idx = idx;
      info.pid = mcsh.PdgCode();
      info.origin = mcsh.Origin();
      if ( info.origin==1 && abs(info.pid)==11 )
        info.priority = 0;
      else if ( info.origin==1 && abs(info.pid)==22 )
        info.priority = 1;
      else if ( info.origin==1 )
        info.priority = 2;
      else if ( info.origin==2 && abs(info.pid)==11 )
        info.priority = 3;
      else
        info.priority = 4;

      info.shower_dir.resize(3,0);
      info.shower_vtx.resize(3,0);
      info.shower_dir_sce.resize(3,0);
      info.shower_vtx_sce.resize(3,0);
      auto const& detprofdir = mcsh.DetProfile().Momentum().Vect();
      for (int i=0; i<3; i++) {
        info.shower_dir[i] = detprofdir[i]/detprofdir.Mag();
        info.shower_vtx[i] = mcsh.DetProfile().Position()[i];
      }

      float dirnorm = 0.;
      for (int v=0; v<3; v++)
        dirnorm += info.shower_dir[v]*info.shower_dir[v];
      dirnorm = sqrt(dirnorm);
      for (int v=0; v<3; v++)
        info.shower_dir[v] /= dirnorm;

      // adjust shower dir due to SCE
      float cos_sce = 1.0;      
      if ( abs(info.pid)==11 ) {
        
        // SCE corrected direction
        std::vector<float> shower_end(3,0);
        for (int p=0; p<3; p++ ) shower_end[p] = info.shower_vtx[p] + 3.0*info.shower_dir[p];
        
        std::vector<double> offset = _psce->GetPosOffsets( info.shower_vtx[0], info.shower_vtx[1], info.shower_vtx[2] );
        info.shower_vtx_sce  = { info.shower_vtx[0] - (float)offset[0] + (float)0.6,
                                 info.shower_vtx[1] + (float)offset[1],
                                 info.shower_vtx[2] + (float)offset[2] };
        offset = _psce->GetPosOffsets( shower_end[0], shower_end[1], shower_end[2] );
        std::vector<float> shower_end_sce  = { shower_end[0] - (float)offset[0] + (float)0.6,
                                               shower_end[1] + (float)offset[1],
                                               shower_end[2] + (float)offset[2] };
        float scenorm = 0.;
        for (int i=0; i<3; i++ ) {
          info.shower_dir_sce[i] = shower_end_sce[i]-info.shower_vtx_sce[i];
          scenorm += info.shower_dir_sce[i]*info.shower_dir_sce[i];
          cos_sce += info.shower_dir[i]*info.shower_dir_sce[i];
        }
        
        scenorm = sqrt(scenorm);
        for (int i=0; i<3; i++ ) info.shower_dir_sce[i] /= scenorm;
        cos_sce /= scenorm;
      }
      else {
        for (int i=0; i<3; i++ ) {
          info.shower_dir_sce[i] = info.shower_dir[i];
          info.shower_vtx_sce[i] = info.shower_vtx[i];
        }
      }

      std::cout << "[ShowerLikelihoodBuilder] true shower " << std::endl;
      std::cout << " pid=" << info.pid << std::endl;
      std::cout << " dir-truth=(" << info.shower_dir[0] << "," << info.shower_dir[1] << "," << info.shower_dir[2] << ")" << std::endl;
      std::cout << " dir-sce=(" << info.shower_dir_sce[0] << "," << info.shower_dir_sce[1] << "," << info.shower_dir_sce[2] << ")" << std::endl;
      std::cout << " cos(truth*sce)=" << cos_sce << std::endl;
      std::cout << " vertex-truth=(" << info.shower_vtx[0] << "," << info.shower_vtx[1] << "," << info.shower_vtx[2] << ")" << std::endl;
      std::cout << " vertex-sce=(" << info.shower_vtx_sce[0] << "," << info.shower_vtx_sce[1] << "," << info.shower_vtx_sce[2] << ")" << std::endl;

      shower_info_v.push_back( info );
    }

    std::sort( shower_info_v.begin(), shower_info_v.end() );    
    std::cout << "[ShowerLikelihoodBuilder] saved " << shower_info_v.size() << " showers" << std::endl;

    std::vector<int> claimed_cluster_v( cluster_v.size(), 0 );
    std::vector<int> cluster_match_v( shower_info_v.size(), -1 );
    int iidx = 0;
    for ( auto& info : shower_info_v ) {
      std::cout << "[" << iidx << "] pid=" << info.pid
                << " origin=" << info.origin
                << " vtx=(" << info.shower_vtx_sce[0] << "," << info.shower_vtx_sce[1] << "," << info.shower_vtx_sce[2] << ")"
                << std::endl;
      int match_cluster_idx = _find_closest_cluster( claimed_cluster_v, info.shower_vtx_sce, info.shower_dir_sce );
      shower_info_v[iidx].matched_cluster = match_cluster_idx;
      std::cout << "   matched cluster index: " << match_cluster_idx << std::endl;
      iidx++;        
    }

    /*

    // cluster hits into cluster_t objects, using dbscan
    _analyze_clusters( truehit_v, shower_dir_sce, shower_vtx_sce );
    */

    std::vector< larlite::larflow3dhit > clustered_truehits_v;
    for ( auto& cluster : cluster_v ) {
      for ( auto& hitidx : cluster.hitidx_v ) {
        clustered_truehits_v.push_back( truehit_v[hitidx] );
      }
    }
    
    // fill profile histogram

    // break into clusters
    // truth ID the trunk cluster
    // save vars for the trunk verus non-trunk clutsters
    // 
    // _fillProfileHist( truehit_v, shower_dir_sce, shower_vtx_sce );

    // mcpg.printGraph();
    
    // save the larflow hits
    larlite::event_larflow3dhit* evout = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "trueshowerhits" );
    for (auto& hit : clustered_truehits_v )
      evout->emplace_back( std::move(hit) );

    // save the shower object we are basing the info on
    larlite::event_mcshower* mcshowerout = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower, "truthshower" );
    for ( auto& info : shower_info_v ) {
      mcshowerout->push_back( ev_mcshower->at( info.idx ) );
    }

    // save the pca's of the clusters
    larlite::event_pcaxis*  evout_pcaxis = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis, "truthshower");
    for ( size_t cidx=0; cidx<cluster_v.size(); cidx++ ) {
      auto const& c = cluster_v[cidx];
      larlite::pcaxis llpca = cluster_make_pcaxis( c, cidx );
      evout_pcaxis->push_back( llpca );
    }
    
    // saved masked shower image
    larcv::EventImage2D* evimgout = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "trueshoweradc" );
    evimgout->Emplace( std::move(masked_v) );

  }

  /**
   * @brief Build a shower profile histogram using true shower hits.
   *
   * we assume all hits belong to the shower. 
   * note: this code was intended to run on single shower events, 
   * in order to build a proper
   * profile that we can use on multi-shower events.
   *
   * this method populates the member histograms, _hll_weighted and _hll.
   * to do: create profiles for different energies
   * 
   * @param[in] truehit_v Collection of true shower hits
   * @param[in] shower_dir Vector describing initial 3D shower direction
   * @param[in] shower_vtx Vector giving 3D shower start point/vertex
   * 
   */ 
  void ShowerLikelihoodBuilder::_fillProfileHist( const std::vector<larlite::larflow3dhit>& truehit_v,
                                                  std::vector<float>& shower_dir,
                                                  std::vector<float>& shower_vtx )
  {

    // get distance of point from pca-axis
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    std::cout << "Fill hits for shower[ "
              << "dir=(" << shower_dir[0] << "," << shower_dir[1] << "," << shower_dir[2] << ") "
              << "vtx=(" << shower_vtx[0] << "," << shower_vtx[1] << "," << shower_vtx[2] << ") "
              << "] with nhits=" << truehit_v.size()
              << std::endl;

    std::vector<float> shower_end(3);
    std::vector<float> d3(3);
    float len3 = 1.0;
    for (int i=0; i<3; i++ ) {
      shower_end[i] = shower_vtx[i] + shower_dir[i];
      d3[i] = shower_dir[i];
    }
      
    for ( auto const& hit : truehit_v ) {
      
      std::vector<float> pt = { hit[0], hit[1], hit[2] };

      std::vector<float> d1(3);
      std::vector<float> d2(3);

      float len1 = 0.;
      for (int i=0; i<3; i++ ) {
        d1[i] = pt[i] - shower_vtx[i];
        d2[i] = pt[i] - shower_end[i];
        len1 += d1[i]*d1[i];
      }
      len1 = sqrt(len1);

      // cross-product
      std::vector<float> d1xd2(3);
      d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
      d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
      d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
      float len1x2 = 0.;
      for ( int i=0; i<3; i++ ) {
        len1x2 += d1xd2[i]*d1xd2[i];
      }
      len1x2 = sqrt(len1x2);
      float rad  = len1x2/len3; // distance of point from PCA-axis

      float proj = 0.;
      for ( int i=0; i<3; i++ )
        proj += shower_dir[i]*d1[i];

      // std::cout << "  hit: (" << pt[0] <<  "," << pt[1] << "," << pt[2] << ") "
      //           << " dist=" << len1
      //           << " proj=" << proj
      //           << " rad=" << rad
      //           << std::endl;
      float w=1.;
      if ( rad>0 ) w = 1.0/rad;
      _hll_weighted->Fill( proj, rad, w );
      _hll->Fill( proj, rad );
    }
    
  }

  /**
   * @brief Make truehit clusters
   *
   * populates the member cluster container, cluster_v.
   * 
   * @param[in] truehit_v Collection of true shower hits
   * 
   */ 
  void ShowerLikelihoodBuilder::_make_truehit_clusters( std::vector< larlite::larflow3dhit >& truehit_v )
  {

    cluster_v.clear();
    float maxdist = 1.0;
    int minsize = 50;
    int maxkd = 5;
    cluster_larflow3dhits( truehit_v, cluster_v, maxdist, minsize, maxkd );
    for ( auto& cluster : cluster_v )
      cluster_pca( cluster );
  }

  /**
   * @brief Find closest cluster to given shower start an direction
   *
   * @param[in] claimed_cluster_v Clusters to search
   * @param[in] shower_vtx Vector giving 3D shower start point/vertex for which we find the closest cluster
   * @param[in] shower_dir Vector describing initial 3D shower direction. Not used.
   * 
   */ 
  int ShowerLikelihoodBuilder::_find_closest_cluster( std::vector<int>& claimed_cluster_v,
                                                      std::vector<float>& shower_vtx,
                                                      std::vector<float>& shower_dir )
  {

    // we define the trunk of the cluster as the one closest to the shower start

    float min_dist2vtx = 1.0e9;
    std::vector<float> trunk_endpt(3,0);
    int best_matched_cluster = -1;
    for ( size_t idx=0; idx<cluster_v.size(); idx++ ) {

      auto& cluster = cluster_v[idx];
      float dist2vtx[2] = {0.};
      for (int e=0; e<2; e++) {
        dist2vtx[e] = 0.;
        for (int i=0; i<3; i++) {
          dist2vtx[e] += (cluster.pca_ends_v[e][i]-shower_vtx[i])*(cluster.pca_ends_v[e][i]-shower_vtx[i]);
        }
        
        if ( dist2vtx[e]<min_dist2vtx ) {
          best_matched_cluster = idx;
          min_dist2vtx = dist2vtx[e];
          trunk_endpt  = cluster.pca_ends_v[e];
        }
      }
      float mindist2vtx = (dist2vtx[0]<dist2vtx[1]) ? dist2vtx[0] : dist2vtx[1];
      // std::cout << "[cluster " << idx << "] mindist2vtx=" << mindist2vtx
      //           << " start=(" << cluster.pca_ends_v[0][0] << "," << cluster.pca_ends_v[0][1] << "," << cluster.pca_ends_v[0][2] << ") "
      //           << " end=("   << cluster.pca_ends_v[1][0] << "," << cluster.pca_ends_v[1][1] << "," << cluster.pca_ends_v[1][2] << ") "        
      //           << std::endl;
    }

    std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] "
              << "trunk cluster index=" << best_matched_cluster << " "
              << "dist2vertex=" << min_dist2vtx
              << std::endl;

    if ( min_dist2vtx>3.0 )
      return -1;

    claimed_cluster_v[ best_matched_cluster ] += 1;
    return best_matched_cluster;
  }  
  
  /**
   * @brief Performs cluster-based analysis of true shower.
   *
   * Gathers statistics we hope to use to relate shower fragment to shower trunks.
   * note: this code was intended to run on single shower events, 
   * in order to build a proper
   * profile that we can use on multi-shower events.
   *
   * 
   * @param[in] truehit_v Collection of true shower hits
   * @param[in] shower_dir Vector describing initial 3D shower direction
   * @param[in] shower_vtx Vector giving 3D shower start point/vertex
   * 
   */ 
  void ShowerLikelihoodBuilder::_analyze_clusters( std::vector< larlite::larflow3dhit >& truehit_v,
                                                   std::vector<float>& shower_dir,
                                                   std::vector<float>& shower_vtx )
  {

    cluster_v.clear();
    _trunk_cluster = -1;
    
    float maxdist = 1.0;
    int minsize = 50;
    int maxkd = 5;
    cluster_larflow3dhits( truehit_v, cluster_v, maxdist, minsize, maxkd );

    std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] number of clusters: " << cluster_v.size() << std::endl;

    // we define the trunk of the cluster as the one closest to the shower start

    float min_dist2vtx = 1.0e9;
    std::vector<float> trunk_endpt(3,0);
    for ( size_t idx=0; idx<cluster_v.size(); idx++ ) {
      auto& cluster = cluster_v[idx];
      std::cout << " [" << idx << "] analyze cluster size=" << cluster.points_v.size() << std::endl;      
      cluster_pca( cluster );
      float dist2vtx[2] = {0.};
      for (int e=0; e<2; e++) {
        for (int i=0; i<3; i++) {
          dist2vtx[e] += (cluster.pca_ends_v[e][i]-shower_vtx[i])*(cluster.pca_ends_v[e][i]-shower_vtx[i]);
        }

        if ( dist2vtx[e]<min_dist2vtx ) {
          _trunk_cluster = idx;
          min_dist2vtx = dist2vtx[e];
          trunk_endpt = cluster.pca_ends_v[e];
        }
      }
    }

    std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] "
              << "trunk cluster index=" << _trunk_cluster << " "
              << "dist2vertex=" << min_dist2vtx
              << std::endl;

    if ( min_dist2vtx>3.0 )
      return;
    
    // get points near the vertex of the trunk cluster, within 5 cm
    cluster_t near_vtx;
    for ( int idx=0; idx<(int)cluster_v[_trunk_cluster].points_v.size(); idx++ ) {
      float dist = 0.;
      for (int i=0; i<3; i++) {
        dist += (cluster_v[_trunk_cluster].points_v[idx][i]-shower_vtx[i])*(cluster_v[_trunk_cluster].points_v[idx][i]-shower_vtx[i]);
      }
      dist = sqrt(dist);
      if ( dist<5.0 ) {
        near_vtx.points_v.push_back( cluster_v[_trunk_cluster].points_v[idx] );
      }
    }
    cluster_pca( near_vtx );
    std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] "
              << "size of trunk portion: " << near_vtx.points_v.size() << " spacepoints" << std::endl;

    // calculate cosine between trunk pca and trunk start direction
    float trunkpcacos = 0.;
    for (int i=0; i<3; i++ ) {
      trunkpcacos += shower_dir[i]*near_vtx.pca_axis_v[0][i];
    }
    std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] "
              << " cos(shower_dir * firstpca)=" << trunkpcacos
              << std::endl;

    // now we measure relationships to the main cluster
    // (1) distance of cluster-endpoint to trunk pca-axis of nearest endpt
    // (2) cosine of pca-axes
    // (3) impact parameter: smallest distance between trunk and cluster pca-axis

    // loop over each cluster
    int idx=-1;
    for ( auto& clust : cluster_v ) {
      idx++;
      // skip trunk cluster
      if ( idx==_trunk_cluster ) continue;
      std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] compare trunk with cluster[" << idx << "]" << std::endl;

      // [2: cos(pca)]
      float cospca = 0.;
      for (int i=0; i<3; i++) {
        cospca += near_vtx.pca_axis_v[0][i]*clust.pca_axis_v[0][i];
      }
      std::cout << "  cluster-trunk pca-cosine: " << cospca << std::endl;

      // [3: impact param]
      float impact_dist, proj_trunk, proj_clust;
      _impactdist( trunk_endpt, near_vtx.pca_axis_v[0],
                   clust.pca_center, clust.pca_axis_v[0],
                   impact_dist, proj_trunk, proj_clust );
      std::cout << "  impact dist: " << impact_dist << std::endl;
      std::cout << "  impact pt trunk proj: " << proj_trunk << std::endl;
      std::vector<float> impactpt(3,0);
      for (int i=0; i<3; i++) {
        impactpt[i] = trunk_endpt[i] + proj_trunk*near_vtx.pca_axis_v[0][i];
      }

      // [1: distance]
      // we use distance between impact point on trunk and endpoints
      float enddist[2] = { 0, 0 };
      for (int e=0; e<2; e++) {      
        for(int i=0; i<3; i++) {
          enddist[e] += (impactpt[i]-clust.pca_ends_v[e][i])*(impactpt[i]-clust.pca_ends_v[e][i]);
        }
        enddist[e] = sqrt(enddist[e]);
      }
      float clust_dist = (enddist[0]<enddist[1]) ? enddist[0] : enddist[1];
      std::cout << "  dist of cluster to trunk: " << clust_dist << std::endl;
      
    }
    
  }

  /**
   * @brief function that calculates shortest distance from a point to a line.
   *
   * the the line is defined by a start point and a ray.
   *
   * @param[in]  ray_start Line 3D start point
   * @param[in]  ray_dir   Line 3D direction
   * @param[in]  pt        Test 3D point 
   * @param[out] radial_dist Shortest distance between pt and line.
   * @param[out] projection  Distance from start point of line to the closest distance segment along line.
   */
  void ShowerLikelihoodBuilder::_dist2line( const std::vector<float>& ray_start,
                                            const std::vector<float>& ray_dir,
                                            const std::vector<float>& pt,
                                            float& radial_dist, float& projection )
  {

    std::vector<float> d1(3);
    std::vector<float> d2(3);

    float len3 = 0.;
    std::vector<float> end(3,0);
    for (int i=0; i<3; i++) {
      end[i] = ray_start[i] + ray_dir[i];
      len3 += ray_dir[i]*ray_dir[i];
    }
    len3 = sqrt(len3);

    float len1 = 0.;
    for (int i=0; i<3; i++ ) {
      d1[i] = pt[i] - ray_start[i];
      d2[i] = pt[i] - end[i];
      len1 += d1[i]*d1[i];
    }
    len1 = sqrt(len1);

    // cross-product
    std::vector<float> d1xd2(3);
    d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
    d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
    d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
    float len1x2 = 0.;
    for ( int i=0; i<3; i++ ) {
      len1x2 += d1xd2[i]*d1xd2[i];
    }
    len1x2 = sqrt(len1x2);
    radial_dist  = len1x2/len3; // distance of point from PCA-axis
    
    projection = 0.;
    for ( int i=0; i<3; i++ )
      projection += ray_dir[i]*d1[i]/len3;
    
  }

  /**
   * @brief shortest distance between lines
   *
   * Calculations referenced from:
   * https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d
   * 
   * @param[in] l_start Start of one line
   * @param[in] l_dir   Direction of one line
   * @param[in] m_start Start of the other line
   * @param[in] m_dir   Direction of the other line
   * @param[out] impact_dist Shortest distance between the two lines
   * @param[out] proj_l Where the shortest distance line is on the l-line relative to l_start
   * @param[out] proj_m where the shortest distance line is on the m-line relative to m_start
   *
   */
  void ShowerLikelihoodBuilder::_impactdist( const std::vector<float>& l_start,
                                             const std::vector<float>& l_dir,
                                             const std::vector<float>& m_start,
                                             const std::vector<float>& m_dir,
                                             float& impact_dist,
                                             float& proj_l,
                                             float& proj_m )
  {

    impact_dist = -1.0;
    proj_l  = 0.;
    proj_m  = 0.;
    
    // L = a + bs where a,b in R3, s in R
    // M = c + dt where c,d in R3, t in R
    
    float b2 = 0.; // l dir squared
    float d2 = 0.; // m dir squred
    float bd = 0.; // innerproduct l and m
    std::vector<float> e(3,0.); // line segment of start points m->l
    float be = 0.; // inner product l_dir and e
    float de = 0.; // inner product m_dir and e
    for (int i=0; i<3; i++) {
      b2 += l_dir[i]*l_dir[i];
      d2 += m_dir[i]*m_dir[i];
      bd += l_dir[i]*m_dir[i];
      e[i] = l_start[i]-m_start[i];
      be += e[i]*l_dir[i];
      de += e[i]*m_dir[i];
    }

    float A = -1.0*(b2*d2 - bd*bd);
    if (A==0) { return; } // parallel lines

    float s = (-1.0*b2*de + be*bd)/A;
    float t = (d2*be - be*bd)/A;

    float D = 0.; // distance
    for (int i=0; i<3; i++ ) {
      float dd = e[i] + l_dir[i]*t - m_dir[i]*s;
      D += dd*dd;
    }
    D = sqrt(D);
    impact_dist = D;

    proj_l = s;
    proj_m = t;
    
  }

}
}
