#include "NuVertexFitter.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"
#include "NuVertexCandidate.h"
#include "cluster_functions.h"
#include "TrackOTFit.h"

namespace larflow {
namespace reco {

  void NuVertexFitter::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll,
                                const std::vector< larflow::reco::NuVertexCandidate >& vertex_v )
  {


    // load adc images
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    auto const& adc_v = ev_adc->Image2DArray();


    const float _prong_radius_ = 10.0;

    _fitted_pos_v.clear();
    _fitted_pos_v.reserve( vertex_v.size() );    

    for ( int ivtx=0; ivtx<(int)vertex_v.size(); ivtx++ )  {

      std::cout << "=====[ FITTER VERTEX " << ivtx << " ]========" << std::endl;
      
      const NuVertexCandidate& cand = vertex_v.at(ivtx);

      int nclusters = (int)cand.cluster_v.size();

      std::vector< const larlite::larflowcluster* > cluster_v;
      std::vector< const larlite::pcaxis* > pcaxis_v;
      
      // get the clusters, keep pixels within 10 cm of keypoint
      std::vector<Prong_t> prong_v;

      for (int icluster=0; icluster<(int)nclusters; icluster++ ) {

        auto const& vtxcluster = cand.cluster_v[icluster];
        larlite::event_larflowcluster* ev_cluster =
          (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer );
        const larlite::larflowcluster* lfcluster = &(ev_cluster->at(vtxcluster.index));
        
        larlite::event_pcaxis* ev_pcaxis =
          (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis, vtxcluster.producer );
        const larlite::pcaxis* lfpca = &(ev_pcaxis->at(vtxcluster.index));

        cluster_v.push_back( lfcluster );
        pcaxis_v.push_back( lfpca );

        std::cout << "cluster[" << icluster << "]--------------" << std::endl;
        std::cout << "num hits: " << lfcluster->size() << std::endl;

        Prong_t prong;
        prong.orig_cluster = lfcluster;
        prong.orig_pcaxis  = lfpca;
        prong.feat_v.clear();
        prong.endpt.resize(3,0);
        prong.startpt.resize(3,0);

        cluster_t clust;

        int npts_in_radius = 0;

        int hitidx = -1;
        for ( auto const& lfhit : *lfcluster ) {
          hitidx++;
          
          float dist = 0.;
          for (int i=0; i<3; i++)
            dist += ( lfhit[i]-cand.pos[i] )*( lfhit[i]-cand.pos[i] );
          dist = std::sqrt(dist);
          
          if ( dist<_prong_radius_ ) {

            npts_in_radius += 1;

            std::vector<float> ptpos(7,0); /// (x,y,z,lm,qu,qv,qy)
            for (int i=0; i<3; i++)
              ptpos[i] = lfhit[i];
            ptpos[3] = lfhit[9];

            int row = adc_v[0].meta().row( lfhit.tick, __FILE__, __LINE__ );            
            for (size_t p=0; p<3; p++) {

              auto const& img = adc_v[p];

              int col = lfhit.targetwire[p];
              float planeq = 0.;
              for (int dr=-1;dr<=1; dr++) {
                int r = row+dr;
                if ( r<0 || r>=(int)img.meta().rows() ) continue;
                for (int dc=-1; dc<=1; dc++) {
                  int c = col+dc;
                  if ( c<0 || c>=(int)img.meta().cols() ) continue;
                  planeq += img.pixel(r,c);
                }
              }
              ptpos[4+p] =  planeq;
            }
            std::cout << " pronghit[" << hitidx << "]"
                      << " pos=(" << lfhit[0] << "," << lfhit[1] << "," << lfhit[2] << ")"
                      << " row=" << row << " tick=" << lfhit.tick
                      << " wire=(" << lfhit.targetwire[0] << "," << lfhit.targetwire[1] << "," << lfhit.targetwire[2] << ")"              
                      << " lmscore=" << lfhit[9]
                      << " q=(" << ptpos[4] << "," << ptpos[5] << "," << ptpos[6] << ")"
                      << std::endl;
            prong.feat_v.push_back( ptpos );
            clust.points_v.push_back( ptpos );
            clust.imgcoord_v.push_back( lfhit.targetwire );
            clust.hitidx_v.push_back( hitidx );
          }          
        }
        std::cout << "num with rad of vertex: " << npts_in_radius << std::endl;

        if ( npts_in_radius<5 )
          continue;

        // calculate pca of the prong
        cluster_pca( clust );
        float startdist = 0;
        float enddist = 0;
        for (int i=0; i<3; i++) {
          startdist += ( clust.pca_ends_v[0][i]-cand.pos[i] )*( clust.pca_ends_v[0][i]-cand.pos[i] );
          enddist   += ( clust.pca_ends_v[1][i]-cand.pos[i] )*( clust.pca_ends_v[1][i]-cand.pos[i] );
        }
        if ( startdist<enddist ) {
          prong.startpt = clust.pca_ends_v[0];
          prong.endpt   = clust.pca_ends_v[1];
        }
        else {
          prong.startpt = clust.pca_ends_v[1];
          prong.endpt   = clust.pca_ends_v[0];
        }
        
        prong_v.emplace_back( std::move(prong) );
      }//end of cluster loop

      std::cout << "Fitting Vertex: number of prongs " << prong_v.size() << std::endl;

      if ( prong_v.size()<1 ) {
        // can't do anything
        _fitted_pos_v.push_back( cand.pos );
        continue;
      }

      std::vector<float> fitted_pos;
      float delta_loss;
      try {
        _fit_vertex( cand.pos, prong_v, fitted_pos, delta_loss );
        std::cout << "[fit returned.] deltaL=" << delta_loss << std::endl;        
      }
      catch (...) {
        std::cout << "[fit failed. using old pos]" << std::endl;
        fitted_pos = cand.pos;
      }
      
      if ( delta_loss<0 ) {
        _fitted_pos_v.push_back( fitted_pos );
      }
      else {
        _fitted_pos_v.push_back( cand.pos );
      }
    }//end of vertex loops
    
  }


  void NuVertexFitter::_fit_vertex( const std::vector<float>& initial_vertex_pos,
                                    const std::vector<NuVertexFitter::Prong_t>& prong_v,
                                    std::vector<float>& fitted_pos,
                                    float& delta_loss )
  {

    // each prong defines a segment from the endpt (made by the pca) to the vertex
    // we calculate the gradient for the vertex from each prong, and update!

    int iter=0;
    const int _maxiters_ = 1000;
    float lr = 0.1;

    std::vector<float> current_vertex = initial_vertex_pos;
    float first_loss = -1;
    float current_loss = -1;
    
    while ( iter<_maxiters_ ) {

      std::vector<float> grad(3,0);
      float tot_loss = 0.;
      
      for (int iprong=0; iprong<(int)prong_v.size(); iprong++ ) {
        float prong_loss = 0;
        std::vector<float> prong_grad(3,0);
        std::vector< std::vector<float> > prong_seg(2);
        prong_seg[0] = prong_v[iprong].endpt;
        prong_seg[1] = current_vertex;
        larflow::reco::TrackOTFit::getLossAndGradient( prong_seg, prong_v[iprong].feat_v, prong_loss, prong_grad );
        
        for (int i=0; i<3; i++ )
          grad[i] += prong_grad[i];
        tot_loss += prong_loss;
      }

      if ( first_loss<0 )
        first_loss = tot_loss;
      
      // update
      float gradlen = 0.;
      for (int i=0; i<3; i++ ) {
        current_vertex[i] += -lr*grad[i];
        gradlen += grad[i]*grad[i];
      }
      current_loss = tot_loss;

      std::cout << "[NuVertexFitter::_fit_vertex] iter[" << iter << "] "
                << " grad=(" << grad[0] << "," << grad[1] << "," << grad[2] << ")"
                << " len=" << sqrt(gradlen)
                << " currentvtx=(" << current_vertex[0] << "," << current_vertex[1] << "," << current_vertex[2] << ")"
                << " loss=" << current_loss
                << std::endl;

      if ( sqrt(gradlen)<1.0e-2 )
        break;
      iter++;
    }

    std::cout << "[NuVertexFitter::_fit_vertex] FIT RESULTS -----------------" << std::endl;
    std::cout << "  num iterations: " << iter << std::endl;
    std::cout << "  original vertex: (" << initial_vertex_pos[0] << "," << initial_vertex_pos[1] << "," << initial_vertex_pos[2] << ")" << std::endl;
    std::cout << "  final vertex: (" << current_vertex[0] << "," << current_vertex[1] << "," << current_vertex[2] << ")" << std::endl;
    std::cout << "  original loss: " << first_loss << std::endl;
    std::cout << "  current loss: " << current_loss << std::endl;    
    std::cout << "-----------------------------------------------------------" << std::endl;

    fitted_pos = current_vertex;
    delta_loss = current_loss-first_loss;
    
  }
  
  
}
}
