#ifndef __LARFLOW_RECO_NUVERTEX_RESTORE_KP_HITS_H__
#define  __LARFLOW_RECO_NUVERTEX_RESTORE_KP_HITS_H__

/**
* @brief Readd the hits to end of clusters near vertices that were vetoed to help cluster tracks/showers
* 
* In ProjectionDefectSplitter used to cluster track particles, we used the reconstructed keypoints to
* remove hits. This helped break the clusters into straight-ish sub-cluster fragments. 
* The goal was to prevent clusters with low-purity which is hard to deal with 
* (our approach assumes its easier to combine pure fragments than to split impure clusters.)
* 
* But one downside to this is that there are now gaps between the location of reco vertices
* and the start of particle prongs. So we now go back and add these hits back to 
* the ends of these prongs.
* 
*/

#include <vector>
#include <string>
#include <algorithm>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflow3dhit.h"
#include "larflow/Reco/NuVertexCandidate.h"

namespace larflow {
namespace reco {

    class NuVertexRestoreKPHits : public larcv::larcv_base {

        public:

        NuVertexRestoreKPHits()
        : larcv::larcv_base("NuVertexRestoreKPHits"),
        _input_kpvetoed_hit_treename("projsplitvetoed"),
        _collection_radius_cm(5.0)
        {};

        virtual ~NuVertexRestoreKPHits() {};

        void process( std::vector<larflow::reco::NuVertexCandidate>& nuvtx_v,
                      larlite::storage_manager& ioll, larcv::IOManager& iolcv ); 

        std::vector<larlite::larflow3dhit> 
        gatherKPVetoedHitsNearVertex( larflow::reco::NuVertexCandidate& nuvtx, 
                                      const larlite::event_larflow3dhit& kpvetoed_hits_v,
                                      const float collection_radius_cm  );

        std::vector<float> 
        getHitDistancesFromProngEnds( const std::vector<float>& vtxpos,
                                      const std::vector<float>& prong_start, 
                                      const std::vector<float>& prong_dir,
                                      const std::vector<larlite::larflow3dhit>& nearby_kpvetoed_hits_v );

        void restoreVertexHits( larflow::reco::NuVertexCandidate& nuvtx,
                                larlite::event_larflow3dhit& ev_kpvetoed );

        protected:

        std::string _input_kpvetoed_hit_treename;
        float _collection_radius_cm;

#ifndef __CLING__
#ifndef __CINT__
        // hide this internal class from ROOTs interpretter and dictionary maker
        class KPDistArray_t {
        
        public:

            KPDistArray_t( int npts, int nprongs ) 
            : _nprongs(nprongs), 
            _npts(npts)
            {
                _array = new float[nprongs*npts];
                std::fill( _array, _array + nprongs*npts, 999.0f);
            };

            ~KPDistArray_t() {
                delete [] _array;
            };

        protected:

            int _nprongs;
            int _npts;
            float* _array;

            float* pget(int ipt, int iprong ) {
                // set stride so that we keep the dist values across prongs are in sequence
                // not that it matters
                return _array + ipt*_nprongs + iprong;
            };

        public:

            void set(int ipt, int iprong, float dist ) {
                *(pget(ipt,iprong)) = dist; 
            };

            float get(int ipt, int iprong) {
                return *(pget(ipt,iprong));
            };

            int get_closest_prong( int ipt ) {
                // why using raw arrays?
                // maybe i'm bored.
                // this is going to cause some horrible segfault or bug one day
                float* start = pget( ipt, 0);
                float* end   = pget( ipt, _nprongs);
                auto min_iter = std::min_element(start, end);
                size_t min_index = std::distance(start, min_iter);
                return (int)min_index;
            };

        };
#endif
#endif

    };

}
}

#endif