#ifndef __FLOW_MATCH_HIT_MAKER_H__
#define __FLOW_MATCH_HIT_MAKER_H__

//struct _object;
//typedef _object PyObject;

#include <Python.h>
#include "bytesobject.h"
#include <vector>
#include <map>
#include <array>
#include "larcv/core/DataFormat/EventChStatus.h"
#include "core/DataFormat/larflow3dhit.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/Image2D.h"


namespace larflow {

  class FlowMatchHitMaker {
  public:
    
    FlowMatchHitMaker()
      : _match_score_threshold(0.5) {};
    virtual ~FlowMatchHitMaker() {};

    typedef struct match_t {
      
      int Y; // Y-wire
      int U; // U-wire
      int V; // V-wire

      float YU; // match between Y to U
      float YV; // match between Y to V
      float UV; // match between U to V
      float UY; // match between U to Y
      float VU; // match between V to U
      float VY; // match between V to Y

      std::array<float,3> tyz;

      match_t()
      : Y(0),U(0),V(0),
        YU(0),YV(0),UV(0),UY(0),VU(0),VY(0)
      {};
      
      bool operator<(const match_t& rhs) const {
        if (Y<rhs.Y) return true;
        else if ( Y==rhs.Y && U<rhs.U ) return true;
        else if ( Y==rhs.Y && U==rhs.U && V<rhs.V ) return true;
        return false;
      };

      void set_wire( int plane, int wire ) {
        if ( plane==0 )      U = wire;
        else if ( plane==1 ) V = wire;
        else if ( plane==2 ) Y = wire;
      };

      void set_score ( int source_plane, int target_plane, float prob ) {
        if ( source_plane==2 ) {
          if ( target_plane==0 && prob>YU )      YU = prob;
          else if ( target_plane==1 && prob>YV ) YV = prob;
        }
        else if ( source_plane==1 ) {
          if ( target_plane==0 && prob>VU )      VU = prob;
          else if ( target_plane==2 && prob>VY ) VY = prob;
        }
        else if ( source_plane==0 ) {
          if ( target_plane==1 && prob>UV )      UV = prob;
          else if ( target_plane==2 && prob>UY ) UY = prob;
        };
      };

      std::vector<float> get_scores() const {
        std::vector<float> s(6,0);
        s[0] = YU;
        s[1] = YV;
        s[2] = UV;
        s[3] = UY;
        s[4] = VU;
        s[5] = VY;
        return s;
      };
      
    } match_t;


    float _match_score_threshold;
    std::vector<match_t> _matches_v; //< vector of match information
    std::map< std::vector<int>, int > _match_map; // map of (Y,U,V) triple to positin in matches_v
    

    void clear() { _matches_v.clear(); _match_map.clear(); };
    int add_match_data( PyObject* pair_probs,
                        PyObject* source_sparseimg, PyObject* target_sparseimg,
                        PyObject* matchpairs,
                        const int source_plane, const int target_plane,
                        const larcv::ImageMeta& source_meta,
                        const std::vector<larcv::Image2D>& img_v,
                        const larcv::EventChStatus& ev_chstatus );
    void set_score_threshold( float score ) { _match_score_threshold = score; };

    void make_hits( const larcv::EventChStatus& ev_chstatus,
                    std::vector<larlite::larflow3dhit>& hit_v )  const;

  };

}

#endif
