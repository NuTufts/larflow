#ifndef __LARFLOW_PREP_PREPLARMATCHDATA_H__
#define __LARFLOW_PREP_PREPLARMATCHDATA_H__


#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/mcshower.h"
#include "larlite/DataFormat/mctrack.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/Processor/ProcessBase.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "FlowTriples.h"
#include "MatchTriplets.h"
#include "TH2D.h"
#include "TMatrixD.h"

namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData 
   *
   * @class PrepMatchTriplets
   *
   * @brief Prepares potential spacepoints deriving from intersection of three wires.
   *
   * @author Taritree Wongjirad (taritree.wongjirad@tuts.edu)
   * @date $Data 2020/07/22 17:00$
   *
   * Revision history
   * 2020/07/22: Added doxygen documentation. 
   * 
   *
   */  
  class PrepMatchTriplets : public larcv::larcv_base {

  public:

    PrepMatchTriplets()
      : larcv::larcv_base("PrepMatchTriplets"),
	// _kStopAtTripletMax(false),
	// _kTripletLimit(1000000),
	// _kshuffle_indices_when_sampling(true),
	// _do_deadch_bug(false),
	_input_overlap_filepath("output_icarus_wireoverlap_matrices.root"),
	_overlap_matrices_loaded(false),
	_kAllowInductionPass(false)
    {};
    virtual ~PrepMatchTriplets() {};
    
    void process_tpc_v2( const std::vector<const larcv::Image2D*>& adc_v,
			 const std::vector<const larcv::Image2D*>& badch_v,
			 const float adc_threshold,
			 const int tpcid, const int cryoid );

    
    void process( larcv::IOManager& iolcv,
                  std::string wire_producer,
                  std::string chstatus_producer,
                  const float adc_threshold=10.0,
                  const bool check_wire_intersection=false );

    void process_truth_labels( larcv::IOManager& iolcv, larlite::storage_manager& ioll, std::string wire_producer="wire" );
    void process_truth_labels_fromsimch( larcv::IOManager& iolcv, larlite::storage_manager& ioll, std::string wire_producer="wire" );
    
    void make_truth_vector( const std::vector<larcv::Image2D>& larflow_v,
			    const std::vector<larcv::Image2D>& instance_v );
    void make_instanceid_vector( const std::vector<larcv::Image2D>& instance_v );
    void make_ancestorid_vector( const std::vector<larcv::Image2D>& ancestor_v );    
    void make_segmentid_vector( const std::vector<larcv::Image2D>& segment_img_v,
                                const std::vector<larcv::Image2D>& adc_v );
    void make_origin_vector_frommcreco( larlite::storage_manager& ioll );    

    // void setStopAtTripletMax( bool stop, int limit=1000000) { _kStopAtTripletMax = stop; _kTripletLimit = limit; };
    // void setShuffleWhenSampling( bool shuffle ) { _kshuffle_indices_when_sampling = shuffle; };
    // void setDoDeadChannelBug( bool doit ) { _do_deadch_bug = doit; };

    // std::vector<int> get_triplet_imgcoord_rowcol( int idx_triplet );
    
    // std::vector<TH2D> plot_sparse_images( std::string hist_stem_name );
                                          
    // std::vector<TH2D> plot_truth_images( std::string hist_stem_name );

    std::vector< MatchTriplets > _match_triplet_v; ///< data objects produced by this code    
    void clear();

    void allowInductionPass( bool doit=true ) { _kAllowInductionPass = doit; };
    
  protected:

    // bool _kStopAtTripletMax;
    // int  _kTripletLimit;
    // bool _kshuffle_indices_when_sampling;
    // bool _do_deadch_bug;

    // map from shower daughter IDs to mother IDs
    std::map<unsigned long, unsigned long> _shower_daughter2mother;
    void fill_daughter2mother_map( const std::vector<larlite::mcshower>& shower_v );

    std::map<unsigned long, int> _instance2class_map;
    void fill_class_map( const std::vector<larlite::mctrack>&  track_v,
                         const std::vector<larlite::mcshower>& shower_v );

    // wire overlap matrices
    std::vector< TMatrixD >           _matrix_list_v;
    std::map< std::vector<int>, int > _m_planeid_to_tree_entry;
    std::string                       _input_overlap_filepath;
    bool                              _overlap_matrices_loaded;
    bool                              _kAllowInductionPass;
    void _load_overlap_matrices( bool force_reload=false );

  public:
    
    void clear_wireoverlap_matrices() {
      _matrix_list_v.clear();
      _m_planeid_to_tree_entry.clear();
      _overlap_matrices_loaded=false;
    };
    void set_wireoverlap_filepath( std::string fpath ) { _input_overlap_filepath=fpath; };

  private:
    
    static bool _setup_numpy; ///< true if numpy has been setup
    
  };

}
}

#endif
