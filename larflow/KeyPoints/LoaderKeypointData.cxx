#include "LoaderKeypointData.h"
#include <iostream>
#include "larflow/PrepFlowMatchData/PrepSSNetTriplet.h"

namespace larflow {
namespace keypoints {

  bool LoaderKeypointData::_setup_numpy = false;
  
  /**
   * @brief constructor given list of input files
   *
   * @param[in] input_v List of paths to input ROOT files containing ground truth data
   *
   */
  LoaderKeypointData::LoaderKeypointData( std::vector<std::string>& input_v )
    : larcv::larcv_base("LoaderKeypointData"),
      _exclude_neg_examples(false),	      
      ttriplet(nullptr),
      tkeypoint(nullptr),
      tssnet(nullptr),
      tlarbysmc(nullptr)
  {
    input_files.clear();
    input_files = input_v;
    load_tree();
  }

  LoaderKeypointData::~LoaderKeypointData()
  {
    if ( ttriplet )  delete ttriplet;
    if ( tkeypoint)  delete tkeypoint;
    if ( tssnet )    delete tssnet;
    if ( tlarbysmc ) delete tlarbysmc;
  }

  /**
   * @brief load TTree class data members and define TBranch variables
   *
   */
  void LoaderKeypointData::load_tree() {
    LARCV_INFO() << "start" << std::endl;
    
    ttriplet  = new TChain("larmatchtriplet");
    tkeypoint = new TChain("keypointlabels");
    tssnet    = new TChain("ssnetlabels");
    tlarbysmc = new TChain("LArbysMCTree");
    for (auto const& infile : input_files ) {
      //std::cout << "add " << infile << " to chains" << std::endl;
      ttriplet->Add(infile.c_str());
      tkeypoint->Add(infile.c_str());
      tssnet->Add(infile.c_str());
      tlarbysmc->Add(infile.c_str());
    }
    LARCV_INFO() << "[LoaderKeypointData::load_tree()] " << input_files.size() << "files added" << std::endl;
    
    triplet_v = 0;
    kpdata_v = 0;
    ssnet_v = 0;
    
    _run    = 0;
    _subrun = 0;
    _event  = 0;
    
    ttriplet->SetBranchAddress(  "triplet_v",           &triplet_v );
    tkeypoint->SetBranchAddress( "keypoint_tpclabel_v", &kpdata_v );
    tssnet->SetBranchAddress( "ssnet_labeldata_v",      &ssnet_v );

    tkeypoint->SetBranchAddress("run",    &_run );
    tkeypoint->SetBranchAddress("subrun", &_subrun );
    tkeypoint->SetBranchAddress("event",  &_event ); 
    
    if ( tlarbysmc->GetEntries()>0 ) {
      has_larbysmc = true;
      tlarbysmc->SetBranchAddress( "vtx_sce_x", &vtx_sce_x );
      tlarbysmc->SetBranchAddress( "vtx_sce_y", &vtx_sce_y );
      tlarbysmc->SetBranchAddress( "vtx_sce_z", &vtx_sce_z );      
    }
    else {
      has_larbysmc = false;
    }

  }

  /**
   * @brief load event data for the different trees
   *
   * @param[in] entry number
   * @return number of bytes loaded from the tkeypoint tree data. returns 0 if end of file or error.
   */
  unsigned long LoaderKeypointData::load_entry( int entry )
  {
    unsigned long bytes = ttriplet->GetEntry(entry);
    bytes += tssnet->GetEntry(entry);
    bytes += tkeypoint->GetEntry(entry);
    if ( has_larbysmc )
      bytes += tlarbysmc->GetEntry(entry);

    LARCV_INFO() << "Loaded trees (ttriplet,tssnet,tkeypoint)" << std::endl;
    return bytes;
  }

  /**
   * @brief get total entries
   *
   * @return number of entries in the ttrplet ROOT tree (chain)
   */
  unsigned long LoaderKeypointData::GetEntries()
  {
    return ttriplet->GetEntries();
  }

  /**
   * @brief return a ground truth data, return a subsample of all truth matches
   *
   * returns a python dictionary. The dictionary contents are:
   * \verbatim embed:rst:leading-asterisk
   *  * "matchtriplet":numpy array with sparse image indices for each place, representing pixels a candidate space point project into
   *  * "match_weight":weight of "matchtriplet" examples
   *  * "positive_indices":indices of entries in "matchtriplet" array that correspond to good/true spacepoints
   *  * "ssnet_label":class label for space point
   *  * "ssnet_top_weight":weight based on topology (i.e. on boundary, near nu-vertex)
   *  * "ssnet_class_weight":weight based on class frequency
   *  * "kplabel":keypoint score numpy array
   *  * "kplabel_weight":weight for keypoint label
   *  * "kpshift":shift in 3D from space point position to nearest keypoint
   * \endverbatim
   *
   * @param[in]  num_max_samples maximum number of space points for which we return ground truth data
   * @param[out] nfilled The number of space points, for which we actually return data
   * @param[in]  withtruth withtruth If true, return info on whether space point is true (i.e. good)
   * @return Python dictionary object with various numpy arrays
   *                        
   */
  PyObject* LoaderKeypointData::sample_data( const int& num_max_samples,
                                             int& nfilled,
                                             bool withtruth,
					     int tpcid,
					     int cryoid )
  {


    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int index_col = (withtruth) ? 4 : 3;
    
    // make match index array
    LARCV_DEBUG() << "make triplets (cryo,tpcid)=(" << cryoid << "," << tpcid << ")" << std::endl;
    if ( _exclude_neg_examples )
      LARCV_DEBUG() << "exclude negative examples" << std::endl;
    else
      LARCV_DEBUG() << "include both negative and positive examples" << std::endl;

    int imatchdata = 0;
    for (int i=0; i<(int)triplet_v->size();i++) {
      if ( triplet_v->at(i)._tpcid==tpcid && triplet_v->at(i)._cryoid==cryoid ) {
	imatchdata = i;
	break;
      }
    }

    auto const& matchdata = triplet_v->at(imatchdata);
    auto const& kpdata    = kpdata_v->at(imatchdata);
    auto const& ssdata    = ssnet_v->at(imatchdata);
    
    PyArrayObject* matches = (PyArrayObject*)matchdata.sample_triplet_matches( num_max_samples, nfilled, withtruth );

    // count npos, nneg examples
    // also make list of indices of positive examples, these are the ones we will evaluate ssnet not
    int npos=0;
    int nneg=0;
    std::vector<int> pos_index_v;
    for (size_t i=0; i<nfilled; i++) {
      long ispositive = *((long*)PyArray_GETPTR2(matches,i,3));
      if (ispositive==1) {
        npos++;
        pos_index_v.push_back(i);
      }
      else {
        nneg++;
      }
    }
    LARCV_DEBUG() << " npos=" << npos << " nneg=" <<  nneg << std::endl;
    PyObject *match_key = Py_BuildValue("s", "matchtriplet");

    // make match weight array
    npy_intp match_weight_dim[] = { num_max_samples };
    PyArrayObject* match_weights = (PyArrayObject*)PyArray_SimpleNew( 1, match_weight_dim, NPY_FLOAT );
    float w_pos = (npos) ? float(npos+nneg)/float(npos) : 0.;
    float w_neg = (nneg) ? float(npos+nneg)/float(nneg) : 0.;
    float w_norm = w_pos*npos + w_neg*nneg;
    for (int i=0; i<num_max_samples; i++ ) {
      long ispositive = *((long*)PyArray_GETPTR2(matches,i,3));      
      if ( ispositive )
        *((float*)PyArray_GETPTR1(match_weights,i)) = w_pos/w_norm;
      else
        *((float*)PyArray_GETPTR1(match_weights,i)) = w_neg/w_norm;
    }
    PyObject *match_weight_key = Py_BuildValue("s", "match_weight");
    
    // make index array
    npy_intp pos_dim[] = { (long)pos_index_v.size() };
    PyArrayObject* positive_index = (PyArrayObject*)PyArray_SimpleNew( 1, pos_dim, NPY_LONG );
    for (size_t i=0; i<pos_index_v.size(); i++) {
      *((long*)PyArray_GETPTR1(positive_index,i)) = pos_index_v[i];
    }
    PyObject *pos_indices_key = Py_BuildValue("s", "positive_indices");


    // SSNET Arrays
    LARCV_DEBUG() << "call make_ssnet_arrays" << std::endl;
    PyArrayObject* ssnet_label  = nullptr;
    PyArrayObject* ssnet_weight = nullptr;
    PyArrayObject* ssnet_class_weight = nullptr;
    try {
      make_ssnet_arrays( num_max_samples, imatchdata, nfilled, withtruth, pos_index_v,
			 matches, ssnet_label, ssnet_weight, ssnet_class_weight );
    }catch (std::exception& e ) {
      LARCV_CRITICAL() << "error: " << e.what() << std::endl;
      throw std::runtime_error(e.what());
    }
    PyObject *ssnet_label_key        = Py_BuildValue("s", "ssnet_label" );
    PyObject *ssnet_top_weight_key   = Py_BuildValue("s", "ssnet_top_weight" );
    PyObject *ssnet_class_weight_key = Py_BuildValue("s", "ssnet_class_weight" );        

    // KP-LABEL ARRAY
    PyArrayObject* kplabel_label  = nullptr;
    PyArrayObject* kplabel_weight = nullptr;
    make_kplabel_arrays( num_max_samples, imatchdata, nfilled, withtruth, pos_index_v,
                         matches, kplabel_label, kplabel_weight );
    PyObject *kp_label_key     = Py_BuildValue("s", "kplabel" );
    PyObject *kp_weight_key    = Py_BuildValue("s", "kplabel_weight" );

    // KP-SHIFT ARRAY
    // PyArrayObject* kpshift_label = nullptr;
    // make_kpshift_arrays( num_max_samples, nfilled, withtruth,
    //                      matches, kpshift_label );
    // PyObject *kp_shift_key     = Py_BuildValue("s", "kpshift" );


    PyObject *d = PyDict_New();
    PyDict_SetItem(d, match_key,              (PyObject*)matches);        
    PyDict_SetItem(d, match_weight_key,       (PyObject*)match_weights);    
    PyDict_SetItem(d, pos_indices_key,        (PyObject*)positive_index);
    PyDict_SetItem(d, ssnet_label_key,        (PyObject*)ssnet_label );
    PyDict_SetItem(d, ssnet_top_weight_key,   (PyObject*)ssnet_weight );
    PyDict_SetItem(d, ssnet_class_weight_key, (PyObject*)ssnet_class_weight );
    PyDict_SetItem(d, kp_label_key,           (PyObject*)kplabel_label );
    PyDict_SetItem(d, kp_weight_key,          (PyObject*)kplabel_weight ); 
    //PyDict_SetItem(d, kp_shift_key,           (PyObject*)kpshift_label );

    Py_DECREF(match_key);
    Py_DECREF(match_weight_key);
    Py_DECREF(pos_indices_key);
    Py_DECREF(ssnet_label_key);
    Py_DECREF(ssnet_top_weight_key);
    Py_DECREF(ssnet_class_weight_key);
    Py_DECREF(kp_label_key);
    Py_DECREF(kp_weight_key);
    //Py_DECREF(kp_shift_key); 
    
    Py_DECREF(matches);
    Py_DECREF(match_weights);
    Py_DECREF(positive_index);
    Py_DECREF(ssnet_label);
    Py_DECREF(ssnet_weight);
    Py_DECREF(ssnet_class_weight);
    Py_DECREF(kplabel_label);
    Py_DECREF(kplabel_weight);
    //Py_DECREF(kpshift_label);    

    return d;
  }

  /**
   * @brief make the ssnet numpy arrays 
   *
   * @param[in]  num_max_samples Max number of samples to return
   * @param[out] nfilled number of samples actually returned
   * @param[in]  withtruth if true, return flag indicating if true/good space point
   * @param[out] pos_match_index vector index in return samples for space points which are true/good
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @param[out] ssnet_label numpy array containing ssnet class labels for each spacepoint
   * @param[out] ssnet_top_weight numpy array containing topological weight
   * @param[out] ssnet_class_weight numpy array containing class weights
   * @return always returns 0
   *
   */
  int LoaderKeypointData::make_ssnet_arrays( const int& num_max_samples,
					     const int imatchdata,
                                             int& nfilled,
                                             bool withtruth,
                                             std::vector<int>& pos_match_index,
                                             PyArrayObject* match_array,
                                             PyArrayObject*& ssnet_label,
                                             PyArrayObject*& ssnet_top_weight,
                                             PyArrayObject*& ssnet_class_weight )    
  {

    int index_col = (withtruth) ? 4 : 3;

    LARCV_DEBUG() << "pos_match_index=" << pos_match_index.size() << " withtruth=" << withtruth << " num_max_samples=" << num_max_samples << std::endl;
    //std::cout << "pos_match_index=" << pos_match_index.size() << " withtruth=" << withtruth << " num_max_samples=" << num_max_samples << std::endl;

    auto const& ssnetdata = ssnet_v->at(imatchdata); /// get data for a certain tpc
    
    // make ssnet label array
    int ssnet_label_nd = 1;
    npy_intp ssnet_label_dims1[] = { (long)pos_match_index.size() };
    npy_intp ssnet_label_dims2[] = { (long)pos_match_index.size() };
    npy_intp ssnet_label_dims3[] = { (long)pos_match_index.size() };

    if ( !_exclude_neg_examples ) {
      // we're going to load negative triplet examples too
      ssnet_label_dims1[0] = num_max_samples;
      ssnet_label_dims2[0] = num_max_samples;
      ssnet_label_dims3[0] = num_max_samples;
    }

    ssnet_label        = (PyArrayObject*)PyArray_SimpleNew( ssnet_label_nd, ssnet_label_dims1, NPY_LONG );
    ssnet_top_weight   = (PyArrayObject*)PyArray_SimpleNew( ssnet_label_nd, ssnet_label_dims2, NPY_FLOAT );
    ssnet_class_weight = (PyArrayObject*)PyArray_SimpleNew( ssnet_label_nd, ssnet_label_dims3, NPY_FLOAT );

    std::vector<int> nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    LARCV_DEBUG() << "make class labels and topological weight arrays. nelems=" << ssnet_label_dims1[0] << std::endl;
    if (_exclude_neg_examples)
      LARCV_DEBUG() << "EXCLUDING NEG EXAMPLES" << std::endl;
    
    int nbad_labels = 0;
    for ( int i=0; i<(int)ssnet_label_dims1[0]; i++ ) {

      // get the sample index
      int idx = (_exclude_neg_examples ) ? pos_match_index[i] : i;

      //LARCV_DEBUG() << " i=" << i << " idx=" << idx << " " << _exclude_neg_examples << std::endl;

      //if ( idx<0 || idx>=match
      
      // get the triplet index
      long index = *((long*)PyArray_GETPTR2(match_array,idx,index_col));

      // get ssnet index
      if (index<0 || index>=(int)ssnetdata._ssnet_label_v.size()) {
	std::stringstream msg;
	msg << "invalid index for ssnet_label_v. index=" << index
	    << " size=" << ssnetdata._ssnet_label_v.size()
	    << " i=" << i	  
	    << " idx=" << idx
	    << " exclude=" << _exclude_neg_examples
	    << std::endl;
	LARCV_CRITICAL() << msg.str() << std::endl;
	throw std::runtime_error( msg.str() );
      }
      
      int label = ssnetdata._ssnet_label_v.at( index );
      if (label<0 || label>=larflow::prep::PrepSSNetTriplet::kNumClasses) {
	std::stringstream msg;
	msg << "invalid class label=" << label << " from the Tree" << std::endl;
	//throw std::runtime_error( msg.str() );
	label = 0;
	nbad_labels++;
	//std::cout << msg.str() << std::endl;
      }
      nclass[label]++;

      *((long*)PyArray_GETPTR1(ssnet_label,i))       = (long)label; // class label
      *((float*)PyArray_GETPTR1(ssnet_top_weight,i)) = (float)ssnetdata._ssnet_weight_v.at( index ); // topological weight
    }
    
    LARCV_DEBUG() << "make class balancing weights" << std::endl;
    
    // calculate class-balancing weights
    int ntot = (int)ssnet_label_dims1[0];
    std::vector<float> w_class( larflow::prep::PrepSSNetTriplet::kNumClasses, 0.0 );
    float w_norm  = 0.;
    for (int i=0; i<(int)nclass.size(); i++) {
      if ( nclass[i]>0 )
	w_class[i] = 1.0/float(nclass[i]);
      else
	w_class[i] = 0.0;
      w_norm += w_class[i];
    }
    if ( w_norm>0 ) {
      for (int i=0; i<(int)nclass.size(); i++)
	w_class[i] /= w_norm;
    }
    
    for ( int i=0; i<(int)ssnet_label_dims1[0]; i++ ) {
      long label = *((long*)PyArray_GETPTR1(ssnet_label,i));
      if (label<0 || label>=larflow::prep::PrepSSNetTriplet::kNumClasses) {
	std::stringstream msg;
	msg << "invalid class label=" << label << " from the Tree" << std::endl;
	throw std::runtime_error( msg.str() );
      }      
      *((float*)PyArray_GETPTR1(ssnet_class_weight,i)) = w_class[label];
    }

    LARCV_DEBUG() << "Num bad labels: " << nbad_labels << std::endl;
    
    return 0;
  }

  /**
   * @brief make keypoint ground truth numpy arrays
   *
   * @param[in]  num_max_samples Max number of samples to return
   * @param[out] nfilled number of samples actually returned
   * @param[in]  withtruth if true, return flag indicating if true/good space point
   * @param[out] pos_match_index vector index in return samples for space points which are true/good
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @param[out] kplabel_label numpy array containing ssnet class labels for each spacepoint
   * @param[out] kplabel_weight numpy array containing weight for each spacepoint
   * @return always returns 0  
   */
  int LoaderKeypointData::make_kplabel_arrays( const int& num_max_samples,
					       const int imatchdata,
					       int& nfilled,
					       bool withtruth,
					       std::vector<int>& pos_match_index,
					       PyArrayObject* match_array,
					       PyArrayObject*& kplabel_label,
					       PyArrayObject*& kplabel_weight )
  {

    auto const& kpdata = kpdata_v->at(imatchdata);
    
    int index_col = (withtruth) ? 4 : 3;
    float sigma = 2.0; // cm

    int kplabel_nd = 2;
    int nclasses = 6; //number of keypoint classes
    npy_intp kplabel_dims[] = { (long)pos_match_index.size(), (long)nclasses };

    if ( !_exclude_neg_examples ) {
      kplabel_dims[0] = num_max_samples;
    }
    
    //std::cout << "make kplabel: " << kplabel_dims[0] << std::endl;    
    kplabel_label = (PyArrayObject*)kpdata.get_triplet_score_array(sigma);
    //(PyArrayObject*)PyArray_SimpleNew( kplabel_nd, kplabel_dims, NPY_FLOAT );

    std::vector<int> npos(nclasses,0);
    std::vector<int> nneg(nclasses,0);

    for (int i=0; i<(int)kplabel_dims[0]; i++ ) {
      // sample array index
      int idx = (_exclude_neg_examples) ? pos_match_index[i] : (int)i;

      // triplet index
      long index = *((long*)PyArray_GETPTR2(match_array,idx,index_col));

      for (int c=0; c<nclasses; c++) {
	
	//std::cout << "[" << i << "," << c << "] index=" << index << " size=" << kplabel_v[c]->at( index ).size() << std::endl;	
	float label = *((float*)PyArray_GETPTR2(kplabel_label,i,c));
	if( label>0 )
	  // increment number of positive keypoint labels
	  npos[c]++;
	else 
	  nneg[c]++;
      }
    }

    // weights to balance positive and negative examples
    int kpweight_nd = 2;
    npy_intp kpweight_dims[] = { (long)pos_match_index.size(), nclasses };
    if ( !_exclude_neg_examples )
      kpweight_dims[0] = num_max_samples;
    kplabel_weight = (PyArrayObject*)PyArray_SimpleNew( kpweight_nd, kpweight_dims, NPY_FLOAT );

    for (int c=0; c<nclasses; c++ ) {
      float w_pos = (npos[c]) ? float(npos[c]+nneg[c])/float(npos[c]) : 0.0;
      float w_neg = (nneg[c]) ? float(npos[c]+nneg[c])/float(nneg[c]) : 0.0;
      float w_norm = w_pos*npos[c] + w_neg*nneg[c];

      //std::cout << "Keypoint class[" << c << "] WEIGHT: W(POS)=" << w_pos/w_norm << " W(NEG)=" << w_neg/w_norm << std::endl;
    
      for (int i=0; i<kpweight_dims[0]; i++ ) {
        // sample array index
        int idx = (_exclude_neg_examples) ? pos_match_index[i] : i;
        // triplet index
        long index = *((long*)PyArray_GETPTR2(match_array,idx,index_col));

	float label = *((float*)PyArray_GETPTR2(kplabel_label,i,c));	
	if (label>0) {
	  *((float*)PyArray_GETPTR2(kplabel_weight,i,c)) = w_pos/w_norm;
	}
        else {
          *((float*)PyArray_GETPTR2(kplabel_weight,i,c))  = w_neg/w_norm;
        }
      }
    }//end of class loop
    
    return 0;
  }

  /**
   * @brief make keypoint shift ground truth numpy arrays
   *
   * @param[in]  num_max_samples Max number of samples to return
   * @param[out] nfilled number of samples actually returned
   * @param[in]  withtruth if true, return flag indicating if true/good space point
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @param[out] kpshift_label numpy array containing ground truth position shifts
   * @return always returns 0  
   */  
  // int LoaderKeypointData::make_kpshift_arrays( const int& num_max_samples,
  // 					       int& nfilled,
  // 					       bool withtruth,
  // 					       PyArrayObject* match_array,
  // 					       PyArrayObject*& kpshift_label )
  // {

  //   int index_col = (withtruth) ? 4 : 3;

  //   // make keypoint shift array
  //   int kpshift_nd = 3;
  //   int nclasses = 3;
  //   npy_intp kpshift_dims[] = { num_max_samples, nclasses, 3 };
  //   //std::cout << "make kpshift: " << kpshift_dims[0] << "," << kpshift_dims[1] << std::endl;    
  //   kpshift_label = (PyArrayObject*)PyArray_SimpleNew( kpshift_nd, kpshift_dims, NPY_FLOAT );

  //   for (int i=0; i<num_max_samples; i++ ) {
  //     long index = *((long*)PyArray_GETPTR2(match_array,i,index_col));
  //     for (int c=0; c<nclasses; c++ ) {
	
  //       if ( i<nfilled && kplabel_v[c]->at( index ).size()>0 ) {
  //         for (int j=0; j<3; j++ )
  //           *((float*)PyArray_GETPTR3(kpshift_label,i,c,j)) = kplabel_v[c]->at( index )[1+j];
  //       }
  //       else{
  //         for (int j=0; j<3; j++ )
  //           *((float*)PyArray_GETPTR3(kpshift_label,i,c,j)) = 0.;
  //       }
  //     }
  //   }
    
  //   return 0;
  // }

  std::vector< std::vector<float> > LoaderKeypointData::get_keypoint_pos(const int imatchdata) const
  {

    auto const& kpdata = kpdata_v->at(imatchdata);
    
    std::vector< std::vector<float> > kp_pos;
    for ( size_t ikpd=0; ikpd<kpdata._kpd_v.size(); ikpd++ ) {    
      kp_pos.push_back( kpdata._kpd_v.at(ikpd).keypt );
    }
    return kp_pos;
  }

  std::vector< int > LoaderKeypointData::get_keypoint_types( const int imatchdata ) const
  {
    auto const& kpdata = kpdata_v->at(imatchdata);    
    std::vector< int > kp_types;
    for ( size_t ikpd=0; ikpd<kpdata._kpd_v.size(); ikpd++ ) {
      kp_types.push_back( kpdata._kpd_v[ikpd].kptype );
    }
    
    return kp_types;
  }

  // std::vector< std::vector<int> > LoaderKeypointData::get_keypoint_pdg_and_trackid( const int imatchdata ) const
  // {
  //   auto const& kpdata = kpdata_v->at(imatchdata);        
  //   std::vector< std::vector<int> > kp_pdgtrackid;
  //   for (int n=0; n<6; n++) {
  //     int nkp = (int)kpdata.kptruth_v[n].size();
  //     for (int i=0; i<nkp; i++) {
  // 	const std::vector<int>& pdgtrackid = kptruth_v[n].at(i);	
  // 	kp_pdgtrackid.push_back( pdgtrackid );
  //     }
  //   }
  //   return kp_pdgtrackid;
  // }
  
}
}
