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
      tlarbysmc(nullptr),
      triplet_v(nullptr),
      kpshift_v(nullptr),
      ssnet_label_v(nullptr),
      ssnet_weight_v(nullptr),
      kpflow_labels_v(nullptr),
      _use_data_from_ttree(true)
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
    for (int i=0; i<6; i++) {
      kplabel_v[i] = 0;
      kppos_v[i] = 0;
      kptruth_v[i] = 0;
    }
    kpflow_labels_v = 0;
    ssnet_label_v = 0;
    ssnet_weight_v = 0;
    _run    = 0;
    _subrun = 0;
    _event  = 0;
    
    ttriplet->SetBranchAddress(  "triplet_v",           &triplet_v );

    tkeypoint->SetBranchAddress("run",    &_run );
    tkeypoint->SetBranchAddress("subrun", &_subrun );
    tkeypoint->SetBranchAddress("event",  &_event ); 
    
    tkeypoint->SetBranchAddress("kplabel_nuvertex",     &kplabel_v[0]);
    tkeypoint->SetBranchAddress("kplabel_trackstart",   &kplabel_v[1]);
    tkeypoint->SetBranchAddress("kplabel_trackend",     &kplabel_v[2]);    
    tkeypoint->SetBranchAddress("kplabel_showerstart",  &kplabel_v[3]);
    tkeypoint->SetBranchAddress("kplabel_showermichel", &kplabel_v[4]);
    tkeypoint->SetBranchAddress("kplabel_showerdelta",  &kplabel_v[5]);    

    tkeypoint->SetBranchAddress("kppos_nuvertex",     &kppos_v[0]);
    tkeypoint->SetBranchAddress("kppos_trackstart",   &kppos_v[1]);
    tkeypoint->SetBranchAddress("kppos_trackend",     &kppos_v[2]);    
    tkeypoint->SetBranchAddress("kppos_showerstart",  &kppos_v[3]);
    tkeypoint->SetBranchAddress("kppos_showermichel", &kppos_v[4]);
    tkeypoint->SetBranchAddress("kppos_showerdelta",  &kppos_v[5]);    

    tkeypoint->SetBranchAddress("kptruth_nuvertex",     &kptruth_v[0]);
    tkeypoint->SetBranchAddress("kptruth_trackstart",   &kptruth_v[1]);
    tkeypoint->SetBranchAddress("kptruth_trackend",     &kptruth_v[2]);    
    tkeypoint->SetBranchAddress("kptruth_showerstart",  &kptruth_v[3]);
    tkeypoint->SetBranchAddress("kptruth_showermichel", &kptruth_v[4]);
    tkeypoint->SetBranchAddress("kptruth_showerdelta",  &kptruth_v[5]);    
    
    tssnet->SetBranchAddress( "ssnet_label_v",    &ssnet_label_v );
    tssnet->SetBranchAddress( "ssnet_weight_v",   &ssnet_weight_v );

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
    for (int n=0; n<6; n++) {
      std::cout << " [" << n << "] num=" << kppos_v[n]->size() << std::endl;
    }
    
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
                                             bool withtruth )
  {


    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int index_col = (withtruth) ? 4 : 3;
    
    // make match index array
    LARCV_DEBUG() << "make triplets" << std::endl;
    if ( _exclude_neg_examples )
      LARCV_DEBUG() << "exclude negative examples" << std::endl;
    else
      LARCV_DEBUG() << "include both negative and positive examples" << std::endl;

    // get pointer to the PrepMatchTriplets instance that has made and stored our spacepoints and labels
    larflow::prep::PrepMatchTriplets* ptripletmaker = nullptr;
    if ( _use_data_from_ttree )
      ptripletmaker = &(triplet_v->at(0));
    else
      ptripletmaker = ptriplet_v.at(0);

    LARCV_NORMAL() << "Sample labels for triplets." << std::endl;   
    PyArrayObject* matches =
      (PyArrayObject*)ptripletmaker->sample_triplet_matches( num_max_samples, nfilled, withtruth );

    LARCV_NORMAL() << "Copying labels for " << nfilled << " triplets to numpy arrays" << std::endl;    
    
    // count npos, nneg examples
    // also make list of indices of positive examples, these are the ones we will evaluate ssnet not
    int npos=0;
    int nneg=0;
    std::vector<int> pos_index_v;
    pos_index_v.reserve(nfilled);
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
    npy_intp match_weight_dim[] = { nfilled };
    PyArrayObject* match_weights = (PyArrayObject*)PyArray_SimpleNew( 1, match_weight_dim, NPY_FLOAT );
    float w_pos = (npos) ? float(npos+nneg)/float(npos) : 0.;
    float w_neg = (nneg) ? float(npos+nneg)/float(nneg) : 0.;
    float w_norm = w_pos*npos + w_neg*nneg;
    for (int i=0; i<nfilled; i++ ) {
      long ispositive = *((long*)PyArray_GETPTR2(matches,i,3));      
      if ( ispositive )
        *((float*)PyArray_GETPTR1(match_weights,i)) = w_pos/w_norm;
      else
        *((float*)PyArray_GETPTR1(match_weights,i)) = w_neg/w_norm;
    }
    PyObject *match_weight_key = Py_BuildValue("s", "match_weight");

    LARCV_NORMAL() << "Prepare 3D positions of triplets" << std::endl;

    // make spacepoint position array
    int spacepoint_nd = 2;
    npy_intp spacepoint_dims[] = { (long)nfilled, 3 };    
    PyArrayObject* spacepoint_array = (PyArrayObject*)PyArray_SimpleNew( 2, spacepoint_dims, NPY_FLOAT );
    for (int ii=0; ii<nfilled; ii++) {
      // get the index of the triplet at array row ii
      long idx = *((long*)PyArray_GETPTR2(matches,ii,4));
      // fill the array row ii with the (x,y,z) position of the triplet with index idx
      for (int v=0; v<3; v++) {
	*((float*)PyArray_GETPTR2(spacepoint_array,ii,v)) = ptripletmaker->_pos_v.at(idx)[v];
      }
    }
    PyObject* spacepoint_key = Py_BuildValue("s", "spacepoints");

    LARCV_NORMAL() << "make positive (i.e. non-ghost) index array" << std::endl;
    
    // make index array
    npy_intp pos_dim[] = { (long)pos_index_v.size() };
    PyArrayObject* positive_index = (PyArrayObject*)PyArray_SimpleNew( 1, pos_dim, NPY_LONG );
    for (size_t i=0; i<pos_index_v.size(); i++) {
      *((long*)PyArray_GETPTR1(positive_index,i)) = pos_index_v[i];
    }
    PyObject *pos_indices_key = Py_BuildValue("s", "positive_indices");

    LARCV_NORMAL() << "there are " << pos_dim[0] << " positive spacepoints" << std::endl;

    // SSNET Arrays
    LARCV_NORMAL() << "call make_ssnet_arrays" << std::endl;
    PyArrayObject* ssnet_label  = nullptr;
    PyArrayObject* ssnet_weight = nullptr;
    PyArrayObject* ssnet_class_weight = nullptr;
    try {
      make_ssnet_arrays( num_max_samples, nfilled, withtruth, pos_index_v,
			 matches, ssnet_label, ssnet_weight, ssnet_class_weight );
    }catch (std::exception& e ) {
      LARCV_CRITICAL() << "error: " << e.what() << std::endl;
      throw std::runtime_error(e.what());
    }
    PyObject *ssnet_label_key        = Py_BuildValue("s", "ssnet_label" );
    PyObject *ssnet_top_weight_key   = Py_BuildValue("s", "ssnet_top_weight" );
    PyObject *ssnet_class_weight_key = Py_BuildValue("s", "ssnet_class_weight" );        

    // KP-LABEL ARRAY
    LARCV_NORMAL() << "make keypoint labels" << std::endl;    
    PyArrayObject* kplabel_label  = nullptr;
    PyArrayObject* kplabel_weight = nullptr;
    make_kplabel_arrays( num_max_samples, nfilled, withtruth, pos_index_v,
                         matches, kplabel_label, kplabel_weight );
    PyObject *kp_label_key     = Py_BuildValue("s", "kplabel" );
    PyObject *kp_weight_key    = Py_BuildValue("s", "kplabel_weight" );

    // KP-SHIFT ARRAY
    LARCV_NORMAL() << "make keypoint shift labels" << std::endl;        
    PyArrayObject* kpshift_label = nullptr;
    make_kpshift_arrays( num_max_samples, nfilled, withtruth,
                         matches, kpshift_label );
    PyObject *kp_shift_key     = Py_BuildValue("s", "kpshift" );

    // KP-ENERGY FLOW ARRAY (PAF: Particle affinity flow)
    LARCV_NORMAL() << "make spacepoint momentum flow array" << std::endl;            
    PyArrayObject* paf_label  = nullptr;
    PyArrayObject* paf_weight = nullptr;
    bool exclude_neg_examples = false;
    make_paf_arrays( nfilled, ///number of spacepoints to provide labels for
		     pos_index_v, // vector<int> where 1=true spaceoint and 0=ghost
		     exclude_neg_examples, //option to remove labels for ghost points (not used)
                     matches,
		     paf_label,
		     paf_weight );
    PyObject *paf_label_key     = Py_BuildValue("s", "paf_label" );
    PyObject *paf_weight_key    = Py_BuildValue("s", "paf_weight" );

    // ORIGIN FLAG: 0=noise, 1=neutrino, 2=cosmic
    LARCV_NORMAL() << "make origin flag array" << std::endl;            
    PyArrayObject* origin_array = nullptr;
    int err_origin = make_origin_array( nfilled,
					pos_index_v,
					exclude_neg_examples,
					matches,
					origin_array );
    PyObject* origin_key = Py_BuildValue("s","origin_label");

    // INSTANCE IDS
    LARCV_NORMAL() << "make instance ID array" << std::endl;
    PyArrayObject* instanceid_array = nullptr;
    int err_instanceid = make_instanceid_array( nfilled,
                          pos_index_v,
                          exclude_neg_examples,
                          matches,
                          instanceid_array );
    PyObject* instanceid_key = Py_BuildValue("s","instanceid_label");

    // List of true keypoint positions
    LARCV_NORMAL() << "Gathering true keypoint positions" << std::endl;
    std::vector< std::vector<float> > kp_pos_v = get_keypoint_pos();
    LARCV_NORMAL() << "Gathering true keypoint types" << std::endl;    
    std::vector< int >  kp_class               = get_keypoint_types();    
    LARCV_NORMAL() << "Gathering true keypoint PDGs and TrackIDs" << std::endl;    
    std::vector< std::vector<int> >   kp_ids   = get_keypoint_pdg_and_trackid();
    
    int nkps = kp_class.size();
    LARCV_NORMAL() << "Gathered truth for " << nkps << "keypoints" << std::endl;

    if ( nkps!=kp_pos_v.size() )
      LARCV_ERROR() << "number of keypoint types does not match number of keypoint positions" << std::endl;
    if ( nkps!=kp_ids.size() )
      LARCV_ERROR() << "number of keypoint types does not match number of keypoint PDGs and Trackids" << std::endl;
    
    npy_intp kptruth_dims[] = { nkps, 3 };
    npy_intp kppos_dims[]   = { nkps, 3 };
    PyArrayObject* kptruth_ids = (PyArrayObject*)PyArray_SimpleNew( 2, kptruth_dims, NPY_LONG );
    PyArrayObject* kptruth_pos = (PyArrayObject*)PyArray_SimpleNew( 2, kppos_dims, NPY_FLOAT );        
    for (int ikp=0; ikp<nkps; ikp++) {
      *((long*)PyArray_GETPTR2(kptruth_ids,ikp,0)) = (long)kp_class[ikp];  // keypoint class
      *((long*)PyArray_GETPTR2(kptruth_ids,ikp,1)) = (long)kp_ids[ikp][0]; // keypoint pdg
      *((long*)PyArray_GETPTR2(kptruth_ids,ikp,2)) = (long)kp_ids[ikp][1]; // keypoint geant4 trackid
      for (int v=0; v<3; v++) {
	*((float*)PyArray_GETPTR2(kptruth_pos,ikp,v)) = (float)kp_pos_v.at(ikp)[v];
      }
    }
    PyObject* kp_truth_ids_key = Py_BuildValue("s","keypoint_truth_kptype_pdg_trackid");
    PyObject* kp_truth_pos_key = Py_BuildValue("s","keypoint_truth_pos");

    // Need the ADC images
    PyObject* wireimg_plane0 = ptripletmaker->make_sparse_image(0);
    PyObject* wireimg_plane1 = ptripletmaker->make_sparse_image(1);
    PyObject* wireimg_plane2 = ptripletmaker->make_sparse_image(2);
    PyObject* wireimgkey_p0  = Py_BuildValue("s","wireimage_plane0");
    PyObject* wireimgkey_p1  = Py_BuildValue("s","wireimage_plane1");
    PyObject* wireimgkey_p2  = Py_BuildValue("s","wireimage_plane2");    


    PyObject *d = PyDict_New();
    PyDict_SetItem(d, match_key,              (PyObject*)matches);        
    PyDict_SetItem(d, match_weight_key,       (PyObject*)match_weights);
    PyDict_SetItem(d, spacepoint_key,         (PyObject*)spacepoint_array);
    PyDict_SetItem(d, pos_indices_key,        (PyObject*)positive_index);
    PyDict_SetItem(d, ssnet_label_key,        (PyObject*)ssnet_label );
    PyDict_SetItem(d, ssnet_top_weight_key,   (PyObject*)ssnet_weight );
    PyDict_SetItem(d, ssnet_class_weight_key, (PyObject*)ssnet_class_weight );
    PyDict_SetItem(d, kp_label_key,           (PyObject*)kplabel_label );
    PyDict_SetItem(d, kp_weight_key,          (PyObject*)kplabel_weight ); 
    PyDict_SetItem(d, kp_shift_key,           (PyObject*)kpshift_label );
    PyDict_SetItem(d, paf_label_key,          (PyObject*)paf_label );
    PyDict_SetItem(d, paf_weight_key,         (PyObject*)paf_weight );
    PyDict_SetItem(d, origin_key,             (PyObject*)origin_array );
    PyDict_SetItem(d, instanceid_key,         (PyObject*)instanceid_array);
    PyDict_SetItem(d, kp_truth_ids_key,       (PyObject*)kptruth_ids );
    PyDict_SetItem(d, kp_truth_pos_key,       (PyObject*)kptruth_pos );
    PyDict_SetItem(d, wireimgkey_p0,          wireimg_plane0 );
    PyDict_SetItem(d, wireimgkey_p1,          wireimg_plane1 );
    PyDict_SetItem(d, wireimgkey_p2,          wireimg_plane2 );

    // decrease reference counter to account for this function creating the objects
    Py_DECREF(match_key);
    Py_DECREF(match_weight_key);
    Py_DECREF(spacepoint_key);
    Py_DECREF(pos_indices_key);
    Py_DECREF(ssnet_label_key);
    Py_DECREF(ssnet_top_weight_key);
    Py_DECREF(ssnet_class_weight_key);
    Py_DECREF(kp_label_key);
    Py_DECREF(kp_weight_key);
    Py_DECREF(kp_shift_key);
    Py_DECREF(paf_label_key);
    Py_DECREF(paf_weight_key);
    Py_DECREF(origin_key);
    Py_DECREF(instanceid_key);
    Py_DECREF(kp_truth_ids_key);
    Py_DECREF(kp_truth_pos_key);
    Py_DECREF(wireimgkey_p0);
    Py_DECREF(wireimgkey_p1);
    Py_DECREF(wireimgkey_p2);    
    
    Py_DECREF(matches);
    Py_DECREF(match_weights);
    Py_DECREF(spacepoint_array);
    Py_DECREF(positive_index);
    Py_DECREF(ssnet_label);
    Py_DECREF(ssnet_weight);
    Py_DECREF(ssnet_class_weight);
    Py_DECREF(kplabel_label);
    Py_DECREF(kplabel_weight);
    Py_DECREF(kpshift_label);
    Py_DECREF(paf_label);
    Py_DECREF(paf_weight);
    Py_DECREF(origin_array);
    Py_DECREF(instanceid_array);
    Py_DECREF(kptruth_pos);
    Py_DECREF(kptruth_ids);
    Py_DECREF(wireimg_plane0);
    Py_DECREF(wireimg_plane1);
    Py_DECREF(wireimg_plane2);    
    
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
    
    // make ssnet label array
    int ssnet_label_nd = 1;
    npy_intp ssnet_label_dims1[] = { (long)pos_match_index.size() };
    npy_intp ssnet_label_dims2[] = { (long)pos_match_index.size() };
    npy_intp ssnet_label_dims3[] = { (long)pos_match_index.size() };

    if ( !_exclude_neg_examples ) {
      // we're going to load negative triplet examples too
      ssnet_label_dims1[0] = nfilled;
      ssnet_label_dims2[0] = nfilled;
      ssnet_label_dims3[0] = nfilled;
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
      if (index<0 || index>=(int)ssnet_label_v->size()) {
        std::stringstream msg;
        msg << "invalid index for ssnet_label_v. index=" << index
            << " size=" << ssnet_label_v->size()
            << " i=" << i	  
            << " idx=" << idx
            << " exclude=" << _exclude_neg_examples
            << std::endl;
        LARCV_CRITICAL() << msg.str() << std::endl;
        throw std::runtime_error( msg.str() );
      }
      
      int label = ssnet_label_v->at( index );
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
      *((float*)PyArray_GETPTR1(ssnet_top_weight,i)) = (float)ssnet_weight_v->at( index ); // topological weight
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
                                                int& nfilled,
                                                bool withtruth,
                                                std::vector<int>& pos_match_index,
                                                PyArrayObject* match_array,
                                                PyArrayObject*& kplabel_label,
                                                PyArrayObject*& kplabel_weight )
  {

    int index_col = (withtruth) ? 4 : 3;
    float sigma = 2.0; // cm

    int kplabel_nd = 2;
    int nclasses = 6; //number of keypoint classes
    npy_intp kplabel_dims[] = { (long)pos_match_index.size(), (long)nclasses };

    if ( !_exclude_neg_examples ) {
      kplabel_dims[0] = nfilled;
    }
    
    std::cout << "make kplabel array with " << kplabel_dims[0] << " rows" << std::endl;    
    kplabel_label = (PyArrayObject*)PyArray_SimpleNew( kplabel_nd, kplabel_dims, NPY_FLOAT );

    std::vector<int> npos(nclasses,0);
    std::vector<int> nneg(nclasses,0);

    for (int i=0; i<(int)kplabel_dims[0]; i++ ) {
      // sample array index
      int idx = (_exclude_neg_examples) ? pos_match_index[i] : (int)i;

      // triplet index
      long index = *((long*)PyArray_GETPTR2(match_array,idx,index_col));

      for (int c=0; c<nclasses; c++) {
	
	//std::cout << "[" << i << "," << c << "] index=" << index << " size=" << kplabel_v[c]->at( index ).size() << std::endl;
	
	if ( kplabel_v[c]->at( index ).size()==0 ) {
          // zero label
          *((float*)PyArray_GETPTR2(kplabel_label,i,c)) = 0.0;
          nneg[c]++;
	}
	else {
	  // hard label	  
	  long label = kplabel_v[c]->at( index )[0]; // [0] indicates if within some radius of keypoint
	  if (label==1) {
	    // make soft label
	    float dist = 0.;
	    for (int j=0; j<3; j++) {
	      float dx = kplabel_v[c]->at( index )[1+j]; // [1+j] is distance to closest keypoint in j-coordinate
	      dist += dx*dx;
	    }
	    // reassign hard label with value based on distance from closest keypoint
	    *((float*)PyArray_GETPTR2(kplabel_label,i,c)) = exp( -dist/(sigma*sigma) );
	    // increment number of positive keypoint labels
	    npos[c]++;
	  }
	  else {
	    // zero label
	    *((float*)PyArray_GETPTR2(kplabel_label,i,c)) = 0.0;
	    nneg[c]++;
	  }
	}//has labels
      }
    }

    // weights to balance positive and negative examples
    int kpweight_nd = 2;
    npy_intp kpweight_dims[] = { (long)pos_match_index.size(), nclasses };
    if ( !_exclude_neg_examples )
      kpweight_dims[0] = nfilled;
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

	long label = 0;
	if ( kplabel_v[c]->at( index ).size()>0 ) {
	  // hard label	
	  label = kplabel_v[c]->at( index )[0];
	}
	
	if (label==1) {
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
  int LoaderKeypointData::make_kpshift_arrays( const int& num_max_samples,
                                                int& nfilled,
                                                bool withtruth,
                                                PyArrayObject* match_array,
                                                PyArrayObject*& kpshift_label )
  {

    int index_col = (withtruth) ? 4 : 3;

    // make keypoint shift array
    int kpshift_nd = 3;
    int nclasses = 3;
    npy_intp kpshift_dims[] = { nfilled, nclasses, 3 };
    //std::cout << "make kpshift: " << kpshift_dims[0] << "," << kpshift_dims[1] << std::endl;    
    kpshift_label = (PyArrayObject*)PyArray_SimpleNew( kpshift_nd, kpshift_dims, NPY_FLOAT );

    for (int i=0; i<kpshift_dims[0]; i++ ) {
      long index = *((long*)PyArray_GETPTR2(match_array,i,index_col));
      for (int c=0; c<nclasses; c++ ) {
	
        if ( i<nfilled && kplabel_v[c]->at( index ).size()>0 ) {
          for (int j=0; j<3; j++ )
            *((float*)PyArray_GETPTR3(kpshift_label,i,c,j)) = kplabel_v[c]->at( index )[1+j];
        }
        else{
          for (int j=0; j<3; j++ )
            *((float*)PyArray_GETPTR3(kpshift_label,i,c,j)) = 0.;
        }
      }
    }
    
    return 0;
  }

  std::vector< std::vector<float> > LoaderKeypointData::get_keypoint_pos() const
  {
    std::vector< std::vector<float> > kp_pos;
    for (int n=0; n<6; n++) {
      int nkp = (int)kppos_v[n]->size();
      for (int i=0; i<nkp; i++) {
	kp_pos.push_back( kppos_v[n]->at(i) );
      }
    }
    return kp_pos;
  }

  std::vector< int > LoaderKeypointData::get_keypoint_types() const
  {
    std::vector< int > kp_types;
    for (int n=0; n<6; n++) {
      int nkp = (int)kppos_v[n]->size();
      for (int i=0; i<nkp; i++) {
	kp_types.push_back(n);
      }
    }
    return kp_types;
  }

  std::vector< std::vector<int> > LoaderKeypointData::get_keypoint_pdg_and_trackid() const
  {
    std::vector< std::vector<int> > kp_pdgtrackid;
    for (int n=0; n<6; n++) {
      int nkp = (int)kptruth_v[n]->size();
      for (int i=0; i<nkp; i++) {
	const std::vector<int>& pdgtrackid = kptruth_v[n]->at(i);	
	kp_pdgtrackid.push_back( pdgtrackid );
      }
    }
    return kp_pdgtrackid;
  }

  void LoaderKeypointData::provide_entry_data( larflow::prep::PrepMatchTriplets& triplets,
					       larflow::keypoints::PrepKeypointData& kpdata,
					       larflow::prep::PrepSSNetTriplet& ssnetdata,
					       larflow::keypoints::PrepAffinityField& kpdirflow )
  {
    
    ptriplet_v.clear();    
    ptriplet_v.push_back( &triplets );
    
    // transfer info from these members in PrepKeypointData
    //std::vector< std::vector<float> > _kppos_v[6]; ///< container of true keypoint 3D positions in cm, for each of the 6 classes
    //std::vector< std::vector<int> >   _kp_pdg_trackid_v[6]; ///< each entry maps (pdg, trackid) for truth meta-data matching
    for (int ikptype=0; ikptype<6; ikptype++) {
      kppos_v[ikptype]   = &(kpdata._kppos_v[ikptype]);
      kplabel_v[ikptype] = &(kpdata._match_proposal_labels_v[ikptype]);
      kptruth_v[ikptype] = &(kpdata._kp_pdg_trackid_v[ikptype]);
    }

    // ssnet containers
    ssnet_label_v   = &(ssnetdata._ssnet_label_v);
    ssnet_weight_v  = &(ssnetdata._ssnet_weight_v);

    // spacepoint direction containers
    kpflow_labels_v = &(kpdirflow._match_labels_v);

  }

  /**
   * @brief make particle affinity field ground truth numpy arrays
   *
   * @param[in]  nfilled number of samples actually returned
   * @param[in]  pos_match_index vector index in return samples for space points which are true/good
   * @param[in]  exclude_neg_examples If true, training samples return do not have negative/bad spacepoint examples
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @param[out] paf_label  numpy array containing target direction for each spacepoint. shape (N,3)
   * @param[out] paf_weight numpy array containing weight for each spacepoint. shape (N,)
   * @return always returns 0  
   */
  int LoaderKeypointData::make_paf_arrays( const int nfilled,
                                           const std::vector<int>& pos_match_index,
                                           const bool exclude_neg_examples,
                                           PyArrayObject* match_array,
                                           PyArrayObject*& paf_label,
                                           PyArrayObject*& paf_weight )
  {

    int index_col = 4;

    int nd = 2;
    npy_intp dims[] = { (long)pos_match_index.size(), 3 };
    
    if ( !exclude_neg_examples ) {
      dims[0] = (long)nfilled;
    }
    paf_label  = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );
    
    int nd_weight = 1;
    npy_intp dims_weight[] = { dims[0] };
    paf_weight = (PyArrayObject*)PyArray_SimpleNew( nd_weight, dims_weight, NPY_FLOAT ); 
    
    int npos = 0;
    int nneg = 0;
    int nmissing = 0;

    std::vector<int> pixtype(dims[0],0);
    for (int i=0; i<(int)dims[0]; i++ ) {
      // sample array index
      int idx = (exclude_neg_examples) ? pos_match_index[i] : (int)i;
      // triplet index
      long index = *((long*)PyArray_GETPTR2(match_array,idx,index_col));
      // ground truth for triplet
      long isgood = *((long*)PyArray_GETPTR2(match_array,idx,3));

      const std::vector<float>& label_v = kpflow_labels_v->at(index);

      // if good spacepoint but we don't have a direction label, we zero out the event (missing)
      // if good spacepoint and has direction label, counted as positive example
      // if bad spacepoint point but doesnt matter if have a label, counted as negative example,
      //   correct answer will be zero vector
                      
      if ( isgood==1 && label_v.size()==10 ) {
        // positive examples
        npos++;
        for (int j=0; j<3; j++)
          *((float*)PyArray_GETPTR2(paf_label,i,j)) = label_v[j];
        pixtype[i] = 1;
      }
      else if (isgood==0) {
        // negative examples
        nneg++;
        for (int j=0; j<3; j++)
          *((float*)PyArray_GETPTR2(paf_label,i,j)) = 0.0;
        pixtype[i] = 0;        
      }
      else if (isgood==1 && label_v.size()<10) {
        nmissing++;
        for (int j=0; j<3; j++)
          *((float*)PyArray_GETPTR2(paf_label,i,j)) = 0.0;
        pixtype[i] = 2;        
      }        
    }
    nneg = 0;

    // weights for positive and negative examples
    float w_pos = (npos) ? float(npos+nneg)/float(npos) : 0.0;
    float w_neg = (nneg) ? float(npos+nneg)/float(nneg) : 0.0;
    float w_norm = w_pos*npos + w_neg*nneg;

    //std::cout << "KPWEIGHT: W(POS)=" << w_pos/w_norm << " W(NEG)=" << w_neg/w_norm << std::endl;
    
    for (int i=0; i<dims[0]; i++ ) {
      if (pixtype[i]==1) {
        *((float*)PyArray_GETPTR1(paf_weight,i)) = w_pos/w_norm;
      }
      else if (pixtype[i]==0) {
        *((float*)PyArray_GETPTR1(paf_weight,i))  = w_neg/w_norm;
      }
      else if (pixtype[i]==2) {
        *((float*)PyArray_GETPTR1(paf_weight,i))  = 0.0;
      }
    }

    return 0;
  }


  /**
   * @brief Get the neutrino origin flag for spacepoints
   * 
   * Go into the PrepMatchTriplet class and transfer the neutrino origin flags for each spacepoint
   * and put them into a numpy array
   *
   */
  int LoaderKeypointData::make_origin_array( const int nfilled,
					     const std::vector<int>& pos_match_index,
					     const bool exclude_neg_examples,
					     PyArrayObject* match_array,					     
					     PyArrayObject*& origin_array )
  {
    
    int nd = 1;
    npy_intp dims[] = { nfilled };
    
    if ( !exclude_neg_examples ) {
      dims[0] = (long)nfilled;
    }
    origin_array  = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_LONG );
    
    larflow::prep::PrepMatchTriplets* ptriplet = nullptr;
    if ( _use_data_from_ttree ) {
      ptriplet = &(triplet_v->at(0));
    }
    else {
      ptriplet = ptriplet_v.at(0);
    }
    
    if ( ptriplet->_origin_v.size()!=dims[0] ) {
      LARCV_ERROR() << "Size of origin_v container (with " << ptriplet->_origin_v.size() << " entries) does not match"
		    << " the expected number of entries (" << dims[0] << ")"
		    << std::endl;
    }

    
    for (int i=0; i<(int)dims[0]; i++ ) {
      int tripletidx = *((int*)PyArray_GETPTR2(match_array,i,4));
      if ( tripletidx>=0 && tripletidx < (long)ptriplet->_origin_v.size() ) {
	long origin_flag = (long)ptriplet->_origin_v[tripletidx];
	*((long*)PyArray_GETPTR1(origin_array,i)) = origin_flag;
      }
      else {
	*((long*)PyArray_GETPTR1(origin_array,i)) = 0;	
      }
    }
    
    return 0;
  }
  
  /**
   * @brief Return particle instance IDs for each spacepoint
   * 
   * Go into the larflow::prep::PrepMatchTriplet class (code in the larflow/PrepFlowMatchData/ folder) 
   * and transfer the information in the std::vector<int> _instance_id_v container 
   * into a numpy array.
   * 
   * These labels are for use in particle-level clustering (object detection).
   *
   * @param[in] nfilled The number of expected labels. This is used to check for
   *                    consistency between spacepoint array and the label array.
   * @param[in] pos_match_index 
   * @param[in] exclude_neg_examples Ignore 'ghost' spacepoint proposals by only return labels for true spacepoints.
   * @param[in] match_array (N,3) Numpy array containing the index to the pixels of each of the three (sparse) wire-plane image arrays.
   * @param[inout] instanceid_array Pointer where the address of the new numpy array will be made. (NPY_LONG)
   */
  int LoaderKeypointData::make_instanceid_array( const int nfilled,
					     const std::vector<int>& pos_match_index,
					     const bool exclude_neg_examples,
					     PyArrayObject* match_array,					     
					     PyArrayObject*& instanceid_array )
  {
    
    // get the triplet data class
    // should encapsulate this and hide the detail of how we loaded
    // the class from this function
    larflow::prep::PrepMatchTriplets* ptriplet = nullptr;
    if ( _use_data_from_ttree ) {
      // we get the data from a ttree loaded from file
      ptriplet = &(triplet_v->at(0));
    }
    else {
      // we get the data from an instance we made
      ptriplet = ptriplet_v.at(0);
    }
    
    // check that we have the info
    if ( !ptriplet || ptriplet->_instance_id_v.size()==0 ) {
      LARCV_ERROR() << "The tripletdata is bad or the _instance_v container is empty." << std::endl;
    }

    int nd = 1;
    npy_intp dims[] = { (npy_intp)ptriplet->_instance_id_v.size() };
    
    if ( exclude_neg_examples ) {
      // we need to count the number of true points
      size_t n_true = 0;
      for ( size_t i=0; i<ptriplet->_instance_id_v.size(); i++ ) {
        long tripletidx = *((long*)PyArray_GETPTR2(match_array,i,4)); 
        // don't remember why we need to pass through another layer of re-indexing
        int istruept = ptriplet->_truth_v.at(tripletidx);
        if (istruept==1)
          n_true += 1;
      }
      // update the number of points
      dims[0] = n_true;
    }

    // Do a check of the expected number of points in the container.
    // This is to help maintain proper consistency between the different arrays describing the spacepoints.
    if ( ptriplet->_instance_id_v.size()!=nfilled ) {
      LARCV_ERROR() << "Size of origin_v container (with " << ptriplet->_origin_v.size() << " entries) does not match"
		    << " the expected number of entries (" << nfilled << ")"
		    << std::endl;
    }

    // Make the array and pass the address to the output pointer
    instanceid_array  = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_LONG );
    
    // Loop through again to copy instance ID
    size_t fill_idx = 0;
    for (size_t i=0; i<ptriplet->_instance_id_v.size(); i++ ) {
      long tripletidx = *((long*)PyArray_GETPTR2(match_array,i,4));

      if ( exclude_neg_examples ) {
        int istruept = ptriplet->_truth_v.at(tripletidx);
        if (istruept==0)
          continue;
      }

      if ( tripletidx>=0 && tripletidx < (long)ptriplet->_instance_id_v.size() ) {
	      long instanceid = (long)ptriplet->_instance_id_v[tripletidx];
	      *((long*)PyArray_GETPTR1(instanceid_array,fill_idx)) = instanceid;
      }
      else {
	      *((long*)PyArray_GETPTR1(instanceid_array,fill_idx)) = -1;	///< default value when spacepoint not part of labled true cluster
      }

      fill_idx++;
    }
    
    return 0;
  }
  
  
  
  
}
}
