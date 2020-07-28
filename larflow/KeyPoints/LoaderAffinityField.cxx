#include "LoaderAffinityField.h"

#include <iostream>

namespace larflow {
namespace keypoints {

  bool LoaderAffinityField::_setup_numpy = false;
  
  /**
   * @brief constructor with list of input ROOT files
   *
   * @param[in] input_v List of paths to input ROOT files containing ground truth data
   *
   */
  LoaderAffinityField::LoaderAffinityField( std::vector<std::string>& input_v )
    : tpaf(nullptr)
  {
    input_files.clear();
    input_files = input_v;
    load_tree();
  }

  LoaderAffinityField::~LoaderAffinityField()
  {
    if ( tpaf ) delete tpaf;
  }

  /**
   * @brief load TTree class data members and define TBranches
   *
   */
  void LoaderAffinityField::load_tree() {
    std::cout << "[LoaderAffinityField::load_tree()]" << std::endl;
    
    tpaf = new TChain("AffinityFieldTree");
    for (auto const& infile : input_files ) {
      std::cout << "add " << infile << " to AffinityFieldTree chain" << std::endl;
      tpaf->Add(infile.c_str());
    }
    
    _label_v = 0;    
    tpaf->SetBranchAddress( "label_v", &_label_v );
  }

  /**
   * @brief load data for a given entry
   *
   * @param[in] entry entry number in TChain
   * @return number of bytes loaded from the file. zero is no more entries or there was an error.
   */
  unsigned long LoaderAffinityField::load_entry( int entry )
  {
    unsigned long bytes = tpaf->GetEntry(entry);
    return bytes;
  }

  /**
   * @brief get total entries
   *
   * @return number of entries in the TChain _paf
   */
  unsigned long LoaderAffinityField::GetEntries()
  {
    return tpaf->GetEntries();
  }

  /**
   * @brief return a ground truth data, return a subsample of all truth matches
   *
   * returns a python dictionary. Contents include (list starts with key of entry)
   * \verbatim embed:rst:loading-asterisk
   *  * `paf_label`:affinity field ground truth (direction of particle at spacepoint)
   *  * `paf_weight`:weight for affinity field labels}
   * \endverbatim
   *
   * @param[in] triplet_matches_pyobj The sampled larmatch triplets. Expected shape: (N,4).
   * @param[in] exclude_neg_examples  Exclude negative examples
   * @return Python dictionary with numpy arrays
   *                        
   */
  PyObject* LoaderAffinityField::get_match_data( PyObject* triplet_matches_pyobj,
                                                bool exclude_neg_examples )
  {

    _exclude_neg_examples = exclude_neg_examples;
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    int index_col = 4;
    
    // cast to numpy array
    PyArrayObject* matches = (PyArrayObject*)triplet_matches_pyobj;

    // get the dimensions
    int ndims = PyArray_NDIM( matches );
    npy_intp* shape = PyArray_SHAPE( matches );
    int nfilled = shape[0];

    // count npos, nneg examples
    // also make list of indices of positive examples, these are the ones we will evaluate ssnet not
    int npos=0;
    int nneg=0;
    std::vector<int> pos_index_v;
    for (int i=0; i<nfilled; i++) {
      long ispositive = *((long*)PyArray_GETPTR2(matches,i,3));
      if (ispositive==1) {
        npos++;
        pos_index_v.push_back(i);
      }
      else {
        nneg++;
      }
    }

    // PARTICLE AFFINITY FIELD GROUND TRUTH
    PyArrayObject* paf_label  = nullptr;
    PyArrayObject* paf_weight = nullptr;
    make_paf_arrays( nfilled, pos_index_v, exclude_neg_examples,
                     matches, paf_label, paf_weight );
    PyObject *paf_label_key     = Py_BuildValue("s", "paf_label" );
    PyObject *paf_weight_key    = Py_BuildValue("s", "paf_weight" );

    PyObject *d = PyDict_New();
    PyDict_SetItem(d, paf_label_key,  (PyObject*)paf_label );
    PyDict_SetItem(d, paf_weight_key, (PyObject*)paf_weight ); 

    Py_DECREF(paf_label_key);
    Py_DECREF(paf_weight_key);
    
    Py_DECREF(paf_label);
    Py_DECREF(paf_weight);

    return d;
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
  int LoaderAffinityField::make_paf_arrays( const int nfilled,
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

      const std::vector<float>& label_v = _label_v->at(index);

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

  
}
}
