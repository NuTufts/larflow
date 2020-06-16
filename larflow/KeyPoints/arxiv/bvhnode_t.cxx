#include "bvhnode_t.h"
#include <iostream>
#include <sstream>

namespace larflow {
namespace keypoints {

  /**
   * comparator to sort in X dimension
   *
   */
  bool compare_x( const bvhnode_t* lhs, const bvhnode_t* rhs ) {
    if ( lhs->bounds[0][0] < rhs->bounds[0][0] )
      return true;
    return false;
  }

  /**
   * comparator to sort in Y dimension
   *
   */  
  bool compare_y( const bvhnode_t* lhs, const bvhnode_t* rhs ) {
    if ( lhs->bounds[1][0] < rhs->bounds[1][0] )
        return true;
    return false;
  }

  /**
   * comparator to sort in Z dimension
   *
   */    
  bool compare_z( const bvhnode_t* lhs, const bvhnode_t* rhs ) {
    if ( lhs->bounds[2][0] < rhs->bounds[2][0] )
        return true;
    return false;
  }

  /**
   * make string with info about bvhnode_t instance
   *
   * @param[in] node BVH node to make info for.
   * @return         string with info in it.
   *
   */    
  std::string strnode( const bvhnode_t* node ) {
    std::stringstream ss;
    if ( node->children.size()>0 ) {
      ss << "node: x[" << node->bounds[0][0] << "," << node->bounds[0][1] << "] "
         << "y[" << node->bounds[1][0] << "," << node->bounds[1][1] << "] "
         << "z[" << node->bounds[2][0] << "," << node->bounds[2][1] << "] "
         << "splitdim=" << node->splitdim;
      
    }
    else {
      ss << "LEAF: "
         << "x[" << node->bounds[0][0] << "] "
         << "y[" << node->bounds[1][0] << "] " 
         << "z[" << node->bounds[2][0] << "] "
         << "kpdata-index=" << node->kpdidx;
    }
    return ss.str();
  }

  /**
   * print graph starting from given node
   *
   * @param[in] node starting node to print tree
   *
   */      
  void print_graph( const bvhnode_t* node ) {
    int depth=0;
    _recurse_printgraph( node, depth );
  }

  /**
   * recursive function for print the BVH graph
   *
   * Not intended to be called by user.
   *
   * @param[in]    node  starting node to print tree
   * @param[inout] depth depth of node, relative to starting node.
   *
   */        
  void _recurse_printgraph( const bvhnode_t* node, int& depth ) {
    std::string info =  strnode(node);
    std::string branch = "";
    for (int i=0; i<depth; i++)
      branch += " |";
    if ( depth>0 ) 
      branch += "-- ";

    std::cout << branch << info << std::endl;

    // we loop through our daughters
    for ( auto& child : node->children ) {
      ++depth;
      _recurse_printgraph( child, depth );
    }
    --depth;
  }

  /**
   * recursive function for getting leaf in BVH tree that is in same 
   * boundary volume as a given test point
   *
   * @param[in] testpt  The test point.
   * @param[in] node    Current node.
   * @return            The leaf node.
   *
   */          
  const bvhnode_t* recurse_findleaf( const std::vector<float>& testpt,
                                     const bvhnode_t* node ) {

    // is leaf?
    if ( node->children.size()==0 ) return node;
    // if child leaf?
    if ( node->children.size()==1 ) return node->children[0];

    // choose child to descend into
    if ( testpt[node->splitdim] < node->children[0]->bounds[node->splitdim][1] )
      return recurse_findleaf( testpt, node->children[0] );
    else
      return recurse_findleaf( testpt, node->children[1] );
    
  }
  
  
}
}
