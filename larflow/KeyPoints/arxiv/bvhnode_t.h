#ifndef __BVHNODE_T_H__
#define __BVHNODE_T_H__

#include <string>
#include <vector>

namespace larflow {
namespace keypoints {

  struct bvhnode_t {
    float bounds[3][2]; //bounding box
    int splitdim;       // dimension we split with (-1) is a leaf
    int kpdidx;         // index to point in the boundary volume
    bvhnode_t* mother;
    std::vector<bvhnode_t*> children;
    bvhnode_t( float xmin, float xmax, float ymin, float ymax, float zmin, float zmax )
    : splitdim(-1),
      kpdidx(-1)
    {
      bounds[0][0] = xmin;
      bounds[0][1] = xmax;
      bounds[1][0] = ymin;
      bounds[1][1] = ymax;
      bounds[2][0] = zmin;
      bounds[2][1] = zmax;
    };
  };
  
  bool compare_x( const bvhnode_t* lhs, const bvhnode_t* rhs );
  bool compare_y( const bvhnode_t* lhs, const bvhnode_t* rhs );
  bool compare_z( const bvhnode_t* lhs, const bvhnode_t* rhs );
  std::string strnode( const bvhnode_t* node );
  void print_graph( const bvhnode_t* node );
  void _recurse_printgraph( const bvhnode_t* node, int& depth );
  const bvhnode_t* recurse_findleaf( const std::vector<float>& testpt, const bvhnode_t* node );
  
  
}
}

#endif
