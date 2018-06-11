#ifndef __CONTOUR_CLUSTER_H__
#define __CONTOUR_CLUSTER_H__

#include <vector>
#include <set>

#include "ContourShapeMeta.h"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#endif

namespace larlitecv {

  class ContourCluster : public std::vector< std::vector<ContourShapeMeta> > {
    friend class ContourClusterAlgo;
  public:
    ContourCluster() {};
    ContourCluster( const std::vector< const ContourShapeMeta* >& plane_contours );
    virtual ~ContourCluster() {};

    void addEarlyContours( const std::vector< const ContourShapeMeta*>& plane_contours );
    void addLateContours( const std::vector< const ContourShapeMeta*>& plane_contours );

    std::vector< std::vector<cv::Point> > earlyEnd;
    std::vector< std::vector<std::vector<float> > > earlyDir;
    std::vector< std::vector< const ContourShapeMeta*> > earlyContours;
    
    std::vector< cv::Point > lateEnd;
    std::vector< std::vector<float> > lateDir;
    std::vector< const ContourShapeMeta* > lateContours;

    std::vector< std::set<int> > indices;

  protected:
    
    void _addFirstContours( const std::vector< const ContourShapeMeta*>& plane_contours );
    
  };

  class ContourGraphNode {
  public:
    ContourGraphNode( int iidx ) {
      idx = iidx; // refers to contour list position
      mother = NULL;
      mother_edge_length = 0;
      daughters.clear();
      iter = -1;
      visited = false;
    };
    virtual ~ContourGraphNode() {};

    int idx; // contour list position
    ContourGraphNode* mother; // pointer back to mother node
    float mother_edge_length; // length back to mother
    std::vector< ContourGraphNode* > daughters; // pointer to daugher nodes
    int iter; // used for recursive traversal algorithms
    bool visited; // used in the past
  };

  bool IsContourLinkValid( const ContourShapeMeta& conta, const ContourShapeMeta& contb, float& connection_dist );  
  bool RecursiveSearch( ContourGraphNode* node, const std::vector<ContourShapeMeta>& contours_v, std::vector<ContourGraphNode*>& nodebank );
  
}

#endif
