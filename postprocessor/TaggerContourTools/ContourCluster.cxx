#include "ContourCluster.h"

namespace larlitecv {

  ContourCluster::ContourCluster( const std::vector< const ContourShapeMeta* >& plane_contours ) {
    _addFirstContours( plane_contours );
  }

  void ContourCluster::_addFirstContours( const std::vector< const ContourShapeMeta*>& plane_contours ) {
    resize( plane_contours.size() );
    indices.resize( plane_contours.size() );
    earlyEnd.resize( plane_contours.size() );
    earlyContours.resize( plane_contours.size() );
    earlyDir.resize( plane_contours.size() );    
    
    for (size_t p=0; p<plane_contours.size(); p++) {
      if ( plane_contours[p]!=NULL ) {
	at(p).push_back( *(plane_contours[p]) );
	
	earlyEnd[p].push_back( plane_contours[p]->getFitSegmentStart() );
	earlyContours[p].push_back( plane_contours[p] );
	earlyDir[p].push_back( plane_contours[p]->getStartDir() );
	
	lateEnd.push_back( plane_contours[p]->getFitSegmentEnd() );
	lateContours.push_back( plane_contours[p] );
	lateDir.push_back( plane_contours[p]->getEndDir() );
      }
    }    
  }

  void ContourCluster::addEarlyContours( const std::vector< const ContourShapeMeta*>& plane_contours ) {
    if ( size()==0 ) {
      _addFirstContours( plane_contours );
      return;
    }
  }

  void ContourCluster::addLateContours( const std::vector< const ContourShapeMeta*>& plane_contours ) {
    if ( size()==0 ) {
      _addFirstContours( plane_contours );
      return;
    }
  }
  
  // =======================================================================================================

  bool IsContourLinkValid( const ContourShapeMeta& conta, const ContourShapeMeta& contb, float& connection_dist ) {

    float dir[4][2];
    float dir_starta_endb[2];
    float dir_enda_startb[2];
    float dir_enda_endb[2];

    dir[0][0] = conta.getFitSegmentStart().x - contb.getFitSegmentStart().x;
    dir[0][1] = conta.getFitSegmentStart().y - contb.getFitSegmentStart().y;
    
    dir[1][0] = conta.getFitSegmentStart().x - contb.getFitSegmentEnd().x;
    dir[1][1] = conta.getFitSegmentStart().y - contb.getFitSegmentEnd().y;

    dir[2][0] = conta.getFitSegmentEnd().x   - contb.getFitSegmentStart().x;
    dir[2][1] = conta.getFitSegmentEnd().y   - contb.getFitSegmentStart().y;

    dir[3][0] = conta.getFitSegmentEnd().x   - contb.getFitSegmentEnd().x;
    dir[3][1] = conta.getFitSegmentEnd().y   - contb.getFitSegmentEnd().y;

    float dists[4];
    for (int v=0; v<4; v++) {
      dists[v] = 0.;
      for (int i=0; i<2; i++)
	dists[v] += dir[v][i]*dir[v][i];
      dists[v] = sqrt(dists[v]);
    }

    // normalize the dirs
    for (int v=0; v<4; v++) {
      for (int i=0; i<2; i++) {
	dir[v][i] /= dists[v];
      }
    }

    int mincon = -1;
    float mindist = 1.0e9;
    for (int i=0; i<4; i++) {
      if ( mincon==-1 || mindist>dists[i] ) {
	mincon  = i;
	mindist = dists[i];
      }
    }
    connection_dist = mindist;
    
    // if close enough. it's valid, for better or worse
    //if ( mindist<5 )
    //return true;

    if ( mindist>150 )
      return false; // too far

    // get the cosines. angles between connection, and the angles between the pieces
    float segsegcos = 0;
    float segacos = 0.;
    float segbcos = 0.;
    if ( mincon==0 ) {
      // start-start
      for (int i=0; i<2; i++) {
	segsegcos += conta.getStartDir()[i]*contb.getStartDir()[i];
	segacos   += conta.getStartDir()[i]*dir[mincon][i];
	segbcos   += contb.getStartDir()[i]*dir[mincon][i];
      }
    }
    else if ( mincon==1 ) {
      // start-end
      for (int i=0; i<2; i++) {
	segsegcos += conta.getStartDir()[i]*contb.getEndDir()[i];
	segacos   += conta.getStartDir()[i]*dir[mincon][i];
	segbcos   += contb.getEndDir()[i]*dir[mincon][i];
      }
    }
    else if ( mincon==2 ) {
      // end-start
      for (int i=0; i<2; i++) {
	segsegcos += conta.getEndDir()[i]*contb.getStartDir()[i];
	segacos   += conta.getEndDir()[i]*dir[mincon][i];
	segbcos   += contb.getStartDir()[i]*dir[mincon][i];
      }
    }
    else if ( mincon==3 ) {
      // end-end
      for (int i=0; i<2; i++) {
	segsegcos += conta.getEndDir()[i]*contb.getEndDir()[i];
	segacos   += conta.getEndDir()[i]*dir[mincon][i];
	segbcos   += contb.getEndDir()[i]*dir[mincon][i];
      }      
    }
    segsegcos *= -1.0;
    segacos   *= -1.0;

    std::cout << "  contour test: segseg=" << segsegcos << " segacos=" << segacos << " segbcos=" << segbcos << " mindist=" << mindist << " contype=" << mincon << std::endl;
    
    if ( segsegcos>0.7 && segacos>0.7 && segbcos>0.7 )
      return true;

    return false;
  }
  
  bool RecursiveSearch( ContourGraphNode* node, const std::vector<ContourShapeMeta>& contours_v, std::vector<ContourGraphNode*>& nodebank ) {
    // we have to add the next daugher node. else go back to the mother
    // we DO NOT want a cylic graph, so first we back to the mother and record indices

    std::cout << "RecursiveSearch: current node=" << node->idx << std::endl;
    node->visited = true;
    
    std::set<int> past_indices; // past node idx
    float current_path_length = 0;
    ContourGraphNode* pos = node;
    past_indices.insert( pos->idx );
    int numsteps = 0;
    std::cout << "  mothers chain: ";
    while ( pos->mother!=NULL ) {
      current_path_length += pos->mother_edge_length;
      pos = pos->mother;
      past_indices.insert( pos->idx );      
      std::cout << pos->idx << " ";
      numsteps++;
      if ( numsteps>=1000 ) {
	throw std::runtime_error("ContourClusterAlgo.h Recursive Search. Bad loop has happened.");
      }
    }
    std::cout << std::endl;
    
    for (int idx=0; idx<(int)contours_v.size(); idx++) {

      if ( node->idx==idx )
	continue;
      
      if ( past_indices.find(idx)!=past_indices.end() )
	continue;

      ContourGraphNode* current = nodebank[idx];
      if ( current->visited )
	continue;
      
      // go to next valid node
      const ContourShapeMeta& contour = contours_v[idx];
      float connection_dist = 0;
      if ( !IsContourLinkValid( contours_v[ node->idx ], contour, connection_dist ) )
	continue;

      std::cout << "  valid link: idx=" << idx << " dist=" << connection_dist << std::endl;
      


      // is it used? (unique path strategy)      
      if ( current->mother!=NULL ) {
      	// what's this current path back?
      	pos = current;
	float test_path = 0.;
	int first_common_mother = -1;
      	while ( pos->mother!=NULL ) {
	  test_path += pos->mother_edge_length;
	  pos = pos->mother;
	  if ( first_common_mother==-1 ) {
	    if ( past_indices.find( pos->idx )!=past_indices.end() ) {
	      first_common_mother = pos->idx;
	    }
	  }
      	}

	// the path is shorter. so don't connect.
	if ( test_path<current_path_length+connection_dist ) {
	  continue;
	}

	// this path is shorter, so we unclaim the path back to the common mother, unconnect those nodes
	pos = current;
	while ( pos->mother!=NULL ) {
	  ContourGraphNode* lastnode = pos;
	  pos = pos->mother; // go to mother
	  lastnode->mother = NULL; // disconnect from mother
	  // reset the daughters
	  std::vector<ContourGraphNode*> daughters = pos->daughters;
	  pos->daughters.clear();
	  for (int i=0; i<(int)daughters.size(); i++) {
	    if ( daughters[i]!=lastnode ) {
	      pos->daughters.push_back( daughters[i] );
	    }
	  }
	  if ( pos->idx==first_common_mother )
	    break;
	}
      }//if has mother link
      
      // valid: link back to mother. if we change the mother, we break the current graph...
      current->mother = node;
      node->daughters.push_back( current );
      node->mother_edge_length = connection_dist;

      // go down
      RecursiveSearch( current, contours_v, nodebank );

      // if we come back and we've been disconnected from a path. return.
      if ( node->mother==NULL )
	return false;
    }

    return false;
  }

  // =======================================================================================================
  
}
