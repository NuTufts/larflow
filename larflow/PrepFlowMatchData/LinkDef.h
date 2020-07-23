/** \defgroup PrepFlowMatchData PrepFlowMatchData
 *
 * \brief Tools to prepare larmatch spacepoints for training and inference.
 *
 *
 * cint script to generate libraries and python bindings.
 * Declare namespace & classes you defined
 * pragma statement: order matters! Google it ;)
 *
 */

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::prepflowmatchdata;
#pragma link C++ namespace std;

#pragma link C++ class std::vector< std::vector<int> >+;
#pragma link C++ class larflow::WireOverlap+;
#pragma link C++ class larflow::FlowTriples+;
#pragma link C++ class larflow::FlowMatchMap+;
#pragma link C++ class larflow::PrepFlowMatchData+;
#pragma link C++ class larflow::PrepMatchTriplets+;
#pragma link C++ class std::vector<larflow::PrepMatchTriplets>+;
#pragma link C++ class larflow::MatchTripletProcessor+;
#pragma link C++ class larflow::FlowMatchHitMaker+;
#pragma link C++ class larflow::prepflowmatchdata::PrepSSNetTriplet+;
#pragma link C++ function larflow::sample_pair_array( const int&, const FlowMatchMap&, int&, bool )+;
#pragma link C++ function larflow::get_chunk_pair_array( const int&, const int&, const FlowMatchMap&, int&, int&, bool )+;
#pragma link C++ function larflow::make_larflow_hits( PyObject*, PyObject*, PyObject*, const larcv::ImageMeta&, larlite::event_larflow3dhit&, const larcv::EventChStatus* )+;
#pragma link C++ function larflow::make_larflow_hits_with_deadchs( PyObject*, PyObject*, PyObject*, const larcv::ImageMeta&, const larcv::EventChStatus&, larlite::event_larflow3dhit& )+;

//ADD_NEW_CLASS ... do not change this line
#endif
