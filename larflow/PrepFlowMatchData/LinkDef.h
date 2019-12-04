//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace larflow;

#pragma link C++ class larflow::FlowMatchMap+;
#pragma link C++ class larflow::PrepFlowMatchData+;
#pragma link C++ class larflow::FlowMatchHitMaker+;
#pragma link C++ function larflow::sample_pair_array( const int&, const FlowMatchMap&, int&, bool )+;
#pragma link C++ function larflow::get_chunk_pair_array( const int&, const int&, const FlowMatchMap&, int&, int&, bool )+;
#pragma link C++ function larflow::make_larflow_hits( PyObject*, PyObject*, PyObject*, const larcv::ImageMeta&, larlite::event_larflow3dhit&, const larcv::EventChStatus* )+;
#pragma link C++ function larflow::make_larflow_hits_with_deadchs( PyObject*, PyObject*, PyObject*, const larcv::ImageMeta&, const larcv::EventChStatus&, larlite::event_larflow3dhit& )+;

//ADD_NEW_CLASS ... do not change this line
#endif
