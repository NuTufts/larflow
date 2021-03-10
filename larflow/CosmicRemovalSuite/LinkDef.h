/** \defgroup Reco Reco
 *
 * \brief Algorithms that use Network outputs to reconstruct the interactions in the TPC
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
#pragma link off all namespaces;

//#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::cosmicremovalsuite;
#pragma link C++ class larflow::cosmicremovalsuite::CRVarsMaker+;
#pragma link C++ class larflow::cosmicremovalsuite::CRSuiteFunctions+;
//ADD_NEW_CLASS ... do not change this line
#endif
