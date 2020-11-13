/** \defgroup LightModel LightModel
 *
 * \brief Tools to prepare lightmodel truth labels
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
#pragma link C++ namespace larflow::lightmodel;
//#pragma link C++ class larflow::lightmodel::KPdata+;
//#pragma link C++ class larflow::lightmodel::PrepKeypointData+;
//#pragma link C++ class larflow::lightmodel::PrepAffinityField+;
#pragma link C++ class larflow::lightmodel::DataLoader+;
//#pragma link C++ class larflow::lightmodel::LoaderAffinityField+;

//ADD_NEW_CLASS ... do not change this line
#endif
