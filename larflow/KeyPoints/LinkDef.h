/** \defgroup Keypoints Keypoints
 *
 * \brief Tools to prepare keypoint truth labels
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
#pragma link C++ namespace larflow::keypoints;
#pragma link C++ class larflow::keypoints::KPdata+;
#pragma link C++ class larflow::keypoints::PrepKeypointData+;
#pragma link C++ class larflow::keypoints::PrepAffinityField+;
#pragma link C++ class larflow::keypoints::LoaderKeypointData+;
#pragma link C++ class larflow::keypoints::LoaderAffinityField+;

//ADD_NEW_CLASS ... do not change this line
#endif
