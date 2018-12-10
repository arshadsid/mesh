/***********************************************************************************************************************
 * Header file defines the thresholding methods to be used in program
***********************************************************************************************************************/
// C++ include files
#include <map>
#include <string>
using namespace std;

// OpenCV include files
#include <opencv2/core/core.hpp>
using namespace cv;

#ifndef __THRESHOLD_HH_INCLUDED__
#define __THRESHOLD_HH_INCLUDED__

multimap< int, Point > Threshold (Mat&, int, std::string);

#endif
