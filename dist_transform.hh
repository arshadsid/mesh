//
// Created by arshad on 01/07/18.
//
// C++ include files
#include <iostream>
#include <string>
#include <set>
#include <map>

// OpenCV include files
#include <opencv2/core/core.hpp>
using namespace cv;

#ifndef PLANAR_GRAPH_FOR_MICROSCOPIC_IMAGE_DIST_TRANSFORM_HH
#define PLANAR_GRAPH_FOR_MICROSCOPIC_IMAGE_DIST_TRANSFORM_HH

struct Compare_Points
{
    inline bool operator() (cv::Point const& a, cv::Point const& b) const
    {
        if ( a.x < b.x ) return true;
        if ( a.x == b.x and a.y < b.y ) return true;
        return false;
    }
};

extern std::vector<Point> shifts8;
extern std::vector<Point> shifts4;

bool Distance_Transform (Mat const&, Mat_<bool>&, std::vector<int>&, bool&, std::string, std::string output_folder);
Mat Save_Mask (Mat const&, Mat const&, std::string);
Mat Save_Binary (Mat const&, Mat const&, std::string);
bool Remove_External_Pixels (Mat_<bool>&);
bool Add_Internal_Pixels (Mat_<bool>&);
bool Neighbors (Mat_<bool>const&, Point, std::vector<Point>const&, int&, std::vector<bool>&);
bool Remove_Small_Components (bool, int, Mat_<bool>&, int&);


#endif //PLANAR_GRAPH_FOR_MICROSCOPIC_IMAGE_DIST_TRANSFORM_H
