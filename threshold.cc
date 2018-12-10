/***********************************************************************************************************************
 * This file implements threshold.cpp
***********************************************************************************************************************/
// C++ include files
#include <iostream>
#include <set>
#include <map>
#include <string>

// OpenCV include files
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


const double big_constant = 1e+32;


# include "threshold.hh"


bool Variance (std::vector<double>const& sums, double& var)
{
    if ( sums[0] == 0 )
        return false;
    double m = (double)sums[1] / sums[0];
    var = double ( sums[2] - 2 * m * sums[1] ) / sums[0] + m * m;
    return true;
}

    bool Otsu_Threshold (Mat const& image, int& threshold_value)
    {
        bool print = false;
        // Histogramp;
                int temp = 0;
        int bound = 160;
        std::vector<long> histogram( 256, 0 );
        for ( int row = 0; row < image.rows; row++ )
            for ( int col = 0; col < image.cols; col++ )
            {
                int v = (int)image.at<uchar>( row, col );
                if ( v < bound ) v = ( bound + v ) / 2;
                histogram[ v ]++;
            }
        //Otsu's binarization
        std::vector<double> sums0( 3, 0 ), sums1( 3, 0 );
        int length = (int)histogram.size();
        for ( int i = 0; i < length; i++ )
            for ( int p = 0; p < 3; p++ )
                sums1[ p ] += histogram[i] * pow( i, p );
        double cost, cost_min = big_constant, var0 = 0, var1 = 0;
        for ( int n = 0; n+1 < length; n++ ) // n+1 = number of values on the first class
        {
            if ( print ) std::cout<<"\nn="<<n<< std::endl;
            for ( int p = 0; p < 3; p++ )
            {
                int v = histogram[ n ] * pow( n, p );
                sums0[ p ] += v;
                sums1[ p ] -= v;
                if ( print ) std::cout<<" v="<<v<<" s0_"<<p<<"="<<sums0[p]<<" s1_"<<p<<"="<<sums1[p]<< std::endl;
            }
            if ( ! Variance( sums0, var0 ) ) continue;
            if ( ! Variance( sums1, var1 ) ) continue;
            cost = sums0[0] * var0 + sums1[0] * var1;
            if ( print ) std::cout<<" v0="<<var0<<" v1="<<var1<<" c="<<cost<< std::endl;
            if ( cost_min > cost ) { cost_min = cost; threshold_value = n; }
        }
        if ( print ) std::cout<<" threshold="<<threshold_value<<std::endl;
        return true;
    }

/***********************************|| THRESHOLD FUNCTION STARTS HERE ||************************************************
 *
 * @param image
 * @param size_small
 * @return
 *
***********************************************************************************************************************/
multimap< int, Point > Threshold (Mat& image, int size_small, std::string name)
{
    Point grid_sizes ( 1 + int( image.cols / size_small ), 1 + int( image.rows / size_small ) );
    std::multimap< int, Point > values_pixels;
    Mat_<int> thresholds( grid_sizes.y, grid_sizes.x );
    Mat_<Point> shifts( grid_sizes.y, grid_sizes.x );
    for ( int row = 0; row < shifts.rows; row++ )
        for ( int col = 0; col < shifts.cols; col++ )
            shifts.at<Point>( row, col ) = Point( col, row ) * size_small;
    //int threshold_value = 0;
    values_pixels.clear();
    for ( int i = 0; i < grid_sizes.y; i++ )
        for ( int j = 0; j < grid_sizes.x; j++ )
        {
            int size_x = min( size_small, image.cols - shifts( i, j ).x );
            int size_y = min( size_small, image.rows - shifts( i, j ).y );
            //std::cout<<"\nr="<<row<<" c="<<col<<" s="<<shifts( row, col )<<" size_x="<<size_x<<" size_y="<<size_y;
            Mat image_small = image( Rect( shifts( i, j ).x, shifts( i, j ).y, size_x, size_y ) );
            Otsu_Threshold( image_small, thresholds( i, j ) );
            for ( int row = 0; row < image_small.rows; row++ )
                for ( int col = 0; col < image_small.cols; col++ )
                {
                    int v = (int)image_small.at<uchar>( row, col );
                    if ( v <= thresholds( i, j ) ) {
                        values_pixels.insert( std::make_pair( v, shifts( i, j ) + Point( col, row ) ) );
                    }
                }
        }
    Point image_sizes = Point( image.cols, image.rows );
    cv::Mat_<bool> threshold_mask = cv::Mat_<bool>( image_sizes.y, image_sizes.x, false );
    for ( auto v : values_pixels ) threshold_mask.at<bool>( v.second ) = true;

    return values_pixels;
}
