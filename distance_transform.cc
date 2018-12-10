// C++ include files
#include <fstream>
#include <iostream>
#include <deque>
#include <set>
#include <string>
#include <iterator>
#include <utility>
#include <algorithm>
#include <limits>
#include <map>


// OpenCV include files
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


// Boost include files
#include <boost/graph/graphviz.hpp>
#include "boost/graph/topological_sort.hpp"
#include <boost/graph/graph_traits.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>
typedef boost::adjacency_list< boost::listS, boost::vecS, boost::undirectedS, Point > Graph;


// Project header files
# include "constants.hh"
# include "threshold.hh"
# include "dist_transform.hh"

// Constants

std::vector<CvScalar> colors = { Lime, Orange, Gray, Cyan, Yellow, Red, Green, Olive, Magenta, Teal, Silver, Coral, Salmon, Khaki, Plum, orchid, beige, lavender, mocassin, honeydew, ivory, azure, crimson, gold, sky_Blue, aqua_marine, bisque, peach_puff, corn_silk, wheat, violet, lawn_green};
// dark colors: Blue, Black, Brown, Maroon, Navy, Purple, Indigo, peru, sienna,

std::vector<Point> shifts_diagonal = { Point(1,1), Point(-1,1), Point(-1,-1), Point(1,-1) };
std::vector<Point> shifts_side = { Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
/***********************************************************************************************************************
************************************************************************************************************************
const std::vector<Vec3b> Colormap_jet{ // from bright red to light blue through green, yellow
    Vec3b(139, 0, 0),
    Vec3b(0, 0, 0),
    Vec3b(0, 69, 139),
    Vec3b(0, 0, 128),
    Vec3b(128, 0, 0),
    Vec3b(128, 0, 128),
    Vec3b(130, 0, 75),
    Vec3b(63, 133, 205),
    Vec3b(45, 82, 160),
    Vec3b(0, 0, 139) };

class Pixel
{
public:
    Point point;
    int value;
    Pixel (Point p, int v) { point = p; value = v; }
    void Print() { std::cout<<" v"<<point<<"="<<value<< std::endl; }

};
bool Decreasing_Values (Pixel const& p1, Pixel const& p2){ return p1.value >= p2.value; }
bool Increasing_Keys (std::pair< double, Point >const& p1, std::pair< double, Point >const& p2) { return p1.first < p2.first; }

struct Decreasing { bool operator() (int i0, int i1) { return (i0 >= i1 ); } };

struct Decreasing_Double
{
    bool operator() (double const& v1, double const& v2) const { return v1 >= v2; }
};



Point Root (std::map< Point, Point, Compare_Points >const& edgel_parents, Point p)
{
    return p;
}

void Print (std::vector<int>const& v)
{
    for ( auto e : v ) std::cout<<e<<" "<< std::endl;
}

void Print (std::vector<Vec3b>const& v)
{
    for ( auto e : v ) std::cout<<e<<" "<< std::endl;
}

void Plot (Mat& img, std::vector<int>const& v)
{
    for ( int i = 0; i < v.size(); i++ )
        circle( img, Point( i, 255 - v[i] ), 1, black, -1 );
}

void Plot (Mat& img, std::vector<Vec3b>const& v)
{
    for ( int i = 0; i < v.size(); i++ )
        for ( int k = 0; k < 3; k++ )
            circle( img, Point( i, v[i][k] ), 1, BGR[k], -1 );
}

bool Extract_Row (Mat const& image, int row, std::vector<int>& line)
{
    for ( int j = 0; j < image.cols; j++ )
        line.push_back( image.at<uchar>( row, j ) );
    return true;
}

bool Extract_Row (Mat const& image, int row, std::vector<Vec3b>& line)
{
    for ( int j = 0; j < image.cols; j++ )
        line.push_back( image.at<Vec3b>( row, j ) );
    return true;
}

void Plot_Row (std::string name, Mat const& image, int row)
{
    Mat img( 256, (int)image.cols, CV_8UC3, white );
    std::vector<Vec3b> line_color;
    //Extract_Row( image, row, line_color );
    //Plot( img, line_color );
    Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    std::vector<int> line_gray;
    Extract_Row( image_gray, row, line_gray );
    //Print( line_gray );
    Plot( img, line_gray );
    imwrite( name, img );
}

void Sqrt (Mat const& x, Mat const& y, Mat& sqrt)
{
    sqrt = Mat( x.rows, x.cols, CV_64F );
    for ( int i = 0; i < sqrt.rows; i++ )
        for ( int j = 0; j < sqrt.cols; j++ )
            sqrt.at<double>( i, j ) = std::sqrt( pow( x.at<double>(i,j), 2 ) + pow( y.at<double>(i,j), 2 ) ); // short = CV_16S
}

void Draw_Gradients (std::multimap< double, Point >const& edgels, Mat const& image_dx, Mat const& image_dy, int scale_factor, Mat& image_grad)
{
    resize(image_grad, image_grad, image_grad.size() * scale_factor);
    for ( auto it = edgels.begin(); it != edgels.end(); it++ )
    {
        Point p = it->second;
        int x = image_dx.at<double>( p ) * scale_factor * 10;
        int y = image_dy.at<double>( p ) * scale_factor * 10;
        line( image_grad, scale_factor * p, scale_factor * p + Point( x, y ), Green, 1 );
    }
    for ( auto it = edgels.begin(); it != edgels.end(); it++ )
        circle( image_grad, scale_factor * it->second, 1, Blue, -1 );
}

bool Acceptable_Edgel_Small_Corner (Point pixel, Mat_<bool>const& mask, Point& edgel1, Point& edgel2)
{
    std::vector<Point> arrows{ Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
    for ( int k = 0; k < 4; k++ )
        if ( mask.at<bool>( pixel + arrows[ k ] ) and mask.at<bool>( pixel + arrows[ (k+1)%4 ] ) )
        {
            Point a0 = arrows[ (k+2)%4 ], a1= arrows[ (k+3)%4 ];
            if ( mask.at<bool>( pixel + a0 ) or mask.at<bool>( pixel + a1 ) or mask.at<bool>( pixel + a0 + a1 ) ) return true; // acceptable edgel
            edgel1 = pixel + arrows[ k ];
            edgel2 = pixel + arrows[ (k+1)%4 ];
            return false;
        }
    return true;
}

bool Acceptable_Edgel_Small_Corner (Point pixel, Mat_<bool>const& mask)
{
    std::vector<Point> arrows{ Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
    for ( int k = 0; k < 4; k++ )
        if ( mask.at<bool>( pixel + arrows[ k ] ) and mask.at<bool>( pixel + arrows[ (k+1)%4 ] ) )
        {
            Point a0 = arrows[ (k+2)%4 ], a1= arrows[ (k+3)%4 ];
            if ( mask.at<bool>( pixel + a0 ) or mask.at<bool>( pixel + a1 ) or mask.at<bool>( pixel + a0 + a1 ) ) return true; // acceptable edgel
            return false; // unacceptable edgel = small corner
        }
    return true;
}

bool Acceptable_Edgel_Big_Corner (Point pixel, Mat_<bool>const& mask)
{
    std::vector<Point> arrows_diag{ Point(1,1), Point(-1,1), Point(-1,-1), Point(1,-1) };
    for ( int k = 0; k < 4; k++ )
    {
        Point a0 = arrows_diag[ k ], a1= arrows_diag[ (k+1)%4 ], a01 = 0.5 * ( a0 + a1 );
        if ( mask.at<bool>( pixel + a0 ) and mask.at<bool>( pixel + a1 ) and mask.at<bool>( pixel + a01 ))
        {
            if ( mask.at<bool>( pixel - a0 ) or mask.at<bool>( pixel - a1 ) or mask.at<bool>( pixel - a01 ) ) return true; // acceptable edgel
            return false; // unacceptable edgel = big corner
        }
    }
    return true;
}

bool Acceptable_Edgel (Point pixel, Mat_<bool>const& mask, Point& edgel1, Point& edgel2)
{
    if ( pixel.x == 0 or pixel.y == 0 or pixel.x+1 == mask.cols or pixel.y+1 == mask.rows ) return true;
    if ( ! Acceptable_Edgel_Big_Corner( pixel, mask ) ) return false;
    if ( ! Acceptable_Edgel_Small_Corner( pixel, mask, edgel1, edgel2 ) ) return false;
    return true;
}

bool Acceptable_Edgel (Point pixel, Mat_<bool>const& mask)
{
    if ( pixel.x == 0 or pixel.y == 0 or pixel.x+1 == mask.cols or pixel.y+1 == mask.rows ) return true;
    if ( ! Acceptable_Edgel_Small_Corner( pixel, mask ) ) return false;
    if ( ! Acceptable_Edgel_Big_Corner( pixel, mask ) ) return false;
    return true;
}

bool Try_Remove_Edgel (Point edgel, Mat_<bool>& live_mask, int& removed_edgels)
{
    Point edgel1, edgel2;
    if ( Acceptable_Edgel( edgel, live_mask, edgel1, edgel2 ) ) return false;
    //std::cout<<" -"<<edgel;
    live_mask.at<bool>( edgel ) = false;
    removed_edgels++;
    Try_Remove_Edgel( edgel1, live_mask, removed_edgels );
    Try_Remove_Edgel( edgel2, live_mask, removed_edgels );
    return true;
}

bool Find_Edgels (Mat const& image_magnitude, double edgels_ratio, std::multimap< double, Point >& edgels)
{
    // Order all pixels according their magnitudes
    int num_pixels = image_magnitude.rows * image_magnitude.cols;
    Mat_<bool> edgels_mask( image_magnitude.rows, image_magnitude.cols, false );
    Mat_<bool> live_mask = edgels_mask.clone();
    std::vector< std::pair< double, Point > > pixels;
    for ( int i = 0; i < image_magnitude.rows; i++ )
        for ( int j = 0; j < image_magnitude.cols; j++ )
            pixels.push_back( std::make_pair( image_magnitude.at<double>( i, j ), Point( j, i ) ) );
    sort( pixels.begin(), pixels.end(), Increasing_Keys );
    for ( int k = num_pixels-1; k >= 0 and edgels.size() < edgels_ratio * num_pixels; k-- )
    {
        edgels_mask.at<bool>( pixels[k].second ) = true;
        if ( Acceptable_Edgel_Small_Corner( pixels[k].second, edgels_mask ) )
        {
            edgels.insert( pixels[k] );
            live_mask.at<bool>( pixels[k].second ) = true;
        }
        pixels.erase( pixels.begin() + k );
    }
    //std::cout<<" pixels="<<pixels.size();

    // removing superfluous edgels
    Point edgel1, edgel2; // empty variables needed for Acceptable_Edgel
    int removed_edgels = 0;
    //
    for ( auto it = edgels.begin(); it != edgels.end(); it++ ) // over initial edgels in the increasing order of magnitude
    {
        if ( live_mask.at<bool>( it->second ) )
            Try_Remove_Edgel( it->second, live_mask, removed_edgels ); // recursively, live_mask can become smaller
        while ( pixels.size() > 0 and removed_edgels > 0 )
        {
            Point p = pixels.rbegin()->second; // strongest pixel that isn't an edgel
            if ( !edgels_mask.at<bool>( p ) and Acceptable_Edgel( p, edgels_mask, edgel1, edgel2 ) )
            {
                //std::cout<<" +"<<p;
                live_mask.at<bool>( p ) = true;
                edgels_mask.at<bool>( p ) = true; // mark strongest pixels from the remaining map as edgels
                removed_edgels--;
            }
            pixels.erase( pixels.begin() + (int)pixels.size()-1 );
        }
    }//
    std::cout<<"Tried="<<num_pixels-(int)pixels.size()<< std::endl;
    edgels.clear();
    for ( int i = 0; i < image_magnitude.rows; i++ )
        for ( int j = 0; j < image_magnitude.cols; j++ )
            if ( live_mask.at<bool>( i, j ) )
                edgels.insert( std::make_pair( image_magnitude.at<double>( i, j ), Point( j, i ) ) );
    //
    return true;
}


bool Draw_Graphs (Mat const& image, std::multimap< Point, Graph, Compare_Points >const& pixels_graphs, Mat& image_graphs)
{
    for ( auto g : pixels_graphs )
    {
        int i = 0;
        for ( auto pair = vertices( g.second ); pair.first != pair.second; ++pair.first, i++)
        {
        //    circle( image, g.second[ *pair.first ], 1, colors[ i % colors.size() ], -1 );
            image_graphs.at<Vec3b>( g.second[ *pair.first ] ) = image.at<Vec3b>( g.second[ *pair.first ] );
        //circle( image, g.second[ *pair.first ], 1, black, -1 );
        }
    }
    return true;
}

bool Draw_Graph (Mat const& image, Graph const& graph, Mat& image_graphs)
{
    for ( auto pair = vertices( graph ); pair.first != pair.second; ++pair.first )
            image_graphs.at<Vec3b>( graph[ *pair.first ] ) = image.at<Vec3b>( graph[ *pair.first ] );
    return true;
}



bool Add_Isolated_Pixel (Point point, Mat_<bool>& mask)
{
    if ( mask.at<bool>( point ) ) return false; // already added
    int num_neighbors = 0;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    if ( num_neighbors == neighbors.size() ) { mask.at<bool>( point ) = true; return true; }
    return false;
}
*********************************************************************************************************************
********************************************************************************************************************/



/**************************************|| Planar_Graph CLASS STARTS HERE ||**************************************************
 * Variables:
 *          -boolean test
 *          -int size_small
 *          -double area_min
 *          -vector<int> min_areas
 *          -String input_folder, output_folder, name_base, ext, name
 *          -Mat_<bool> mask
 *          -Mat image
 *
 * Constructor:
 *          -Initialize variable values for: input_folder, name_base, ext, output_folder, min_areas
 *
 * Functions:
 * bool Image_to_Graph (int image_ind)
 *                  -loads image
 *                  -performs thresholding using threshold.h
 *                  -performs edge thinning using edge_thinning.h
 *
***********************************************************************************************************************/
class Planar_Graph
{
public:
    // parameters
    bool test = false;
    int size_small = 200;
    double area_min = 100;
    std::vector<int> min_areas;
    Point image_sizes;
    String input_folder, name_base, ext, output_folder, name;
    std::multimap< int, Point > values_pixels;
    Mat_<bool> mask;
    Mat_<Vec3b> mask_skeleton;
    cv::Mat image;

    Planar_Graph (const String &_input_folder, const String &_name_base, const String &_ext, const String &_output_folder)
    {
        input_folder = _input_folder;
        name_base = _name_base;
        ext = _ext;
        output_folder = _output_folder;
        for ( int i = 1; i < 6; i++ ) min_areas.push_back( 100* i );
    }



    bool Image_to_Graph (int image_ind_upper, int image_ind)
    {
        bool debug = true, save_images = false; //false;
        if ( debug ) save_images = false;
        name = name_base/* + std::to_string( image_ind_upper ) + "_" */+ std::to_string( image_ind );
        std::cout<<"\n"<<name<< std::endl;
        image = cv::imread( input_folder + name + "." + ext, CV_LOAD_IMAGE_COLOR );

        //cv::Mat frame = cv::imread( input_folder + name + "." + ext, CV_LOAD_IMAGE_COLOR );
        //cv::GaussianBlur(frame, image, cv::Size(0, 0), 2);
        //cv::addWeighted(frame, 1.5, image, -0.5, 0, image);

        if ( !image.data ) { std::cout<<" not found"<< std::endl; return false; }
        if ( test ) image = image( cv::Rect( 0, 0, size_small, size_small ) );
        image_sizes = Point( image.cols, image.rows );
        name = output_folder /*+ to_string(image_ind_upper) + "_" */+ std::to_string( image_ind ) + "/";

        std::cout<<name;
        /*int t;
        std::cout<<image_sizes.x/3<<", "<<image_sizes.y/2<<"\n"<<image_sizes.x/3<<", "<<image_sizes.y
                <<"\n"<<2*image_sizes.x/3<<", "<<image_sizes.y/2<<"\n"<<2*image_sizes.x/3<<", "<<image_sizes.y
                <<"\n"<<image_sizes.x<<", "<<image_sizes.y/2<<"\n"<<image_sizes.x<<", "<<image_sizes.y<<std::endl;
        Mat image_small = image( Rect( 0, 0, (image_sizes.x/3), (image_sizes.y/2) ) );
        cv::imwrite(  name + "FC 100x"+ to_string(image_ind_upper) +"_1.png" , image_small );
        image_small = image( Rect( 0, image_sizes.y/2, (image_sizes.x/3), (image_sizes.y/2) ) );
        cv::imwrite(  name + "FC 100x"+ to_string(image_ind_upper) +"_2.png" , image_small );

        image_small = image( Rect( image_sizes.x/3, 0, (image_sizes.x/3), (image_sizes.y/2) ) );
        cv::imwrite(  name + "FC 100x"+ to_string(image_ind_upper) +"_3.png" , image_small );
        image_small = image( Rect( image_sizes.x/3, image_sizes.y/2, (image_sizes.x/3), (image_sizes.y/2) ) );
        cv::imwrite(  name + "FC 100x"+ to_string(image_ind_upper) +"_4.png" , image_small );

        image_small = image( Rect( 2*image_sizes.x/3, 0, (image_sizes.x-(2*image_sizes.x/3)), (image_sizes.y/2) ) );
        cv::imwrite(  name + "FC 100x"+ to_string(image_ind_upper) +"_5.png" , image_small );
        image_small = image( Rect( 2*image_sizes.x/3, image_sizes.y/2, (image_sizes.x-(2*image_sizes.x/3)), (image_sizes.y/2) ) );
        cv::imwrite(  name + "FC 100x"+ to_string(image_ind_upper) +"_6.png" , image_small );

        std::cin>>t;*/

        //int num_pixels = image.rows * image.cols;

        // Convert to grayscale
        cv::Mat image_gray;
        cv::cvtColor( image, image_gray, CV_BGR2GRAY );

        /*cv::imwrite(  name + "gray_1.png" , image_gray );
        Mat image_gray2( image_gray.size(), 0 );
        for ( int row = 0; row < image_gray.rows; row++ )
            for ( int col = 0; col < image_gray.cols; col++ ){
                int avg = 0;
                int n = 9;
                for (const auto &i : shifts_side){
                    Point p = Point(col, row) + i;
                    if(p.x<0 or p.x>image_gray.cols-1 or p.y<0 or p.y>image_gray.rows-1) {
                        n -= 1;
                        continue;
                    }
                    avg += (int)image_gray.at<uchar>( p );
                }
                for (const auto &i : shifts_diagonal){
                    Point p = Point(col, row) + i;
                    if(p.x<0 or p.x>image_gray.cols-1 or p.y<0 or p.y>image_gray.rows-1) {
                        n -= 1;
                        continue;
                    }
                    avg += (int)image_gray.at<uchar>( p );
                }
                avg += (int)image_gray.at<uchar>( row, col );
                image_gray2.at<uchar>( row, col ) = avg/n;
            }
        cv::imwrite(  name + "gray_2.png" , image_gray2 );
        Mat image_gray3( image_gray.size() , 0);
        for ( int row = 0; row < image_gray.rows; row++ )
            for ( int col = 0; col < image_gray.cols; col++ ){
                int avg = 0;
                int n = 16;
                for (const auto &i : shifts_side){
                    Point p = Point(col, row) + i;
                    if(p.x<0 or p.x==image_gray.cols or p.y<0 or p.y==image_gray.rows) {
                        n -= 2;
                        continue;
                    }
                    avg += 2*(int)image_gray.at<uchar>( p );
                }
                for (const auto &i : shifts_diagonal){
                    Point p = Point(col, row) + i;
                    if(p.x<0 or p.x>image_gray.cols-1 or p.y<0 or p.y>image_gray.rows-1) {
                        n -= 1;
                        continue;
                    }
                    avg += (int)image_gray.at<uchar>( p );
                }
                avg += 4*(int)image_gray.at<uchar>( row, col );
                image_gray3.at<uchar>( row, col ) = avg/n;
            }
        cv::imwrite(  name + "gray_3.png" , image_gray3 );*/

        std::cout<<" Thresholding image..."<<std::endl;
        // Thresholds || This function is in threshold.h
        values_pixels = Threshold( image_gray, size_small, name );

        // Generating the mask
        mask = cv::Mat_<bool>( image_sizes.y, image_sizes.x, false );
        for ( auto v : values_pixels ) mask.at<bool>( v.second ) = true;
        if (save_images) mask_skeleton = Save_Mask( image, mask, name + "_threshold" + std::to_string( size_small ) + ".png" );
        //if (save_images) Save_Binary( image, mask, name + "_binary" + std::to_string( size_small ) + ".png" );

        // r = ration of pixels in the edge to total number of edges
        if ( debug ) std::cout<<"  ratio: "<<(double)values_pixels.size() / ( image.rows * image.cols )<< std::endl;

        // Edge thinning base function (Check definition in headers/edge_thinning.cpp)
        //cv::Mat orig_image = cv::imread( input_folder + name + "." + ext, CV_LOAD_IMAGE_COLOR );
        Distance_Transform(image, mask, min_areas, save_images, name, input_folder);

        if ( save_images ){
            for ( int row = 0; row < mask.rows; row++ )
                for ( int col = 0; col < mask.cols; col++ ){
                    if ( mask.at<bool>( row, col ) ){
                        mask_skeleton.at<Vec3b>( row, col ) = black;
                    }
                }

            cv::imwrite(  name + "_skeleton_mask.png" , mask_skeleton );
        }

        return true;
    }
};


/**************************************|| MAIN FUNCTION STARTS HERE ||**************************************************
 * Defines input and output directories
 * Defines image extension and name
 * Defines Number of graphs and starting index
 *
 * Creates Planar_Graph object
 * Runs function Planar_Graph::Image_to_Graph( image_ind ) for all images
 *      where image_ind is the image index
 * @return
 *
***********************************************************************************************************************/
int main ()
{
    String ext = "tiff";                // "jpeg"; Defining extension of the images
    String name_base = "FC 100x";       // "vortex_image1";
    int num_graphs = 1;                 // Number of graphs in input folder
    int start_ind = 2;                  // Start index of graphs

    // for smaller cropped graphs
    int start_ind_upper = 5;                  // Start index of graphs
    int num_graphs_upper = 1;                 // Number of graphs in input folder
    String input_folder = "../../../src/OpenMesh/Apps/planar/input/";      // Path to input folder
    String output_folder = "../../../src/OpenMesh/Apps/planar/output/";    // Path to output folder

    Planar_Graph planar_graph( input_folder, name_base, ext, output_folder );   // Initializing using constructor
    std::cout<<"image (min_area, domains, vertices) ..."<< std::endl;

    // for loop runs for each image invoking Image_to_Graph function
    for ( int image_ind_upper = start_ind_upper; image_ind_upper < start_ind_upper + num_graphs_upper; image_ind_upper++ )
        for ( int image_ind = start_ind; image_ind < start_ind + num_graphs; image_ind++ )
            if ( ! planar_graph.Image_to_Graph( image_ind_upper, image_ind ) ) continue;
    return 3;
}

/**********************************************|| CODE ENDS HERE ||****************************************************/