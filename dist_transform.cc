/***********************************************************************************************************************
 * This file implements dist_transform.h
***********************************************************************************************************************/

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
#include <list>
#include <cmath>


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
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/connected_components.hpp>
typedef boost::adjacency_list< boost::listS, boost::vecS, boost::undirectedS, Point > Graph;

// OpenMesh include files
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
struct MyTraits : public OpenMesh::DefaultTraits
{
VertexAttributes(OpenMesh::Attributes::Status);
FaceAttributes(OpenMesh::Attributes::Status);
EdgeAttributes(OpenMesh::Attributes::Status);
};
typedef OpenMesh::PolyMesh_ArrayKernelT<MyTraits>  MyMesh;

#include "dist_transform.hh"
#include "constants.hh"
std::vector<Point> shifts24 = { Point(2,-2), Point(2,-1), Point(2,0), Point(2,1), Point(2,2), Point(1,-2), Point(1,-1), Point(1,0), Point(1,1), Point(1,2), Point(0,-2), Point(0,-1), Point(0,1), Point(0,2), Point(-1,-2), Point(-1,-1), Point(-1,0), Point(-1,1), Point(-1,2), Point(-2,-2), Point(-2,-1), Point(-2,0), Point(-2,1), Point(-2,2) };
std::vector<Point> shifts8 = { Point(1,0), Point(1,1), Point(0,1), Point(-1,1), Point(-1,0), Point(-1,-1), Point(0,-1), Point(1,-1) };
std::vector<Point> shifts4 = { Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
std::vector<Point> shifts4_diagonal = { Point(1,1), Point(-1,1), Point(-1,-1), Point(1,-1) };
MyMesh::VertexHandle vhandle[4075]; // NOLINT
std::list<std::list<Point>> removed_structre;


struct ComparePixels {
    bool operator()(const Point a, const Point b) const {
        if (a.x != b.x )return a.x < b.x;
        else return a.y < b.y;
    }
};

std::multimap< Point, Point, ComparePixels > remove_loops_points;
// typedef std::multimap<Point, bool, ComparePixels>::iterator MIterator;
typedef std::multimap<Point, std::pair<Point, std::pair<Point, Point>>, ComparePixels>::iterator MMAPIterator;
typedef std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >::iterator MMAPEdgeIterator;


// Function to get the values of neighbours
bool Neighbors (Mat_<bool>const& mask, Point point, std::vector<Point>const& shifts, int& num_neighbors, std::vector<bool>& neighbors)
{
    Point p;
    num_neighbors = 0;
    neighbors.assign( shifts.size(), true );
    for ( int i = 0; i < shifts.size(); i++ )
    {
        p = point + shifts[i];
        if ( p.x >= 0 and p.x < mask.cols and p.y >= 0 and p.y < mask.rows )
            neighbors[i] = mask.at<bool>( p ); // the presence of neighbor
        if ( neighbors[i] ) num_neighbors++;
    }
    return true;
}


// Functions to remove an isolated pixel || where number of Neighbours is zero
bool Remove_Isolated_Pixel (const Point &point, Mat_<bool>& mask)
{
    if ( ! mask.at<bool>( point ) ) return false; // already removed
    int num_neighbors = 0;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    if ( num_neighbors == 0 ) { mask.at<bool>( point ) = false; return true; }
    return false;
}


// Remove an external pixel
bool Remove_External_Pixel (Point point, Mat_<bool>& mask)
{
    // exceptional cases
    if ( point.x < 0 or point.y < 0 or point.x >= mask.cols or point.y >= mask.rows ) return false;
    if ( Remove_Isolated_Pixel( point, mask ) ) return true;
    // corners aren't removed
    if ( point == Point(0,0) ) return false;
    if ( point == Point(0,mask.rows-1) ) return false;
    if ( point == Point(mask.cols-1,0) ) return false;
    if ( point == Point(mask.cols-1,mask.rows-1) ) return false;
    //bool debug = false; if ( point.x == mask.cols-1 ) debug = true;
    int num_neighbors;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    int changes = 0;
    for ( int i = 0; i < shifts8.size(); i++ )
        if ( neighbors[i] != neighbors[ (i+1) % neighbors.size() ] ) changes++;
    //if ( debug ) std::cout<<"\n"<<point<<" n="<<num_neighbors<<" c="<<changes;
    if ( changes == 2 and num_neighbors <= 4 )
    {
        mask.at<bool>( point ) = false;
        for ( int i = 0; i < shifts8.size(); i++ )
            if ( neighbors[i] )
                Remove_External_Pixel( point + shifts8[i], mask );
    }
    else{
        return false;
    }
    return true;
}


// Mark_External
bool Mark_External (Point point, Mat_<bool>& mask, Mat_<Vec3b>& image_mask, const Vec3b &colour1, const Vec3b &colour2);

bool Mark_External(Point point, Mat_<bool> &mask, Mat_<Vec3b> &image_mask, const Vec3b &colour1, const Vec3b &colour2) {
    // exceptional cases
    if ( point.x < 0 or point.y < 0 or point.x >= mask.cols or point.y >= mask.rows ) return false;
    if ( Remove_Isolated_Pixel( point, mask ) ) return true;
    if ( point == Point(0,0) ) return false;
    if ( point == Point(0,mask.rows-1) ) return false;
    if ( point == Point(mask.cols-1,0) ) return false;
    //bool debug = false; if ( point.x == mask.cols-1 ) debug = true;
    int c = 0;
    if (colour2 == red) {
        for (const auto &i : shifts4) {
            if (image_mask.at<Vec3b>(point + i) == colour1) {
                c++;
            }
        }
        if (c == 0) return false;
    }
    c=0;
    for (const auto &i : shifts8) {
        if ( image_mask.at<Vec3b>( point + i) == colour1 ){
            if (colour2 == red)
            {
                c++;
                if (c == 1){
                    image_mask.at<Vec3b>( point ) = colour2;
                    return true;
                }
            }
            else{
                image_mask.at<Vec3b>( point ) = colour2;
                return true;
            }
        }
    }

}

// Function marks a layer
bool Mark_External_Pixels (Mat_<bool>& mask, Mat_<Vec3b>& image_mask, const Vec3b &colour1, const Vec3b &colour2)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( image_mask.at<Vec3b>( Point( col, row ) ) == black ){
                //std::cout<<"\n("<<col<<", "<<row<<")";
                //std::cout<<"\nm="<<m<<" distance_colors[m]="<<temp2<<" distance_colors[m-1]="<<temp;
                Mark_External( Point( col, row ), mask, image_mask, colour1, colour2);
            }
    return true;
}

bool Connectivity(Point p1, Point p2, Mat_<bool>& mask, const Point &point)
{
    if ( p1 == Point(0,0) ) return false;
    if ( p1 == Point(0,mask.rows-1) ) return false;
    if ( p1 == Point(mask.cols-1,0) ) return false;
    if ( p1 == Point(mask.cols-1,mask.rows-1) ) return false;
    if ( p2 == Point(0,0) ) return false;
    if ( p2 == Point(0,mask.rows-1) ) return false;
    if ( p2 == Point(mask.cols-1,0) ) return false;
    if ( p2 == Point(mask.cols-1,mask.rows-1) ) return false;
    // int x;
    if (debug) std::cout<<"\ninside "<< p1.x <<","<<p1.y<<" "<<p2.x <<","<<p2.y<< std::endl;
    //std::cin>>x;

    for ( int i = 0; i<shifts8.size(); i++)
    {
        Point p = p1+shifts8[i];
        if (debug) std::cout<<"\ninside for"<< p.x <<","<<p.y<<" "<<p2.x <<","<<p2.y<< std::endl;
        if ( p.x < 0 or p.y < 0 or p.x >= mask.cols or p.y >= mask.rows ) continue;
        else if (mask.at<bool>(p))
        {
            if (p == p2 and p!=point)
            {
                if (debug) std::cout<<"\ninside else if"<< p.x <<","<<p.y<<" "<<p2.x <<","<<p2.y<< std::endl;
                return true;
            }
            for (const auto &j : shifts8) {
                Point q = p2 + j;
                if ( q.x < 0 or q.y < 0 or q.x >= mask.cols or q.y >= mask.rows ) continue;
                else if (mask.at<bool>(q) and p==q and p!=point)
                {
                    if (debug) std::cout<<"\ninside else if for"<< p.x <<","<<p.y<<" "<<q.x <<","<<q.y<< std::endl;
                    return true;
                }
            }
        }
    }
    return false;
}

bool Remove_Border_Connectivity(Point point, Mat_<bool>& mask, Mat_<Vec3b>& image_mask)
{
    bool connectivity;
    int counter;
    bool connectivity_overall = true;
    //std::cout<<"Point "<<point<<std::endl;
    for (int i = 0; i<shifts8.size(); i++)
    {
        if(( point.x + shifts8[i].x)>-1 and ( point.y + shifts8[i].y)>-1 and ( point.x + shifts8[i].x) < mask.cols and ( point.y + shifts8[i].y) < mask.rows)
            if(mask.at<bool>( point + shifts8[i]))
            {
                Point point1 = point + shifts8[i];
                counter = 0;
                connectivity = false;
                for (int j = 0; j<shifts8.size(); j++)
                {
                    if(i == j) continue;
                    else if(( point.x + shifts8[j].x)>-1 and ( point.y + shifts8[j].y)>-1 and ( point.x + shifts8[j].x) < mask.cols and ( point.y + shifts8[j].y) < mask.rows)
                        if(mask.at<bool>( point + shifts8[j]))
                        {
                            Point point2 = point + shifts8[j];
                            double dist = sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2));
                            if(dist>1) connectivity = Connectivity(point1, point2, mask, point);
                            if (connectivity) counter++;
                        }
                }
                if (counter==0) connectivity_overall = false;
            }
    }


    if(connectivity_overall)
    {
        mask.at<bool>( point ) = false;
        image_mask.at<Vec3b>( point ) = white;
        // check = true;
        // if (debug) std::cout<<"\n ("<< point.x <<","<<point.y<<") removed because of connectivity preserved"<<std::endl;
    }
    return true;
}

// Remove a pixel from a layer
bool Remove_Layer_Pixel (Point point, Mat_<bool>& mask, Mat_<Vec3b>& image_mask, bool& check, int row, int col, bool debug)
{
    // exceptional cases
    if ( point.x < 0 or point.y < 0 or point.x >= mask.cols or point.y >= mask.rows ) return false;
    if ( Remove_Isolated_Pixel( point, mask ) ) return true;
    // corners aren't removed
    if ( point == Point(0,0) ) return false;
    if ( point == Point(0,mask.rows-1) ) return false;
    if ( point == Point(mask.cols-1,0) ) return false;
    if ( point == Point(mask.cols-1,mask.rows-1) ) return false;

    int count = 0;
    bool connectivity_overall = true;
    int num_neighbors;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    int changes = 0;
    for ( int i = 0; i < shifts8.size(); i++ )
        if ( neighbors[i] != neighbors[ (i+1) % neighbors.size() ] ) changes++;

    for (const auto &i : shifts8)
        if (mask.at<bool>( point + i)) count++;

    if(count<2)
    {
        mask.at<bool>( point ) = false;
        image_mask.at<Vec3b>( point ) = white;
        check = true;
        if (debug) std::cout<<"\n ("<< point.x <<","<<point.y<<") removed because of count<2"<<std::endl;
        return true;
    }
    else if(changes==2)
    {
        mask.at<bool>( point ) = false;
        image_mask.at<Vec3b>( point ) = white;
        check = true;
        if (debug) std::cout<<"\n ("<< point.x <<","<<point.y<<") removed because of changes==2"<<std::endl;
        return true;
    }
    else if ( count > 1 )
    {
        bool connectivity;
        int counter;
        for (int i = 0; i<shifts8.size(); i++)
        {
            if(mask.at<bool>( point + shifts8[i]))
            {
                Point point1 = point + shifts8[i];
                counter = 0;
                connectivity = false;
                for (int j = 0; j<shifts8.size(); j++)
                {
                    if(i == j) continue;
                    else if(mask.at<bool>( point + shifts8[j]))
                    {
                        Point point2 = point + shifts8[j];
                        double dist = sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2));
                        if(dist>1) connectivity = Connectivity(point1, point2, mask, point);
                        if (connectivity) counter++;
                    }
                }
                if (counter==0) connectivity_overall = false;
            }
        }


        if(connectivity_overall)
        {
            mask.at<bool>( point ) = false;
            image_mask.at<Vec3b>( point ) = white;
            check = true;
            if (debug) std::cout<<"\n ("<< point.x <<","<<point.y<<") removed because of connectivity preserved"<<std::endl;
        }
    }
    return true;
}


// Function removes a layer
bool Remove_Layer (Mat_<bool>& mask, Mat_<Vec3b>& image_mask, std::string name, int m, int& c)
{
    bool check = false;
    bool iteration_check = false;
    int count = 0, start = 1;
    do{
        check = false;
        for ( int row = start; row < mask.rows-start; row++ )
            for ( int col = start; col < mask.cols-start; col++ )
                if ( image_mask.at<Vec3b>( Point( col, row ) ) == red ){
                    //std::cout<<"\n("<<col<<", "<<row<<")";
                    // std::cin>>x;
                    //if (col == 252 and row == 102) debug = true;
                    //else debug =false;
                    Remove_Layer_Pixel( Point( col, row ), mask, image_mask, check, row, col, false);
                    //
                    // c++;
                    //if (m==11) cv::imwrite( name +"_distance m=" + std::to_string( m ) + " c=" + std::to_string( c ) + "col"+ std::to_string( col ) + "row"+ std::to_string( row ) + ".png" , image_mask );
                }
        if (check){
            iteration_check = true;
        }
    }while (check);

    return iteration_check;
}


// Function adds a missing internal pixel inside the edge
bool Add_Internal_Pixel (Point point, Mat_<bool>& mask)
{
    // exceptional cases
    if ( point.x < 0 or point.y < 0 or point.x >= mask.cols or point.y >= mask.rows ) return false;
    int num_neighbors;
    std::vector<bool> neighbors( shifts4.size(), true );
    Neighbors( mask, point, shifts4, num_neighbors, neighbors );
    if ( num_neighbors >= 3 ) // a pixel with at least 3 of 4 potential neighbors is called external
    {
        mask.at<bool>( point ) = true;
        neighbors.assign( shifts8.size(), true );
        Neighbors( mask, point, shifts8, num_neighbors, neighbors );
        for ( int i = 0; i < shifts8.size(); i++ )
            if ( neighbors[i] ) Remove_External_Pixel( point + shifts8[i], mask );
            else Add_Internal_Pixel( point + shifts8[i], mask );
    }
    return true;
}

// Function adds all missing internal pixels inside the edges
bool Add_Internal_Pixels (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( ! mask.at<bool>( Point( col, row ) ) )
                Add_Internal_Pixel( Point( col, row ), mask );
    return true;
}

bool Remove_Pixel( Point p,  Mat_<Vec3b>& binary_skeleton,  Mat_<Vec3b>& image_mask,  Mat_<bool>& mask, Mat image, bool& check )
{
    int count = 0;
    if(binary_skeleton.at<Vec3b>(p) == green){
        for (const auto &i : shifts8){
            if (binary_skeleton.at<Vec3b>(p + i) == red or binary_skeleton.at<Vec3b>(p + i) == green)
                count++;
        }
        if (count == 1){
            //std::cout << "Removed 3"<<p << std::endl;
            binary_skeleton.at<Vec3b>(p) = black;
            image_mask.at<Vec3b>(p) = image.at<Vec3b>(p);
            mask.at<bool>(p) = false;
            check = true;
            for (auto &i : shifts8) {
                if (binary_skeleton.at<Vec3b>(p + i) == red or binary_skeleton.at<Vec3b>(p + i) == green) {
                    Remove_Pixel(p + i, binary_skeleton, image_mask, mask, image, check);
                }
            }
        }
        return true;
    }
    else{
        count = 0;
        for (const auto &i : shifts8) {
            if (binary_skeleton.at<Vec3b>(p + i) == red or binary_skeleton.at<Vec3b>(p + i) == green) {
                if (binary_skeleton.at<Vec3b>(p + i) == red) {
                    bool t = true;
                    for (const auto &j : shifts8)
                        for (const auto &k : shifts8) {
                            if ((p + i + j).x >= 0 and (p + i + j).y >= 0 and
                                (p + i + j).x < binary_skeleton.cols and
                                (p + i + j).y < binary_skeleton.rows)
                                if ((binary_skeleton.at<Vec3b>(p + i + j) == green or
                                     binary_skeleton.at<Vec3b>(p + i + j) == red) and
                                    (p + i + j == p + k))
                                    t = false;
                        }
                    if (t) count++;
                } else count++;
            }
        }
        if (count == 1) {
            //std::cout << "Removed 3"<<p << std::endl;
            binary_skeleton.at<Vec3b>(p) = black;
            image_mask.at<Vec3b>(p) = image.at<Vec3b>(p);
            mask.at<bool>(p) = false;
            check = true;
            for (auto &i : shifts8) {
                if (binary_skeleton.at<Vec3b>(p + i) == red or binary_skeleton.at<Vec3b>(p + i) == green) {
                    Remove_Pixel(p + i, binary_skeleton, image_mask, mask, image, check);
                }
            }
        }
        return true;
    }
}

bool Pixels_to_Graph (Mat_<bool>const& mask, bool object, Graph& graph)
{
    graph.clear();
    std::map< Point, Graph::vertex_descriptor, Compare_Points > pixels_vertices;
    // Add vertices
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            if ( mask.at<bool>( row, col ) != object ) continue; // irrelevant pixel
            auto vertex = boost::add_vertex( graph );
            graph[ vertex ] = Point( col, row );
            pixels_vertices.insert( std::make_pair( Point( col, row ), vertex ) );
        }
    // Add horizontal edges
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col+1 < mask.cols; col++ )
        {
            Point p0( col, row ), p1(  col+1, row );
            if ( mask.at<bool>( p0 ) != object or mask.at<bool>( p1 ) != object ) continue;
            boost::add_edge( pixels_vertices[ p0 ], pixels_vertices[ p1 ], graph );
        }
    // Add vertical edges
    for ( int row = 0; row+1 < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            Point p0( col, row ), p1(  col, row+1 );
            if ( mask.at<bool>( p0 ) != object or mask.at<bool>( p1 ) != object ) continue;
            boost::add_edge( pixels_vertices[ p0 ], pixels_vertices[ p1 ], graph );
        }
    //Add diagonal edges from top left
    for (int row = 0; row < mask.rows - 1; row++)
        for(int col=0; col < mask.cols - 1; col++)
        {
            Point p0(col, row), p1(col + 1, row), p2(col + 1, row + 1), p3(col, row + 1);
            //NOT add diagonal edges if already connected.
            if ( mask.at<bool>( p0 ) != object or mask.at<bool>( p2 ) != object or mask.at<bool>(p1 ) == object or mask.at<bool>(p3 ) == object) continue;
            boost::add_edge(pixels_vertices[ p0 ], pixels_vertices[ p2 ], graph);
        }
    //Add diagonal edges from top-right
    for (int row = 0; row < mask.rows - 1; row++)
        for(int col = mask.cols-1 ; col > 0; col--)
        {
            Point p0(col, row), p1(col - 1, row), p2(col - 1, row + 1), p3(col, row + 1);
            //NOT add diagonal edges if already connected.
            if ( mask.at<bool>( p0 ) != object or mask.at<bool>( p2 ) != object or mask.at<bool>(p1 ) == object or mask.at<bool>(p3 ) == object) continue;
            boost::add_edge(pixels_vertices[ p0 ], pixels_vertices[ p2 ], graph);
        }
    return true;
}


// Check if pixel is a boundry pixel
bool Is_Boundary (Point p, Point size)
{
    return p.x == 0 or p.y == 0 or p.x == size.x - 1 or p.y == size.y - 1;
}

bool Is_Boundary (Point p, Mat image)
{
    return p.x == 1 or p.y == 1 or p.x == image.cols - 2 or p.y == image.rows - 2;
}

bool Is_Boundary (MyMesh::Point p, Mat image)
{
    return p[0] == 1 or p[1] == 1 or p[0] == image.cols - 2 or p[1] == image.rows - 2;
}

bool Is_Boundary (std::vector<Point>const& points, const Point &size)
{
    for (const auto &p : points )
        if ( Is_Boundary( p, size ) ) return true;
    return false;
}

// Remove all small components
bool Remove_Small_Components (bool object, int area_min, Mat_<bool>& mask, int& num_components, bool flag)
{
    Graph graph;
    Pixels_to_Graph( mask, object, graph );
    std::vector<int> vertex_components( boost::num_vertices( graph ) );
    // Count the sizes of components
    num_components = boost::connected_components( graph, &vertex_components[0]);
    std::vector< int > component_sizes( num_components, 0 );
    for ( int i = 0; i != vertex_components.size(); ++i )
        component_sizes[ vertex_components[ i ] ]++;
    //std::cout<<"\nComponents:"; for ( int i = 0; i < component_sizes.size(); i++ ) std::cout<<" c"<<i<<"="<<component_sizes[i];
    // Select small components to remove
    std::map< int, std::vector<Point> > small_components;
    std::vector<Point> empty;
    // if ( object ) area_min = *max_element( component_sizes.begin(), component_sizes.end() ); // keep only the largest foreground object
    for ( int i = 0; i < component_sizes.size(); i++ )
        if ( component_sizes[i] < area_min ) small_components.insert( std::make_pair( i, empty ) );
    // Mark pixels from small components
    for ( int i = 0; i != vertex_components.size(); ++i )
    {
        auto it = small_components.find( vertex_components[ i ] );
        if ( it != small_components.end() ) (it->second).push_back( graph[i] );
    }
    num_components -= (int)small_components.size();
    // Check if any small components touches the boundary
    if ( object ) for (auto &small_component : small_components)
            if ( Is_Boundary(small_component.second, Point( mask.cols, mask.rows ) ) ) // keep components touching the boundary
            {
                (small_component.second).clear();
                num_components++;
            }//
    // Remove superfluous pixels

    if(flag){
        for (auto &small_component : small_components)
            for (const auto &p : small_component.second ) mask.at<bool>( p ) = ! object;
    }
    return true;
}

std::list<Point> Get_Disconnected(Mat_<bool>& mask, Mat_<Vec3b>& binary_skeleton){
    Graph graph;
    /*cv::imshow("bin", binary_skeleton);
    cv::waitKey(0);*/
    std::list<Point> small_component_vertex, visualize_list;
    Pixels_to_Graph( mask, true, graph );
    std::vector<int> vertex_components( boost::num_vertices( graph ) );
    // Count the sizes of components
    int num_components = boost::connected_components( graph, &vertex_components[0]);
    std::vector< int > component_sizes( num_components, 0 );
    for ( int i = 0; i != vertex_components.size(); ++i )
        component_sizes[ vertex_components[ i ] ]++;
    //std::cout<<"\nComponents:"; for ( int i = 0; i < component_sizes.size(); i++ ) std::cout<<" c"<<i<<"="<<component_sizes[i];
    // Select small components to remove
    std::map< int, std::vector<Point> > small_components;
    std::vector<Point> empty;
    for ( int i = 0; i < component_sizes.size(); i++ )
        if ( component_sizes[i] < 1000 ) small_components.insert( std::make_pair( i, empty ) );
    // Mark pixels from small components
    for ( int i = 0; i != vertex_components.size(); ++i )
    {
        auto it = small_components.find( vertex_components[ i ] );
        if ( it != small_components.end() ) (it->second).push_back( graph[i] );
    }
    num_components -= (int)small_components.size();

        for (auto &small_component : small_components)
        {
            bool keep = false;
            for (const auto &p : small_component.second ) {
                if(binary_skeleton.at<Vec3b>(p) == green)
                {
                    small_component_vertex.push_back(p);
                    if(debug) std::cout<<p;
                    keep = true;
                    break;
                }
            }
            visualize_list.clear();
            if(!keep)
            for (const auto &p : small_component.second ) {
                visualize_list.push_back(p);
            }
            removed_structre.push_back(visualize_list);
            if(debug) std::cout<<std::endl;
        }
        //std::cin>>num_components;
    return small_component_vertex;
}


int Count_Componenets (Mat binary_skeleton, Mat image){
    Graph graph;
    std::list<Point> small_component_vertex;
    std::vector<int> num_vertices( 5 ), num_domains( 5 ), num_walls( 5 );
    int i = 0, area_min;
    Mat_<bool> mask( binary_skeleton.size(), false );
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( binary_skeleton.at<Vec3b>( row, col ) == black or  binary_skeleton.at<Vec3b>( row, col ) == green)
                mask.at<bool>( row, col ) = true;
    Pixels_to_Graph( mask, true, graph );
    std::vector<int> vertex_components( boost::num_vertices( graph ) );

    // Count the sizes of components
    int num_components = boost::connected_components( graph, &vertex_components[0]);
    std::vector< int > component_sizes( num_components, 0 );
    for ( int i = 0; i != vertex_components.size(); ++i )
        component_sizes[ vertex_components[ i ] ]++;
    //std::cout<<"\nComponents:"; for ( int i = 0; i < component_sizes.size(); i++ ) std::cout<<" c"<<i<<"="<<component_sizes[i];
    // Select small components to remove
    /*std::map< int, std::vector<Point> > small_components;
    std::vector<Point> empty;
    //area_min = *max_element( component_sizes.begin(), component_sizes.end() ); // keep only the largest foreground object
    for ( int i = 0; i < component_sizes.size(); i++ )
        if ( component_sizes[i] < area_min ) {
        small_components.insert( std::make_pair( i, empty ) );
    }
    for ( int i = 0; i != vertex_components.size(); ++i )
    {
        auto it = small_components.find( vertex_components[ i ] );
        if ( it != small_components.end() ) (it->second).push_back( graph[i] );
    }

    // num_components -= (int)small_components.size();
    for (auto &small_component : small_components)
        for (const auto &p : small_component.second ) {
        mask.at<bool>( p ) = true;
        std::cout<<p;
    }*/

    std::cout<<"Number of componenets: "<<num_components<<std::endl;
    cv::imshow("src", binary_skeleton);
    cv::waitKey(0);

    int one_nbr = 0, nbr_3 = 0;
    int counter = 0, nbr_count; float total = 0; Point p;
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            if(mask.at<bool>(row, col))
            {
                nbr_count = 0;
                binary_skeleton.at<Vec3b>( row, col ) = blue;
                p = Point(col, row);
                for(auto nbr : shifts8){
                    if(mask.at<bool>(p + nbr)) nbr_count++;
                }
                if(nbr_count == 1) one_nbr++;
                //if(nbr_count >= 3) nbr_3++;
                if(nbr_count == 2) {
                    Point p11 = Point(0, 0), q11;
                    for(auto nbr : shifts8){
                        if(mask.at<bool>(p + nbr)) {
                            if(p11 == Point(0, 0))
                            {
                                p11 = p + nbr;
                            }
                            else q11 = p + nbr;
                        }
                        if (Point((p11.x + q11.x), (p11.y + q11.y)) == Point(2*p.x, 2*p.y))
                            nbr_3++;

                    }
                }
                counter++;
                total += int(image.at<uchar>( row, col ));
            }
        }

    std::cout<<"Total intensity: "<<total<<std::endl;
    std::cout<<"Total pixels: "<<counter<<std::endl;
    std::cout<<"Total one-nbr: "<<one_nbr<<std::endl;
    std::cout<<"Total three-nbr: "<<nbr_3<<std::endl;
    std::cout<<"average intensity: "<<total/counter<<std::endl;

    cv::imshow("src2", binary_skeleton);
    cv::waitKey(0);

    std::cin>>num_components;
    return num_components;
}


bool Reset_Mask (Mat& image_mask)
{
    for ( int row = 0; row < image_mask.rows; row++ )
        for ( int col = 0; col < image_mask.cols; col++ )
            if ( image_mask.at<Vec3b>( row, col ) == red or image_mask.at<Vec3b>( row, col ) == green )
                image_mask.at<Vec3b>( row, col ) = black;
    return true;
}


bool Draw_Mask (Mat const& image, Mat const& mask, Mat& image_mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                image_mask.at<Vec3b>( row, col ) = image.at<Vec3b>( row, col );
    return true;
}


Mat Save_Mask (Mat const& image, Mat const& mask, std::string name)
{
    Mat_<Vec3b> image_mask( image.size(), white );
    Draw_Mask( image, mask, image_mask );
    cv::imwrite( name, image_mask );
    return image_mask;
}

bool Draw_Binary (Mat const& image, Mat const& mask, Mat& image_mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                image_mask.at<Vec3b>( row, col ) = black;
    return true;
}


Mat Save_Binary (Mat const& image, Mat const& mask, std::string name)
{
    Mat_<Vec3b> image_mask( image.size(), white );
    Draw_Binary( image, mask, image_mask );
    cv::imwrite( name, image_mask );
    return image_mask;
}

void Image_to_BG (Mat_<Vec3b>& image_straight, std::multimap< Point, std::pair<Point, std::pair<Point, Point>>, ComparePixels >& vertices_pairs)
{
    typedef boost::adjacency_list < boost::listS, boost::vecS, boost::undirectedS, Point, boost::property < boost::edge_weight_t, int > > graph_t;
    typedef boost::graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
    typedef boost::graph_traits < graph_t >::edge_descriptor edge_descriptor;
    typedef std::pair<int, int> Edge;


    graph_t g;
    std::map< Point, Graph::vertex_descriptor, Compare_Points > pixels_vertices;

    Point ver[2000];
    int ver_i = 0;
    for(auto v : vertices_pairs)
    {
        bool t = true;
        for (int i = 0; i < ver_i; i++)
            if(ver[i] == v.first) t = false;
        if (t) {
            if(false) std::cout<<"vertex added: "<<v.first<<std::endl;
            auto vertex = boost::add_vertex(g);
            g[vertex] = v.first;
            pixels_vertices.insert(std::make_pair(v.first, vertex));
            ver[ver_i] = v.first;
            ver_i++;
        }
        t = true;
        for (int i = 0; i < ver_i; i++)
            if(ver[i] == v.second.first) t = false;
        if (t) {
            if(false) std::cout<<"vertex added: "<<v.second.first<<std::endl;
            auto vertex2 = boost::add_vertex(g);
            g[vertex2] = v.second.first;
            pixels_vertices.insert(std::make_pair(v.second.first, vertex2));
            ver[ver_i] = v.second.first;
            ver_i++;
        }
    }

    for(auto v : vertices_pairs)
        boost::add_edge( pixels_vertices[ v.first ], pixels_vertices[ v.second.first ], 1, g );

    // for dijkstra uncomment the block below
    /******************************************************************************************************************
    boost::property_map<graph_t, boost::edge_weight_t>::type weightmap = get(boost::edge_weight, g);
    std::vector<vertex_descriptor> p(num_vertices(g));
    std::vector<int> d(num_vertices(g));
    typedef boost::graph_traits<Graph>::vertices_size_type vertices_size_type;
    vertex_descriptor s = vertex(0, g);

    dijkstra_shortest_paths(g, s,
                            predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
                                    distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));

    std::cout << "distances and parents:" << std::endl;
    boost::graph_traits < graph_t >::vertex_iterator vi, vend;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
        std::cout << "distance(" << ver[*vi] << ") = " << d[*vi] << ", ";
        std::cout << "parent(" << ver[*vi] << ") = " << ver[p[*vi]] << std::
        endl;
    }
    std::cout << std::endl;

    std::ofstream dot_file("output/dijkstra-eg.dot");

    dot_file << "graph D {\n"
             << "  rankdir=LR\n"
             << "  size=\"4,3\"\n"
             << "  ratio=\"fill\"\n"
             << "  edge[style=\"bold\"]\n" << "  node[shape=\"circle\"]\n";

    boost::graph_traits < graph_t >::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei) {
        boost::graph_traits < graph_t >::edge_descriptor e = *ei;
        boost::graph_traits < graph_t >::vertex_descriptor
                u = source(e, g), v = target(e, g);
        dot_file << "x" << ver[u].x << "y" << ver[u].y << " -- " << "x" << ver[v].x << "y" << ver[v].y
                 << "[label=\"" << get(weightmap, e) << "\"";
        //if (p[v] == u)
            dot_file << ", color=\"black\"";
        //else
            //dot_file << ", color=\"grey\"";
        dot_file << "]";
    }
    dot_file << "}";
     ******************************************************************************************************************/

}


bool Boundry_points (Mat_<bool>& mask, Mat_<Vec3b>& image_mask, bool& check){
    int count = 0;
    check = false;
    for ( int row = 0; row < mask.rows; row++ ){
        /*if(mask.at<bool>(row, 0)){
            Point p = Point(0, row);
            count = 0;
            for(int i =0; i<shifts8.size(); i++)
                if ( (p+shifts8[i]).x >= 0 and (p+shifts8[i]).x < mask.cols and (p+shifts8[i]).y >= 0 and (p+shifts8[i]).y < mask.rows and mask.at<bool>(p+shifts8[i]))
                    count++;
            if (count<2){
                mask.at<bool>(row, 0) = false;
                image_mask.at<Vec3b>(row, 0) = white;
                check = true;
            }
            else Remove_Border_Connectivity(p, mask, image_mask);
        }*/
        Point p = Point(0, row);
        mask.at<bool>(row, 0) = true;
        image_mask.at<Vec3b>(row, 0) = black;
        if(row != 0)Remove_Border_Connectivity(p, mask, image_mask);
        /*if(mask.at<bool>(row, mask.cols-1)){
            Point p = Point(mask.cols-1, row);
            count = 0;
            for(int i =0; i<shifts8.size(); i++)
                if ( (p+shifts8[i]).x >= 0 and (p+shifts8[i]).x < mask.cols and (p+shifts8[i]).y >= 0 and (p+shifts8[i]).y < mask.rows and mask.at<bool>(p+shifts8[i]))
                    count++;
            if (count<2){
                mask.at<bool>(row, mask.cols-1) = false;
                image_mask.at<Vec3b>(row, mask.cols-1) = white;
                check = true;
            }
            else Remove_Border_Connectivity(p, mask, image_mask);
        }*/
        p = Point(mask.cols-1, row);
        mask.at<bool>(row, mask.cols-1) = true;
        image_mask.at<Vec3b>(p) = black;
        if(row != 0)Remove_Border_Connectivity(p, mask, image_mask);
    }

    for ( int col = 0; col < mask.cols; col++ ){
        /*if(mask.at<bool>(0, col)){
            Point p = Point(col, 0);
            count = 0;
            bool vertex_nbr = false;
            for(int i = 0; i<shifts8.size(); i++)
                if ( (p+shifts8[i]).x >= 0 and (p+shifts8[i]).x < mask.cols and (p+shifts8[i]).y >= 0 and (p+shifts8[i]).y < mask.rows and mask.at<bool>(p+shifts8[i]))
                    count++;
            if (count<2){
                mask.at<bool>(0, col) = false;
                image_mask.at<Vec3b>(0, col) = white;
                check = true;
            }
            else Remove_Border_Connectivity(p, mask, image_mask);
        }*/
        Point p = Point(col, 0);
        mask.at<bool>(0, col) = true;
        image_mask.at<Vec3b>(0, col) = black;
        if(col != 0) Remove_Border_Connectivity(p, mask, image_mask);
        /*if(mask.at<bool>(mask.rows-1, col)){
            Point p = Point(col, mask.rows-1);
            count = 0;
            for(int i = 0; i<shifts8.size(); i++)
                if ( (p+shifts8[i]).x >= 0 and (p+shifts8[i]).x < mask.cols and (p+shifts8[i]).y >= 0 and (p+shifts8[i]).y < mask.rows and mask.at<bool>(p+shifts8[i]))
                    count++;
            if (count<2){
                mask.at<bool>(mask.rows-1, col) = false;
                image_mask.at<Vec3b>(mask.rows-1, col) = white;
                check = true;
            }
            else Remove_Border_Connectivity(p, mask, image_mask);
        }*/
        p = Point(col, mask.rows-1);
        mask.at<bool>(mask.rows-1, col) = true;
        image_mask.at<Vec3b>(mask.rows-1, col) = black;
        if(col != 0) Remove_Border_Connectivity(p, mask, image_mask);
    }
    mask.at<bool>(mask.rows-1, 0) = false;
    image_mask.at<Vec3b>(mask.rows-1, mask.cols-1) = white;
    mask.at<bool>(0, 0) = false;
    image_mask.at<Vec3b>(0, mask.cols-1) = white;
}

bool Close_Vertex_removal (Mat_<bool>& mask, Mat_<Vec3b>& image_mask, Mat_<Vec3b>& binary_skeleton, Mat image, std::string name){


    // This function removes different abberations caused because of vertex being too close to each other

    // ABBERATION 1:
    // Remove 3 adjacent vertices vertices
    // Uncomment this part of the code to remove the above abberation
    /*
    int vertex_count;
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( image_mask.at<Vec3b>( row, col )==green )
            {
                Point p = Point(col, row);
                Point replace[3];
                vertex_count = 0;
                for(int i =0; i<shifts8.size(); i++)
                    if (image_mask.at<Vec3b>(p+shifts8[i])==green)
                    {
                        replace[vertex_count++] = p+shifts8[i];
                    }
                if(vertex_count > 1){
                    mask.at<bool>(p) = false;
                    image_mask.at<Vec3b>(p) = image.at<Vec3b>(p);
                    binary_skeleton.at<Vec3b>(p) = black;
                    int replace_index;
                    int replace_counter = 0;
                    for (int i = 0; i <vertex_count; i++)
                        for(int j = 0; j <shifts4.size() ; j++)
                        if(binary_skeleton.at<Vec3b>(replace[i]+shifts4[j]) == red)
                        {
                            mask.at<bool>(replace[i]+shifts4[j]) = false;
                            image_mask.at<Vec3b>(replace[i]+shifts4[j]) = image.at<Vec3b>(p);
                            binary_skeleton.at<Vec3b>(replace[i]+shifts4[j]) = black;
                            for(int k = 0; k <shifts4.size() ; k++){
                                int green_count =0;
                                if(binary_skeleton.at<Vec3b>(replace[i]+shifts4[j]+shifts4[k]) == black){
                                    for(int l = 0; l <shifts8.size() ; l++){
                                        if(binary_skeleton.at<Vec3b>(replace[i]+shifts4[j]+shifts4[k]+shifts8[l]) == green)
                                            green_count++;
                                    }
                                    if(green_count == 1 ){
                                        mask.at<bool>(replace[i]+shifts4[j]+shifts4[k]) = true;
                                        image_mask.at<Vec3b>(replace[i]+shifts4[j]+shifts4[k]) = red;
                                        binary_skeleton.at<Vec3b>(replace[i]+shifts4[j]+shifts4[k]) = red;
                                    }
                                }
                            }
                        }
                    for (int i = 0; i <vertex_count; i++) {
                        replace_counter=0;
                        for (int j = 0; j < shifts8.size(); j++) {
                            if (binary_skeleton.at<Vec3b>(replace[i] + shifts8[j]) == red)
                                replace_counter++;
                        }
                        if (replace_counter==1) {
                            replace_index = i;
                            break;
                        }
                    }

                    for(int i = 0; i<shifts8.size(); i++)
                        if (binary_skeleton.at<Vec3b>(p+shifts8[i])==red){
                        Point connect = p + shifts8[i];
                        bool connected = false;
                        {
                            for(int  j=0; j<shifts8.size(); j++)
                                for(int  k=0; k<shifts8.size(); k++)
                                    {
                                        Point common1 = connect+shifts8[j];
                                        Point common2 = replace[replace_index]+shifts8[k];
                                        if(common1 == common2 and common1 != p)
                                            {
                                                int count2 = 0;
                                                for(int l = 0; l<shifts8.size(); l++){
                                                    if(binary_skeleton.at<Vec3b>(common1+shifts8[l])==green)
                                                        count2++;
                                                }
                                                if(count2 == 1){
                                                    mask.at<bool>(connect+shifts8[j]) = true;
                                                    image_mask.at<Vec3b>(connect+shifts8[j]) = red;
                                                    binary_skeleton.at<Vec3b>(connect+shifts8[j]) = red;
                                                    connected = true;
                                                    break;
                                                }
                                            }
                                    }
                                    if (connected) break;
                        }
                    }
                }
            }
            */


    // ABBERATION 2
    // This part of the code removes abberations caused due to removal process making holes that do not exist.
    Mat_<Vec3b> small_image(10, 10, white);
    int num_vertex_window = 0;
    int vertex_count = 0;
    std::multimap< int, Point > values_pixels;
    for ( int row = 0; row < binary_skeleton.rows; row++ )
        for ( int col = 0; col < binary_skeleton.cols; col++ )
            if ( binary_skeleton.at<Vec3b>( row, col )==green ) {
                num_vertex_window = 0;
                ++vertex_count;
                int boundry_point;
                // std::cout<<"yaayy "<<row<<" "<<col<<std::endl;
                for (int small_row = -5; small_row < 5; small_row++)
                    for (int small_col = -5; small_col < 5; small_col++)
                        if (row + small_row >= 0 and col + small_col >= 0 and row + small_row < binary_skeleton.rows and
                            col + small_col < binary_skeleton.cols) {
                            if (binary_skeleton.at<Vec3b>(row + small_row, col + small_col) == green and small_col+5>0 and small_row+5>0 and small_row+5<small_image.rows-1 and small_col+5<small_image.cols-1)
                                num_vertex_window++;
                            small_image.at<Vec3b>(small_row + 5, small_col + 5) = binary_skeleton.at<Vec3b>(
                                    row + small_row, col + small_col);
                            if (small_image.at<Vec3b>(small_row + 5, small_col + 5) == green and (small_col!=0 or small_row!=0)) {
                                small_image.at<Vec3b>(small_row + 5, small_col + 5) = blue;
                                binary_skeleton.at<Vec3b>(row + small_row, col + small_col) = blue;
                            }
                        }
                if (num_vertex_window > 1) {
                    boundry_point = 0;
                    for (int small_row = -5; small_row < 5; small_row++)
                        for (int small_col = -5; small_col < 5; small_col++)
                            if (small_image.at<Vec3b>(small_row+5, small_col+5)==red or small_image.at<Vec3b>(small_row+5, small_col+5)==green){
                                Point p = Point(5 + small_col, 5 + small_row);
                                int p_nbrs = 0;
                                if (small_col + 5 == 0 or small_row + 5 == 0 or small_col + 5 == 9 or small_row + 5 == 9)
                                    for (auto &i : shifts8) {
                                        if (small_image.at<Vec3b>(p + i)==green or small_image.at<Vec3b>(p + i)==blue){
                                            p_nbrs = 1;
                                            break;
                                        }
                                        else if ((small_image.at<Vec3b>(p + i)==red or small_image.at<Vec3b>(p + i)==blue) and p.x+ i.x>-1 and p.x+ i.x<10 and p.y+ i.y>-1 and p.y+ i.y<10 ) {
                                            p_nbrs++;
                                        }
                                    }
                                if (p_nbrs == 1) boundry_point++;
                                //std::cout<<"\t\t boundary at: "<<small_row+5<<" "<<small_col+5<<std::endl;
                            }
                    // checking in 5x5 neighbourhood
                    // for (int i = 0 ; i < shifts8.size() ; i++)
                    if (boundry_point - 2 < num_vertex_window) {
                        cv::imwrite(name + "_vertex_" + std::to_string(vertex_count) + ".png", small_image);
                        for (int small_row = -5; small_row < 5; small_row++)
                            for (int small_col = -5; small_col < 5; small_col++)
                                if (small_col == -5 or small_col == 4 or small_row == -5 or small_row == 4){
                                    binary_skeleton.at<Vec3b>(row + small_row, col + small_col) = blue;
                                }
                    }
                    else {
                        for (int small_row = -5; small_row < 5; small_row++)
                            for (int small_col = -5; small_col < 5; small_col++)
                                if (row + small_row >= 0 and col + small_col >= 0 and row + small_row < binary_skeleton.rows and
                                    col + small_col < binary_skeleton.cols){
                                    if (binary_skeleton.at<Vec3b>(row + small_row, col + small_col) == blue)
                                        binary_skeleton.at<Vec3b>(row + small_row, col + small_col) = green;
                                }
                    }
                }
            }

    // Uncomment below function to save all the images of size 10x10 in which there are 2 or more vertices in 5x5 neighbourhood
    /*
vertex_count = 0;
Mat_<Vec3b> small_image(10, 10, white);
for ( int row = 0; row < binary_skeleton.rows; row++ )
for ( int col = 0; col < binary_skeleton.cols; col++ )
    if ( binary_skeleton.at<Vec3b>( row, col )==green ) {
        ++vertex_count;
        Point p = Point(col, row);
        int num_vertex = 0;
        for(int i =0; i<shifts24.size(); i++){
            if(binary_skeleton.at<Vec3b>(p+shifts24[i])==green)
                num_vertex++;
        }
        if (num_vertex>0) {
            for (int small_row = -5; small_row < 5; small_row++)
                for (int small_col = -5; small_col < 5; small_col++) {
                    small_image.at<Vec3b>(small_row + 5, small_col + 5) = binary_skeleton.at<Vec3b>(
                            row + small_row, col + small_col);
                }
            cv::imwrite(name + "_vertex_close_" + std::to_string(vertex_count) + ".png", small_image);
        }
}*/

}

Mat_<Vec3b> Mark_Vertex(Mat_<bool>& mask, Mat image, Mat_<Vec3b>& binary_skeleton, std::multimap< Point, bool, ComparePixels >& vertices_pixels)
{
    Mat_<Vec3b> image_mask2( image.size(), white );
    Point p;
    int vertex_count = 0;
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ ){
            image_mask2.at<Vec3b>( row, col ) = image.at<Vec3b>( row, col );
            int num_neighbors;
            std::vector<bool> neighbors( shifts8.size(), true );
            Neighbors( mask, Point(col, row), shifts8, num_neighbors, neighbors );
            if ( mask.at<bool>( row, col ) ){
                image_mask2.at<Vec3b>( row, col ) = black;
                binary_skeleton.at<Vec3b>( row, col ) = red;
                if (num_neighbors == 3 and row >= 1 and col >= 1 and col <= mask.cols-1 and row <= mask.rows-1 ){
                    p = Point(col, row);
                    Point nbr;
                    int count = 0;
                    for (const auto &i : shifts8)
                        if(image_mask2.at<Vec3b>( p + i) == green) {
                            count++;
                            nbr = p + i;
                        }
                    if (count == 0)
                    {
                        binary_skeleton.at<Vec3b>( row, col ) = green;
                        image_mask2.at<Vec3b>( row, col ) = green;
                        vertex_count++;
                        vertices_pixels.insert( std::make_pair( Point( col, row ), false ) );
                    }
                    else if(count == 1){
                        count = 0;
                        for (const auto &i : shifts8) {
                            for (const auto &j : shifts8) {
                                if(mask.at<bool>( p + i ))
                                    if(p+i == nbr+j) count++;
                            }
                        }
                        if (count == 0)
                        {
                            binary_skeleton.at<Vec3b>( row, col ) = green;
                            image_mask2.at<Vec3b>( row, col ) = green;
                            vertex_count++;
                            vertices_pixels.insert( std::make_pair( Point( col, row ), false ) );
                        }
                    }
                }
                else if (num_neighbors == 4){
                        binary_skeleton.at<Vec3b>(row, col) = green;
                        image_mask2.at<Vec3b>(row, col) = green;
                        vertex_count++;
                        vertices_pixels.insert(std::make_pair(Point(col, row), false));
                }
            }
        }
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ ){
        int count = 0;
            Point p = Point(col, row);
            for(const auto &i : shifts4)
            {
                if (image_mask2.at<Vec3b>(p + i) == green)
                    count++;
            }
            if (count>3) {
                for (const auto &i : shifts4) {
                    image_mask2.at<Vec3b>(p + i) = black;
                    binary_skeleton.at<Vec3b>(p + i) = red;
                    mask.at<bool>(p + i) = false;
                    vertices_pixels.erase(p + i);
                    vertex_count--;
                }
                image_mask2.at<Vec3b>(p + shifts4_diagonal[1]) = green;
                binary_skeleton.at<Vec3b>(p + shifts4_diagonal[1]) = green;
                mask.at<bool>(p + shifts4_diagonal[1]) = true;
                vertices_pixels.insert(std::make_pair(p + shifts4_diagonal[1], false));
                image_mask2.at<Vec3b>(p + shifts4_diagonal[3]) = green;
                binary_skeleton.at<Vec3b>(p + shifts4_diagonal[3]) = green;
                mask.at<bool>(p + shifts4_diagonal[3]) = true;
                vertices_pixels.insert(std::make_pair(p + shifts4_diagonal[3], false));
                vertex_count++;

                image_mask2.at<Vec3b>(p) = black;
                binary_skeleton.at<Vec3b>(p) = red;
                vertices_pixels.erase(p);
            }
    }
    std::cout<<" Number of vertices = "<<vertex_count<<std::endl;

    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            if ( binary_skeleton.at<Vec3b>( row, col )==red ){
                int green_nbrs = 0;
                Point green1, green2;
                p = Point(col, row);
                for(auto i :shifts4){
                    if ( binary_skeleton.at<Vec3b>( p+i )==green ) {
                        green_nbrs++;
                        green1 = p+i;
                        break;
                    }
                }
                for(const auto &i :shifts8){
                    if ( binary_skeleton.at<Vec3b>( p+i )==green and p+i!=green1 ) {
                        green_nbrs++;
                        green2 = p+i;
                    }
                }
                if(green_nbrs == 2){
                    int red_count = 0;
                    for(auto i :shifts8){
                        if(green1+i == green2) green_nbrs=3;
                        if(binary_skeleton.at<Vec3b>( green1+i )==red) red_count++;
                    }
                    if(green_nbrs == 3) {
                        Point replace;
                        if(debug) std::cout<<red_count;
                        if(red_count == 3){
                            if(debug) std::cout<<red_count;
                            replace = green1;
                            green1 = green2;
                            green2 = replace;
                        }
                        for(const auto &i :shifts8){
                            for(const auto &j:shifts8){
                                if(i+green1==j+p and j+p != green2){
                                    for(auto k:shifts8){
                                        if(green1+i == green2+k) green_nbrs = 4;
                                    }
                                    if(green_nbrs != 4 ) {
                                        bool red_nbr = false;
                                        for(auto l : shifts8) {
                                            if (binary_skeleton.at<Vec3b>(p + l) == red) {
                                                red_nbr = true;
                                                for (auto m : shifts8) {
                                                    if (p + l + m == green1 + i) {
                                                        replace = green1 + i;
                                                        break;
                                                    }
                                                    else continue;
                                                }
                                            }
                                        }
                                        if (!red_nbr){
                                            replace = green1+i;
                                            break;
                                        }

                                    }
                                    else green_nbrs = 3;
                                }
                            }
                        }
                        if(debug) std::cout << p<<replace<<std::endl;
                        // std::cin >> green_nbrs;
                        binary_skeleton.at<Vec3b>( p ) = black;
                        image_mask2.at<Vec3b>( p ) = image.at<Vec3b>( p );
                        mask.at<bool>(p) = false;
                        binary_skeleton.at<Vec3b>( replace ) = red;
                        image_mask2.at<Vec3b>( replace ) = black;
                        mask.at<bool>(p) = true;
                    }
                }
            }
        }
    return image_mask2;
}

bool Mark_Vertex(Mat_<Vec3b>& image_straight, std::multimap< Point, bool, ComparePixels >& vertices_pixels)
{
    for (auto v : vertices_pixels)
    {
        image_straight.at<Vec3b>(v.first) = green;
    }
}

int Image_Factor(Mat image, const Point &p){
    return (int(image.at<Vec3b>( p )[1])+int(image.at<Vec3b>( p )[0]));
}

int Image_Factor2(Mat image, const Point &p){
    return (int(image.at<Vec3b>( p )[1])*int(image.at<Vec3b>( p )[1]));
}

int Image_Factor3(Mat image, const Point &p){
    return (int(image.at<uchar>( p )));
}

double Standard_Deviation(std::list<Point> point_list, const Mat &image){
    double mean = 0, sd = 0;
    for(const auto &v : point_list){
        mean += Image_Factor3(image, v);
    }
    mean /= point_list.size();
    for(const auto &v : point_list){
        sd += (Image_Factor3(image, v)-mean)*(Image_Factor3(image, v)-mean);
    }
    return sqrt(sd/(point_list.size()-1));
}

double Standard_Deviation1(std::list<Point> point_list, const Mat &image){
    double mean = 0, sd = 0;
    for(const auto &v : point_list){
        mean += int(image.at<Vec3b>( v )[0]);
    }
    mean /= point_list.size();
    for(const auto &v : point_list){
        sd += (int(image.at<Vec3b>( v )[0])-mean)*(int(image.at<Vec3b>( v )[0])-mean);
    }
    return sqrt(sd/(point_list.size()-1));
}

double Standard_Deviation2(std::list<Point> point_list, const Mat &image){
    double mean = 0, sd = 0;
    for(const auto &v : point_list){
        mean += int(image.at<Vec3b>( v )[1]);
    }
    mean /= point_list.size();
    for(const auto &v : point_list){
        sd += (int(image.at<Vec3b>( v )[1])-mean)*(int(image.at<Vec3b>( v )[1])-mean);
    }
    return sqrt(sd/(point_list.size()-1));
}

double Standard_Deviation3(std::list<Point> point_list, const Mat &image){
    double mean = 0, sd = 0;
    for(const auto &v : point_list){
        mean += int(image.at<Vec3b>( v )[2]);
    }
    mean /= point_list.size();
    for(const auto &v : point_list){
        sd += (int(image.at<Vec3b>( v )[2])-mean)*(int(image.at<Vec3b>( v )[2])-mean);
    }
    return sqrt(sd/(point_list.size()-1));
}

double Mean(std::list<Point> point_list, const Mat &image){
    double mean = 0;
    for(const auto &v : point_list){
        mean += Image_Factor3(image, v);
    }
    mean /= point_list.size();
    return mean;
}


int Median(std::list<Point> point_list, const Mat &image, int& min, int& max, int& range)
{
    std::vector<int> intensities;

    for(const auto &v : point_list)
        intensities.push_back(Image_Factor3(image, v));
    size_t size = intensities.size();
    if (size == 0)
    {
        return 0;  // Undefined, really.
    }
    else
    {
        sort(intensities.begin(), intensities.end());
        min = intensities.front();
        max = intensities.back();
        range = max - min;
        return size % 2 == 0 ? (intensities[size / 2 - 1] + intensities[size / 2]) / 2 : intensities[size / 2];
    }
}

int Median (std::list<Point> point_list, const Mat &image)
{
    std::vector<int> intensities;

    for(const auto &v : point_list)
        intensities.push_back(Image_Factor3(image, v));
    size_t size = intensities.size();
    if (size == 0)
    {
        return 0;  // Undefined, really.
    }
    else
    {
        sort(intensities.begin(), intensities.end());
        for(const auto &v : intensities)
            std::cout<<v<<std::endl;
        return size % 2 == 0 ? (intensities[size / 2 - 1] + intensities[size / 2]) / 2 : intensities[size / 2];
    }
}

long Mode(std::list<Point> point_list, const Mat &image){
    std::vector<int> histogram(256,0);
    std::cout<<point_list.front()<<point_list.back()<<" "<<point_list.size()<<std::endl;
    for(const auto &i : point_list )
        ++histogram[ Image_Factor3(image, i) ];
    return std::max_element( histogram.begin(), histogram.end() ) - histogram.begin();
}


double Mean_Line(std::list<Point> edge, int size, Mat image){
    double mean = 0;
    int i = 0;
    for(const auto &e:edge){
        if(i == size) break;
        i++;
        mean += int(image.at<uchar>( e ));
        for(const auto &it:shifts24)
            mean += int(image.at<uchar>( e+it ));
    }
    return mean/(i*25);
}

double Mean_Centre(std::list<Point> edge, Mat image){
    double mean = 0;
    int i = 0;
    unsigned long s1 = edge.size()/4, s2 = 3*edge.size()/4;
    for(const auto &e:edge){
        if(i == s1) continue;
        else if(i == s2) break;
        i++;
        mean += int(image.at<uchar>( e ));
        for(const auto &it:shifts24)
            mean += int(image.at<uchar>( e+it ));
    }
    return mean/(i*25);
}

int Median_Line(std::list<Point> edge, int size, const Mat &image){
    std::list<Point>Neighbours;
    int i = 0;
    for(const auto &e:edge){
        if(i == size) break;
        i++;
        Neighbours.push_back(e);
        //for(const auto &it:shifts24)
        //Neighbours.push_back(e+it);
    }
    return Median(Neighbours, image);
}

int Median_Line(std::list<Point> edge, const Mat &image){
    std::list<Point>Neighbours;
    int i = 0;
    unsigned long s1 = edge.size()/4, s2 = 3*edge.size()/4;
    for(const auto &e:edge){
        if(i == s1) continue;
        else if(i == s2) break;
        i++;
        Neighbours.push_back(e);
        //for(const auto &it:shifts24)
        //Neighbours.push_back(e+it);
    }
    return Median(Neighbours, image);
}

bool Next_Vertex ( Mat image, Mat_<Vec3b>& image_mask, Mat_<Vec3b>& binary_skeleton, std::multimap< Point, bool, ComparePixels >& vertices_pixels, std::multimap< Point, int, ComparePixels >& all_pixels, Point p,
                   const Point &prev, int& distance, float& sd, float& sd2, float& sd3, float& mean, bool& boundary, Point& min, Point& max){
    distance++;
    if (p.x == 1 or p.y == 1 or p.x == image.cols-2 or p.y == image.rows-2 ) boundary = true;
    if (distance>41) return false;
    int image_factor = Image_Factor(image, p);
    //mean = mean + image_factor;
    all_pixels.insert( std::make_pair( p, image_factor ) );
    binary_skeleton.at<Vec3b>(p)=blue;
    image_mask.at<Vec3b>(p)=white;
    if (image_factor<Image_Factor(image, min)) min = p;
    else if (image_factor>Image_Factor(image, max)) max = p;

    for (const auto &i : shifts8) {
        if(binary_skeleton.at<Vec3b>(p + i)==green and distance>2 and p + i !=prev){
            return true;
        }
    }
    for (int i = 0; i < shifts8.size(); i++) {
        bool valid = true;
        if(binary_skeleton.at<Vec3b>(p+shifts8[i])==red and p+shifts8[i]!=prev){
            Point nbr = p + shifts8[i];
            for (const auto &j : shifts8) {
                if(binary_skeleton.at<Vec3b>(nbr+ j)==green and distance<3) valid = false;
            }
            if (valid) Next_Vertex(image, image_mask, binary_skeleton, vertices_pixels, all_pixels, nbr, p, distance, sd, sd2, sd3, mean, boundary, min, max);
        }
    }
    if (sd==0){
        for ( auto temp : all_pixels ){
            mean = mean + temp.second;
        }
        mean = mean / distance;
        for ( auto temp : all_pixels ){
            sd = sd + (temp.second-mean)*(temp.second-mean);
        }
        sd = sqrt(sd/distance);

        mean = 0;
        for ( auto temp : all_pixels ){
            mean = mean + Image_Factor2(image, temp.first);
        }
        mean = mean / distance;
        for ( auto temp : all_pixels ){
            sd2 = sd2 + (Image_Factor2(image, temp.first)-mean)*(Image_Factor2(image, temp.first)-mean);
        }
        sd2 = sqrt(sd2/distance);

        mean = 0;
        for ( auto temp : all_pixels ){
            mean = mean + Image_Factor3(image, temp.first);
        }
        mean = mean / distance;
        for ( auto temp : all_pixels ){
            sd3 = sd3 + (Image_Factor3(image, temp.first)-mean)*(Image_Factor3(image, temp.first)-mean);
        }
        sd3 = sqrt(sd3/distance);
        sd3 = (sd3 - sd) * Image_Factor(image, min);
    }
    if (distance>38 or distance<10 or boundary or sd<20 or sd2<2000 or sd3<-1) {
        binary_skeleton.at<Vec3b>(p)=red;
        image_mask.at<Vec3b>(p)=black;
    }
    //else if (Image_Factor(image, max)-Image_Factor(image, min)<190) {
    //image_mask.at<Vec3b>(p)=blue;
    //}
    return true;
}



bool Saddle_Removal (const Mat &image, Mat_<bool>& mask, Mat_<Vec3b>& image_mask, Mat_<Vec3b>& binary_skeleton, std::multimap< Point, bool, ComparePixels >& vertices_pixels, std::multimap< Point, int, ComparePixels >& all_pixels){
    int distance;
    float sd, sd2, sd3;
    float mean;
    bool boundary;
    Point min;
    Point max;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    for ( auto v : vertices_pixels ){
        for (const auto &i : shifts8) {
            if ( binary_skeleton.at<Vec3b>( v.first + i)==red ) {
                distance = 0;
                all_pixels.clear();
                sd = 0;
                sd2 = 0;
                sd3 = 0;
                mean = 0;
                boundary = false;
                Point nbr = v.first + i;
                min = nbr;
                max = nbr;
                Next_Vertex(image, image_mask, binary_skeleton, vertices_pixels, all_pixels, nbr, v.first, distance, sd, sd2, sd3, mean, boundary, min, max);
                if (distance<40 and distance>10 and !boundary and sd>20 and sd2>2000 and sd3>-1) std::cout<<v.first<<" Dist: "<<distance<<" SD: "<<sd<<" SD2: "<<sd2<<" SD3: "<<sd3<<" min: "<<Image_Factor(image, min)<<" max: "<<Image_Factor(image, max)<<" Dif: "<<Image_Factor(image, max)-Image_Factor(image, min)<<std::endl;
            }
        }

    }
    //for ( auto v : all_pixels ) {
    //std::cout<<"pixel"<<v.first<<std::endl;
    //}
}
void Remove_Loops(const Mat &image, Mat_<Vec3b>& binary_skeleton, Mat_<Vec3b>& image_mask, Mat_<bool>& mask, std::multimap< Point, bool, ComparePixels >& vertices_pixels, String name) {
    bool check, found, loop = true;
    int count;
    std::list<Point> point_list, remove_list;
    Point p, prev, next_vertex, mid, nbr;
    int nbr_count;
    // cv::imwrite(  name + "_skeleton_original_straight_test2.png" , binary_skeleton );
    while(loop) {
        loop = false;
        for (auto it = vertices_pixels.begin(); it != vertices_pixels.end(); ++it) {
            for (int i = 0; i < shifts8.size(); i++) {
                p = it->first + shifts8[i];
                if ((it->first + shifts8[i]).x > -1 and (it->first + shifts8[i]).y > -1 and
                    (it->first + shifts8[i]).x < binary_skeleton.cols and
                    (it->first + shifts8[i]).y < binary_skeleton.rows and
                    (binary_skeleton.at<Vec3b>(it->first + shifts8[i]) == red or
                     binary_skeleton.at<Vec3b>(it->first + shifts8[i]) == green)) {
                    p = it->first + shifts8[i];
                    found = false;
                    prev = it->first;
                    next_vertex;
                    point_list.clear();
                    mid = p;
                    point_list.push_back(it->first);
                    point_list.push_back(p);
                    if (binary_skeleton.at<Vec3b>(p) == green) {
                        found = true;
                        next_vertex = p;
                    }
                    while (!found) {
                        nbr_count = 0;
                        if(true) std::cout << it->first << std::endl;
                        for (const auto &i1 : shifts8) {
                            nbr = p + i1;
                            if (binary_skeleton.at<Vec3b>(nbr) == red or binary_skeleton.at<Vec3b>(nbr) == green)
                                nbr_count++;

                            std::cout << it->first << p << nbr << prev << " " << binary_skeleton.at<Vec3b>(nbr)
                                      <<" is green = "<< (binary_skeleton.at<Vec3b>(nbr) == green) << " " << (nbr != prev) << std::endl;
                            if (binary_skeleton.at<Vec3b>(nbr) == green and nbr != prev) {
                                found = true;
                                prev = p;
                                next_vertex = nbr;
                                if(true) std::cout << " r" << it->first << next_vertex << std::endl;
                                break;
                            }
                        }
                        if (nbr_count == 1) {
                            found = true;
                            prev = p;
                            next_vertex = nbr;
                            point_list.push_back(next_vertex);
                            if(true) std::cout << " tr" << it->first << next_vertex << p << std::endl;
                            break;
                        }
                        if (!found) {
                            for (int i2 = 0; i2 < shifts8.size(); i2++) {
                                bool valid = true;
                                if (binary_skeleton.at<Vec3b>(p + shifts8[i2]) == red and p + shifts8[i2] != prev) {
                                    nbr = p + shifts8[i2];

                                    for (const auto &j : shifts8) {
                                        if ((binary_skeleton.at<Vec3b>(nbr + j) == green and nbr + j == prev) or
                                            nbr.x < 0 or nbr.y < 0 or nbr.x > binary_skeleton.cols - 1 or
                                            nbr.y > binary_skeleton.rows - 1)
                                            valid = false;
                                    }
                                    if (valid) {
                                        prev = p;
                                        p = nbr;
                                        point_list.push_back(p);
                                        break;
                                    }
                                }
                            }
                        }
                        if(true) std::cout <<"Finding Loop at: "<< it->first << p << prev << loop << std::endl;
                        /*if(it->first.x == 859 and it->first.y == 189){
                            cv::imwrite(  name + "_skeleton_original_straight_test2.png" , binary_skeleton );
                            std::cin>>count;
                        }*/
                    }
                    if (next_vertex == it->first) {
                        if(true) std::cout << "Loop found at: " << it->first << std::endl;
                        loop = true;
                        for (const auto &t : point_list) {
                            std::cout<<"Removed: "<<t<<std::endl;
                            binary_skeleton.at<Vec3b>(t) = black;
                            mask.at<bool>(t) = false;
                            image_mask.at<Vec3b>(t) = image.at<Vec3b>(t);

                        }
                        //0point_list.push_back(point_list.front());
                        //removed_structre.push_back(point_list);
                        // std::cin>>loop;
                        cv::imwrite(  name + "_skeleton_original_straight_test.png" , image_mask );
                        remove_list.push_back(next_vertex);
                        //std::cin>>count;
                        continue;
                    }
                }
            }
        }

        // std::cout<<"done"<<std::endl;
        do {
            // std::cout<<"done"<<std::endl;
            check = false;
            for (int row = 0; row < binary_skeleton.rows - 1; row++)
                for (int col = 0; col < binary_skeleton.cols - 1; col++) {
                    if (binary_skeleton.at<Vec3b>(row, col) == red or binary_skeleton.at<Vec3b>(row, col) == green) {
                        p = Point(col, row);
                        Remove_Pixel(p, binary_skeleton, image_mask, mask, image, check);
                    }
                }

        } while (check);
        for (const auto &v : remove_list) {
            vertices_pixels.erase(v);
        }
        for (auto it = vertices_pixels.begin(); it != vertices_pixels.end(); ++it) {
            count = 0;
            for (const auto &i : shifts8) {
                if (binary_skeleton.at<Vec3b>(it->first + i) == red or
                    binary_skeleton.at<Vec3b>(it->first + i) == green) {
                    count++;
                }
            }
            if (count == 2) {
                bool t = true;
                for (const auto &i : shifts8) {
                    if (binary_skeleton.at<Vec3b>(it->first + i) == red)
                        for (const auto &j : shifts8)
                            for (const auto &k : shifts8) {
                                if ((it->first + i + j).x >= 0 and (it->first + i + j).y >= 0 and
                                    (it->first + i + j).x < binary_skeleton.cols and
                                    (it->first + i + j).y < binary_skeleton.rows) {
                                    //std::cout << it->first << (it->first + i + j) << (it->first + k) << (binary_skeleton.at<Vec3b>(it->first + i + j) == red) << (it->first + i + j == it->first + k) << std::endl;
                                    if (binary_skeleton.at<Vec3b>(it->first + i + j) == red and
                                        (it->first + i + j == it->first + k)) {
                                        t = false;
                                        if(true) std::cout << it->first << (it->first + i + j) << (it->first + k) << std::endl;
                                    }
                                }
                            }
                }
                if (!t) {
                    if(true) std::cout << "Removed2: " << it->first << std::endl;
                    binary_skeleton.at<Vec3b>(it->first) = black;
                    image_mask.at<Vec3b>(it->first) = image.at<Vec3b>(it->first);
                    vertices_pixels.erase(it);
                    continue;
                }
                binary_skeleton.at<Vec3b>(it->first) = red;
                image_mask.at<Vec3b>(it->first) = black;
                vertices_pixels.erase(it);
                continue;
            }
        }
        if(false) cv::imwrite(  name + "_skeleton_original_straight_test2.png" , image_mask );

    }
    // std::cin>>count;

}

double polygon_area(std::list<Point> poly)
{

    double area = 0.0;
    unsigned long j;

    for (int i = 0; i < poly.size(); ++i)
    {
        j = (i + 1)%poly.size();
        auto it = poly.begin();
        std::advance(it, i);
        auto it2 = poly.begin();
        std::advance(it2, j);
        area += 0.5 * ((*it).x*(*it2).y - (*it2).x*(*it).y);
    }

    return (abs(area));
}

bool Disconnect_Small_Components_Outer2(std::list<std::list<Point>>& face_list, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::list<Point> loop_list){
    bool removed, disconnected_found;
    int face1_count, face2_count, face3_count;
    Point loop_vertex;
    do{
        removed = false;
        for (auto face1=face_list.begin(); face1!=face_list.end(); ++face1){
            disconnected_found = false;
            for(const auto &point:*face1){
                for(auto loop_point : loop_list){
                    if(point == loop_point) {
                        disconnected_found = true;
                        loop_vertex = point;
                        break;
                    }
                }
                if(disconnected_found) break;
            }
            if (disconnected_found){
                if(debug) std::cout<<loop_vertex<<std::endl;
                bool second_face_found;
                for (auto face2=face_list.begin(); face2!=face_list.end(); ++face2){
                    second_face_found = false;
                    if(*face1 == *face2) continue;
                    for(const auto &point:*face2){
                        if(point == loop_vertex) {
                            second_face_found = true;
                            break;
                        }
                    }
                    if(second_face_found){

                        bool third_face_found;
                        for (auto face3=face_list.begin(); face3!=face_list.end(); ++face3){
                            third_face_found = false;
                            if(*face1 == *face3 or *face2 == *face3) continue;
                            for(const auto &point:*face3){
                                if(point == loop_vertex) {
                                    third_face_found = true;
                                    break;
                                }
                            }
                            if(third_face_found){
                                if(debug) {
                                    std::cout << loop_vertex << std::endl;
                                    std::cout << "face1: ";
                                    for (auto point:*face1)
                                        std::cout << point;
                                    std::cout << std::endl;
                                    std::cout << "face2: ";
                                    for (auto point:*face2)
                                        std::cout << point;
                                    std::cout << std::endl;
                                    std::cout << "face3: ";
                                    for (auto point:*face3)
                                        std::cout << point;
                                    std::cout << std::endl;
                                }
                                double area1 = polygon_area((*face1)), area2 = polygon_area((*face2)), area3 = polygon_area((*face3));
                                if(debug)std::cout<<"Area1: "<<area1<<" Area2: "<<area2<<" Area3: "<<area3<<std::endl;
                                if(area1>area2 and area1>area3)
                                {
                                    if(debug)std::cout<<area1<<"face1"<<std::endl;
                                    int count = 0;
                                    for(const auto &point:*face1){
                                        if ( edge_structure.find(point) != edge_structure.end()) count++;
                                    }
                                    if(count<20) face_list.erase(face1);
                                    else break;
                                }
                                /*else if ((*face1).size()==(*face2).size() and ((*face1).size()<(*face3).size() or (*face1).size()==(*face3).size())){
                                    int max1 = 0, max2 = 0, temp;
                                    int min1 = 1000, min2 = 1000;
                                    for(auto point1: *face1)
                                        for(auto point2: *face1){
                                        temp = (int)sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2));
                                        max1 = (max1>temp?max1:temp);
                                        min1 = (min1<temp?min1:temp);
                                    }
                                    for(auto point1: *face2)
                                        for(auto point2: *face2){
                                            max2 = (max1>(int)sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2))?max1:(int)sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)));
                                    }
                                    if(max1>max2) face_list.erase(face1);
                                    else face_list.erase(face2);
                                }*/
                                else if (area2>area3) {
                                    if(debug)std::cout<<area2<<"face2"<<std::endl;
                                    int count = 0;
                                    for(const auto &point:*face2){
                                        if ( edge_structure.find(point) != edge_structure.end()) count++;
                                    }
                                    if(count<20)
                                    face_list.erase(face2);
                                    else break;
                                }
                                else {
                                    if(debug)std::cout<<area2<<"face3"<<std::endl;
                                    int count = 0;
                                    for(const auto &point:*face3){
                                        if ( edge_structure.find(point) != edge_structure.end()) count++;
                                    }
                                    if(count<20) face_list.erase(face3);
                                    else break;
                                }
                                removed = true;
                                break;

                            }
                        }
                        if (removed) break;
                    }
                }
                if(removed) break;
            }
        }
    }while(removed);
}

bool Disconnect_Small_Components_Outer(std::list<std::list<Point>>& face_list, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure){
    int count, count1, count2;
    Point p1, p2;
    bool removed = false;

    int face1_count, face2_count, same_count, non_vertex_count_common, non_vertex_count1, non_vertex_count2;
    std::list<Point> common_points;
    do{
        removed = false;
        for (auto face1=face_list.begin(); face1!=face_list.end(); ++face1){
            face1_count = 0;
            non_vertex_count1 = 0;
            for(const auto &point:*face1)
                if ( edge_structure.find(point) != edge_structure.end()) face1_count++;
                else non_vertex_count1++;
            for (auto face2=face_list.begin(); face2!=face_list.end(); ++face2){
                if(*face1 == *face2) continue;
                face2_count = 0;
                same_count = 0;
                non_vertex_count2 = 0;
                non_vertex_count_common = 0;
                common_points.clear();
                for(const auto &point:*face2)
                {
                    if ( edge_structure.find(point) != edge_structure.end()) {
                        face2_count++;
                        for(const auto &point_same:*face1)
                            if ( point_same == point) {
                            same_count++;
                            common_points.push_back(point);
                        }
                    }
                    else {
                        for (const auto &point_same:*face1)
                            if (point_same == point) non_vertex_count_common++;
                        non_vertex_count2++;
                    }
                }
                int non_vertex = (non_vertex_count1>non_vertex_count2?non_vertex_count1:non_vertex_count2);
                int face_count = (face1_count>face2_count?face1_count:face2_count);
                if(((face_count>=3 and abs(same_count-face_count)<=2 and abs(face1_count-face2_count)<=2) or ((face_count<=3) and same_count==face1_count and face1_count==face2_count)) and  (non_vertex_count_common>2)){
                    std::cout<<"face1: ";
                    for(auto point:*face1)
                        std::cout<<point;
                    std::cout<<std::endl;
                    std::cout<<"face2: ";
                    for(auto point:*face2)
                        std::cout<<point;
                    std::cout<<std::endl;
                    int face3_count = 0, face3_count_common = 0;
                    for (auto face3=face_list.begin(); face3!=face_list.end(); ++face3){
                        face3_count = 0;
                        face3_count_common = 0;
                        if(*face1 == *face3 or *face2 == *face3) continue;
                        if((*face1).size()<(*face3).size())continue;
                        for(const auto &point:*face3){
                            for(auto v : common_points){
                                if(point == v)face3_count_common++;
                            }
                            if ( edge_structure.find(point) != edge_structure.end()) face3_count++;
                        }
                        if(face3_count>face1_count) continue;
                        if(face3_count_common>1) {
                            std::cout<<"face3: ";
                            for(auto point:*face3)
                                std::cout<<point;
                            std::cout<<std::endl;
                            break;
                        }
                    }
                    if(face3_count_common<2) continue;
                    if((*face1).size()>(*face2).size())
                        face_list.erase(face1);

                    else face_list.erase((face2));
                    /*;*/
                    removed = true;
                    break;
                }
            }if(removed) break;
        }
    }while(removed);
}

bool Remove_Edge_Structure(Point p1, Point p2, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure){
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result = edge_structure.equal_range(p2), result2;
    Point connect1, connect2;
    std::list<Point> temp, temp2;
    bool added = false;
    for (auto it = result.first; it != result.second; it++)
        for (auto it2 = result.first; it2 != result.second; it2++)
    {

        if (it->second.first != it2->second.first and it->second.first != p1 and it2->second.first != p1)
        {
            /*std::cout<<"Point: "<<it->first<<" Connected to: "<<it->second.first << it->second.second.second.first<<std::endl;
            for(auto point:it->second.second.first)
            {
                std::cout<<point;
            }
            std::cout<<std::endl;

            std::cout<<"Point: "<<it2->first<<" Connected to: "<<it2->second.first<<std::endl;
            for(auto point:it2->second.second.first)
            {
                std::cout<<point;
            }
            std::cout<<std::endl;*/

            temp.clear();
            temp2.clear();
            temp = it->second.second.first;
            temp.reverse();
            temp2 = it2->second.second.first;
            temp2.pop_front();
            temp.insert(temp.end(), temp2.begin(), temp2.end());
            /*std::cout<<"Replace Point: "<<temp.front()<<" Connected to: "<<temp.back()<<std::endl;
            for(auto point:temp)
            {
                std::cout<<point;
            }
            std::cout<<std::endl;
            std::cout<<std::endl;*/
            edge_structure.insert(std::make_pair(temp.front(), std::make_pair(temp.back(), std::make_pair(temp, std::make_pair(it->second.second.second.second.first, std::make_pair(it2->second.second.second.second.first, std::make_pair(it->second.second.second.second.second.first+it2->second.second.second.second.second.first, std::make_pair(it->second.second.second.second.second.second.first+it2->second.second.second.second.second.second.first, it->second.second.second.second.second.second.second))))))));
            added = true;
            break;
        }
        if(added) break;
    }
    for (auto iter = edge_structure.begin(); iter != edge_structure.end();)
    {
        // you have to do this because iterators are invalidated
        auto erase_iter = iter++;

        // removes
        if (erase_iter->first == p2)
            edge_structure.erase(erase_iter);
    }

    added = false;
    result = edge_structure.equal_range(p1);
    if (edge_structure.find(p1) != edge_structure.end())
    for (auto it = result.first; it != result.second; it++)
        for (auto it2 = result.first; it2 != result.second; it2++)
        {
            if (it->second.first != it2->second.first and it->second.first != p2 and it2->second.first != p2)
            {
                /*std::cout<<"Point: "<<it->first<<" Connected to: "<<it->second.first<<std::endl;
                for(auto point:it->second.second.first)
                {
                    std::cout<<point;
                }
                std::cout<<std::endl;

                std::cout<<"Point: "<<it2->first<<" Connected to: "<<it2->second.first<<std::endl;
                for(auto point:it2->second.second.first)
                {
                    std::cout<<point;
                }
                std::cout<<std::endl;*/

                temp.clear();
                temp2.clear();
                temp = it->second.second.first;
                temp.reverse();
                temp2 = it2->second.second.first;
                temp2.pop_front();
                temp.insert(temp.end(), temp2.begin(), temp2.end());
                /*std::cout<<"Replace Point: "<<temp.front()<<" Connected to: "<<temp.back()<<std::endl;
                for(auto point:temp)
                {
                    std::cout<<point;
                }
                std::cout<<std::endl;
                std::cout<<std::endl;*/
                edge_structure.insert(std::make_pair(temp.front(), std::make_pair(temp.back(), std::make_pair(temp, std::make_pair(it->second.second.second.second.first, std::make_pair(it2->second.second.second.second.first, std::make_pair(it->second.second.second.second.second.first+it2->second.second.second.second.second.first, std::make_pair(it->second.second.second.second.second.second.first+it2->second.second.second.second.second.second.first, it->second.second.second.second.second.second.second))))))));
                added = true;
                break;
            }
            if(added) break;
        }

    for (auto iter = edge_structure.begin(); iter != edge_structure.end();)
    {
        if(debug)std::cout<<"loop inner: "<<std::endl;
        // you have to do this because iterators are invalidated
        auto erase_iter = iter++;

        // removes
        if (erase_iter->first == p1)
        {
            edge_structure.erase(erase_iter);
        }
    }

}
bool Disconnect_Small_Components(std::list<std::list<Point>>& face_list, std::multimap< Point, Point, ComparePixels >& points_decided, std::multimap< Point, bool, ComparePixels >& vertices_pixels, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::list<Point>& loop_list){
    std::list<Point> temp;
    std::list<Point> remove;
    Point error_at, visual1, visual2, remove1, remove2;
    int count;
    bool error_found, to_remove, removed, is_vertex;
    /*for (auto v:points_decided){
        std::cout<<v.first<<v.second<<std::endl;
    }*/
    for(auto& face : face_list)
    {
        is_vertex = true;
        to_remove = false;
        removed = false;
        remove.clear();
        face.pop_back();
        temp  = face;
        error_found = false;
        error_at = Point(-1, -1);
        if(debug) {
            for (auto point : face) {
                std::cout << point;
            }
            std::cout << std::endl;
        }
        for(const auto &point_face:face)
        {
            count = 0;
            for(const auto &point_temp:temp)
                if (point_temp == point_face) count++;
            if(count==2 or count==3){
                if (error_at == point_face){
                    error_found = false;
                    error_at = Point(-1, -1);
                    remove.push_back(point_face);
                    if(debug)std::cout<<"Repeat end at:"<<point_face<<std::endl;
                }
                else if (error_at == Point(-1, -1)) {
                    // std::cout << "Error at: " << point_face << std::endl;
                    error_found = true;
                    error_at = point_face;
                    visual1 = point_face;
                    remove1 = point_face;
                    to_remove = true;
                    if(debug)std::cout<<"Repeat start at:"<<point_face<<std::endl;
                }
                else if ( edge_structure.find(point_face) != edge_structure.end()) {
                    visual2 = point_face;
                    remove2 = point_face;
                    if(debug)std::cout<<"Repeat continue at:"<<point_face<<std::endl;
                }
                else remove2 = point_face;
                if(removed) {
                    auto result = points_decided.equal_range(remove1);
                    for (auto it = result.first; it != result.second; it++)
                        if(it->second == remove2) {
                            points_decided.erase(it);
                            break;
                        }
                    result = points_decided.equal_range(remove2);
                    for (auto it = result.first; it != result.second; it++)
                        if(it->second == remove1) {
                            points_decided.erase(it);
                            break;
                        }
                    /*if ( points_decided.find(remove1)->second == remove2) points_decided.erase(remove1);
                    if ( points_decided.find(remove2)->second == remove1) points_decided.erase(remove2);*/
                    if ( vertices_pixels.find(remove1) != vertices_pixels.end()) vertices_pixels.erase(remove1);
                    if ( vertices_pixels.find(remove2) != vertices_pixels.end()) vertices_pixels.erase(remove2);
                    Remove_Edge_Structure(remove1, remove2, edge_structure);
                    if(debug)std::cout<<"Points removed: "<<remove1<<remove2<<std::endl;
                    remove1 = remove2;
                }
                else removed = true;
                // std::cout<<remove1<<remove2<<std::endl;
            }
            if (error_found)
            {
                if(debug)std::cout<<"Removed: "<<point_face<<std::endl;
                remove.push_back(point_face);
            }

        }
        if(to_remove){

            if(debug)std::cout<<"Removing: "<<visual1<<visual2<<std::endl;
            auto result = points_decided.equal_range(visual1);
            for (auto it = result.first; it != result.second; it++)
                if(it->second == visual2) {
                    points_decided.erase(it);
                    break;
                }
            result = points_decided.equal_range(visual2);
            for (auto it = result.first; it != result.second; it++)
                if(it->second == visual1) {
                    points_decided.erase(it);
                    break;
                }
            /*if ( points_decided.find(visual1)->second == visual2) points_decided.erase(visual1);
            if ( points_decided.find(visual2)->second == visual1) points_decided.erase(visual2);*/
            if ( vertices_pixels.find(visual1) != vertices_pixels.end()) vertices_pixels.erase(visual1);
            if ( vertices_pixels.find(visual2) != vertices_pixels.end()) vertices_pixels.erase(visual2);
            Remove_Edge_Structure(visual1, visual2, edge_structure);
            count = 0;
            bool visual1_removed = false;
            for (const auto &point : remove)
                if(edge_structure.find(point) != edge_structure.end()) count++;
            if(count>10) is_vertex = false;
            for (const auto &point : remove)
                for (auto it=face.begin(); it!=face.end(); ++it){

                    if(point == *it) {
                        if(point == visual1 and visual1_removed) continue;
                        else if (point == visual1 and !visual1_removed) visual1_removed = true;
                        face.erase(it);
                        if ( edge_structure.find(point) != edge_structure.end() and is_vertex) {
                            loop_list.push_back(point);
                            if(debug)std::cout<<point;
                            // std::cin>>count;
                            is_vertex = false;
                        }
                        break;
                    }
            }
            if(debug) {
                for (auto point : remove) {
                    std::cout << point;
                }
                std::cout << std::endl;
            }
        }
    }

    if(debug) {
        for (auto v:points_decided) {
            std::cout << v.first << v.second << std::endl;
        }
        std::cout << "After loop found at: " << std::endl;
        for (auto v:loop_list) {
            std::cout << v << std::endl;
        }
    }
    // std::cin>>count;
    do {
        to_remove = false;
        for (auto face=face_list.begin(); face!=face_list.end(); ++face) {
            count = 0;
            for (const auto &point_face:*face) {
                if (edge_structure.find(point_face) != edge_structure.end()) count++;
            }
            if (count <= 1) {
                for (const auto &point_face1:*face) {
                    for (const auto &point_face2:*face){
                        auto result = points_decided.equal_range(point_face1);
                        for (auto it = result.first; it != result.second; it++)
                            if(it->second == point_face2) {
                                points_decided.erase(it);
                                break;
                            }
                        result = points_decided.equal_range(point_face2);
                        for (auto it = result.first; it != result.second; it++)
                            if(it->second == point_face1) {
                                points_decided.erase(it);
                                break;
                            }
                    }
                }
                face_list.erase(face);
                to_remove = true;
            }
            if(to_remove) break;
        }
    }while(to_remove);

    return true;
}

std::list<Point> Find_Faces(Mat_<Vec3b>& binary_skeleton, std::list<Point> edge, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > reverse_edge_structure, std::list<std::list<Point>> face_list, bool& valid){

    //std::cout<<edge.back()<<std::endl;
    bool completed = false;
    Point start = edge.front();
    std::list<Point> current_list = edge;
    Point next, nbr, next_nbr, next1;
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result = edge_structure.equal_range(start);
    std::pair<MMAPIterator, MMAPIterator> result2;
    bool found = false;
    int t, once = 0, green_nbrs;
    for (auto it = result.first; it != result.second; it++)
        if(it->second.second.first == current_list)
        {
            found = true;
            next = it->second.second.first.back();
            nbr = it->second.second.second.second.first;
        }
    if (next == start) completed = true;
    else {
        int temp = 0;
        next_nbr = nbr; // initialization
        for (const auto &i : shifts8)
        {
            if (binary_skeleton.at<Vec3b>(next+i) == red or binary_skeleton.at<Vec3b>(next+i) == green)
                if (next+i == nbr) {
                    if(temp>0)break;
            }
                else {
                temp++;
                next_nbr = next+i;
            }
        }
    }
    if(debug)std::cout<<"Finding: "<<start<<" Next: "<<next<<" nbr: "<<nbr<<" next nbr"<<next_nbr<<std::endl;
    int present_counter = 0;
    while (!completed){
        if(edge.front().x == 285 and edge.front().y == 1493)std::cout<<"loop face "<<edge.front()<<edge.back()<<std::endl;
        start = next;
        found = false;
        result = edge_structure.equal_range(start);
        for (auto it = result.first; it != result.second; it++)
            if(it->second.second.second.first == next_nbr)
            {
                found = true;
                next = it->second.second.first.back();
                nbr = it->second.second.second.second.first;
                current_list.insert(current_list.end(), it->second.second.first.begin(), it->second.second.first.end());
            }
        if(!found){
            result = reverse_edge_structure.equal_range(start);
            for (auto it = result.first; it != result.second; it++)
                if(it->second.second.second.first == next_nbr)
                {
                    found = true;
                    next = it->second.second.first.back();
                    nbr = it->second.second.second.second.first;
                    current_list.insert(current_list.end(), it->second.second.first.begin(), it->second.second.first.end());
                }
        }
        if(binary_skeleton.at<Vec3b>(next) != green) {
            valid = false;
            return edge;
        }
        if (next == edge.front()) completed = true;
        else {
            if(next == nbr)
                for (const auto &i : shifts8)
                    for (const auto &j : shifts8) {
                        if (binary_skeleton.at<Vec3b>(next + j) == red and (next + j != start + j)) {
                            next_nbr = next + j;
                        }
                    }

            else{
                //next_nbr = nbr; // initialization
                int temp = 0;
                for (const auto &i2 : shifts8) {
                    if (binary_skeleton.at<Vec3b>(next+i2) == red or binary_skeleton.at<Vec3b>(next+i2) == green)
                        if (next+i2 == nbr) {
                            if(temp>0)break;
                        }
                        else {
                            temp++;
                            next_nbr = next+i2;
                        }
                }
            }
        }
        while(once==1) {
            next1 = start;
            once += 10;
        }
        once++;

        // std::cout<<"Finding: "<<start<<" Next: "<<next<<" nbr: "<<nbr<<" next nbr"<<next_nbr<<std::endl;
        // if(edge.front().x == 2539 and edge.front().y == 1464) std::cin>>t;
        if(!found)std::cout<<"\t\t NOT FOUND"<<next<<std::endl;
    }
    current_list.unique();
    /*for (const auto &v : current_list)
    {
        std::cout<<" "<<v;
    }
    std::cout<<std::endl;*/
    // std::cout<<"Start: "<<edge.front()<<" Next: "<<next1<<" end: "<<edge.back()<<std::endl;
    if (edge.front().x == 1 or edge.front().x == binary_skeleton.cols-2 or edge.front().y == 1 or edge.front().y == binary_skeleton.rows-2)
        if(edge.back().x == 1 or edge.back().x == binary_skeleton.cols-2 or edge.back().y == 1 or edge.back().y == binary_skeleton.rows-2)
            if(next.x == 1 or next.x == binary_skeleton.cols-2 or next.y == 1 or next.y == binary_skeleton.rows-2)
                if(start.x == 1 or start.x == binary_skeleton.cols-2 or start.y == 1 or start.y == binary_skeleton.rows-2)
                if(current_list.size()>20)
                {
                    valid = false;
                    if(debug) std::cout<<"Start2: "<<edge.front()<<" Next: "<<next<<" end: "<<next<<std::endl;
                    return edge;
                }

    bool p1, p2, p3;
    int test_vertces, current_vertices=0;
    for(auto point:current_list){
        if (edge_structure.find(point) != edge_structure.end()) current_vertices++;
    }
    if(debug)std::cout<<"Start: "<<edge.front()<<" Next: "<<next1<<" end: "<<edge.back()<<"Size: "<<current_vertices<<std::endl;

    for (auto face : face_list){
        test_vertces = 0;
        present_counter = 0;
        p1 = p2 = p3 = true;
        for (const auto &point : face)
        {
            if (point == edge.front() and p1){
                present_counter++;
                p1 = false;
            }
            else if (point == edge.back() and p2){
                present_counter++;
                p2 = false;
            }
            else if (point == next1 and p3){
                present_counter++;
                p3 = false;
            }
            if (edge_structure.find(point) != edge_structure.end()) test_vertces++;
        }
        // std::cout<<present_counter<<" While Start: "<<edge.front()<<" Next: "<<next1<<" end: "<<edge.back()<<face.size()<<current_list.size()<<std::endl;
        if (present_counter >= 3 and (face.size() == current_list.size() or current_list.size()>50)){
            valid = false;
            if(debug) {
                if (edge.front().x == 1180 and edge.front().y == 1206)
                    for (const auto &point : face) {
                        std::cout << point;
                    }
                std::cout << std::endl;
                std::cout << "Start: " << edge.front() << " Next: " << next1 << " end: " << edge.back()
                              << std::endl;
            }
            return edge;
        }
    }
    return current_list;

}


bool DouglasPeucker(Mat_<Vec3b> &image_mask, const Point x1, Point x2, std::list<Point> point_list,
                    std::multimap<Point, Point, ComparePixels>& points_decided, int epsilon, std::list<Point>& edge, int& breaks) {
    float dmax = 0;
    Point max;
    breaks++;
    std::list<Point> point_list1;
    std::list<Point> point_list2;
    point_list1.clear();
    point_list2.clear();

    float a, b, c;
    a = x2.y - x1.y;
    b = x1.x - x2.x;
    c = (x1.y - x2.y)*x1.x + (x2.x - x1.x)*x1.y;

    float d;
    for (auto v : point_list)
    {
        d = abs(a*v.x + b*v.y + c)/sqrt((a*a)+(b*b));
        if (d>dmax){
            dmax = d;
            max = v;
        }
    }

    bool list_choice = true;
    for (const auto &v : point_list)
    {
        if ( v == max) {
            list_choice = false;
            point_list1.push_back (v);
            point_list2.push_back (v);
        }
        else
        {
            if (list_choice) point_list1.push_back (v);
            else point_list2.push_back (v);
        }
    }
    //std::cout<<"before"<<std::endl;
    if (dmax>epsilon){
        DouglasPeucker(image_mask, x1, max, point_list1,
                       points_decided, epsilon, edge, breaks);
        DouglasPeucker(image_mask, max, x2, point_list2,
                       points_decided, epsilon, edge, breaks);
    }
    else {
        // if(x1.x ==48)std::cout<<"\t\t"<<x1<<" "<<x2<<std::endl;
        points_decided.insert( std::make_pair( x1, x2 ) );
        bool f = false;
        //for (auto v : edge)
        //{
          //  if (v == x1) f = true;
        //}
        //if(!f)
                edge.push_back (x1);
        edge.push_back (x2);
    }

}


bool Vertex_Pairs (Mat image, Mat_<Vec3b>& binary_skeleton, Mat_<Vec3b>& image_mask, Mat_<Vec3b>& image_straight, MyMesh& mesh, std::multimap< Point, bool, ComparePixels >& vertices_pixels, std::multimap< Point, std::pair<Point, std::pair<Point, Point>>, ComparePixels >& vertex_pairs, std::multimap< Point, std::pair<Point, std::pair<Point, Point>>, ComparePixels >& reverse_vertex_pairs, String name, std::list<Point> loop_list, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added) {
    std::multimap< Point, Point, ComparePixels > points_decided;
    bool present, found, present_mid;
    int border_counter;
    Point p, m, prev, mid, next_vertex, nbr, last;
    std::list<Point> point_list, point_list1, point_list2;
    std::list<std::list<Point>> edges;
    std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > reverse_edge_structure;
    int temp;
    if(debug)
    for ( auto v : vertices_pixels ) {
        std::cout << "loop " << v.first << " " << std::endl;
    }
    //std::cin>>temp;
    for ( auto v : vertices_pixels ){
        //std::cout<<"loop1 "<<v.first<<" "<<std::endl;
        for (int i = 0; i < shifts8.size(); i++) {
            p = v.first + shifts8[i];
            if(v.first.x == 332 or v.first.y == 1199 and debug) std::cout<<"loop startig at"<<v.first<<" "<<p<<std::endl;
            //std::cout<<"\t loop "<<v.first<<" "<<v.first+shifts8[i]<<" "<<binary_skeleton.rows<<" "<<binary_skeleton.cols<<std::endl;
            if ( ( v.first + shifts8[i] ).x > -1 and ( v.first + shifts8[i] ).y > -1 and ( v.first + shifts8[i] ).x < binary_skeleton.cols and ( v.first + shifts8[i] ).y < binary_skeleton.rows and (binary_skeleton.at<Vec3b>( v.first + shifts8[i] )==red or binary_skeleton.at<Vec3b>( v.first + shifts8[i] )==green) ) {
                p = v.first + shifts8[i];
                found = false;
                prev = v.first;
                next_vertex;
                point_list.clear();
                present = false;
                mid = p;
                point_list.push_back (v.first);
                if(binary_skeleton.at<Vec3b>(p) == green) {
                    found = true;
                    next_vertex = p;
                }
                if(v.first.x == 332 or v.first.y == 1199 and debug) std::cout<<"\t loop startig at"<<v.first<<" "<<p<<std::endl;
                while(!found) {
                    //std::cout<<"loop3 "<<v.firFst<<" "<<std::endl;
                    if(v.first.x == 332 or v.first.y == 1199 and debug) std::cout<<"\t\t loop"<<v.first<<" "<<p<<std::endl;
                    border_counter = 0;

                    for (int i1 = 0; i1 < shifts8.size(); i1++) {

                        nbr = p + shifts8[i1];
                        if (binary_skeleton.at<Vec3b>(nbr) == red and nbr != prev)
                            border_counter++;
                        if ((nbr.x == 0 or nbr.y == 0 or nbr.x == binary_skeleton.cols - 1 or nbr.y == binary_skeleton.rows - 1) and binary_skeleton.at<Vec3b>(nbr)==red) {
                            temp = 0;
                            for (const auto &j : shifts8)
                                if(binary_skeleton.at<Vec3b>(nbr+ j)==green or binary_skeleton.at<Vec3b>(nbr+ j)==red)
                                    temp++;
                            if (temp==1)
                            {
                                found = true;
                                prev = p;
                                next_vertex = nbr;
                            }
                        }
                        if (binary_skeleton.at<Vec3b>(nbr) == green and nbr != prev) {
                            if(v.first.x == 332 or v.first.y == 1199 and debug) std::cout<<"green found at p:"<<p<<" nbr:"<<nbr<<" prev"<<prev<<" "<<binary_skeleton.at<Vec3b>(nbr)<<std::endl;
                            found = true;
                            prev = p;
                            next_vertex = nbr;
                            break;
                        }
                    }
                    if(border_counter==0 and !found){
                        if(v.first.x == 332 or v.first.y == 1199 and debug) std::cout<<"border: p:"<<p<<" prev"<<prev<<" "<<border_counter<<std::endl;
                        found = true;
                        next_vertex = p;
                    }
                    if(!found)
                        for (int i2 = 0; i2 < shifts8.size(); i2++) {
                            bool valid = true;
                            if (binary_skeleton.at<Vec3b>(p + shifts8[i2]) == red and p + shifts8[i2] != prev) {
                                nbr = p + shifts8[i2];

                                for (const auto &j : shifts8) {
                                    if((binary_skeleton.at<Vec3b>(nbr+ j)==green and nbr+ j == prev) or nbr.x<0 or nbr.y<0 or nbr.x>binary_skeleton.cols-1 or nbr.y>binary_skeleton.rows-1) valid = false;
                                }if (valid){
                                    if(v.first.x == 332 or v.first.y == 1199 and debug) std::cout<<" p:"<<p<<" nbr:"<<nbr<<" prev"<<prev<<" "<<binary_skeleton.at<Vec3b>(nbr)<<std::endl;
                                    // if(v.first.x == 309 and v.first.y == 164 and nbr.x>310)std::cout<<"\t\t\t"<<nbr<<" "<<binary_skeleton.at<Vec3b>(nbr)<<" "<<binary_skeleton.rows<<" "<<binary_skeleton.cols<<std::endl;
                                    prev = p;
                                    p = nbr;
                                    point_list.push_back (p);
                                    break;

                                }
                            }
                        }
                }
                //if(v.first.x == 2568 and v.first.y == 828) std::cout<<"\t loop ttt"<<v.first<<" "<<next_vertex<<" "<<present<<std::endl;
                point_list.push_back (next_vertex);
                // std::cout<<v.first<<" present :"<<next_vertex<<mid<<std::endl;
                //vertex_pairs.insert( std::make_pair( v.first, std::make_pair(next_vertex, mid) ) );
                for (auto j : edge_structure){
                    if (j.second.first == v.first and j.first == next_vertex and (j.second.second.second.first == mid or j.second.second.second.first == prev)) {
                        present = true;
                        if((v.first.x == 1 or v.first.y == 60) and debug) std::cout<<v.first<<" present inside:"<<point_list.size()<<std::endl;
                    }
                }
                if(!present)
                {
                    if(next_vertex == v.first and false) { // this will add loops
                        /*
                        if(v.first.x == 1 or v.first.y == 60) std::cout<<v.first<<" size:"<<point_list.size()<<std::endl;
                        temp = 0;
                        point_list1.clear();
                        point_list2.clear();
                        present_mid = false;
                        for (auto middle : point_list)
                        {
                            temp++;
                            if (temp < point_list.size()/2)
                                point_list1.push_back (middle);
                            else if (temp == point_list.size()/2)
                            {
                                point_list1.push_back (middle);
                                for (auto j : vertex_pairs){
                                    if (j.second.first == middle and j.first == v.first)
                                        present_mid = true;
                                    else
                                        for (const auto &ii : shifts8)
                                            if (j.second.first == (middle+ ii) and j.first == v.first) present_mid = true;

                                }
                                if(!present_mid)
                                {
                                    vertex_pairs.insert( std::make_pair( v.first, std::make_pair(middle, std::make_pair(mid, prev)) ) );
                                    reverse_vertex_pairs.insert( std::make_pair( middle, std::make_pair(v.first, std::make_pair(prev, mid)) ) );
                                    std::list<Point> edge;
                                    edge.clear();
                                    int breaks;
                                    DouglasPeucker(image_mask, v.first, middle, point_list1, points_decided, 5, edge, breaks);
                                    edge.unique();
                                    edge_structure.insert( std::make_pair( v.first, std::make_pair(middle, std::make_pair(edge, std::make_pair(mid, std::make_pair(prev, std::make_pair(0, std::make_pair((int)point_list.size(), Standard_Deviation(point_list, image))))))) ) );
                                    edges.push_back(edge);
                                    edge.reverse();
                                    // reverse_edge_structure.insert( std::make_pair( middle, std::make_pair(v.first, std::make_pair(edge, std::make_pair(prev, std::make_pair(mid, std::make_pair(0, std::make_pair((int)point_list.size(), Standard_Deviation(point_list, image))))))) ) );
                                }
                                point_list2.push_back (middle);
                                m = middle;
                            }
                            else point_list2.push_back (middle);
                        }
                        if(!present_mid) {
                            vertex_pairs.insert(std::make_pair(m, std::make_pair(v.first, std::make_pair(mid, prev))));
                            reverse_vertex_pairs.insert(std::make_pair(v.first, std::make_pair(m, std::make_pair(prev, mid))));
                            std::list<Point> edge;
                            edge.clear();
                            int breaks = 0;
                            DouglasPeucker(image_mask, m, v.first, point_list2,
                                           points_decided, 5, edge, breaks);
                            edge.unique();
                            edge_structure.insert(std::make_pair(m, std::make_pair(v.first, std::make_pair(edge, std::make_pair(mid, std::make_pair(prev, std::make_pair((int)point_list.size(), std::make_pair((int)point_list.size(), Standard_Deviation(point_list, image)))))))));
                            edges.push_back(edge);
                            edge.reverse();
                            // reverse_edge_structure.insert(std::make_pair(v.first, std::make_pair(m, std::make_pair(edge, std::make_pair(prev, std::make_pair(mid, std::make_pair((int)point_list.size(), std::make_pair((int)point_list.size(), Standard_Deviation(point_list, image)))))))));
                        }*/
                    }
                    else
                    {
                        if((v.first.x == 1 or v.first.y == 60) and debug) std::cout<<" found next"<<std::endl;
                        vertex_pairs.insert( std::make_pair( v.first, std::make_pair(next_vertex, std::make_pair(mid, prev)) ) );
                        reverse_vertex_pairs.insert( std::make_pair( next_vertex, std::make_pair(v.first, std::make_pair(prev, mid)) ) );
                        std::list<Point> edge;
                        edge.clear();
                        int breaks = 0;
                        DouglasPeucker(image_mask, v.first, next_vertex, point_list,
                                       points_decided, 3, edge, breaks);
                        edge.unique();
                        edge_structure.insert( std::make_pair( v.first, std::make_pair(next_vertex, std::make_pair(edge, std::make_pair(mid, std::make_pair(prev, std::make_pair((int)point_list.size(), std::make_pair(breaks, point_list)))))) ) );
                        edges.push_back(edge);
                        edge.reverse();
                        edge_structure.insert( std::make_pair( next_vertex, std::make_pair(v.first, std::make_pair(edge, std::make_pair(prev, std::make_pair(mid, std::make_pair((int)point_list.size(), std::make_pair(breaks, point_list)))))) ) );
                    }
                }
            }
        }

    }
    if(!debug)
    for(auto edge : edge_structure){
        std::cout<<edge.first<<edge.second.first<<" "<<edge.second.second.second.second.second.first<<" "<<edge.second.second.second.second.second.second.first<<std::endl;
    }
    if(!debug)std::cout<<"loop ends "<<std::endl;


    if(debug)
        std::cout<<"Reverse"<<std::endl;
        for(auto edge : reverse_edge_structure){
            std::cout<<edge.first<<edge.second.first<<" "<<edge.second.second.second.second.second.first<<std::endl;
        }
    if(!debug)std::cout<<"loop ends "<<std::endl;

    // std::cin>>temp;
    std::list<std::list<Point>> face_list;
    for(auto list : edges)
    {
        if(debug)std::cout<<"\tloop6 "<<list.front()<<" "<<list.back()<<std::endl;
        std::list<Point> face;
        face.clear();
        bool valid = true;
        if(!(list.back().x <= 1 or list.back().y <= 1 or list.back().x >= binary_skeleton.cols-2 or list.back().y >= binary_skeleton.rows-2) or binary_skeleton.at<Vec3b>(list.back()) == green) {
            face = Find_Faces(binary_skeleton, list, edge_structure, reverse_edge_structure, face_list, valid);
            if (valid) {
                // face.pop_back()
                face_list.push_back(face);
            }
            if(debug) std::cout<<list.front()<<list.back()<<valid<<std::endl;
        }
    }

    /*for(const auto &structure : edge_structure)
    {
        std::cout<<"Vertex: "<<structure.first<<" Next: "<<structure.second.first<<" Attached to: "<<structure.second.second.second.first<<" "<<structure.second.second.second.second.first<<std::endl;
        for(const auto &point : structure.second.second.first)
        {
            std::cout<<point<<" ";
        }
        std::cout<<std::endl;
    }*/
    if(debug)std::cout<<"loop ends again"<<std::endl;
    if(debug)std::cout<<"\n\n\n";
    int c=0;
    int vhandle_counter = 0;
    std::vector<MyMesh::VertexHandle>  face_vhandles;
    std::vector<MyMesh::Point>  points_added;
    bool f;

    Disconnect_Small_Components (face_list, points_decided, vertices_pixels, edge_structure, loop_list);

    Disconnect_Small_Components_Outer2(face_list, edge_structure, loop_list);
    // Disconnect_Small_Components_Outer(face_list, edge_structure);
    for (auto &face : face_list) {

        if (face.size()<2) continue;
        for (const auto &point : face) {
            f = false;
            for (auto vert : points_added) {
                if (vert == MyMesh::Point(point.x, point.y, 0)) f = true;
            }
            if (!f) {
                vhandle[vhandle_counter++] = mesh.add_vertex(MyMesh::Point(point.x, point.y, 0));
                points_added.emplace_back(MyMesh::Point(point.x, point.y, 0));
                vertices_added.insert(std::make_pair(point, vhandle[vhandle_counter-1]));
            }

        }
    }

    std::cout<<std::endl;
    c = 0;
    std::cout<<" Initial domain list "<<face_list.size()<<" :"<<std::endl;
    int starting_face = 0;
    for (auto &face : face_list)
    {
        if (face.size()<2) continue;
        face_vhandles.clear();
        std::cout<<"\tlist "<<++c<<": ";
        int x_avg = 0, y_avg = 0, c1 = 0, vertices=0;
        for(const auto &point : face)
        {
            std::cout<<point;
            face_vhandles.insert(face_vhandles.begin(), vertices_added.find(point)->second);
            std::cout<<", ";
            x_avg += point.x;
            y_avg += point.y;
            c1++;

            if ( edge_structure.find(point) != edge_structure.end())vertices++;
        }
        std::cout<<std::endl;
        //vhandle[vhandle_counter++] = mesh.add_vertex(MyMesh::Point(x_avg/c1, y_avg/c1, 10));
        //face_vhandles.push_back (vhandle[vhandle_counter-1]);
        mesh.add_face(face_vhandles);
        // circle(image_straight, Point(x_avg/c1, y_avg/c1), 2, black, 2);

        putText(image_straight, std::to_string(vertices), Point(x_avg/c1, y_avg/c1), 0, 0.3, ((vertices%2==0) ? black : blue), 1, LINE_AA);


        /*face_vhandles.clear();
        face_vhandles.push_back(vertices_added.find(face.back())->second);
        face_vhandles.push_back(vertices_added.find(face.front())->second);
        face_vhandles.push_back (vhandle[vhandle_counter-1]);
        mesh.add_face(face_vhandles);*/
    }

    for ( auto v : points_decided ){
        cv::line(image_straight, v.first, v.second, black, 1);
    }
    //cv::imwrite(  name + "_skeleton_original_straight_intermediate2.png" , image_straight );
    return true;

}

Point Openmesh_to_opencv (MyMesh::Point p) {
    return (Point((int) p[0], (int) p[1]));
}

bool Is_Vertex (MyMesh& mesh, MyMesh::VertexHandle p, const Mat &image){
    int c = 0;
    for(auto vf = mesh.vv_iter(p); vf.is_valid(); ++vf){
        c++;
    }
    return c >= 3;
}

bool Visualize (MyMesh& mesh, const Mat &image, const String &name){
    int c = 0;
    cv::Mat_<Vec3b> image_pruned;
    image.copyTo(image_pruned);
    cv::Mat_<Vec3b> image_pruned2;
    image.copyTo(image_pruned2);
    Mat_<Vec3b> image_mask2( image.size(), white );
    Point p, first, last, avg;
    for(auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        std::cout<<"Face "<<++c<<": ";
        bool print = false;
        avg = Point(0, 0);
        int vertices = 0, c1 = 0;
        for(auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it){
            std::cout<<Openmesh_to_opencv(mesh.point(*fv_it));
            if(print){
                cv::line(image_pruned, p, Openmesh_to_opencv(mesh.point(*fv_it)), green, 1);
                cv::line(image_pruned2, p, Openmesh_to_opencv(mesh.point(*fv_it)), blue, 1);
                cv::line(image_mask2, p, Openmesh_to_opencv(mesh.point(*fv_it)), black, 1);
                p = Openmesh_to_opencv(mesh.point(*fv_it));
                last = p;
            }
            else {
                p = Openmesh_to_opencv(mesh.point(*fv_it));
                first = p;
                print = true;
            }
            avg.x += Openmesh_to_opencv(mesh.point(*fv_it)).x;
            avg.y += Openmesh_to_opencv(mesh.point(*fv_it)).y;
            c1++;
            if(Is_Vertex(mesh, *fv_it, image))vertices++;
        }
        cv::line(image_pruned, last, first, green, 1);
        cv::line(image_pruned2, last, first, blue, 1);
        cv::line(image_mask2, last, first, black, 1);
        std::cout<<std::endl;
        putText(image_pruned, std::to_string(vertices), Point(avg.x/c1, avg.y/c1), 0, 0.3, ((vertices%2==0) ? black : blue), 1, LINE_AA);
    }

    c = 0;
    for (auto v_it=mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it){
        //std::cout<<Openmesh_to_opencv(mesh.point(*v_it))<<std::endl;
        c++;
        if ( Is_Vertex(mesh, *v_it, image)) {
            image_pruned.at<Vec3b>(Openmesh_to_opencv(mesh.point(*v_it))) = blue;
            //image_pruned2.at<Vec3b>(Openmesh_to_opencv(mesh.point(*v_it))) = blue;
            //image_mask2.at<Vec3b>(Openmesh_to_opencv(mesh.point(*v_it))) = green;
        }
    }
    for(auto list : remove_loops_points){
        c++;
        c++;
        cv::line(image_pruned2, list.first, list.second, blue, 1);
        cv::line(image_mask2, list.first, list.second, black, 1);
    }

    std::cout<<c;
    //std::cin>>c;
    //cv::imwrite(  name + "_pruned_visual.png" , image_pruned );
    cv::imwrite(  name + " Final mesh (blue) on original image.png" , image_pruned2 );
    cv::imwrite(  name + " Final mesh on white background.png" , image_mask2 );
}


bool Visualize (MyMesh& mesh, const Mat &image, String name, std::list<std::pair<Point, Point>> deleted){
    int c = 0;
    cv::Mat_<Vec3b> image_pruned;
    image.copyTo(image_pruned);
    Point p, first, last, avg;
    for(auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        bool print = false;
        avg = Point(0, 0);
        int vertices = 0, c1 = 0;
        for(auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it){
            if(print){
                cv::line(image_pruned, p, Openmesh_to_opencv(mesh.point(*fv_it)), black, 1);
                p = Openmesh_to_opencv(mesh.point(*fv_it));
                last = p;
            }
            else {
                p = Openmesh_to_opencv(mesh.point(*fv_it));
                first = p;
                print = true;
            }
            avg.x += Openmesh_to_opencv(mesh.point(*fv_it)).x;
            avg.y += Openmesh_to_opencv(mesh.point(*fv_it)).y;
            c1++;
            if(Is_Vertex(mesh, *fv_it, image))vertices++;
        }
        cv::line(image_pruned, last, first, black, 1);
        putText(image_pruned, std::to_string(vertices), Point(avg.x/c1, avg.y/c1), 0, 0.3, ((vertices%2==0) ? black : blue), 1, LINE_AA);
    }

    c = 0;
    for (auto v_it=mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it){
        //std::cout<<Openmesh_to_opencv(mesh.point(*v_it))<<std::endl;
        c++;
        if ( Is_Vertex(mesh, *v_it, image))
            image_pruned.at<Vec3b>(Openmesh_to_opencv(mesh.point(*v_it))) = green;
    }

    for(auto v : deleted){
        cv::line(image_pruned, v.first, v.second, blue, 1);
    }
    //cv::imwrite(  name + "_pruned_visual_with_deleted.png" , image_pruned );
}

MyMesh::FaceHandle Start_Face(MyMesh& mesh, Point middle, std::multimap< Point, bool, ComparePixels >& vertices_pixels, Mat image){
    MyMesh::FaceHandle start_face = *mesh.faces_begin();
    int min = middle.y, c;
    Point temp;
    for (auto v_it=mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it){
        temp = Openmesh_to_opencv(mesh.point(*v_it));
        if(min>=abs(middle.x-temp.x)+abs(middle.y-temp.y) and Is_Vertex(mesh, *v_it, image)){
            for (auto vf_it=mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it){
                c = 0;
                for(auto fv_it = mesh.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it)
                    if ( Is_Vertex(mesh, *fv_it, image)) c++;
                if(c%2!=0 and c<20){
                    start_face = *vf_it;
                    min = abs(middle.x-temp.x)+abs(middle.y-temp.y);
                }
            }
        }
    }
    return start_face;
}

bool Restricted_Removal_Disconnected (MyMesh& mesh, MyMesh::VertexHandle remove1, MyMesh::VertexHandle remove2, const Mat &image){
    if(Is_Boundary(mesh.point(remove1), image))
        return false;
    int count = 0;
    // std::cout<<"disconnected"<<Openmesh_to_opencv(mesh.point(remove1))<<std::endl;
    for(auto vf = mesh.vf_iter(remove1); vf.is_valid(); ++vf) count++;
    return count<=2;
}

bool Restricted_Removal (MyMesh& mesh, MyMesh::VertexHandle remove1, MyMesh::VertexHandle remove2, const Mat &image){
    MyMesh::FaceHandle face1, face2;
    bool face1_found = false, face2_found = false;
    for(auto vf = mesh.vf_iter(remove1); vf.is_valid(); ++vf)
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv)
            if(*fv == remove2)
            {
                face1 = *vf;
                face1_found = true;
                break;
            }
    for(auto vf = mesh.vf_iter(remove2); vf.is_valid(); ++vf)
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv)
            if(*fv == remove1 and *vf != face1)
            {
                face2 = *vf;
                face2_found = true;
                break;
            }
    if(!face1_found or !face2_found) return true;
    int common_vertices = 0;
    for(auto fv1 = mesh.fv_iter(face1); fv1.is_valid(); ++fv1)
        for(auto fv2 = mesh.fv_iter(face2); fv2.is_valid(); ++fv2)
            if(Is_Vertex(mesh, *fv1, image) and Is_Vertex(mesh, *fv2, image) and *fv1 == *fv2) common_vertices++;

    std::cout<<mesh.point(remove1)<<" "<<mesh.point(remove2)<<" "<<common_vertices<<std::endl;
    /*int c = 0;
    for(auto vf = mesh.vf_iter(remove2); vf.is_valid(); ++vf) c++;
    if (c<3) common_vertices = 3;
    c = 0;
    for(auto vf = mesh.vf_iter(remove1); vf.is_valid(); ++vf)c++;
    if (c<3) common_vertices = 3;*/
    return common_vertices != 2;

}

bool Remove_Edge (MyMesh& mesh, MyMesh::VertexHandle& remove1, MyMesh::VertexHandle& remove2, MyMesh::FaceHandle target_face, const Mat &image, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > & edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added){

    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    double max = 0, mean, median, mode, val;
    int length;
    Point r1, r2;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );

    for(auto fv_it = mesh.fv_iter(target_face); fv_it.is_valid(); ++fv_it)
        if(Is_Vertex(mesh, *fv_it, image))
        {
            result = edge_structure.equal_range(Openmesh_to_opencv(mesh.point(*fv_it)));
            for (auto it = result.first; it != result.second; it++)
            {
                median = Median(it->second.second.second.second.second.second.second, image_gray);
                mean = Mean(it->second.second.second.second.second.second.second, image_gray);
                mode = Mode(it->second.second.second.second.second.second.second, image_gray);
                length = it->second.second.second.second.second.first;
                val = mean+median+mode - (length*3);
                if(length<50 and val>max and it->second.second.second.second.second.second.first==1 and median>=70)
                    if ( edge_structure.find(it->second.first) != edge_structure.end()){
                    if ( Is_Boundary(it->first, image) and Is_Boundary(it->second.first, image)) continue;
                    if ( Restricted_Removal(mesh, vertices_added.find(it->first)->second, vertices_added.find(it->second.first)->second, image)) continue;
                    if ( !Is_Vertex(mesh, vertices_added.find(it->first)->second, image) or !Is_Vertex(mesh, vertices_added.find(it->second.first)->second, image) ) continue;
                    remove1 = vertices_added.find(it->first)->second;
                    remove2 = vertices_added.find(it->second.first)->second;
                    max = val;
                    r1 = it->first;
                    r2 = it->second.first;
                    std::cout<<"Considered: "<<it->first<<it->second.first<<" "<<it->second.second.second.second.second.first<<" "<<val<<std::endl;
                }
                std::cout<<it->first<<it->second.first<<" "<<it->second.second.second.second.second.first<<val<<std::endl;

            }

        }

    if(r1 == Point(0, 0))
        return false;
    std::cout<<"Removed: "<<r1<<r2<<std::endl;
    Remove_Edge_Structure(r1, r2, edge_structure);
    return true;
}

bool Join_Faces (MyMesh& mesh, MyMesh::VertexHandle& remove1, MyMesh::VertexHandle& remove2){
    bool tem, tem2;
    std::cout<<"Joining faces: "<<std::endl;
    std::vector<MyMesh::VertexHandle> vhandles, vhandles1, vhandles2;
    int handle = 1;
    auto prev = remove2;
    MyMesh::FaceVertexIter prev1, prev2;

    do {
        tem = false;
    for (auto vf = mesh.vf_iter(remove1); vf.is_valid(); ++vf) {
        int found_first = 0, found2 = 0, count = 0;
        for (auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv) {
            count++;
            if(*fv == remove1) {
                prev1 = fv;
                found_first = 1;
            }
            if(*fv == remove2) {
                prev2 = fv;
                found_first = 2;
                found2++;
            }
        }
        long distance = 0;
        if(found_first == 1 and found2 == 1) distance = std::distance(prev2, prev1);
        else if(found_first == 2 and found2 == 1) distance = std::distance(prev1, prev2);

        if(distance != 0) {
            if (distance == 1 or distance + 1 == count) {
                for (auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv) {
                    std::cout << mesh.point(*fv) << std::endl;
                    if (handle == 1) vhandles1.push_back(*fv);
                    else vhandles2.push_back(*fv);
                }
                std::cout<<handle<<std::endl;
                handle++;
                mesh.delete_face(*vf, false);
                tem = true;
                break;
            }
            std::cout << distance << " " << count << std::endl;
            std::cout << std::endl;
        }
    }
    }while(tem);

    mesh.garbage_collection();

    for (auto &it : vhandles1) {
        std::cout<<"00 "<<mesh.point(it)<<std::endl;
    }

    for (auto &it : vhandles2) {
        std::cout<<"01 "<<mesh.point(it)<<std::endl;
    }

    auto it1 = std::find(vhandles1.begin(), vhandles1.end(), remove1);
    auto it2 = std::find(vhandles1.begin(), vhandles1.end(), remove2);
    std::cout<<"it1: "<<mesh.point(*it1)<<std::endl;
    std::cout<<"it2: "<<mesh.point(*it2)<<std::endl;

    auto it_temp = vhandles1.begin();
    for (; it_temp != vhandles1.end(); it_temp++) {
        if(*it_temp == *it1 and it_temp == vhandles1.begin())
        {
            auto it_temp2 = it_temp;
            it_temp2++;
            if(*it_temp2 == *it2)
                break;
            else
                {
                    it_temp2 = it2;
                    it2 = it1;
                    it1 = it_temp2;
                    break;
                }
        }
        else if(*it_temp == *it2 and it_temp == vhandles1.begin())
        {
            auto it_temp2 = it_temp;
            it_temp2++;
            if(*it_temp2 == *it1)
            {
                it_temp2 = it2;
                it2 = it1;
                it1 = it_temp2;
                break;
            }
            else break;

        }
        if(*it_temp == *it1)
            break;
        else if(*it_temp == *it2){
            auto it_temp2 = it2;
            it2 = it1;
            it1 = it_temp2;
            break;
        }
    }
    std::cout<<"it1: "<<mesh.point(*it1)<<std::endl;
    std::cout<<"it2: "<<mesh.point(*it2)<<std::endl;
    MyMesh::Point p1 = mesh.point(*it1), p2 = mesh.point(*it2);

    bool exit = false;
    while(!exit){
        std::cout<<"Rotating1: "<<mesh.point(*vhandles1.begin())<<" "<<mesh.point(vhandles1.back())<<" "<<p2<<" "<<p1<<std::endl;
        if(mesh.point(*vhandles1.begin()) == p2)
            if(mesh.point(vhandles1.back()) == p1)
                exit = true;
        auto temp_handle = vhandles1.begin();
        temp_handle++;
        if(!exit) std::rotate(vhandles1.begin(), temp_handle, vhandles1.end());
        // std::cin>>handle;
    }
        // std::rotate(vhandles1.begin(), it2, vhandles1.end());


    for (auto &it : vhandles1) {
        std::cout<<"10 "<<mesh.point(it)<<std::endl;
    }

    auto it3 = std::find(vhandles2.begin(), vhandles2.end(), remove2);
    auto it4 = std::find(vhandles2.begin(), vhandles2.end(), remove1);
    std::cout<<"it3: "<<mesh.point(*it3)<<std::endl;
    std::cout<<"it4: "<<mesh.point(*it4)<<std::endl;

    it_temp = vhandles2.begin();
    for (; it_temp != vhandles2.end(); it_temp++) {
        if(*it_temp == *it3 and it_temp == vhandles2.begin())
        {
            auto it_temp2 = it_temp;
            it_temp2++;
            if(*it_temp2 == *it4)
                break;
            else
            {
                it_temp2 = it4;
                it4 = it3;
                it3 = it_temp2;
                break;
            }
        }
        else if(*it_temp == *it4 and it_temp == vhandles2.begin())
        {
            auto it_temp2 = it_temp;
            it_temp2++;
            if(*it_temp2 == *it3)
            {
                it_temp2 = it4;
                it4 = it3;
                it3 = it_temp2;
                break;
            }
            else break;

        }
        if(*it_temp == *it3)
            break;
        else if(*it_temp == *it4){
            auto it_temp2 = it4;
            it4 = it3;
            it3 = it_temp2;
            break;
        }
    }
    std::cout<<"it3: "<<mesh.point(*it3)<<std::endl;
    std::cout<<"it4: "<<mesh.point(*it4)<<std::endl;

    MyMesh::Point p3 = mesh.point(*it3), p4 = mesh.point(*it4);
    exit = false;
    while(!exit){
        std::cout<<"Rotating2: "<<mesh.point(*vhandles2.begin())<<" "<<mesh.point(vhandles2.back())<<" "<<p4<<" "<<p3<<std::endl;
        if(mesh.point(*vhandles2.begin()) == p4)
            if(mesh.point(vhandles2.back()) == p3)
                exit = true;
        auto temp_handle = vhandles2.begin();
        temp_handle++;
        if(!exit)std::rotate(vhandles2.begin(), temp_handle, vhandles2.end());
        // std::cin>>handle;
    }
    // std::rotate(vhandles2.begin(), it4, vhandles2.end());


    for (auto &it : vhandles2) {
        std::cout<<"11 "<<mesh.point(it)<<std::endl;
    }
    vhandles2.pop_back();
    vhandles2.erase(vhandles2.begin());
    vhandles1.insert( vhandles1.end(), vhandles2.begin(), vhandles2.end() );

    for (auto &it : vhandles1) {
        std::cout<<"Final: "<<mesh.point(it)<<std::endl;
    }

    mesh.add_face(vhandles1);

    std::cout<<"done"<<std::endl;
    return true;
}



bool Add_Handles(MyMesh& mesh, MyMesh::VertexHandle& remove1, MyMesh::VertexHandle& remove2, std::list<MyMesh::VertexHandle>& vhandles,
                 const Mat &image){
    for(auto vf = mesh.vf_iter(remove1); vf.is_valid(); ++vf)
    {
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv) {
            bool add = false;
            for (auto v : vhandles) {
                if(v != remove1 and v != remove2)
                if (v == *fv) add = true;
            }
            if(!add and Is_Vertex(mesh, *fv, image)) {
                vhandles.push_back(*fv);
                vhandles.push_back(*fv);
                vhandles.push_back(*fv);
            }
        }
    }

    for(auto vf = mesh.vf_iter(remove2); vf.is_valid(); ++vf)
    {
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv) {
            bool add = false;
            for (auto v : vhandles) {
                if(v != remove1 and v != remove2)
                    if (v == *fv) add = true;
            }
            if(!add and Is_Vertex(mesh, *fv, image)) {
                vhandles.push_back(*fv);
                vhandles.push_back(*fv);
                vhandles.push_back(*fv);
            }
        }
    }
}
void perm(int n, int& permutations, int size, std::list<std::vector<unsigned int>>& combination_list)
{
    /*
    if(n == arr.size())
    {
        permutations++;
        combination_list.push_back(arr);
        return;
    }
    for(unsigned int i = 0; i<2;i++)
    {
        arr.at(i)= 0 + i;
        perm(n+1, permutations, arr, combination_list);
    }*/
    std::vector<unsigned int> arr(size, 0);
    for(int j = 0; j<size; j++) {
        arr.clear();
        for (int i = 0; i < j; i++) {
            arr.push_back(1);
        }
        for (int i = 0; i < size-j; i++) {
            arr.insert(arr.begin(), 0);
        }
        do {
            permutations++;
            combination_list.push_back(arr);
        } while (std::next_permutation(std::begin(arr), std::end(arr)));
    }
    arr.clear();
    for (int i = 0; i < size; i++) {
        arr.push_back(1);
    }
    combination_list.push_back(arr);
    permutations++;
}


void Compare_Point(Point& p1, Point& p2){
    if(p1.x<p2.x) return;
    if(p1.x==p2.x and p1.y<p2.y) return;
    if(p1.x==p2.x and p1.y==p2.y) return;
    if(p1.x>p2.x or p1.y>p2.y) {
        Point p = p1;
        p1 = p2;
        p2 = p;
        return;
    }
}
bool Pruning_Skeleton(MyMesh& mesh, std::multimap< Point, bool, ComparePixels >& vertices_pixels, const Mat &image, const String &name, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added)
{
    cv::Mat_<Vec3b> image_pruned;
    image.copyTo(image_pruned);
    Point p, first, last;
    MyMesh::VertexHandle remove1, remove2;
    std::list<MyMesh::VertexHandle> vhandles;
    std::list<MyMesh::FaceHandle> fhandles;
    int c = 0;
    MyMesh::FaceHandle start_face = Start_Face(mesh, Point(image.cols/2, image.rows/2), vertices_pixels, image);
    // mesh.delete_face(start_face, true);

            Visualize(mesh, image, name);
            std::cin>>c;

    bool print = false;
    for(auto fv_it = mesh.fv_iter(start_face); fv_it.is_valid(); ++fv_it){
        if(Is_Vertex(mesh, *fv_it, image)){
            vhandles.push_back(*fv_it);
            vhandles.push_back(*fv_it);
            vhandles.push_back(*fv_it);
        }
        if(print){
            cv::line(image_pruned, p, Openmesh_to_opencv(mesh.point(*fv_it)), black, 1);
            p = Openmesh_to_opencv(mesh.point(*fv_it));
            last = p;
        }
        else {
            p = Openmesh_to_opencv(mesh.point(*fv_it));
            first = p;
            print = true;
        }
    }
    std::cout<<std::endl;
    cv::line(image_pruned, last, first, black, 1);

    Remove_Edge(mesh, remove1, remove2, start_face, image, edge_structure, vertices_added);
    Join_Faces(mesh, remove1, remove2);
    std::cout<<Openmesh_to_opencv(mesh.point(remove1))<<Openmesh_to_opencv(mesh.point(remove2))<<std::endl;
    for(auto fv_it = mesh.fv_iter(start_face); fv_it.is_valid(); ++fv_it)
        if(Is_Vertex(mesh, *fv_it, image))
            image_pruned.at<Vec3b>(Openmesh_to_opencv(mesh.point(*fv_it))) = green;

    auto vhandle_iter = vhandles.begin();

    Visualize(mesh, image, name);
    std::cin>>c;
    while(vhandle_iter != vhandles.end()) {
        vhandle_iter++;
        for (auto vf_it = mesh.vf_iter(*vhandle_iter); vf_it.is_valid(); ++vf_it) {
            std::cout<<"Starting Point: "<<mesh.point(*vhandle_iter)<<std::endl;
            if (*vf_it == start_face) continue;
            int v_count = 0;
            bool should_remove = true;
            for (auto fv_it = mesh.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it)
                if (Is_Vertex(mesh, *fv_it, image)) v_count++;
            if (v_count % 2 == 0) should_remove = false;
            if ( should_remove ) fhandles.push_back(*vf_it);
            print = false;
            for (auto fv_it = mesh.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it) {
                // vhandles.push_back(*fv_it);
                if (print) {
                    cv::line(image_pruned, p, Openmesh_to_opencv(mesh.point(*fv_it)), black, 1);
                    p = Openmesh_to_opencv(mesh.point(*fv_it));
                    last = p;
                } else {
                    p = Openmesh_to_opencv(mesh.point(*fv_it));
                    first = p;
                    print = true;
                }
            }
            cv::line(image_pruned, last, first, black, 1);
            for (auto fv_it = mesh.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it)
                if (Is_Vertex(mesh, *fv_it, image))
                    image_pruned.at<Vec3b>(Openmesh_to_opencv(mesh.point(*fv_it))) = green;
            if(should_remove) should_remove = Remove_Edge(mesh, remove1, remove2, *vf_it, image, edge_structure, vertices_added);
            if(should_remove) Join_Faces(mesh, remove1, remove2);
            // if(should_remove) Add_Handles(mesh, remove1, remove2, vhandles, image);
            if(should_remove) std::cout <<"Remove successful"<<Openmesh_to_opencv(mesh.point(remove1)) << Openmesh_to_opencv(mesh.point(remove2))
                      << std::endl;
            cv::imwrite(name + "_pruned.png", image_pruned);
            try
            {
                if ( !OpenMesh::IO::write_mesh(mesh, name + "_output_pruned.off") )
                {
                    std::cerr << "Cannot write mesh to file 'output_pruned.off'" << std::endl;
                    return true;
                }
            }
            catch( std::exception& x )
            {
                std::cerr << x.what() << std::endl;
                return true;
            }
            if(should_remove) Visualize(mesh, image, name);
            // if(should_remove) std::cin >> c;
            if(should_remove) break;
        }
    }

    std::cout<<"Level 1 done"<<std::endl;
    std::cin>>c;
    bool remove = false;
    do
    for(auto face = mesh.faces_begin() ; face != mesh.faces_end(); ++face){
        int v_count = 0;
        bool should_remove = true;
        remove = false;
        std::cout<<"C"<<std::endl;
        for (auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it)
            if (Is_Vertex(mesh, *fv_it, image)) v_count++;
        if (v_count % 2 == 0) should_remove = false;
        print = false;
        for (auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it) {
            // vhandles.push_back(*fv_it);
            if (print) {
                cv::line(image_pruned, p, Openmesh_to_opencv(mesh.point(*fv_it)), black, 1);
                p = Openmesh_to_opencv(mesh.point(*fv_it));
                last = p;
            } else {
                p = Openmesh_to_opencv(mesh.point(*fv_it));
                first = p;
                print = true;
            }
        }
        std::cout << std::endl;
        cv::line(image_pruned, last, first, black, 1);
        for (auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it)
            if (Is_Vertex(mesh, *fv_it, image))
                image_pruned.at<Vec3b>(Openmesh_to_opencv(mesh.point(*fv_it))) = green;

        if(should_remove) should_remove = Remove_Edge(mesh, remove1, remove2, *face, image, edge_structure, vertices_added);
        if(should_remove) remove = Join_Faces(mesh, remove1, remove2);
        std::cout << Openmesh_to_opencv(mesh.point(remove1)) << Openmesh_to_opencv(mesh.point(remove2))
                  << std::endl;
        cv::imwrite(name + "_pruned.png", image_pruned);
        try
        {
            if ( !OpenMesh::IO::write_mesh(mesh, name + "_output_pruned.off") )
            {
                std::cerr << "Cannot write mesh to file 'output_pruned.off'" << std::endl;
                return true;
            }
        }
        catch( std::exception& x )
        {
            std::cerr << x.what() << std::endl;
            return true;
        }
        Visualize(mesh, image, name);
        // if(should_remove) std::cin >> c;
        if (remove) break;
    }while(remove);


    /*
    */

    std::cout<<std::endl;
    cv::imwrite(  name + "_pruned.png" , image_pruned );
}



bool Edge_Parameters(std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, const Mat &image, const String &name, std::multimap< Point, bool, ComparePixels >& vertices_pixels)
{
    std::ofstream myfile;
    std::list<std::pair<Point, Point>> added;
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    bool add;
    long mode;
    int length, min, max, range, median, median_line, median_nbrs = 0;
    double sd, dist, mean, sd_color, sd1, sd2, sd3, mean_line1, mean_line2, mean_centre;
    Point point1, point2;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    cv::Mat_<Vec3b> image_edge_analysis;
    image.copyTo(image_edge_analysis);
    myfile.open (name + "_edge_data.csv");
    myfile << "Point 1,Point 2,Length,Distance,SD Gray,SD Colored,SD1,SD2,SD3,Mean,Median,Mode,Range,Max,Min,Mean_Centre,Mean_Line1,Mean_Line2,Mean_Nbrs,Mean_Ratio,Median_Line,Median_Nbrs,val,Keep\n";
    int vertices = 1;
    for (auto edge : edge_structure){
        if(edge.second.second.second.second.second.second.first > 1) continue;
        if(Is_Boundary (edge.first, image) and Is_Boundary (edge.second.first, image)) continue;
        add = true;
        for (auto v : added)
            if((v.first == edge.first and v.second == edge.second.first) or (v.first == edge.second.first and v.second == edge.first)) {
                add = false;
                break;
            }
        if (!add) continue;
        point1 = edge.first; point2 = edge.second.first;
        added.emplace_back(point1, point2);
        std::cout<<"For points: "<<point1<<point2<<"Other points: ";
        result = edge_structure.equal_range(point1);
        mean_line1 = 0;
        for (auto it = result.first; it != result.second; it++)
            if(it->second.first != point2)
            {
                mean_line1 += Mean_Line(it->second.second.second.second.second.second.second, edge.second.second.second.second.second.first/2, image_gray);
                median_nbrs += Median_Line(it->second.second.second.second.second.second.second, edge.second.second.second.second.second.first/2, image_gray);
                std::cout<<it->first<<it->second.first;
            }
        mean_line1 /= 2;
        result = edge_structure.equal_range(point2);
        mean_line2 = 0;
        for (auto it = result.first; it != result.second; it++)
            if(it->second.first != point1)
            {
                mean_line2 += Mean_Line(it->second.second.second.second.second.second.second, edge.second.second.second.second.second.first/2, image_gray);
                median_nbrs += Median_Line(it->second.second.second.second.second.second.second, edge.second.second.second.second.second.first/2, image_gray);
                std::cout<<it->first<<it->second.first;
            }
        std::cout<<std::endl;
        mean_line2 /= 2;
        median_nbrs /= 4;
        mean_centre = Mean_Centre(edge.second.second.second.second.second.second.second, image);
        length = edge.second.second.second.second.second.first;
        sd = Standard_Deviation(edge.second.second.second.second.second.second.second, image_gray);
        sd1 = Standard_Deviation1(edge.second.second.second.second.second.second.second, image);
        sd2 = Standard_Deviation2(edge.second.second.second.second.second.second.second, image);
        sd3 = Standard_Deviation3(edge.second.second.second.second.second.second.second, image);
        sd_color = Standard_Deviation(edge.second.second.second.second.second.second.second, image);
        dist = sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2));
        mean = Mean(edge.second.second.second.second.second.second.second, image_gray);
        median = Median(edge.second.second.second.second.second.second.second, image_gray, min, max, range);
        mode = Mode(edge.second.second.second.second.second.second.second, image_gray);
        myfile << point1.x <<" " << point1.y <<","<<point2.x<<" "<<point2.y<<","<<length<<","<<dist<<","<<sd<<","<<sd_color<<","
               <<sd1<<","<<sd2<<","<<sd3<<","<<mean<<","<<median<<","<<mode<<","<<range<<","<<max<<","<<min<<","
               <<mean_centre<<","<<mean_line1<<","<<mean_line2<<","<<(mean_line1+mean_line2)/2<<","<<mean_centre/((mean_line1+mean_line2)/2)<<","
               <<Median_Line(edge.second.second.second.second.second.second.second, image)<<","<<(mean+median+mode)-(length*3)<<","<<median_nbrs<<",0,\n";
        cv::line(image_edge_analysis, point1, point2, black, 1);
        Point p = Point((point1.x+point2.x)/2, (point1.y+point2.y)/2);
        std::cout<<"Line: "<<p<<vertices<<"Centre"<<mean_centre<<" Line1:"<<mean_line1<<" Line2:"<<mean_line2<<std::endl;
        putText(image_edge_analysis, std::to_string(vertices), p, 0, 0.3, blue, 1, LINE_AA);
        vertices++;
    }
    for(auto v:vertices_pixels){
        image_edge_analysis.at<Vec3b>(v.first) = green;
        image_edge_analysis.at<Vec3b>(v.second) = green;
    }
    myfile.close();
    cv::imwrite(  name + "_edge_analysis.png" , image_edge_analysis );
    return true;
}


int Count_Odd(MyMesh mesh, const Mat &image)
{
    int vertex_count;
    bool is_boundary;
    int face_odd = 0;
    for(auto f_it=mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
    {
        vertex_count = 0;
        is_boundary = false;
        for(auto fv_it=mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            if(Is_Vertex(mesh, *fv_it, image)) vertex_count++;
            if(Is_Boundary(mesh.point(*fv_it), image)) is_boundary = true;
        }
        //if (vertex_count>=10) is_boundary = false;
        if(vertex_count%2 != 0 and !is_boundary) face_odd++;
    }
    return face_odd;
}

int Cost(MyMesh& mesh, const Mat &image, int& face_odd, int& face_boundary_odd);
int Cost(MyMesh& mesh, const Mat &image, int& face_odd, int& face_boundary_odd)
{
    int vertex_count;
    bool is_boundary;
    for(auto f_it=mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
    {
        vertex_count = 0;
        is_boundary = false;
        for(auto fv_it=mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            if(Is_Vertex(mesh, *fv_it, image)) vertex_count++;
            if(Is_Boundary(mesh.point(*fv_it), image)) is_boundary = true;
        }
        //if (vertex_count>=10) is_boundary = false;
        if(vertex_count%2 != 0 and !is_boundary) face_odd++;
        else if(vertex_count%2 != 0 and is_boundary) face_boundary_odd++;
    }
    return (face_odd*face_odd)+face_boundary_odd;
}


int Cost(int cost, const std::vector<int> &faces, const std::vector<bool> &faces_bool, int odd, int odd_boundary, bool odd_even_bool);
int Cost(int cost, const std::vector<int> &faces, const std::vector<bool> &faces_bool, int odd, int odd_boundary, bool odd_even_bool) {
    if(faces.size() == 3) return cost;
    int odd_temp = 0, odd_boundary_temp = 0, odd_even = 0;
    if(faces.at(0)%2 != 0 and faces.at(1)%2 != 0){
        if((faces_bool.at(0) or faces_bool.at(1))) odd_boundary_temp+=2; else odd_temp+=2;
        //if((faces_bool.at(0) or faces_bool.at(1)) and (faces.at(0)+faces.at(1)<=10)) odd_boundary_temp+=2; else odd_temp+=2;
    }
    else if(faces.at(0)%2 == 0 and faces.at(1)%2 != 0 or faces.at(0)%2 != 0 and faces.at(1)%2 == 0)  odd_even++;
    if(faces.at(2)%2 != 0 and faces.at(2) != 0) if(faces_bool.at(2)) odd_boundary_temp++; else odd_temp++;
    else if(faces.at(2)%2 == 0 ) if(faces_bool.at(2)) odd_boundary_temp--; else odd_temp--;
    if(faces.at(3)%2 != 0 and faces.at(3) != 0 ) if(faces_bool.at(3)) odd_boundary_temp++; else odd_temp++;
    else if(faces.at(3)%2 == 0 ) if(faces_bool.at(3)) odd_boundary_temp--; else odd_temp--;

    /*if(faces.at(2)%2 != 0 and faces.at(2) != 0) if(faces_bool.at(2) and faces.at(2)<=10) odd_boundary_temp++; else odd_temp++;
    else if(faces.at(2)%2 == 0 ) if(faces_bool.at(2) and faces.at(2)<=10) odd_boundary_temp--; else odd_temp--;
    if(faces.at(3)%2 != 0 and faces.at(3) != 0 ) if(faces_bool.at(3) and faces.at(3)<=10) odd_boundary_temp++; else odd_temp++;
    else if(faces.at(3)%2 == 0 ) if(faces_bool.at(3) and faces.at(3)<=10) odd_boundary_temp--; else odd_temp--;*/




    if(odd_even_bool)
        return ((odd-odd_temp)*(odd-odd_temp))+(odd_even);          // 2*odd_even
    if(!odd_even_bool)
        return ((odd-odd_temp)*(odd-odd_temp))+odd_boundary-odd_boundary_temp;
}


int Cost_Level3(int cost, const std::vector<int> &faces, const std::vector<bool> &faces_bool, int odd, int odd_boundary, bool odd_even_bool) {
    if(faces.size() == 3) return cost;
    int odd_temp = 0, odd_boundary_temp = 0, odd_even = 0;
    if(faces.at(0)%2 != 0 and faces.at(1)%2 != 0){
        if((faces_bool.at(0) and faces.at(0)<=10) or (faces_bool.at(1) and faces.at(1)<=10)) odd_boundary_temp+=2; else odd_temp+=2;
    }
    else if(faces.at(0)%2 == 0 and faces.at(1)%2 != 0 or faces.at(0)%2 != 0 and faces.at(1)%2 == 0)  odd_even++;
    if(faces.at(2)%2 != 0 and faces.at(2) != 0) if(faces_bool.at(2) and faces.at(2)<=10) odd_boundary_temp++; else odd_temp++;
    else if(faces.at(2)%2 == 0 ) if(faces_bool.at(2) and faces.at(2)<=10) odd_boundary_temp--; else odd_temp--;

    if(faces.at(3)%2 != 0 and faces.at(3) != 0 ) if(faces_bool.at(3) and faces.at(3)<=10) odd_boundary_temp++; else odd_temp++;
    else if(faces.at(3)%2 == 0 ) if(faces_bool.at(3) and faces.at(3)<=10) odd_boundary_temp--; else odd_temp--;

    if(odd_even_bool)
        return ((odd-odd_temp)*(odd-odd_temp))+odd_boundary-odd_boundary_temp+(2*odd_even);
    if(!odd_even_bool)
        return ((odd-odd_temp)*(odd-odd_temp))+odd_boundary-odd_boundary_temp;
}

bool Update_Straight_Edge_Structure(std::list<std::pair<std::pair<Point, Point>, std::pair<std::vector<int>, std::vector<bool>>>>& straight_edges, MyMesh& mesh, MyMesh::VertexHandle v1, MyMesh::VertexHandle v2,
                                    const Mat &image, std::map< Point, MyMesh::VertexHandle, ComparePixels > vertices_added){
    int temp, i;
    Point p, p1, p2;
    std::list<MyMesh::VertexHandle> update_list;
    std::list<MyMesh::FaceHandle> face_handle;
    int face11_count, face12_count, face21_count, face22_count;
    bool face11_boundary, face12_boundary, face21_boundary, face22_boundary;
    MyMesh::FaceHandle temp_face1[3], temp_face2[3], face11, face12, face21, face22;

    bool removed;
    do {
        removed = false;
        for (auto it:straight_edges) {
            std::cout<<it.first.first<<Openmesh_to_opencv(mesh.point(v2))<<Openmesh_to_opencv(mesh.point(v1))<<std::endl;
            if (it.first.first == Openmesh_to_opencv(mesh.point(v1)) or
                it.first.second == Openmesh_to_opencv(mesh.point(v1)) or
                it.first.second == Openmesh_to_opencv(mesh.point(v2)) or
                it.first.first == Openmesh_to_opencv(mesh.point(v2))) {
                std::cout << "Removed from list point" << it.first.first << it.first.second << std::endl;
                // std::cin >> face11_count;
                straight_edges.remove(it);
                removed = true;
                break;
            }
        }
    }while(removed);

    for(auto it = mesh.vf_iter(v1); it.is_valid(); ++it)
        face_handle.push_back(*it);
    for(auto it = mesh.vf_iter(v2); it.is_valid(); ++it)
    {
        if(!(std::find(face_handle.begin(), face_handle.end(), *it) != face_handle.end()))
            face_handle.push_back(*it);
    }
        /*for(auto fit = mesh.fv_iter(*it); fit.is_valid(); ++fit)
            if(Is_Vertex(mesh, *fit, image)) update_list.push_back(*fit);*/
    for(auto face_it:face_handle){
        /*temp = 0;
        for(auto vert = mesh.fv_iter(face_it); vert.is_valid(); ++vert)
            if(Is_Vertex(mesh, *vert, image)) temp++;
        for(auto vert = mesh.fv_iter(face_it); vert.is_valid(); ++vert)
        if(Is_Vertex(mesh, *vert, image))
        {
            p = Openmesh_to_opencv(mesh.point(*vert));
            for(auto straight:straight_edges)
                if(straight.first.first == p) {straight.second.first.at(2) = temp; std::cout<<"Updated: "<<p<<temp<<std::endl; break;}
                else if (straight.first.second == p) {straight.second.first.at(3) = temp; std::cout<<"Updated: "<<p<<temp<<std::endl; break;}
        }*/
        for(auto vert = mesh.fv_iter(face_it); vert.is_valid(); ++vert) if(Is_Vertex(mesh, *vert, image))
        {
            std::cout<<"Vertex: "<<Openmesh_to_opencv(mesh.point(*vert))<<std::endl;
            for(auto it = straight_edges.begin(); it != straight_edges.end(); it++)
                if(it->first.first == Openmesh_to_opencv(mesh.point(*vert)) or it->first.second == Openmesh_to_opencv(mesh.point(*vert)))
                {
                    face11_count = 0; face12_count = 0; face21_count = 0; face22_count = 0, temp = 0;
                    face11_boundary = false; face12_boundary = false; face21_boundary = false; face22_boundary = false;

                    i = 0;
                    for(auto vf = mesh.vf_iter(vertices_added.find(it->first.first)->second); vf.is_valid() ; ++vf) temp_face1[i++] = *vf;
                    i = 0;
                    for(auto vf = mesh.vf_iter(vertices_added.find(it->first.second)->second); vf.is_valid() ; ++vf) {
                        temp_face2[i++] = *vf;
                        for(auto t:temp_face1)
                            if (t == *vf)
                                if(temp++ == 0) face21 = *vf;
                                else face22 = *vf;
                    }
                    std::cout<<"reached1"<<std::endl;
                    for(auto t:temp_face1) if (t != face21 and t != face22) face11 = t;
                    for(auto t:temp_face2) if (t != face21 and t != face22) face12 = t;


                    for(auto fv = mesh.fv_iter(face21); fv.is_valid(); fv++)
                    {
                        if(Is_Vertex(mesh, *fv, image)) face21_count++;
                        if(Is_Boundary(mesh.point(*fv), image)) face21_boundary = true;
                    }
                    std::cout<<"reached2"<<std::endl;
                    for(auto fv = mesh.fv_iter(face22); fv.is_valid(); fv++) {
                        if(Is_Vertex(mesh, *fv, image)) face22_count++;
                        if(Is_Boundary(mesh.point(*fv), image)) face22_boundary = true;
                    }
                    std::cout<<"reached3"<<" "<<it->first.first<<it->first.second<<std::endl;
                    if(!Is_Boundary (it->first.first, image)) for(auto fv = mesh.fv_iter(face11); fv.is_valid(); fv++) {
                            if(Is_Vertex(mesh, *fv, image)) face11_count++;
                            if(Is_Boundary(mesh.point(*fv), image)) face11_boundary = true;
                        }
                    std::cout<<"reached4"<<std::endl;
                    if(!Is_Boundary (it->first.second, image)) for(auto fv = mesh.fv_iter(face12); fv.is_valid(); fv++) {
                            if(Is_Vertex(mesh, *fv, image)) face12_count++;
                            if(Is_Boundary(mesh.point(*fv), image)) face12_boundary = true;
                        }
                    std::cout<<"reached5"<<std::endl;
                    std::vector<int> temp_list;
                    std::vector<bool> temp_bool_list;
                    temp_list.push_back(face21_count); temp_list.push_back(face22_count);
                    temp_list.push_back(face11_count); temp_list.push_back(face12_count);
                    temp_bool_list.push_back(face21_boundary); temp_bool_list.push_back(face22_boundary);
                    temp_bool_list.push_back(face11_boundary); temp_bool_list.push_back(face12_boundary);
                    it->second.first = temp_list;
                    it->second.second = temp_bool_list;
                    std::cout<< it->first.first << it->first.second << " face21: " << it->second.first.at(0) << " face22: "
                                << it->second.first.at(1)
                                << " face11: " << it->second.first.at(2) << " face12: " << it->second.first.at(3)
                                << std::endl;
                }
        }
    }
    // std::cin>>face11_count;
}

bool Trivial_Removal(MyMesh& mesh, const Mat &image, const String &name, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added, std::list<std::pair<Point, Point>>& deleted)
{
    int c, valid_options, repeated;
    bool is_boundary;
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    Point point1, point2, p1, p2;
    std::list<Point> points;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    MyMesh::FaceHandle facehandle;
    for(auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face) {
        c = 0; repeated = 0;
        is_boundary = false;
        points.clear();
        for(auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it)
            if(Is_Vertex(mesh, *fv_it, image)) {
                std::cout<<Openmesh_to_opencv(mesh.point(*fv_it))<<" ";
                c++;
                points.push_back(Openmesh_to_opencv(mesh.point(*fv_it)));
                if(Is_Boundary(mesh.point(*fv_it), image)) is_boundary = true;
            }std::cout<<std::endl;
        if(c>=10) is_boundary = false;
        if(!is_boundary and c%2 != 0){
            valid_options = 0;
            for(auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it) if(Is_Vertex(mesh, *fv_it, image)) {
                result = edge_structure.equal_range(Openmesh_to_opencv(mesh.point(*fv_it)));
                for (auto it = result.first; it != result.second; it++){
                    if ( it->second.second.second.second.second.second.first > 1) continue;
                    if (edge_structure.find(it->second.first) == edge_structure.end()) continue;
                    if ( Is_Boundary (it->first, image) and Is_Boundary (it->second.first, image)) continue;
                    point1 = it->first; point2 = it->second.first;
                    if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 45) continue;
                    if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it->second.second.second.second.second.second.second, image_gray)<25) continue;
                    if ( Restricted_Removal(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                    if ( Restricted_Removal_Disconnected(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                    valid_options++;
                    //if(sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it->second.second.second.second.second.second.second, image_gray)<25) valid_options--;
                    if(std::find(points.begin(), points.end(), point1) != points.end())
                        if(std::find(points.begin(), points.end(), point2) != points.end())
                            repeated++;
                    p1 = point1;
                    p2 = point2;
                    std::cout<<"\tOption: "<<valid_options<<it->first<<it->second.first<<std::endl;
                }
            }
            valid_options = valid_options-repeated/2;
            std::cout<<"\tOPTIONS: "<<valid_options<<"\n"<<std::endl;
            if(valid_options == 1) {

                bool is_deleted = Join_Faces(mesh, vertices_added.find(p1)->second, vertices_added.find(p2)->second);
                deleted.emplace_back(p1, p2);
                if (is_deleted) Visualize(mesh, image, name, deleted);
                Remove_Edge_Structure(p1, p2, edge_structure);
                std::cout<<"Trivial deletion"<<p1<<p2<<std::endl;
                std::cin>>c;
            }
        }
    }
    // std::cin>>c;
}




bool Trivial_Removal(MyMesh& mesh, const Mat &image, const String &name, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added, std::list<std::pair<Point, Point>>& deleted, std::list<std::pair<std::pair<Point, Point>, std::pair<std::vector<int>, std::vector<bool>>>>& straight_edges)
{
    int c, valid_options, repeated;
    bool is_boundary;
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    Point point1, point2, p1, p2;
    std::list<Point> points;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    MyMesh::FaceHandle facehandle;
    bool removed;
    do {
        removed = false;
        for (auto face = mesh.faces_begin(); face != mesh.faces_end(); ++face) {
            c = 0;
            repeated = 0;
            is_boundary = false;
            points.clear();
            for (auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it)
                if (Is_Vertex(mesh, *fv_it, image)) {
                    std::cout << Openmesh_to_opencv(mesh.point(*fv_it)) << " ";
                    c++;
                    points.push_back(Openmesh_to_opencv(mesh.point(*fv_it)));
                    if (Is_Boundary(mesh.point(*fv_it), image)) is_boundary = true;
                }
            std::cout << std::endl;
            if (c >= 10) is_boundary = false;
            if (!is_boundary and c % 2 != 0) {
                valid_options = 0;
                for (auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it)
                    if (Is_Vertex(mesh, *fv_it, image)) {
                        result = edge_structure.equal_range(Openmesh_to_opencv(mesh.point(*fv_it)));
                        for (auto it = result.first; it != result.second; it++) {
                            if (edge_structure.find(it->second.first) == edge_structure.end()) continue;
                            if (it->second.second.second.second.second.second.first > 1) continue;
                            if (Is_Boundary(it->first, image) and Is_Boundary(it->second.first, image)) continue;
                            point1 = it->first;
                            point2 = it->second.first;
                            if (sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2)) > 45) continue;
                            if (sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2)) > 40 and
                                Standard_Deviation(it->second.second.second.second.second.second.second, image_gray) <
                                25)
                                continue;
                            if (Restricted_Removal(mesh, vertices_added.find(point1)->second,
                                                   vertices_added.find(point2)->second, image))
                                continue;
                            if (Restricted_Removal_Disconnected(mesh, vertices_added.find(point1)->second,
                                                                vertices_added.find(point2)->second, image))
                                continue;
                            valid_options++;
                            //if(sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it->second.second.second.second.second.second.second, image_gray)<25) valid_options--;
                            if (std::find(points.begin(), points.end(), point1) != points.end())
                                if (std::find(points.begin(), points.end(), point2) != points.end())
                                    repeated++;
                            p1 = point1;
                            p2 = point2;
                            std::cout << "\tOption: " << valid_options << it->first << it->second.first << std::endl;
                        }
                    }
                valid_options = valid_options - repeated / 2;
                std::cout << "\tOPTIONS: " << valid_options << "\n" << std::endl;
                if (valid_options == 1) {

                    bool is_deleted = Join_Faces(mesh, vertices_added.find(p1)->second,
                                                 vertices_added.find(p2)->second);
                    deleted.emplace_back(p1, p2);
                    if (is_deleted) Visualize(mesh, image, name, deleted);
                    Remove_Edge_Structure(p1, p2, edge_structure);
                    Update_Straight_Edge_Structure(straight_edges, mesh, vertices_added.find(p1)->second,
                                                   vertices_added.find(p2)->second, image, vertices_added);
                    std::cout << "Trivial deletion" << p1 << p2 << std::endl;
                    removed = true;
                    std::cin >> c;
                    break;
                }
            }
        }
    }while(removed);
    // std::cin>>c;
}

bool FixFace(MyMesh mesh, const Mat &image, MyMesh::FaceHandle face, std::list<std::pair<Point, Point>>& fixface, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added);

bool Voting(MyMesh& mesh, std::multimap< Point, bool, ComparePixels >& vertices_pixels, const Mat &image, const String &name, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels >& edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added)
{
    int edges_considered = 0;
    cv::Mat_<Vec3b> image_pruned;
    image.copyTo(image_pruned);
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    std::list<std::pair<Point, Point>> added;
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    Point point1, point2;
    std::list<std::pair<Point, Point>> deleted;
    MyMesh::FaceHandle temp_face1[3], temp_face2[3] , face11, face12, face21, face22;
    int i, face11_count, face12_count, face21_count, face22_count, temp;
    bool add, face11_boundary, face12_boundary, face21_boundary, face22_boundary;
    std::list<std::pair<std::pair<Point, Point>, std::pair<std::vector<int>, std::vector<bool>>>> straight_edges;

    int odd_initial = Count_Odd(mesh, image);
    Trivial_Removal(mesh, image, name, edge_structure, vertices_added, deleted);
    for (auto edge : edge_structure) {

        /*std::cout << "Detected test 0: " << edge.first << edge.second.first << std::endl;
        if (edge.first.x == 1429 or edge.first.x == 1456) std::cin >> i;*/
        if (edge_structure.find(edge.second.first) != edge_structure.end()) {
            if (edge.second.second.second.second.second.second.first > 1) continue;

            /*std::cout << "Detected test 01: " << edge.first << edge.second.first << std::endl;
            if (edge.first.x == 1429 or edge.first.x == 1456) std::cin >> i;*/
            if (Is_Boundary(edge.first, image) and Is_Boundary(edge.second.first, image)) continue;

            /*std::cout << "Detected test 02: " << edge.first << edge.second.first << std::endl;
            if (edge.first.x == 1429 or edge.first.x == 1456) std::cin >> i;*/
            point1 = edge.first;
            point2 = edge.second.first;
            /*std::cout << "Detected test: " << point1 << point2 << std::endl;
            if (point1.x == 1429 or point1.x == 1456) std::cin >> i;*/
            if (sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2)) > 45 /*or (edge.second.second.second.second.second.first<=2 *//*and Standard_Deviation(edge.second.second.second.second.second.second.second, image_gray) < 2)*/) continue;
            if (sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2)) > 40 and
                Standard_Deviation(edge.second.second.second.second.second.second.second, image_gray) < 25)
                continue;
            if (Restricted_Removal(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second,
                                   image))
                continue;
            if (Restricted_Removal_Disconnected(mesh, vertices_added.find(point1)->second,
                                                vertices_added.find(point2)->second, image))
                continue;
            add = true;
            int c = 0;
            /*for(auto t = mesh.vf_iter(vertices_added.find(edge.first)->second); t.is_valid(); ++t) c++;
            if(c == 2 and !Is_Boundary(point1, image)){

                continue;
            }*/
            /*std::cout << "Detected test2: " << point1 << point2 << std::endl;
            if (point1.x == 1429 or point1.x == 1456) std::cin >> i;*/
            for (auto v : deleted)
                if ((v.first == edge.first or v.second == edge.second.first) or
                    (v.first == edge.second.first or v.second == edge.first)) {
                    add = false;
                    std::cout << "Debug" << point1 << point2 << std::endl;
                    //std::cin>>c;
                    break;
                }
            for (auto v : added)
                if ((v.first == edge.first and v.second == edge.second.first) or
                    (v.first == edge.second.first and v.second == edge.first)) {
                    add = false;
                    break;
                }
            if (!add) continue;
            edges_considered++;
            added.emplace_back(point1, point2);
            face11_count = 0;
            face12_count = 0;
            face21_count = 0;
            face22_count = 0, temp = 0;
            face11_boundary = false;
            face12_boundary = false;
            face21_boundary = false;
            face22_boundary = false;

            i = 0;
            for (auto vf = mesh.vf_iter(vertices_added.find(point1)->second); vf.is_valid(); ++vf)
                temp_face1[i++] = *vf;
            // std::cout<<"face1"<<std::endl;
            //std::cin>>i;

            i = 0;
            for (auto vf = mesh.vf_iter(vertices_added.find(point2)->second); vf.is_valid(); ++vf) {
                temp_face2[i++] = *vf;
                for (auto t:temp_face1)
                    if (t == *vf)
                        if (temp++ == 0) face21 = *vf;
                        else face22 = *vf;
            }
            // std::cout<<"face2"<<std::endl;
            //std::cin>>i;

            for (auto t:temp_face1) if (t != face21 and t != face22) face11 = t;
            for (auto t:temp_face2) if (t != face21 and t != face22) face12 = t;


            for (auto fv = mesh.fv_iter(face21); fv.is_valid(); fv++) {
                if (Is_Vertex(mesh, *fv, image)) face21_count++;
                if (Is_Boundary(mesh.point(*fv), image)) face21_boundary = true;
            }
            // std::cout<<"face21"<<std::endl;
            //std::cin>>i;
            for (auto fv = mesh.fv_iter(face22); fv.is_valid(); fv++) {
                if (Is_Vertex(mesh, *fv, image)) face22_count++;
                if (Is_Boundary(mesh.point(*fv), image)) face22_boundary = true;
            }
            // std::cout<<"face22"<<std::endl;
            //std::cin>>i;

            if (!Is_Boundary(point1, image))
                for (auto fv = mesh.fv_iter(face11); fv.is_valid(); fv++) {
                    if (Is_Vertex(mesh, *fv, image)) face11_count++;
                    if (Is_Boundary(mesh.point(*fv), image)) face11_boundary = true;
                }
            // std::cout<<"face11"<<std::endl;
            //std::cin>>i;
            if (!Is_Boundary(point2, image))
                for (auto fv = mesh.fv_iter(face12); fv.is_valid(); fv++) {
                    if (Is_Vertex(mesh, *fv, image)) face12_count++;
                    if (Is_Boundary(mesh.point(*fv), image)) face12_boundary = true;
                }
            // std::cout<<"face12"<<std::endl;
            //std::cin>>i;
            std::cout << point1 << point2 << "Face count 21: " << face21_count << "Face count 22: " << face22_count
                      << "Face count 11: " << face11_count << "Face count 12: " << face12_count << std::endl;
            Point p1 = point1, p2 = point2;
            // Compare_Point(p1, p2);
            std::vector<int> temp_list;
            std::vector<bool> temp_bool_list;
            temp_list.push_back(face21_count);
            temp_list.push_back(face22_count);
            temp_list.push_back(face11_count);
            temp_list.push_back(face12_count);
            temp_bool_list.push_back(face21_boundary);
            temp_bool_list.push_back(face22_boundary);
            temp_bool_list.push_back(face11_boundary);
            temp_bool_list.push_back(face12_boundary);

            if (face21_count % 2 == 0 or face22_count % 2 == 0) {
                straight_edges.emplace_back(std::make_pair(p1, p2), std::make_pair(temp_list, temp_bool_list));
                continue;
            }
            if (!Is_Boundary(point1, image))
                if (face11_count % 2 == 0) {
                    straight_edges.emplace_back(std::make_pair(p1, p2), std::make_pair(temp_list, temp_bool_list));
                    continue;
                }
            if (!Is_Boundary(point2, image))
                if (face12_count % 2 == 0) {
                    straight_edges.emplace_back(std::make_pair(p1, p2), std::make_pair(temp_list, temp_bool_list));
                    continue;
                }
            std::cout << "Detected: " << point1 << point2 << std::endl;
            bool is_deleted = Join_Faces(mesh, vertices_added.find(point1)->second,
                                         vertices_added.find(point2)->second);
            deleted.emplace_back(point1, point2);
            if (is_deleted) Visualize(mesh, image, name, deleted);
            edges_considered--;
            Update_Straight_Edge_Structure(straight_edges, mesh, vertices_added.find(point1)->second,
                                           vertices_added.find(point2)->second, image, vertices_added);
            Visualize(mesh, image, name);
            std::cout << "Level 1: " << point1 << point2 << std::endl;
            std::cin >> i;
            Trivial_Removal(mesh, image, name, edge_structure, vertices_added, deleted, straight_edges);
        }
    }

    std::cout<<"edges: "<<edges_considered<<std::endl;
    std::list<std::vector<unsigned int>> combination_list;
    int odd = 0, odd_boundary = 0;
    int cost = Cost(mesh, image, odd, odd_boundary), temp_cost;
    std::cout<<"Cost: "<<cost<< " odd: "<<odd<<" boundary: "<<odd_boundary<<std::endl;
    int min_cost = cost;
    auto it = straight_edges.begin();
    bool removed;

    for (auto v:straight_edges) {
        if (!Is_Boundary(v.first.first, image) and !Is_Boundary(v.first.second, image))
            temp_cost = Cost(cost, v.second.first, v.second.second, odd, odd_boundary, true);
        std::cout << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                  << v.second.first.at(1)
                  << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                  << std::endl;
    }
    std::cout<<"before1"<<std::endl;

    do {
        removed = false;
        for (auto v:straight_edges) {
            temp_cost = cost;
            if (!Is_Boundary(v.first.first, image) and !Is_Boundary(v.first.second, image))
                temp_cost = Cost(cost, v.second.first, v.second.second, odd, odd_boundary, true);
            std::cout << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                      << v.second.first.at(1)
                      << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                      << std::endl;
            if (temp_cost < min_cost) min_cost = temp_cost;
            if (temp_cost <= ((odd - 2) * (odd - 2)) + odd_boundary) {
                if(v.second.first.at(0)%2 == 0 and v.second.first.at(1)%2 == 0 and v.second.first.at(2)%2==0) continue;
                if(v.second.first.at(0)%2 == 0 and v.second.first.at(1)%2 == 0 and v.second.first.at(3)%2==0) continue;
                if(v.second.first.at(0)%2 == 0 and v.second.first.at(2)%2 == 0 and v.second.first.at(3)%2==0) continue;
                if(v.second.first.at(1)%2 == 0 and v.second.first.at(2)%2 == 0 and v.second.first.at(3)%2==0) continue;
                /*if(v.second.first.at(0)%2 != 0 and v.second.first.at(1)%2 == 0) continue;
                if(v.second.first.at(1)%2 != 0 and v.second.first.at(0)%2 == 0) continue;
                if(v.second.first.at(2)%2 != 0 and v.second.first.at(3)%2 == 0) continue;
                if(v.second.first.at(3)%2 != 0 and v.second.first.at(2)%2 == 0) continue;*/
                bool removable = false;
                if((v.second.first.at(0)%2 != 0 and !v.second.second.at(0)) or (v.second.first.at(1)%2 != 0 and !v.second.second.at(1))
                        or (v.second.first.at(2)%2 != 0 and !v.second.second.at(2)) or (v.second.first.at(3)%2 != 0 and !v.second.second.at(3)))
                    removable = true;
                if(!removable) continue;
                std::cout << "Detected at Level2: " << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                                                    << v.second.first.at(1)
                                                    << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                                                    << std::endl;
                std::cin >> cost;
                bool is_deleted = Join_Faces(mesh, vertices_added.find(v.first.first)->second, vertices_added.find(v.first.second)->second);
                deleted.emplace_back(v.first.first, v.first.second);
                if (is_deleted) Visualize(mesh, image, name, deleted);
                edges_considered--;
                Update_Straight_Edge_Structure(straight_edges, mesh, vertices_added.find(v.first.first)->second,
                                               vertices_added.find(v.first.second)->second, image, vertices_added);
                Visualize(mesh, image, name);
                std::cin >> cost;
                odd = 0;
                odd_boundary = 0;
                cost = Cost(mesh, image, odd, odd_boundary);
                removed = true;
                std::cout<<"starting trivial removal"<<std::endl;
                Trivial_Removal(mesh, image, name, edge_structure, vertices_added, deleted, straight_edges);
                std::cout<<"ending trivial removal"<<std::endl;
                break;
            }
        }
    }while(removed);
    std::cout<<"Min cost: "<<min_cost<<" Cost: "<<cost<<std::endl;

    std::cout<<"after 1"<<std::endl;
    for (auto v:straight_edges) {
        if (!Is_Boundary(v.first.first, image) and !Is_Boundary(v.first.second, image))
            temp_cost = Cost(cost, v.second.first, v.second.second, odd, odd_boundary, true);
        std::cout << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                  << v.second.first.at(1)
                  << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                  << std::endl;
    }

    std::cout<<"\n\nStarting Level 3 Removal: "<<std::endl;
    min_cost = cost;
    /*for (auto v:straight_edges) {
        temp_cost = cost;
        if (!Is_Boundary(v.first.first, image) and !Is_Boundary(v.first.second, image))
            temp_cost = Cost(cost, v.second.first, v.second.second, odd, odd_boundary, false);
        std::cout << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                  << v.second.first.at(1)
                  << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                  << std::endl;
        if (temp_cost < min_cost) min_cost = temp_cost;
    }*/

    do {
        removed = false;
        // break;
        for (auto v:straight_edges) {
            odd = 0;
            odd_boundary = 0;
            cost = Cost(mesh, image, odd, odd_boundary);
            temp_cost = cost;
            if (!Is_Boundary(v.first.first, image) and !Is_Boundary(v.first.second, image))
                temp_cost = Cost(cost, v.second.first, v.second.second, odd, odd_boundary, false);
            std::cout << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                      << v.second.first.at(1)
                      << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                      << std::endl;
            if (temp_cost < min_cost) min_cost = temp_cost;
            if (temp_cost <= ((odd - 2) * (odd - 2)) + odd_boundary + 1) {
                std::cout << "Detected at Level3: " << v.first.first << v.first.second << " face21: " << v.second.first.at(0) << " face22: "
                          << v.second.first.at(1)
                          << " face11: " << v.second.first.at(2) << " face12: " << v.second.first.at(3) << " " << temp_cost
                          << std::endl;
                std::cin >> cost;
                bool is_deleted = Join_Faces(mesh, vertices_added.find(v.first.first)->second, vertices_added.find(v.first.second)->second);
                deleted.emplace_back(v.first.first, v.first.second);
                if (is_deleted) Visualize(mesh, image, name, deleted);
                edges_considered--;
                Update_Straight_Edge_Structure(straight_edges, mesh, vertices_added.find(v.first.first)->second,
                                               vertices_added.find(v.first.second)->second, image, vertices_added);
                Visualize(mesh, image, name);
                std::cin >> cost;
                odd = 0;
                odd_boundary = 0;
                cost = Cost(mesh, image, odd, odd_boundary);
                removed = true;
                Trivial_Removal(mesh, image, name, edge_structure, vertices_added, deleted, straight_edges);
                break;
            } if(removed) break;
        }
    }while(removed);

    std::cout<<"Original odd:"<<odd_initial<<" Min cost: "<<min_cost<<" Cost: "<<cost<< " odd: "<<odd<<" boundary: "<<odd_boundary<<std::endl;

    int vertices;
    bool is_boundary;
    std::list<Point> points;
    std::list<std::pair<Point, Point>> fixface;
    Point p1, p2;
    do {
        removed = false;
        break;
        for(auto face = mesh.faces_begin(); face!=mesh.faces_end(); ++face)
        {
            fixface.clear();
            points.clear();
            vertices = 0;
            is_boundary = false;
            for(auto v_it=mesh.fv_iter(*face); v_it.is_valid(); ++v_it)
            {
                if(Is_Vertex(mesh, *v_it, image)) {
                    points.push_back(Openmesh_to_opencv(mesh.point(*v_it)));
                    vertices++;
                }
                if(Is_Boundary(mesh.point(*v_it), image)) is_boundary = true;
            }
            if(is_boundary or vertices%2 == 0) continue;
            bool fixed;
            if(vertices % 2 !=0){
                int valid_options = 0, repeated = 0;
                fixed = false;
                for(auto fv_it = mesh.fv_iter(*face); fv_it.is_valid(); ++fv_it) if(Is_Vertex(mesh, *fv_it, image)){
                    result = edge_structure.equal_range(Openmesh_to_opencv(mesh.point(*fv_it)));
                    for (auto it2 = result.first; it2 != result.second; it2++){
                        if ( edge_structure.find(it2->second.first) == edge_structure.end()) continue;
                        if ( it2->second.second.second.second.second.second.first > 1) continue;
                        if ( Is_Boundary (it2->first, image) and Is_Boundary (it2->second.first, image)) continue;
                        point1 = it2->first; point2 = it2->second.first;
                        if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 45) continue;
                        if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it2->second.second.second.second.second.second.second, image_gray)<25) continue;
                        if ( Restricted_Removal(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                        if ( Restricted_Removal_Disconnected(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                        valid_options++;
                        //if(sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it->second.second.second.second.second.second.second, image_gray)<25) valid_options--;
                        if(std::find(points.begin(), points.end(), point1) != points.end())
                            if(std::find(points.begin(), points.end(), point2) != points.end())
                                repeated++;
                        p1 = point1;
                        p2 = point2;
                        std::cout<<"\tOption: "<<valid_options<<it2->first<<it2->second.first<<std::endl;
                        fixface.clear();
                        fixed = FixFace(mesh, image_gray, *face, fixface, edge_structure, vertices_added);
                        if(fixed){
                            std::pair<Point, Point> p = fixface.back();
                            Join_Faces(mesh, vertices_added.find(p.first)->second, vertices_added.find(p.second)->second);
                            Visualize(mesh, image, name);
                        }
                        break;
                    }
                }if (fixed) break;
                valid_options = valid_options-repeated/2;
                std::cout<<"\tOPTIONS: "<<valid_options<<"\n"<<std::endl;
            }
        }
    }while(removed);
    return true;
}

bool Try_Remove(MyMesh mesh, Point point1, Point point2, const Mat &image, MyMesh::FaceHandle face, std::list<std::pair<Point, Point>>& fixface, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added){
    Join_Faces(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second);
    int count;
    for(auto vf = mesh.vf_iter(vertices_added.find(point1)->second); vf.is_valid(); ++vf){
        count = 0;
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv){
            if(Is_Vertex(mesh, *fv, image)) count++;
        }
        if(count%2!=0) return false;
    }

    for(auto vf = mesh.vf_iter(vertices_added.find(point2)->second); vf.is_valid(); ++vf){
        count = 0;
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv){
            if(Is_Vertex(mesh, *fv, image)) count++;
        }
        if(count%2!=0) return false;
    }

    return true;
}


bool FixFace(MyMesh mesh, const Mat &image, MyMesh::FaceHandle face, std::list<std::pair<Point, Point>>& fixface, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added){
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    Point point1, point2;
    bool remove;
    for(auto fv_it = mesh.fv_iter(face); fv_it.is_valid(); ++fv_it) if(Is_Vertex(mesh, *fv_it, image)){
            result = edge_structure.equal_range(Openmesh_to_opencv(mesh.point(*fv_it)));
            remove = false;
            for (auto it = result.first; it != result.second; it++){
                if (edge_structure.find(it->second.first) == edge_structure.end()) continue;
                if ( it->second.second.second.second.second.second.first > 1) continue;
                if ( Is_Boundary (it->first, image) and Is_Boundary (it->second.first, image)) continue;
                point1 = it->first; point2 = it->second.first;
                if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 45) continue;
                if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it->second.second.second.second.second.second.second, image)<25) continue;
                if ( Restricted_Removal(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                if ( Restricted_Removal_Disconnected(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                remove = Try_Remove(mesh, point1, point2, image, face, fixface, edge_structure, vertices_added);
                if (remove) {
                    std::cout<<"Successful remove: "<<point1<<point2<<std::endl;
                    fixface.emplace_back(point1, point2);
                    break;
                }
            }
            if(remove) break;
    }
}

bool Try_Remove2(MyMesh mesh, Point point1, Point point2, const Mat &image, MyMesh::FaceHandle face, std::list<std::pair<Point, Point>>& fixface, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added, int& c){
    Join_Faces(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second);
    c++;
    if(c==2) return false;
    int count;
    for(auto vf = mesh.vf_iter(vertices_added.find(point1)->second); vf.is_valid(); ++vf){
        count = 0;
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv){
            if(Is_Vertex(mesh, *fv, image)) count++;
        }
        if(count%2!=0) {
            bool test_remove = Try_Remove2(mesh, point1, point2, image, face, fixface, edge_structure, vertices_added, count);
            if (test_remove) continue;
            return false;
        }
    }

    for(auto vf = mesh.vf_iter(vertices_added.find(point2)->second); vf.is_valid(); ++vf){
        count = 0;
        for(auto fv = mesh.fv_iter(*vf); fv.is_valid(); ++fv){
            if(Is_Vertex(mesh, *fv, image)) count++;
        }
        if(count%2!=0) return false;
    }

    return true;
}

//void remove_strucutre_order();

bool FixFace2(MyMesh mesh, const Mat &image, MyMesh::FaceHandle face, std::list<std::pair<Point, Point>>& fixface, std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure, std::map< Point, MyMesh::VertexHandle, ComparePixels >& vertices_added){
    std::pair<MMAPEdgeIterator, MMAPEdgeIterator> result;
    Point point1, point2;
    bool remove;
    for(auto fv_it = mesh.fv_iter(face); fv_it.is_valid(); ++fv_it) if(Is_Vertex(mesh, *fv_it, image)){
            result = edge_structure.equal_range(Openmesh_to_opencv(mesh.point(*fv_it)));
            remove = false;
            int count = 0;
            for (auto it = result.first; it != result.second; it++){
                if (edge_structure.find(it->second.first) == edge_structure.end()) continue;
                if ( it->second.second.second.second.second.second.first > 1) continue;
                if ( Is_Boundary (it->first, image) and Is_Boundary (it->second.first, image)) continue;
                point1 = it->first; point2 = it->second.first;
                if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 46) continue;
                if ( sqrt(pow((point1.x-point2.x),2)+pow((point1.y-point2.y),2)) > 40 and Standard_Deviation(it->second.second.second.second.second.second.second, image)<25) continue;
                if ( Restricted_Removal(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                if ( Restricted_Removal_Disconnected(mesh, vertices_added.find(point1)->second, vertices_added.find(point2)->second, image)) continue;
                remove = Try_Remove2(mesh, point1, point2, image, face, fixface, edge_structure, vertices_added, count);
                if (remove) {
                    std::cout<<"Successful remove: "<<point1<<point2<<std::endl;
                    fixface.emplace_back(point1, point2);
                    break;
                }
            }
            if(remove) break;
        }
}
/****************************************|| EDGE_THINNING FUNCTION STARTS HERE ||***************************************
 *
 * @param image
 * @param mask
 * @param min_areas
 * @param save_images
 * @param name
 *
 * This function is called from distance_transform.cpp for using distance transform for edge thinning
 *
 * @return
 */
bool Distance_Transform (Mat const& image, Mat_<bool>& mask, std::vector<int>& min_areas, bool& save_images, std::string name, std::string output_folder)
{
    std::vector<int> num_vertices( min_areas.size() ), num_domains( min_areas.size() ), num_walls( min_areas.size() );
    int i = 0;
    cv::Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    Mat bin = cv::imread( output_folder + "related3_skeleton.png", CV_LOAD_IMAGE_COLOR );
    //Mat bin2 = cv::imread( output_folder + "FC 100x2_3_binary_skeleton_paper.png", CV_LOAD_IMAGE_COLOR );
   /* int t = 0;
    for ( int row = 0; row < bin2.rows; row++ )
        for ( int col = 0; col < bin2.cols; col++ )
            if(bin2.at<Vec3b>(row, col) == green)
                t++;
    std::cout<<"Vertices: "<<t<<std::endl;*/
    cv::Mat_<Vec3b> temp/*(image.size(), white)*/;
        image.copyTo(temp);
    /*for ( int row = 0; row < bin.rows; row++ )
        for ( int col = 0; col < bin.cols; col++ )
            if(bin.at<Vec3b>(row, col) == black)
            {
                Point p = Point(col, row);
                for(auto i : shifts4){
                    if((p+i).x >= 0 and (p+i).y >= 0 and (p+i).y < bin.rows and (p+i).x < bin.cols)
                    if(bin.at<Vec3b>(p+i) != black)
                    {
                        temp.at<Vec3b>(p+i) = black;
                        break;
                    }
                }
                temp.at<Vec3b>(row, col) = black;
            }
    cv::imwrite(  name + "with_symmetry_two_blue_white.png" , temp );

    cv::imshow("", bin);
    cv::waitKey();*/
    //Count_Componenets(bin, image_gray);

    Mat_<Vec3b> image_mask;
    Mat_<Vec3b> binary_skeleton;

    // Remove small objects
    //Remove_Small_Components( true, 100, mask, num_walls[i], false ); // true means foreground
    Remove_Small_Components( false, 70, mask, num_domains[i], true ); // true means foreground
    //if ( save_images ) image_mask = Save_Binary( image, mask, name + "_connected_true" + std::to_string( min_areas[i] ) + ".png" );
    //if ( save_images ) binary_skeleton = Save_Binary( image, mask, name + "_connected_true" + std::to_string( min_areas[i] ) + ".png" );
    if ( false ) binary_skeleton = Save_Binary( image, mask, name + "_holes" + std::to_string( min_areas[i] ) + ".png" );
    // Remove small holes

    //Remove_Small_Components( true, 50, mask, num_domains[i], true ); // true means foreground
    image_mask =Save_Binary( image, mask, name + "_connected" + std::to_string( min_areas[i] ) + ".png" );
    binary_skeleton = Save_Binary( image, mask, name + " Extended Otsu thresholding output" + std::to_string( min_areas[i] ) + ".png" );

    int m = 1;
    bool check;
    std::cout<<" Thinning edges..."<<std::endl;
    // for smoothing the first layer
    Mark_External_Pixels(mask, image_mask, white, red);
    for ( int row = 1; row < mask.rows-1; row++ )
        for ( int col = 1; col < mask.cols-1; col++ )
        {
            if(image_mask.at<Vec3b>(row, col) == red){
                image_mask.at<Vec3b>(row, col) = white;
                mask.at<bool>(row, col) = false;
            }
        }
    // Mark pixels with white neighbours
    std::cout<<"  progress: [";
    do{
        if(m%5) std::cout<<"\033[7m \033[0m"<< std::flush;
        Mark_External_Pixels(mask, image_mask, white, red);
        Mark_External_Pixels(mask, image_mask, red, green);
        // if ( save_images ) cv::imwrite( name + "_distance" + std::to_string( m ) + ".png" , image_mask );
        int c = 0;
        check = Remove_Layer(mask, image_mask, name, m, c);
        //Remove_Small_Components( true, 100, mask, num_walls[i], image_mask ); // true means foreground
        // if ( save_images ) cv::imwrite( name + "_distance" + std::to_string( m+1 ) + ".png" , image_mask );
        Reset_Mask(image_mask);
        m++;
    }while (check);

    // average intensity of the skeleton
    int counter = 0; float total = 0;
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            if(mask.at<bool>(row, col))
            {
                counter++;
                total += int(image_gray.at<uchar>( row, col ));
            }
        }


    do{
        check = false;
        Boundry_points(mask, image_mask, check);
    }while(check);
    while(m<50){
        if(m%5) std::cout<<"\033[7m \033[0m";
        m++;
    }
    std::cout<<"]"<<std::endl;

    std::cout<<total<<std::endl;
    std::cout<<counter<<std::endl;
    std::cout<<total/counter<<std::endl;

    if ( true ) cv::imwrite( name + " Black skeleton with symmetric thinning on white background.png" , image_mask );
    if ( false ) Save_Mask( image, mask, name + "_distance_mask" + std::to_string( min_areas[i] ) + ".png" );



    std::multimap< Point, bool, ComparePixels > vertices_pixels;
    std::multimap< Point, int, ComparePixels > all_pixels;
    Mat_<Vec3b> image_vertex = Mark_Vertex(mask, image, binary_skeleton, vertices_pixels);

    if(false)cv::imwrite(  name + "_vertex_loopless.png" , binary_skeleton );

    //std::cout<<" Removing one vertex loops..."<<std::endl;
    //std::cin>>total;

    Remove_Loops(image, binary_skeleton, image_vertex, mask, vertices_pixels, name);
    std::list<Point> loop_list = Get_Disconnected(mask, binary_skeleton);

    //std::cout<<"done"<<std::endl;
    std::list<std::list<Point>> remove_structure2;
    /*for(auto list : removed_structre){
        for(auto p : list)
            std::cout<<p<<" ";
        std::cout<<std::endl;
    }*/
    for(auto list = removed_structre.begin(); list != removed_structre.end(); list++){
        if((*list).empty()) continue;
        unsigned long size = list->size();
        std::list<Point> point_list;
        bool s = false;
        point_list.push_back((*list).front());
        Point front = (*list).front();
        Point last = list->back(), first = list->front();
        list->erase(list->begin());
        int size2 = 0;
        //std::cout<<"starting"<<front<<" ";
        while(!list->empty()) {
            for (auto p = (*list).begin(); p != (*list).end(); p++) {
                if (abs(p->x - front.x) <= 1 and abs(p->y - front.y) <= 1 and (*p) != front) {
                    std::cout << *p << " ";
                    if (front == last) {
                        s = true;
                        std::cout<<"last "<<*p;
                        point_list.push_back(*p);
                        remove_structure2.push_back(point_list);
                        last = Point(0, 0);
                        std::cout << std::endl;
                        point_list.clear();
                        point_list.push_back(*p);
                        std::cout<<"\n\nnew first: "<<point_list.front()<<std::endl;
                    }
                    size2++;
                    //if(!s)
                        point_list.push_back(*p);
                    /*else
                        point_list2.push_back(*p);*/
                    front = *p;
                    list->erase(p);
                    break;
                }
            }
        }
        point_list.push_back(first);
        std::cout<<first<<std::endl;

        remove_structure2.push_back(point_list);
    }
/*
    for(auto list : remove_structure2){

    }*/

    // std::cout<<"don e"<<std::endl;
    for(auto list : remove_structure2){
        std::list<Point> point_list;
        int breaks = 0;
        DouglasPeucker(image_vertex, list.front(), list.back(), list, remove_loops_points, 3, point_list, breaks);
        for(auto p : list){
            std::cout<<p<<" ";
            temp.at<Vec3b>(p) = green;
        }
        std::cout<<std::endl;
    }
    //cv::imwrite(  name + "removed_structure.png" , temp );

    //std::cout<<"points decided"<<std::endl;
    /*for(auto point : remove_loops_points)
    {
        std::cout<<point.first<<point.second<<std::endl;
    }*/
    /*std::cin>>total;*/


    cv::Mat_<Vec3b> binary_skeleton2;
    binary_skeleton.copyTo(binary_skeleton2);
    cv::Mat_<Vec3b> image_straight_paper;
    image.copyTo(image_straight_paper);
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if(binary_skeleton2.at<Vec3b>(row, col) == green or binary_skeleton2.at<Vec3b>(row, col) == red)
            {
                binary_skeleton2.at<Vec3b>(row, col) = red;
                image_straight_paper.at<Vec3b>(row, col) = black;
            }
    //cv::imwrite(  name + "_vertex_loopless.png" , binary_skeleton );
    cv::imwrite(  name + " Red skeleton on thresholded image.png" , binary_skeleton2 );
    cv::imwrite(  name + " Black skeleton with symmetric thinning on original image.png" , image_straight_paper );

    //std::cin>>total;
    //std::cout<<"remove structure"<<std::endl;
    //std::cin>>total;

    // The function close_vertex will find erroneous in vertices and draw a boundary around them
    // Close_Vertex_removal(mask, image_vertex, binary_skeleton, image, name);

    // This function will mark all edges that form a saddle
    // Saddle_Removal(image, mask, image_vertex, binary_skeleton, vertices_pixels, name, all_pixels);

    // This function will get all the vertex pairs and store them in vertex_pairs
    // This function will perform douglas-peucker algorithm to straighten the edges
    cv::Mat_<Vec3b> image_straight;
    image.copyTo(image_straight);
    std::multimap< Point, std::pair<Point, std::pair<Point, Point>>, ComparePixels > vertex_pairs;
    std::multimap< Point, std::pair<Point, std::pair<Point, Point>>, ComparePixels > reverse_vertex_pairs;
    std::multimap< Point, std::pair<Point, std::pair<std::list<Point>, std::pair<Point, std::pair<Point, std::pair<int, std::pair<int, std::list<Point>>>>>>>, ComparePixels > edge_structure;
    std::map< Point, MyMesh::VertexHandle, ComparePixels > vertices_added;
    MyMesh mesh;
    int starting_face = Vertex_Pairs(image, binary_skeleton, image_vertex, image_straight, mesh, vertices_pixels, vertex_pairs, reverse_vertex_pairs, name, loop_list, edge_structure, vertices_added);
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            mask.at<bool>(row, col) = image_straight.at<Vec3b>(row, col) == black;
    if ( save_images ) Save_Mask( image, mask, name + "_distance_mask_straight" + std::to_string( min_areas[i] ) + ".png" );

    //remove_strucutre_order();
    std::cout<<"Finishing...."<<std::endl;
    Visualize(mesh, image, name);
    //cv::imwrite(  name + " Original image (input).png" , image );
    std::cout<<"Finished...."<<std::endl;
    //std::cin>>total;
    //Mark_Vertex(image_straight, vertices_pixels);

    /*if ( false ){
        cv::imwrite(  name + "_skeleton_original_straight.png" , image_straight );
        cv::imwrite(  name + "_skeleton_original.png" , image_vertex );
        cv::imwrite(  name + "_skeleton_binary.png" , binary_skeleton );
    }*/

    /*try
    {
        if ( !OpenMesh::IO::write_mesh(mesh, name + "_output.off") )
        {
            std::cerr << "Cannot write mesh to file 'output.off'" << std::endl;
            return true;
        }
    }
    catch( std::exception& x )
    {
        std::cerr << x.what() << std::endl;
        return true;
    }*/

    // Converting to boost graph
    //Image_to_BG (image_straight, vertex_pairs);

    //std::cout<<"\n Prunning faces..."<<std::endl;
    //Edge_Parameters(edge_structure, image, name, vertices_pixels);
    //Voting(mesh, vertices_pixels, image, name, edge_structure, vertices_added);
    //std::cout<<"\n Finished: "<<name.substr(name.find_last_of('/') + 1)<<std::endl;

    /*try
    {
        if ( !OpenMesh::IO::write_mesh(mesh, name + "_output_pruned.off") )
        {
            std::cerr << "Cannot write mesh to file 'output_pruned.off'" << std::endl;
            return true;
        }
    }
    catch( std::exception& x )
    {
        std::cerr << x.what() << std::endl;
        return true;
    }*/

    return true;
}
