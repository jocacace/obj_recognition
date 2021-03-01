#include "ros/ros.h"
#include <ros/package.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/thread.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "boost/thread.hpp"
#include "sensor_msgs/CameraInfo.h"
#include <ros/package.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <filesystem>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "std_msgs/Bool.h"
#include "obj_recognition/recognized_object.h"

#define WHILE_LOOP              (0)

std::mutex mtx;         

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

string winname = "recognition";
class OBJ_DETECTION {

    public: 
        OBJ_DETECTION();
        void run();

        //Main functions
        void td2d();
        void save_ds();

        //callbacks
        void depth_cb( sensor_msgs::ImageConstPtr depth );
        void cloudCB(const sensor_msgs::PointCloud2& input);
        void cam_parameters( sensor_msgs::CameraInfo );
        void start_cb( std_msgs::Bool );
        void stop_cb( std_msgs::Bool );
        void reset_cb( std_msgs::Bool );

    private:

        ros::NodeHandle _nh;
        
        //---Input sub
        ros::Subscriber _depth_sub;
        ros::Subscriber _camera_info_sub;
        ros::Subscriber _depth_img_sub;
        ros::Subscriber _start_recognition_sub;
        ros::Subscriber _stop_recognition_sub;
        ros::Subscriber _reset_recognition_sub; //??we need this?
        //---

        //output
        ros::Publisher _recognized_obj_pub;

        pcl::PointCloud<pcl::PointXYZ> _cloud;
           
        //---Flags
        bool _cam_info_first;
        bool _first_cloud;
        bool _depth_ready;
        bool _start_recognition;
        bool _stop_recognition;
        bool _reset_recognition;
        //---

        //---camera parameters
        cv::Mat *_cam_cameraMatrix, *_cam_distCo, *_cam_R, *_cam_P;
        //---
     
        Mat _depth_src;
     
        //---Parameters
        string _points_topic;
        string _depth_topic;
        string _info_topic;
        bool _save_ds;
        string _dataset_path;
        double _camera_frame_rate;
        double _s;
        int _rate;
        int _detection_itr_threshold;
        double _detection_threshold;
        int _cycle_itr_threshold;
        //---
             
};


OBJ_DETECTION::OBJ_DETECTION() {


    if( !_nh.getParam("points_topic", _points_topic) ) {
        _points_topic =  "/camera/depth_registered/points";
    }
    if( !_nh.getParam("depth_topic", _depth_topic) ) {
        _depth_topic =  "/camera/aligned_depth_to_color/image_raw";
    }
    if( !_nh.getParam("info_topic", _info_topic) ) {
        _info_topic =  "/camera/aligned_depth_to_color/camera_info";
    }

    if( !_nh.getParam("camera_framerate", _camera_frame_rate )) {
        _camera_frame_rate = 15.0;
    }
    if ( !_nh.getParam("s", _s) ) {
        _s = 2.0;
    }
    if ( !_nh.getParam( "rate", _rate )) {
        _rate = 15;
    }
    if ( !_nh.getParam("detection_itr_threshold", _detection_itr_threshold)) {
        _detection_itr_threshold = 10;
    }

    _depth_sub = _nh.subscribe( _points_topic.c_str(), 1, &OBJ_DETECTION::cloudCB, this);
    _depth_img_sub = _nh.subscribe( _depth_topic.c_str(), 1, &OBJ_DETECTION::depth_cb, this );
    _camera_info_sub = _nh.subscribe( _info_topic.c_str(), 1, &OBJ_DETECTION::cam_parameters, this );

    _start_recognition_sub = _nh.subscribe( "/obj_recognition/start", 1, &OBJ_DETECTION::start_cb, this);
    _stop_recognition_sub = _nh.subscribe( "/obj_recognition/start", 1, &OBJ_DETECTION::stop_cb, this);
    _reset_recognition_sub = _nh.subscribe( "/obj_recognition/start", 1, &OBJ_DETECTION::reset_cb, this); //Do we need this?
    _recognized_obj_pub = _nh.advertise< obj_recognition::recognized_object >("/obj_recognition/recognized", 1);
    
    //---Load params
    string dataset_name;
    if( !_nh.getParam("dataset_name", dataset_name) ) {
        dataset_name =  "TESTBED";
    }

    if( !_nh.getParam("save_ds", _save_ds) ) {
        _save_ds =  false;
    }
    if( !_nh.getParam("detection_threshold", _detection_threshold )) {
        _detection_threshold = 0.05; //Score
    }
    if( !_nh.getParam("cycle_itr_threshold", _cycle_itr_threshold)) {
        _cycle_itr_threshold = 100;
    }
    //---
    
    _dataset_path = ros::package::getPath("obj_recognition");
    _dataset_path = _dataset_path + "/DATASET/" + dataset_name + "/";
    _depth_ready = false;
    _cam_info_first = true;
    _start_recognition = false;
    _stop_recognition = false;
    _reset_recognition = false;
    _first_cloud = false;
    _cycle_itr_threshold = 0;

}

void OBJ_DETECTION::start_cb( std_msgs::Bool s ) {
    _start_recognition = s.data;
}
void OBJ_DETECTION::stop_cb( std_msgs::Bool s ) {
    _stop_recognition = s.data;
}
void OBJ_DETECTION::reset_cb( std_msgs::Bool res) {    
    _reset_recognition = res.data;
}

//save camera parameters in openCV structs
void OBJ_DETECTION::cam_parameters( sensor_msgs::CameraInfo camera_info) {

    /*
     *  ROS topic data
     *  K = cameraMatrix
     *  D = distCoeffs
     *  R = Rettification
     *  P = Projection
     */


    if( _cam_info_first == true ) {

        ROS_INFO("Start camera parameters initialization...");
        //---resize calibration matrix
        _cam_cameraMatrix = new cv::Mat(3, 3, CV_64FC1);
        _cam_distCo = new cv::Mat(1, 5, CV_64FC1);
        _cam_R = new cv::Mat(3, 3, CV_64FC1);
        _cam_P = new cv::Mat(3, 4, CV_64FC1);
        //---

        //---K
        for(int i=0; i<3;i++) {
            for(int j=0; j<3; j++) {
                _cam_cameraMatrix->at<double>(i,j) = camera_info.K[3*i+j];

                cout << "[" << i << ", " << j << "]: " << _cam_cameraMatrix->at<double>(i,j) << endl;
            }
        }
        //---D
				if( camera_info.D.size() >= 5 ) {
	        for(int i=0; i<5;i++) {
            _cam_distCo->at<double>(0,i) = camera_info.D[i];
  	      }
				}
        //---R
        for(int i=0; i<3;i++) {
            for(int j=0; j<3; j++) {
                _cam_R->at<double>(i,j) = camera_info.R[3*i+j];
            }
        }
        //---P
        for(int i=0; i<3;i++) {
            for(int j=0; j<4; j++) {
                _cam_P->at<double>(i,j) = camera_info.P[4*i+j];
            }
        }
        _cam_info_first = false;

        ROS_INFO("...camera parameters initialization complete!");
    }

}

void OBJ_DETECTION::cloudCB(const sensor_msgs::PointCloud2& input) {
    pcl::fromROSMsg(input, _cloud);
    _first_cloud = true;
}


void OBJ_DETECTION::depth_cb( sensor_msgs::ImageConstPtr depth ) {
    mtx.lock(); // Inizio zona critica
    
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
        _depth_src = cv_ptr->image;
        _depth_ready = true;

        //imshow("d img", _depth_src);
        //waitKey(1);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mtx.unlock();
}

void tokenize(string &str, char delim, vector<string> &out) {
	size_t start;
	size_t end = 0;

	while ((start = str.find_first_not_of(delim, end)) != string::npos) {
		end = str.find(delim, start);
		out.push_back(str.substr(start, end - start));
	}
}

void OBJ_DETECTION::td2d() {

    //---Wait input data: they must be a continous stream
    while( !_depth_ready ) {
        usleep(0.1*1e6);
    }
    ROS_INFO("Depth data arrived!");

    while ( !_first_cloud ) {
        usleep(0.1*1e6);
    }
    ROS_INFO("PointCloud data arrived!");
   
    while( _cam_info_first ) {
        usleep(0.1*1e6);
    }
    ROS_INFO("CameraInfo data arrived!");
    //---

    ros::Rate r(_rate);

    double cx, cy, fx_inv, fy_inv;
    double zd_c1;    
    double cx_c1, cy_c1, cz_c1;

    int dilation_size = 5;
    int dilation_type = 1;
    int dilation_elem = 1;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( dilation_type,
                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        Point( dilation_size, dilation_size ) );

 
    while( ros::ok() ) {

        string p_piece = "";
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

        Mat depth;
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (_cloud.makeShared());

        vector< double > min_dist;
        int min_itr = 0;
        int cycle_itr = 0;

        bool min_found = false;
        double min_cloud = 0.0;
        bool first_cnt = false;
        bool first = true;  
        int pieces_itr = 0;     

        mtx.lock();
        _depth_src.copyTo(depth);
        cv::Mat contourImage(depth.size(), CV_8UC3, cv::Scalar(0,0,0)); 
        mtx.unlock();      
        cv::Scalar colors;
        colors = cv::Scalar(255, 255, 255);
        bool _recognition_done = false;

        //Wait for a signal?
        cout << "[Press enter to start recognition process]" << endl;
        string ll;
        getline(cin, ll);
        destroyAllWindows();

        while( !_recognition_done ) {
            if ( !min_found ) {
                while( min_itr < _s*_camera_frame_rate ) {
                    kdtree.setInputCloud (_cloud.makeShared());
                    std::vector<int> nn_indices (1);
                    std::vector<float> nn_dists (1);
                    kdtree.nearestKSearch(pcl::PointXYZ(0, 0, 0), 1, nn_indices, nn_dists);

                    if( !isnan( _cloud.points[nn_indices[0]].z ) ) {
                        min_dist.push_back( _cloud.points[nn_indices[0]].z );
                        min_itr++;


                    }
                }

                double nmin = 0.0;
                for(int i=0; i<min_dist.size(); i++) {
                    nmin += min_dist[i];
                }
                nmin /= min_dist.size();

                min_cloud = 0.0;
                int norma_size = 0;

                for(int i=0; i<min_dist.size(); i++) {
                    if( min_dist[i] <= nmin ) {
                        min_cloud += min_dist[i];
                        norma_size++;
                    } 
                }

                min_cloud /= norma_size;

                min_found = true;

            } //Searching for the minimum value
            //else cout << "Non cerco piu il min!" << endl;

            mtx.lock();
            _depth_src.copyTo(depth);
            mtx.unlock();      

            Mat img(depth.rows, depth.cols, CV_8UC1, Scalar(0));
            double z_min = min_cloud;
                
            for(int i=0; i<depth.rows; i++) {
                for(int j=0; j<depth.cols; j++)  {
                    float d = depth.at<float>(i,j);
                    d*= 0.001;
                    if(  ( ( d ) >=  z_min - 0.03)  && (( d ) <= z_min + 0.03 ) ) {
                        img.at<char>(i,j) = 255; 
                    }    
                }
            }

            Mat dilation_dst;
            dilate( img, dilation_dst, element );        
            cv::Mat contour1Image(depth.size(), CV_8UC3, cv::Scalar(0,0,0)); 
            std::vector<std::vector<cv::Point> > contours1;
            cv::Mat contour1Output = dilation_dst.clone();
            cv::findContours( contour1Output, contours1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
            
            for (size_t idx = 0; idx < contours1.size(); idx++) {
                cv::drawContours(contour1Image, contours1, idx, colors);
            }


            Mat dst1;
            cvtColor(contour1Image, dst1, cv::COLOR_BGR2GRAY);

            vector< string > pieces;
            vector< double > scores;
            for (const auto & entry : fs::directory_iterator(_dataset_path)) {
                string p = entry.path();
                contourImage = imread(p, IMREAD_COLOR);

                vector<string> a;
                vector<string> b;
                tokenize(p,'/',a);
                tokenize(a[ a.size()-1 ],'_', b);                
                pieces.push_back( b[ 0 ] );
                
                Mat dst;
                cvtColor(contourImage, dst, cv::COLOR_BGR2GRAY);

                double d1 = matchShapes(dst, dst1, CONTOURS_MATCH_I1, 0);

                scores.push_back(d1);
            }

            int index = -1;
            double min_score = 1000;
            for(int i=0; i<scores.size(); i++ ) {
                if( min_score > scores[i] ) {
                    min_score = scores[i];
                    index = i;
                }
            }
        
            vector<vector<Point> > contours_poly( contours1.size() );
            vector<Rect> boundRect( contours1.size() );
            vector<Point2f>centers( contours1.size() );
            vector<float>radius( contours1.size() );
            
            for( size_t i = 0; i < contours1.size(); i++ ) {
                approxPolyDP( contours1[i], contours_poly[i], 3, true );
                boundRect[i] = boundingRect( contours_poly[i] );
            }
            
            for( size_t i = 0; i< contours1.size(); i++ ) {
                drawContours( dst1, contours_poly, (int)i, colors );
                rectangle( dst1, boundRect[i].tl(), boundRect[i].br(), colors, 2 );
            }
            
            //We want t o detect only 1 shape... it this true?            
            if( boundRect.size() == 1 ) {
                if( pieces[index] == p_piece ) {
                    pieces_itr++;
                }
                else {
                    p_piece = pieces[index];
                    pieces_itr = 0;
                }
            }


            if( pieces_itr >= _detection_itr_threshold ) {           
                Point center_of_rect = (boundRect[0].br() + boundRect[0].tl())*0.5;
                circle( dst1, center_of_rect, 3, colors, 2 );        

                cx = _cam_cameraMatrix->at<double>(0,2);
                cy = _cam_cameraMatrix->at<double>(1,2);
                fx_inv = 1.0 / _cam_cameraMatrix->at<double>(0,0);
                fy_inv = 1.0 / _cam_cameraMatrix->at<double>(1,1);
                zd_c1 = depth.at<float>( center_of_rect.y, center_of_rect.x );
                zd_c1 *= 0.001;
                cx_c1 = (zd_c1) * ( (center_of_rect.x - cx) * fx_inv );
                cy_c1 = (zd_c1) * ( (center_of_rect.y - cy) * fy_inv );
                cz_c1 = zd_c1;

                //cout << "3d Point: (" << cx_c1 << ", " << cy_c1 << ", " << cz_c1 << ")" << endl; 
                //cout << "Pezzo: " << pieces[index] << " - " << scores[index] << endl;

                string text = pieces[index] + " (" + to_string(cx_c1) + ", " + to_string(cy_c1) + ", " + to_string( cz_c1 ) + ")";
                cv::putText(dst1, //target image
                    text, //text
                    cv::Point(10, img.rows / 6), //top-left position
                    cv::FONT_HERSHEY_TRIPLEX,
                    0.5,
                    colors, //font color
                    1);

                imshow( winname, dst1 );
                waitKey(100);
                
                obj_recognition::recognized_object obj;
                obj.piece.data = pieces[index];
                obj.center.x = cx_c1;
                obj.center.y = cy_c1;
                obj.center.z = cz_c1;
                _recognized_obj_pub.publish(obj);
                
                _recognition_done = true;
            }

            cycle_itr++;
            if ( cycle_itr > _cycle_itr_threshold ) {
                cycle_itr = 0;
                min_found = false;
            }
            r.sleep();
        }
    }
}



//TODO: fix: image is not updated
void OBJ_DETECTION::save_ds () {

   while( !_depth_ready ) {
        usleep(0.1*1e6);
    }
    while ( !_first_cloud ) {
        usleep(0.1*1e6);
    }
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    ros::Rate r(15);

    Mat depth;    
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (_cloud.makeShared());

    int s = 2;


    string ln;
    ln = "";

    while ( ln != "e" ) {      

        cout << "Save dataset function. Press s to save a new 3d Model, press e to exit" << endl;
        getline(cin, ln);
        destroyAllWindows();


        if( ln == "e" ) exit(1);

        vector< double > min_dist;
        int min_itr = 0;
        bool min_found = false;
        double min_cloud = 0.0;
        bool first_cnt = false;

      
        while( min_itr < _s*_camera_frame_rate ) {
            kdtree.setInputCloud (_cloud.makeShared());
            std::vector<int> nn_indices (1);
            std::vector<float> nn_dists (1);
            kdtree.nearestKSearch(pcl::PointXYZ(0, 0, 0), 1, nn_indices, nn_dists);

            if( !isnan( _cloud.points[nn_indices[0]].z ) ) {
                min_dist.push_back( _cloud.points[nn_indices[0]].z );
                min_itr++;
            }
        }
      
        double nmin = 0.0;
        for(int i=0; i<min_dist.size(); i++) {
            nmin += min_dist[i];
        }
        nmin /= min_dist.size();

        min_cloud = 0.0;
        int norma_size = 0;

        for(int i=0; i<min_dist.size(); i++) {
            if( min_dist[i] <= nmin ) {
                min_cloud += min_dist[i];
                norma_size++;
            } 
        }

        min_cloud /= norma_size;
        min_found = true;

        mtx.lock();
        _depth_src.copyTo(depth);
        mtx.unlock();      
        Mat img(depth.rows, depth.cols, CV_8UC1, Scalar(0));
        double z_min = min_cloud;
    
        mtx.lock();
        _depth_src.copyTo(depth);
        cv::Mat contourImage(depth.size(), CV_8UC3, cv::Scalar(0,0,0)); 
        mtx.unlock();      
      
        for(int i=0; i<depth.rows; i++) {
            for(int j=0; j<depth.cols; j++)  {
                float d = depth.at<float>(i,j);
                d*= 0.001;
                if(  ( ( d ) >=  z_min - 0.03)  && (( d ) <= z_min + 0.03 ) ) {
                    img.at<char>(i,j) = 255; 
                }    
            }
        }

    
        Mat dilation_dst;
        int dilation_size = 5;
        int dilation_type = 1;
        int dilation_elem = 1;
        if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
        else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
        else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
        Mat element = getStructuringElement( dilation_type,
                            Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                            Point( dilation_size, dilation_size ) );
        dilate( img, dilation_dst, element );

        cv::Mat invSrc =  cv::Scalar::all(255) - dilation_dst;
        std::vector<std::vector<cv::Point> > contours;

        cv::Mat contourOutput = dilation_dst.clone();
        cv::findContours( contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );

        cv::Scalar colors;
        colors = cv::Scalar(255, 255, 255);
        for (size_t idx = 0; idx < contours.size(); idx++) {
            cv::drawContours(contourImage, contours, idx, colors);
        }
        
        cv::imshow("OriginalContours", contourImage);
        cv::waitKey(100);
        Moments  M = moments( contours[0] );

        /*
        cout << "Area: " << M.m00 << endl;
        //cout << "Momenti: " << M.mu11 << " " << M.mu20 << " " << M.mu02 << endl;
        cout << "main axies moment: " << (0.5*atanf( (2*M.mu11) / (M.mu20 - M.mu02) ))*180.0/3.1415 << endl;
        */

        cout << "Do you like the acquired model? [Y/n]" << endl;
        getline(cin, ln);
        if( ln == "Y" || ln == "" ) {
            cout << "Insert object type: " << endl;
            getline(cin, ln);
            
            int num = 0;  
            for (const auto & entry : fs::directory_iterator(_dataset_path)) {
                //std::cout << entry.path() -  _dataset_path << std::endl;
                string p = entry.path();
                if ( p.find(ln) != std::string::npos) {
                    num++;
                }
            }
            imwrite( _dataset_path + ln + "_" + to_string(num) + ".jpg", contourImage);
            
            contourImage.release();
            
        } 
        r.sleep();
    }
}

void OBJ_DETECTION::run() {

    if (_save_ds) 
        boost::thread save_ds_t(&OBJ_DETECTION::save_ds, this );
    else
        boost::thread td2d_t(&OBJ_DETECTION::td2d, this );
        
    ros::spin();
}


int main( int argc, char ** argv ) {

    ros::init(argc, argv, "obj_detection");
    OBJ_DETECTION detection;
    detection.run();

    return 0;
}