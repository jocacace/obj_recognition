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

#define WHILE_LOOP              (0)


using namespace std;
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;


class OBJ_DETECTION {
    public: 
        OBJ_DETECTION();
        void run();
        void cloudCB(const sensor_msgs::PointCloud2& input);
        void recognition();
        void save_ds();

    private:
        ros::NodeHandle _nh;
        ros::Subscriber _depth_sub;
        pcl::PointCloud<pcl::PointXYZ> _cloud;
        bool _first_cloud;
        string _dataset_path;
        bool _save_ds;
        double _ss;
        double _descr_rad;
        double _cg_size;
        double _cg_thresh;

           
             
};


OBJ_DETECTION::OBJ_DETECTION() {
    _depth_sub = _nh.subscribe("/camera/depth_registered/points", 1, &OBJ_DETECTION::cloudCB, this);
    _first_cloud = false;


    //Load params
    string dataset_name;
    if( !_nh.getParam("dataset_name", dataset_name) ) {
        dataset_name =  "TESTBED";
    }

    if( !_nh.getParam("save_ds", _save_ds) ) {
        _save_ds =  false;
    }


    if( !_nh.getParam("ss", _ss) ) {
        _ss =  0.01;
    }
    if( !_nh.getParam("desc_rad", _descr_rad) ) {
        _descr_rad =  0.1;
    }
    if( !_nh.getParam("cg_size", _cg_size) ) {
        _cg_size =  0.015;
    }
    if( !_nh.getParam("cg_thresh", _cg_thresh) ) {
        _cg_thresh =  5.0;
    }


    _dataset_path = ros::package::getPath("obj_recognition");
    _dataset_path = _dataset_path + "/DATASET/TESTBED/";
}



void OBJ_DETECTION::cloudCB(const sensor_msgs::PointCloud2& input) {
    pcl::fromROSMsg(input, _cloud);
    _first_cloud = true;
}


void OBJ_DETECTION::recognition() {


    while ( !_first_cloud ) {
        usleep(0.1*1e6);
    }
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    ros::Rate r(10);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<NormalType>::Ptr ds_model_normal (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<NormalType>::Ptr model_normal (new pcl::PointCloud<NormalType> ());

    pcl::PointCloud<PointType>::Ptr ds_model (new pcl::PointCloud<PointType> ());

    std::vector < pcl::PointCloud<PointType>::Ptr > ds_models;
    std::vector < pcl::PointCloud<NormalType>::Ptr > ds_model_normals;
    ds_models.push_back( ds_model );
    ds_model_normals.push_back( ds_model_normal );

    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    std::vector < pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> > descrs_est;
    descrs_est.push_back( descr_est );

    //temp: carico solo 1 modello:
    if (pcl::io::loadPCDFile ( "/home/jcacace/dev/ros_ws/src/DIH2/obj_recognition/DATASET/TESTBED/p1.pcd" , *ds_models[0]) < 0) {
        std::cout << "Error loading model cloud." << std::endl;
        exit (0);
    }

    //view processed cloud
    pcl::visualization::PCLVisualizer viewer ("Dataset p1");
    viewer.addPointCloud ( ds_models[0], "p1");
    viewer.spinOnce ();

    //  Compute Normals
    std::vector < pcl::NormalEstimationOMP<PointType, NormalType> > ds_norms_est;
    pcl::NormalEstimationOMP<PointType, NormalType> ds_norm_est;
    std::vector < pcl::UniformSampling<PointType> > uniforms_sampling;
    pcl::UniformSampling<PointType> uniform_sampling;
    pcl::PointCloud<PointType>::Ptr ds_model_keypoints (new pcl::PointCloud<PointType> ());
    std::vector < pcl::PointCloud<PointType>::Ptr > ds_models_keypoints; // (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<DescriptorType>::Ptr ds_model_descriptors (new pcl::PointCloud<DescriptorType> ());
    std::vector < pcl::PointCloud<DescriptorType>::Ptr > ds_models_descriptors;

    ds_norms_est.push_back(ds_norm_est);
    uniforms_sampling.push_back ( uniform_sampling );
    ds_models_keypoints.push_back( ds_model_keypoints );
    ds_models_descriptors.push_back( ds_model_descriptors );

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    cout << "Objects into the dataset: " << ds_models.size() << endl;
    for(int i=0; i<ds_models.size(); i++ ) {
        cout << "Elaborating Object: " << i+1 << endl;

        ds_norms_est[i].setKSearch (10);
        ds_norms_est[i].setInputCloud (  ds_models[i]  );
        ds_norms_est[i].compute (*ds_model_normals[i]);

        uniforms_sampling[i].setInputCloud (ds_models[i]);
        uniforms_sampling[i].setRadiusSearch (_ss);
        uniforms_sampling[i].filter (*ds_models_keypoints[i] );
        std::cout << "Scene total points: " << ds_models[i]->size () 
            << "; Selected Keypoints: " << ds_models_keypoints[i]->size () << std::endl;
    

        //---Visualize keypoints
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        pcl::transformPointCloud (*ds_models_keypoints[i], *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*ds_models_keypoints[i], *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
        viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (ds_models_keypoints[i], 0, 0, 255);
        viewer.addPointCloud (ds_models_keypoints[i], scene_keypoints_color_handler, "scene_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");
        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
        viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
        viewer.spinOnce ();
        //---

        descrs_est[i].setRadiusSearch  (_descr_rad);
        descrs_est[i].setInputCloud    (ds_models_keypoints[i]);
        descrs_est[i].setInputNormals  (ds_model_normals[i] );
        descrs_est[i].setSearchSurface (ds_models[i]);
        descrs_est[i].compute          (*ds_models_descriptors[i]);    
    }

    pcl::KdTreeFLANN<DescriptorType> match_search;


    while ( ros::ok() ) {
        
        cout << "Press enter!" << endl;
        string ln;
        getline(cin, ln);

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (_cloud.makeShared());

        kdtree.setInputCloud (_cloud.makeShared());
        std::vector<int> nn_indices (1);
        std::vector<float> nn_dists (1);

        kdtree.nearestKSearch(pcl::PointXYZ(0, 0, 0), 1, nn_indices, nn_dists);
      

        if( !isnan( _cloud.points[nn_indices[0]].x) || !isnan(_cloud.points[nn_indices[0]].y) || !isnan(_cloud.points[nn_indices[0]].z) ) {
           
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (_cloud.points[nn_indices[0]].z - 10.0, _cloud.points[nn_indices[0]].z + 10.02);
            pass.filter (*cloud_filtered);


 //view processed cloud
    pcl::visualization::PCLVisualizer viewer ("cloud filter");
    viewer.addPointCloud ( cloud_filtered, "p1");
    viewer.spinOnce ();


            /*MATCH HERE!!! 
            
                ds_***: oggetti elaborati nel dataset            
            */
            pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
            pcl::PointCloud<NormalType>::Ptr cloud_normals (new pcl::PointCloud<NormalType> ());
            pcl::UniformSampling<PointType> unisampling;
            pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
            pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());

            norm_est.setKSearch (10);
            norm_est.setInputCloud (cloud_filtered);
            norm_est.compute (*cloud_normals);
        
            unisampling.setInputCloud (cloud_filtered);
            unisampling.setRadiusSearch (_ss);
            unisampling.filter (*model_keypoints);
            std::cout << "Model total points: " << cloud_filtered->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;




        pcl::transformPointCloud (*model_keypoints, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
        viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (model_keypoints, 0, 0, 255);
        viewer.addPointCloud (model_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");
        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
        viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
        viewer.spinOnce ();




            pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descriptors_est;
            descriptors_est.setRadiusSearch (_descr_rad);
            descriptors_est.setInputCloud (model_keypoints);
            descriptors_est.setInputNormals (cloud_normals);
            descriptors_est.setSearchSurface (cloud_filtered);
            descriptors_est.compute (*model_descriptors);

            match_search.setInputCloud ( model_descriptors );

            for( int m=0; m<ds_models.size(); m++ ) {


                pcl::CorrespondencesPtr model_corrs (new pcl::Correspondences ());
                for (std::size_t i = 0; i < ds_models_descriptors[m]->size(); ++i) {
                    std::vector<int> neigh_indices (1);
                    std::vector<float> neigh_sqr_dists (1);
                    if (!std::isfinite (ds_models_descriptors[m]->at (i).descriptor[0])) //skipping NaNs
                        continue;
                    
                    int found_neighs = match_search.nearestKSearch (ds_models_descriptors[m]->at (i), 1, neigh_indices, neigh_sqr_dists);
                    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) {
                        pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
                        model_corrs->push_back (corr);
                    }
               
                }
                std::cout << "Correspondences found: " << model_corrs->size () << std::endl;



                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
                std::vector<pcl::Correspondences> clustered_corrs;
                pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
                pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

                pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
                float rf_rad_ (0.015f);
                rf_est.setFindHoles (true);
                rf_est.setRadiusSearch (rf_rad_);

                rf_est.setInputCloud (model_keypoints);
                rf_est.setInputNormals (cloud_normals);
                rf_est.setSearchSurface (cloud_filtered);
                rf_est.compute (*model_rf);

                rf_est.setInputCloud (  ds_models_keypoints[m] ); 
                rf_est.setInputNormals (  ds_model_normals[m] ); 
                rf_est.setSearchSurface (  ds_models[m] ); 
                rf_est.compute (*scene_rf);



                //  Clustering
                pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
                clusterer.setHoughBinSize (_cg_size);
                clusterer.setHoughThreshold (_cg_thresh);
                clusterer.setUseInterpolation (true);
                clusterer.setUseDistanceWeight (false);

                clusterer.setInputCloud (  model_keypoints);
                clusterer.setInputRf (model_rf);
                clusterer.setSceneCloud ( ds_models_keypoints[m] );
                clusterer.setSceneRf (scene_rf);
                clusterer.setModelSceneCorrespondences (  model_corrs );

                //clusterer.cluster (clustered_corrs);
                clusterer.recognize (rototranslations, clustered_corrs);

                //
                //  Output results
                //
                std::cout << "Model instances found: " << rototranslations.size () << std::endl;
                for (std::size_t i = 0; i < rototranslations.size (); ++i)
                {
                    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
                    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

                    // Print the rotation matrix and translation vector
                    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
                    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

                    printf ("\n");
                    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
                    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
                    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
                    printf ("\n");
                    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
                }


            }       


        }
        r.sleep();
    }
}

void OBJ_DETECTION::save_ds () {


    while ( !_first_cloud ) {
        usleep(0.1*1e6);
    }
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    ros::Rate r(10);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    string ln;
    ln = "";
    while ( ln != "e" ) {      
        cout << "Save dataset function. Press s to save a new 3d Model, press e to exit" << endl;
        getline(cin, ln);

        if( ln == "e" ) exit(1);

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (_cloud.makeShared());

        kdtree.setInputCloud (_cloud.makeShared());
        std::vector<int> nn_indices (1);
        std::vector<float> nn_dists (1);

        kdtree.nearestKSearch(pcl::PointXYZ(0, 0, 0), 1, nn_indices, nn_dists);


        if( !isnan( _cloud.points[nn_indices[0]].x) || !isnan(_cloud.points[nn_indices[0]].y) || !isnan(_cloud.points[nn_indices[0]].z) ) {
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (_cloud.points[nn_indices[0]].z - 0.02, _cloud.points[nn_indices[0]].z + 0.02);
            pass.filter (*cloud_filtered);

            //view processed cloud
            pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
            viewer.addPointCloud (cloud_filtered, "scene_cloud");
            viewer.spinOnce ();

            cout << "Do you like the acquired model? [Y/n]" << endl;
            getline(cin, ln);
            if( ln == "Y" || ln == "" ) {
                cout << "Insert model name: " << endl;
                getline(cin, ln);
                pcl::io::savePCDFileASCII ( _dataset_path + ln + ".pcd"  , *cloud_filtered);
            }
        }
        else 
            cout << "Error detecting the surface, please, try again"<< endl;
    }
}

void OBJ_DETECTION::run() {
    if (_save_ds) 
        boost::thread save_ds_t( &OBJ_DETECTION::save_ds, this);
    else
        boost::thread recognition_t( &OBJ_DETECTION::recognition, this);
    
    ros::spin();
}


int main( int argc, char ** argv ) {

    ros::init(argc, argv, "obj_detection");
    OBJ_DETECTION detection;
    detection.run();

    return 0;
}