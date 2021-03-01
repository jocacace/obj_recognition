# obj_recognition

This package contains the obj_recognition ROS node. The goal of this package is to allow the recognition of a set of objects based on their shape (contours). This package must be used with a depth sensor (like the Realsense 435d or similar). In the rest of this README we consider that you are using an Intel Realsense. 

### Install the package (pre-requirements)
-   [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
-   [Realsense Lib](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

Then, you can install ROS package to support realsense cameras:

	$ sudo apt-get install ros-noetic-realsense2-*

### Run the package
On three different terminals launch the following commands:

- start the roscore:	 
       
      $ roscore
   
- launch the depth sensor node:
      
      $ roslaunch obj_recognition rs_rgbd.launch 

 - launch the recognition node:

       $ roslaunch obj_recognition depth_recognition.launch

Now the system is ready to work. The system works in two modalities: object recognition and training set generation. To select the mode you have to configure the package using the configuration file.

### Node configuration

The main configuration file is located in the conf/ directory and by default it is called recognition.yaml. Here you can find the following parameters:
-	points_topic: the topic on which the point cloud are streamed
-	depth_topic: the topic on which the depth image matrix is streamed
-	info_topic: the topic in which the calibration of the camera is streamed
-	dataset_name: the name of the training set to generate/use
-	save_ds: set this parameter to True to add elements to the training set, False to recognize the objects

### Output
The node provides its output using a ROS message of type: obj_recognition/recognized_object. In this way it is possible to publish the type of the object and its 3D position in camera frame.

	std_msgs/String piece
		string data
	geometry_msgs/Point center
		float64 x
		float64 y
		float64 z
### Limitation

The node is not able to calculate the orientation of the recognized object.

