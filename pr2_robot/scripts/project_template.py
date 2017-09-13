#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Helper function to do Statistical Outlier Filtering
def statistical_outlier_filter(cloud, k=50, x=1.0):
    # we start by creating a filter object
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(k)

    # Any point with a mean distance larger than
    # global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    return cloud_filtered


def voxel_grid_downsampler(cloud, leaf_size=0.01):
    # Voxel Grid Downsampling

    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Set the voxel (or leaf) size
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)

    # Call the filter function to obtain the resultant downsampled
    # point cloud
    cloud_filtered = vox.filter()

    return cloud_filtered


def passthrough_filter(cloud, filter_axis='z', axis_min=0.6, axis_max=1.1):
    # PassThrough Filter

    # Create a PassThrough filter object.
    passthrough = cloud.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    return cloud_filtered


def ransac_plane_segmentor(cloud, max_distance=0.01):
    # RANSAC Plane Segmentation

    # create the segmentation object
    seg = cloud.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and
    # model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers

    # Extract inliers
    cloud_table = cloud.extract(inliers, negative=False)

    # Extract outliers
    cloud_objects = cloud.extract(inliers, negative=True)

    return cloud_table, cloud_objects


def get_colorful_euclidean_clusters(objects,
                                    tolerance=0.012,
                                    min_size=50,
                                    max_size=50000):
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold as well as minimum and maximum
    # cluster size (in points)
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    return white_cloud, cluster_indices, cluster_cloud


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    # cloud_no_noise = statistical_outlier_filter(pcl_cloud, k=50, x=1.0)

    # TODO: Voxel Grid Downsampling
    cloud_downsampled = voxel_grid_downsampler(pcl_cloud,
                                               leaf_size=0.0027)

    # TODO: PassThrough Filter
    cloud_passthrough = passthrough_filter(cloud_downsampled,
                                           filter_axis='z',
                                           axis_min=0.6,
                                           axis_max=1.1)

    cloud_passthrough = passthrough_filter(cloud_passthrough,
                                           filter_axis='x',
                                           axis_min=0.4,
                                           axis_max=1.7)
    # TODO: RANSAC Plane Segmentation
    # TODO: Extract inliers and outliers
    cloud_table, cloud_objects = ransac_plane_segmentor(
                                     cloud_passthrough, max_distance=0.01)

    # TODO: Euclidean Clustering
    # TODO: Create Cluster-Mask Point Cloud to visualize each
    #       cluster separately
    white_cloud, cluster_indices, cluster_cloud = \
            get_colorful_euclidean_clusters(cloud_objects,
                                            tolerance=0.012,
                                            min_size=50,
                                            max_size=50000)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 TODOs:

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    # loop through each detected cluster one at a time
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        # ros_cluster is of type PointCloud2
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        # retrieve the label for the result
        label = encoder.inverse_transform(prediction)[0]
        # and add it to detected_objects_labels list
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(
        len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def get_pose(x, y, z):
    new_pose = Pose()
    new_pose.position.x = float(x)
    new_pose.position.y = float(y)
    new_pose.position.z = float(z)
    return new_pose


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # TODO: Initialize variables
    dict_list = []
    world_num = {3: 1, 5: 2, 8: 3}

    # TODO: Get/Read parameters
    object_list_params = rospy.get_param('/object_list')
    dropbox_params = rospy.get_param('/dropbox')

    test_scene_num = Int32()
    test_scene_num.data = world_num[len(object_list_params)]

    # TODO: Parse parameters into individual variables
    # making a new dict, use the name as the key to get group as the value
    object_param_dict = {d['name']: d['group'] for d in object_list_params}
    # use the group from above as a key in this dict
    dropbox_param_dict = {d['group']: (d['name'], d['position'])
                          for d in dropbox_params}

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for object_ in object_list:
        #  if object_.label not in object_param_dict:
            #  continue
        object_name = String()
        object_name.data = str(object_.label)

        object_group = object_param_dict[object_.label]
        dropbox_name, dropbox_pos = dropbox_param_dict[object_group]

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pcl_array = ros_to_pcl(object_.cloud).to_array()
        centroid = np.mean(pcl_array, axis=0)[:3]
        pick_pose = get_pose(*centroid)

        # TODO: Create 'place_pose' for the object
        place_pose = get_pose(*dropbox_pos)

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = dropbox_name

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        #  try:
            #  pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            #  # TODO: Insert your message variables to be sent as a service request
            #  resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            #  print ("Response: ", resp.success)

        #  except rospy.ServiceException, e:
            #  print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    yaml_filename = "output_" + str(world_num[len(object_list_params)]) + ".yaml"
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':
    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points",
                               pc2.PointCloud2, pcl_callback,
                               queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects",
                                      PointCloud2,
                                      queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table",
                                    PointCloud2,
                                    queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster",
                                      PointCloud2,
                                      queue_size=1)

    # TODO: here you need to create two publishers
    # Call them object_markers_pub and detected_objects_pub
    # Have them publish to "/object_markers" and "/detected_objects" with
    # Message Types "Marker" and "DetectedObjectsArray" , respectively
    object_markers_pub = rospy.Publisher("/object_markers",
                                         Marker,
                                         queue_size=1)

    detected_objects_pub = rospy.Publisher("/detected_objects",
                                           DetectedObjectsArray,
                                           queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

