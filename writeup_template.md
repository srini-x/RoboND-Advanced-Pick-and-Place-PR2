## Project: Perception Pick & Place
Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify).
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

#### Exercise 1, 2 and 3 pipeline implemented

The exercises 1, 2, and 3 provide a good understanding of the concepts requred
to complete thid project. Below, I talk about the implementation of the exercises.
I mostly used the code provided in the lessons. I will mention the places where
some modifications were made.

#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The main goal of this exercise is Filtering and Segmentation. A RGB-D point
cloud is captured by a sensor_stick in ROS. We use various techniques to filter
out the unwanted points and divide the whole point cloud into objects on the
table and table top.

**Steps to  complete Exercise 1**

- Downsample your point cloud by applying a Voxel Grid Filter.
This step reduces the points in the point cloud without loosing out the
essential information required to identify them. I used `leaf_size = 0.01`.

- Apply a Pass Through Filter to isolate the table and objects.
I used 2 Pass Through Filters, one along `z-axis` and the other along `y-axis`.

```
filter_axis='z', axis_min=0.6, axis_max=1.1
filter_axis='y', axis_min=-2.5, axis_max=-1.4
```

- Perform RANSAC plane fitting to identify the table and Use the ExtractIndices
Filter to create new point clouds containing the table and objects separately.
Code for RANSAC plane fitting is implemented in the function
`ransac_plane_segmentor(cloud, max_distance=0.01):`. Parameter to use in this
function is `max_distance=0.01`. This function separates the point cloud into
an objects cloud and a table cloud.


#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

**This exercise has following top level goals:**

- Create publishers and topics to publish the segmented table and tabletop
objects as separate point clouds

- Apply Euclidean clustering on the table-top objects (after table segmentation
is successful)

- Create a XYZRGB point cloud such that each cluster obtained from the previous
step has its own unique color.

- Finally publish your colored cluster cloud on a separate topic

**Steps to  complete Exercise 2**

- create a ros publisher that publishes to the topic "/pcl_objects".
create this in the main method.

```python
pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
```

- convert pcl cloud data to ros message. data needs to be in the ros message
  format to publish.

```python
ros_cloud_objects = pcl_to_ros(cloud_objects)
```

- publish the ros message.

```python
pcl_objects_pub.publish(ros_cloud_objects)
```

- Statistical Outlier Filtering. This code is implemented in the
  `statistical_outlier_filter(cloud, k=50, x=1.0)` function.

- Euclidean Clustering. Create Cluster-Mask Point Cloud to visualize each
  cluster separately. `get_colorful_euclidean_clusters` function has the code
  to perform Euclidean Clustering. Parameters used are

```python
tolerance=0.012, min_size=50, max_size=50000
```

- Publish the colored point clusters. If everything works, points from each
  object will belong to a cluster and will be in a unique color.

```python
ros_cluster_cloud = pcl_to_ros(cluster_cloud)
pcl_cluster_pub.publish(ros_cluster_cloud)
```


#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

The goal of this section is to identify the object in each of the colored point
clusters.

**Steps to  complete Exercise 2**

- loop through each detected cluster one at a time. the variable
  `cluster_indices` contains a list of point lists. we go through each point
  list and identify the object in it.

- Compute the associated feature vector. compute color histograms and normal
  histograms and join them to get the complete feature vector.

```python
chists = compute_color_histograms(ros_cluster, using_hsv=True)
normals = get_normals(ros_cluster)
nhists = compute_normal_histograms(normals)
feature = np.concatenate((chists, nhists))
```

- Make the prediction, retrieve the label for the result, and add it to
  detected_objects_labels list.

```python
prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
label = encoder.inverse_transform(prediction)[0]
detected_objects_labels.append(label)
```

- Add the detected object to the list of detected objects.

```python
detected_objects.append(do)
```

- At the end of the loop, publish the list of detected objects.

```python
detected_objects_pub.publish(detected_objects)
```


Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

And here's another image!
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

### Pick and Place Setup

The goal of the project is similar to that of Exercise 3. We need to recognize
all the objects in the world and label them. One extra step is to write a .yaml
file with group and location infornation. Also there are 3 different worlds
with different objects in each world.

Following points summerize the goals for the project.

1. For all three tabletop setups (`test*.world`), perform object recognition
2. read in respective pick list (`pick_list_*.yaml`).
3. construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

#### Steps to complete the project

Two major things to solve in this project are:

1. Generate a good `model.sav` using the `capture_features.py`.

I made following changes to the `capture_features.py`:

- modify the `models` variable to match all the objects in the 3rd world.
  Because the world 3 has all the objects from the other worlds, a model
  trained on world 3 will work on all worlds. I increased the number of samples
  to `1000` and number of attempts to `10`. `500` and `5` for number
  of samples and attempts produced similar accuracy results.

2. create the `.yaml` file.

The code for this step is in the `pr2_mover(object_list)` method.

- get the label from the identified object. Use the label as the key to obtain
  the `object_group` which again can be used as a key to obtain the corresponding `dropbox_name`
  and `dropbox_pos`.

```python
object_group = object_param_dict[object_.label]
dropbox_name, dropbox_pos = dropbox_param_dict[object_group]
```

- An array of point cloud data corresponding to the object can be obtained and
  computing a mean of those points will give us the `centroid` of the object.

```python
pcl_array = ros_to_pcl(object_.cloud).to_array()
centroid = np.mean(pcl_array, axis=0)[:3]
pick_pose = get_pose(*centroid)
```

- I created a function `get_pose()` to create a `Pose`. Used this to create
  `pick_pose` and `place_pose`.

- After obtaining all the required information, `make_yaml_dict` method can be
  used to create a yaml dictionary. A yaml dict for each object is added to
  a list.

- Using the function `send_to_yaml` the list of yaml dicts can be written to a
  `.yaml` file.


#### Improvements

- An obvious improvement for the project is to incrase the accuracy of the
  object recognition. This can be done by increasing the size of the feature
  vector. Right now, the feature vector only combines the `hsv` colorspace and
  the norm vectors. Adding `ypbpr` space can improve the prediction accuracy.

