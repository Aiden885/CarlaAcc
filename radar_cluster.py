#!/usr/bin/env python

import numpy as np
from sklearn.cluster import DBSCAN
class RadarClusterNode:
    def __init__(self):
        # rospy.init_node('radar_cluster_node', anonymous=True)
        
        # Parameters
        self.eps = 1.5 #rospy.get_param('~eps', 1.5)  # DBSCAN radius
        self.min_samples =  3#rospy.get_param('~min_samples', 3)  # Min points for a cluster
        
        # # Publisher for cluster centers as PointCloud2
        # self.cluster_pub = rospy.Publisher('/radar_clusters', PointCloud2, queue_size=10)
        
        # # Subscriber for radar PointCloud2
        # self.radar_sub = message_filters.Subscriber('/carla/ego_vehicle/radar', PointCloud2)
        
        # # Time synchronizer
        # self.ts = message_filters.TimeSynchronizer([self.radar_sub], queue_size=10)
        # self.ts.registerCallback(self.radar_callback)
        self.points = []
        
    # def radar_callback(self, radar_msg):
    #     # Extract points from PointCloud2
        
        
    #     for point in pc2.read_points(radar_msg, field_names=("x", "y", "z", "vx", "vy", "vz"), skip_nans=True):
    #         x, y, z, vx, vy, vz = point
    #         self.points.append([x, y, z, vx, vy, vz])
        
    #     if not self.points:
    #         rospy.logwarn("No valid points received")
    #         return
        
    #     cluster_points = self.radar_cluster(self.points)
    #     self.pub_cluster(cluster_points, radar_msg)
    
    def radar_cluster(self, radar_points):
        if radar_points is None:
            return
        points = np.array(radar_points)[:, :3]
        velocities = np.array(radar_points)[:, 3:6]
        # Apply DBSCAN clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = db.labels_
        
        # Prepare cluster points for PointCloud2
        cluster_points = []
        unique_labels = set(labels) - {-1}  # Exclude noise points
        
        for label in unique_labels:
            cluster_points_mask = labels == label
            cluster_points_data = points[cluster_points_mask]
            cluster_velocities = velocities[cluster_points_mask]
            
            # Calculate cluster center and mean velocity
            center = np.mean(cluster_points_data, axis=0)
            mean_velocity = np.mean(cluster_velocities, axis=0)
            
            # Calculate width, length, height (bounding box)
            min_point = np.min(cluster_points_data, axis=0)
            max_point = np.max(cluster_points_data, axis=0)
            w, l, h = max_point - min_point
            
            # Append center point with mean velocity
            cluster_points.append([center[0], center[1], center[2], w, l, h, mean_velocity[0], mean_velocity[1], mean_velocity[2]])

        if not cluster_points:
            return None
        
        return cluster_points

    # def pub_cluster(self, cluster_points, radar_msg):
    #     # Create PointCloud2 message
    #     fields = [
    #         pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='w', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='l', offset=16, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='h', offset=20, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='vx', offset=24, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='vy', offset=28, datatype=pc2.PointField.FLOAT32, count=1),
    #         pc2.PointField(name='vz', offset=31, datatype=pc2.PointField.FLOAT32, count=1)
    #     ]
        
    #     header = Header()
    #     header.stamp = radar_msg.header.stamp
    #     header.frame_id = radar_msg.header.frame_id
        
    #     cluster_cloud = pc2.create_cloud(header, fields, cluster_points)
        
    #     # Publish clusters
    #     self.cluster_pub.publish(cluster_cloud)
    #     rospy.loginfo(f"Published {len(cluster_points)} cluster centers")

def main(data):
    cluster = RadarClusterNode()
    return cluster.radar_cluster(data)

if __name__ == '__main__':
    main()