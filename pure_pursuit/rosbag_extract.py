"""
import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
import tf_transformations  # To handle quaternion to euler conversion

# Function to extract x, y, and theta positions from any Odometry topic in a ROS 2 bag
def extract_odom_positions_ros2(bag_path, topic_name):
    positions = []  # List to store x, y, and theta positions
    timestamps = []  # List to store timestamps
    
    # Open the ROS 2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and types
    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}

    # Check if the topic exists in the bag file
    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag file.")

    # Read messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            # Deserialize data using the Odometry message type
            msg = deserialize_message(data, Odometry)
            
            # Extract position and orientation (quaternion)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
            # Convert quaternion to Euler angles (yaw, pitch, roll)
            _, _, theta = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            # Append the extracted position and orientation along with timestamp
            timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds
            positions.append((x, y, theta))
            timestamps.append(timestamp_sec)

    if not positions:
        print(f"No data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return timestamps, positions


# Function to compute MAE and RMSE
def compute_error(real_positions, deformed_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    deformed_x, deformed_y, deformed_theta = zip(*deformed_positions)

    # Compute MAE and RMSE for X, Y, and Theta
    mae_x = np.mean(np.abs(np.array(real_x) - np.array(deformed_x)))
    rmse_x = np.sqrt(np.mean((np.array(real_x) - np.array(deformed_x)) ** 2))

    mae_y = np.mean(np.abs(np.array(real_y) - np.array(deformed_y)))
    rmse_y = np.sqrt(np.mean((np.array(real_y) - np.array(deformed_y)) ** 2))

    mae_theta = np.mean(np.abs(np.array(real_theta) - np.array(deformed_theta)))
    rmse_theta = np.sqrt(np.mean((np.array(real_theta) - np.array(deformed_theta)) ** 2))

    return {
        "MAE_X": mae_x, "RMSE_X": rmse_x,
        "MAE_Y": mae_y, "RMSE_Y": rmse_y,
        "MAE_Theta": mae_theta, "RMSE_Theta": rmse_theta
    }


# Function to plot the trajectories with landmark IDs
def plot_trajectories(real_positions, ekf_positions, landmarks_ids, landmarks_x, landmarks_y):
    real_x, real_y, _ = zip(*real_positions)
    ekf_x, ekf_y, _ = zip(*ekf_positions)

    plt.figure(figsize=(10, 8))

    # Plot the real trajectory
    plt.plot(real_x, real_y, label='/odom Trajectory', color='red', linestyle='-', linewidth=1)

    # Plot the EKF trajectory
    plt.plot(ekf_x, ekf_y, label='/ekf Trajectory', color='blue', linestyle='--', linewidth=1)

    # Plot landmarks
    plt.scatter(landmarks_x, landmarks_y, label='Landmarks', color='black', marker='x', s=50)

    # Annotate each landmark with its ID
    for i, (x, y, landmarks_id) in enumerate(zip(landmarks_x, landmarks_y, landmarks_ids)):
        plt.text(x+0.05, y+0.05, f'Landmark {str(landmarks_id)}', fontsize=8, color='red')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    plt.title('Trajectory Comparison')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Function to plot x, y, and theta over time
def plot_x_y_theta(timestamps_real, real_positions, timestamps_ekf, ekf_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    ekf_x, ekf_y, ekf_theta = zip(*ekf_positions)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot X position over time
    axs[0].plot(timestamps_ekf, ekf_x, label='/ekf/pose/pose/position/x', color='blue', linestyle='-', linewidth=1)
    axs[0].plot(timestamps_real, real_x, label='/odom/pose/pose/position/x', color='red', linestyle='-', linewidth=1)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='upper right')
    axs[0].set_title('X Position Over Time')
    axs[0].set_ylabel('X (m)')
    axs[0].set_xlabel('Time (s)')

    # Plot Y position over time
    axs[1].plot(timestamps_ekf, ekf_y, label='/ekf/pose/pose/position/y', color='blue', linestyle='-', linewidth=1)
    axs[1].plot(timestamps_real, real_y, label='/odom/pose/pose/position/y', color='red', linestyle='-', linewidth=1)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Y Position Over Time')
    axs[1].set_ylabel('Y (m)')
    axs[1].set_xlabel('Time (s)')

    # Plot Theta (orientation) over time
    axs[2].plot(timestamps_ekf, ekf_theta, label='/ekf/pose/orientation/yaw', color='blue', linestyle='-', linewidth=1)
    axs[2].plot(timestamps_real, real_theta, label='/odom/pose/orientation/yaw', color='red', linestyle='-', linewidth=1)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend(loc='upper right')
    axs[2].set_title('Theta (Orientation) Over Time')
    axs[2].set_ylabel('Theta (rad)')
    axs[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()


from scipy.interpolate import interp1d

# Function to interpolate EKF data to real timestamps
def interpolate_ekf_data(timestamps_real, timestamps_ekf, ekf_positions):
    # Extract x, y, and theta from EKF positions
    ekf_x, ekf_y, ekf_theta = zip(*ekf_positions)

    # Create interpolation functions for x, y, and theta
    interp_x = interp1d(timestamps_ekf, ekf_x, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(timestamps_ekf, ekf_y, kind='linear', fill_value="extrapolate")
    interp_theta = interp1d(timestamps_ekf, ekf_theta, kind='linear', fill_value="extrapolate")

    # Interpolate the EKF data at real timestamps
    ekf_x_interp = interp_x(timestamps_real)
    ekf_y_interp = interp_y(timestamps_real)
    ekf_theta_interp = interp_theta(timestamps_real)

    # Return interpolated EKF data
    return list(zip(ekf_x_interp, ekf_y_interp, ekf_theta_interp))

# Function to compute MAE and RMSE
def compute_error(real_positions, deformed_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    deformed_x, deformed_y, deformed_theta = zip(*deformed_positions)

    # Compute MAE and RMSE for X, Y, and Theta
    mae_x = np.mean(np.abs(np.array(real_x) - np.array(deformed_x)))
    rmse_x = np.sqrt(np.mean((np.array(real_x) - np.array(deformed_x)) ** 2))

    mae_y = np.mean(np.abs(np.array(real_y) - np.array(deformed_y)))
    rmse_y = np.sqrt(np.mean((np.array(real_y) - np.array(deformed_y)) ** 2))

    mae_theta = np.mean(np.abs(np.array(real_theta) - np.array(deformed_theta)))
    rmse_theta = np.sqrt(np.mean((np.array(real_theta) - np.array(deformed_theta)) ** 2))

    return {
        "MAE_X": mae_x, "RMSE_X": rmse_x,
        "MAE_Y": mae_y, "RMSE_Y": rmse_y,
        "MAE_Theta": mae_theta, "RMSE_Theta": rmse_theta
    }


# Main execution
bag_path = "/home/francesco-masin/bag_files/middle_level"

# New landmarks with IDs and coordinates
landmark_ids = [0, 1, 2, 3, 4, 5]
landmarks_x = [1.80, -0.45, -0.2, 1.20, 1.33, -0.09]
landmarks_y = [0.14, -0.11, 1.76, 1.18, -1.59, -1.64] 

# Extract real and EKF positions from the ROS 2 bag
timestamps_real, real_positions = extract_odom_positions_ros2(bag_path, "/odom")
timestamps_ekf, ekf_positions = extract_odom_positions_ros2(bag_path, "/ekf")

# Interpolate the EKF positions to match the real timestamps
if real_positions and ekf_positions:
    interpolated_ekf_positions = interpolate_ekf_data(timestamps_real, timestamps_ekf, ekf_positions)

    # Compute MAE and RMSE between real and EKF data
    errors = compute_error(real_positions, interpolated_ekf_positions)
    print("MAE and RMSE between real and EKF:")
    for key, value in errors.items():
        print(f"{key}: {value}")

    # Plot the trajectories and X, Y, Theta over time
    plot_trajectories(real_positions, interpolated_ekf_positions, landmark_ids, landmarks_x, landmarks_y)
    plot_x_y_theta(timestamps_real, real_positions, timestamps_real, interpolated_ekf_positions)
"""

import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
import tf_transformations  # To handle quaternion to euler conversion
from scipy.interpolate import interp1d

# Function to extract x, y, and theta positions from any Odometry topic in a ROS 2 bag
def extract_odom_positions_ros2(bag_path, topic_name):
    positions = []  # List to store x, y, and theta positions
    velocities = [] # List to store velocities
    timestamps = []  # List to store timestamps
    
    # Open the ROS 2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and types
    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}

    # Check if the topic exists in the bag file
    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag file.")

    # Read messages
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            if topic == "/cmd_vel":
                msg = deserialize_message(data, Twist)
                v = msg.linear.x
                w = msg.angular.z
            else:
                # Deserialize data using the Odometry message type
                msg = deserialize_message(data, Odometry)
                
                # Extract position and orientation (quaternion)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                v = msg.twist.twist.linear.x
                w = msg.twist.twist.angular.z
                quat = msg.pose.pose.orientation
                # Convert quaternion to Euler angles (yaw, pitch, roll)
                _, _, theta = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

                # Append the extracted position and orientation along with timestamp
                timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds
                positions.append((x, y, theta))
            timestamp_sec = t / 1e9
            velocities.append((v,w))
            timestamps.append(timestamp_sec)

    if not positions:
        print(f"No data extracted from topic {topic_name}.")
    else:
        print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return timestamps, positions, velocities


# Function to interpolate data to real timestamps
def interpolate_data(timestamps_real, timestamps_source, source_positions):
    # Extract x, y, and theta
    source_x, source_y, source_theta = zip(*source_positions)

    # Create interpolation functions for x, y, and theta
    interp_x = interp1d(timestamps_source, source_x, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(timestamps_source, source_y, kind='linear', fill_value="extrapolate")
    interp_theta = interp1d(timestamps_source, source_theta, kind='linear', fill_value="extrapolate")

    # Interpolate the source data at real timestamps
    x_interp = interp_x(timestamps_real)
    y_interp = interp_y(timestamps_real)
    theta_interp = interp_theta(timestamps_real)

    # Return interpolated data
    return list(zip(x_interp, y_interp, theta_interp))

def interpolate_velocity_data(timestamps_real, timestamps_source, source_velocities):
    source_v, source_w = zip(*source_velocities)
    interp_v = interp1d(timestamps_source, source_v, kind='linear', fill_value="extrapolate")
    interp_w = interp1d(timestamps_source, source_w, kind='linear', fill_value="extrapolate")
    # Interpolate the source data at real timestamps
    v_interp = interp_v(timestamps_real)
    w_interp = interp_w(timestamps_real)
    # Return interpolated data
    return list(zip(v_interp,w_interp))


# Function to compute MAE and RMSE
def compute_error(real_positions, deformed_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    deformed_x, deformed_y = zip(*deformed_positions)

    # Compute MAE and RMSE for X, Y, and Theta
    mae_x = np.mean(np.abs(np.array(real_x) - np.array(deformed_x)))
    rmse_x = np.sqrt(np.mean((np.array(real_x) - np.array(deformed_x)) ** 2))

    mae_y = np.mean(np.abs(np.array(real_y) - np.array(deformed_y)))
    rmse_y = np.sqrt(np.mean((np.array(real_y) - np.array(deformed_y)) ** 2))

    # mae_theta = np.mean(np.abs(np.array(real_theta) - np.array(deformed_theta)))
    # rmse_theta = np.sqrt(np.mean((np.array(real_theta) - np.array(deformed_theta)) ** 2))

    return {
        "MAE_X": mae_x, "RMSE_X": rmse_x,
        "MAE_Y": mae_y, "RMSE_Y": rmse_y,
        # "MAE_Theta": mae_theta, "RMSE_Theta": rmse_theta
    }

def compute_error_velocities(real_velocities, deformed_velocities):
    real_v, real_w = zip(*real_velocities)
    deformed_v, deformed_w = zip(*deformed_velocities)

    # Compute MAE and RMSE for X, Y, and Theta
    mae_v = np.mean(np.abs(np.array(real_v) - np.array(deformed_v)))
    rmse_v = np.sqrt(np.mean((np.array(real_v) - np.array(deformed_v)) ** 2))

    mae_w = np.mean(np.abs(np.array(real_w) - np.array(deformed_w)))
    rmse_w = np.sqrt(np.mean((np.array(real_w) - np.array(deformed_w)) ** 2))

    return {
        "MAE_V": mae_v, "RMSE_V": rmse_v,
        "MAE_W": mae_w, "RMSE_W": rmse_w,
    }

def extract_path_ros2(bag_path, topic_name):
    positions = []  # List to store x, y, and theta positions
    
    # Open the ROS 2 bag
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and types
    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}

    # Check if the topic exists in the bag file
    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag file.")
    (topic, data, t) = reader.read_next()
    while ( topic != topic_name):
        (topic, data, t) = reader.read_next()
    
    print(topic)
    msg = deserialize_message(data, Path)
    positions = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses])
    # Read messages
    # while reader.has_next():
    #     (topic, data, t) = reader.read_next()
    #     if topic == topic_name:
    #         # Deserialize data using the Odometry message type
    #         msg = deserialize_message(data, Path)
            
    #         # Extract position and orientation (quaternion)
    #         positions = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses])
    #         # y = msg.poses.position.y

    #         # Append the extracted position and orientation along with timestamp
    #         timestamp_sec = t / 1e9  # Convert from nanoseconds to seconds
    #         positions.append((x, y))

    # if not positions:
    #     print(f"No data extracted from topic {topic_name}.")
    # else:
    #     print(f"Extracted {len(positions)} positions from topic {topic_name}.")
        
    return positions

def calculate_total_distance(waypoints):
    total_distance = 0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        total_distance += np.linalg.norm(p2 - p1)
    return total_distance

def find_resolution_for_target_size(waypoints, target_size):
    total_distance = calculate_total_distance(waypoints)
    n_original_points = len(waypoints)
    n_interpolated_points = target_size - n_original_points
    resolution = total_distance / n_interpolated_points
    return resolution

def interpolate_waypoints(waypoints, target_size=1492):
    """
    Interpolate the waypoints to add more points along the path

    Args:
    waypoints: array of waypoints
    target_size: desired number of waypoints in the output

    Return:
    interpolated_waypoints: array of interpolated waypoints
    """
    total_distance = calculate_total_distance(waypoints)
    n_original_points = len(waypoints)
    n_interpolated_points = target_size - n_original_points
    resolution = total_distance / n_interpolated_points

    interpolated_waypoints = [waypoints[0]]  # Start with the first waypoint
    accumulated_points = 1  # Start with the first waypoint already added

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = np.linalg.norm(p2 - p1)
        n_points = int(dist / resolution)
        if n_points > 1:
            x = np.linspace(p1[0], p2[0], n_points, endpoint=False)
            y = np.linspace(p1[1], p2[1], n_points, endpoint=False)
            interpolated_waypoints += list(zip(x, y))
            accumulated_points += n_points
        interpolated_waypoints.append(p2)  # Ensure the last point is included
        accumulated_points += 1

    # If we have more points than needed, truncate the list
    if accumulated_points > target_size:
        interpolated_waypoints = interpolated_waypoints[:target_size]
    # If we have fewer points, add more points at the end
    elif accumulated_points < target_size:
        last_point = waypoints[-1]
        while accumulated_points < target_size:
            interpolated_waypoints.append(last_point)
            accumulated_points += 1

    return np.array(interpolated_waypoints)

# Example usage:
waypoints = np.array([[0, 0], [1, 1], [2, 2]])  # Replace with your actual waypoints
interpolated_waypoints = interpolate_waypoints(waypoints, 1492)
print(f"Number of interpolated waypoints: {len(interpolated_waypoints)}")


# Function to plot the trajectories with landmark IDs
def plot_trajectories(ground_truth_positions, path_positions):
    real_x, real_y = zip(*path_positions)
    # ekf_x, ekf_y, _ = zip(*ekf_positions)
    ground_truth_x, ground_truth_y, _ = zip(*ground_truth_positions)

    plt.figure(figsize=(10, 8))

    # Plot the real trajectory
    plt.plot(real_x, real_y, label='/global_path Trajectory', color='red', linestyle='-', linewidth=1)

    # Plot the EKF trajectory
    # plt.plot(ekf_x, ekf_y, label='/ekf Trajectory', color='blue', linestyle='--', linewidth=1)

    # Plot the ground truth trajectory
    plt.plot(ground_truth_x, ground_truth_y, label='/ground_truth Trajectory', color='green', linestyle=':', linewidth=1)

    # Plot landmarks
    # plt.scatter(landmarks_x, landmarks_y, label='Landmarks', color='black', marker='x', s=50)

    # Annotate each landmark with its ID
    # for i, (x, y, landmarks_id) in enumerate(zip(landmarks_x, landmarks_y, landmark_ids)):
    #     plt.text(x+0.05, y+0.05, f'Landmark {str(landmarks_id)}', fontsize=8, color='red')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    plt.title('Trajectory Comparison')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Function to plot x, y, and theta over time
def plot_x_y_theta(timestamps_real, real_positions, timestamps_ekf, ekf_positions, timestamps_ground_truth, ground_truth_positions):
    real_x, real_y, real_theta = zip(*real_positions)
    # ekf_x, ekf_y, ekf_theta = zip(*ekf_positions)
    ground_truth_x, ground_truth_y, ground_truth_theta = zip(*ground_truth_positions)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot X position over time
    axs[0].plot(timestamps_ekf, ekf_x, label='/ekf/pose/pose/position/x', color='blue', linestyle='-', linewidth=1)
    axs[0].plot(timestamps_real, real_x, label='/odom/pose/pose/position/x', color='red', linestyle='-', linewidth=1)
    axs[0].plot(timestamps_ground_truth, ground_truth_x, label='/ground_truth/pose/pose/position/x', color='green', linestyle=':', linewidth=1)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='upper right')
    axs[0].set_title('X Position Over Time')
    axs[0].set_ylabel('X (m)')
    axs[0].set_xlabel('Time (s)')

    # Plot Y position over time
    axs[1].plot(timestamps_ekf, ekf_y, label='/ekf/pose/pose/position/y', color='blue', linestyle='-', linewidth=1)
    axs[1].plot(timestamps_real, real_y, label='/odom/pose/pose/position/y', color='red', linestyle='-', linewidth=1)
    axs[1].plot(timestamps_ground_truth, ground_truth_y, label='/ground_truth/pose/pose/position/y', color='green', linestyle=':', linewidth=1)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Y Position Over Time')
    axs[1].set_ylabel('Y (m)')
    axs[1].set_xlabel('Time (s)')

    # Plot Theta (orientation) over time
    axs[2].plot(timestamps_ekf, ekf_theta, label='/ekf/pose/orientation/yaw', color='blue', linestyle='-', linewidth=1)
    axs[2].plot(timestamps_real, real_theta, label='/odom/pose/orientation/yaw', color='red', linestyle='-', linewidth=1)
    axs[2].plot(timestamps_ground_truth, ground_truth_theta, label='/ground_truth/pose/orientation/yaw', color='green', linestyle=':', linewidth=1)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend(loc='upper right')
    axs[2].set_title('Theta (Orientation) Over Time')
    axs[2].set_ylabel('Theta (rad)')
    axs[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

def plot_v_w(timestamps_real, real_velocities):
    real_v, real_w = zip(*real_velocities)
    # ekf_v, ekf_w = zip(*ekf_velocities)
    # ground_truth_v, ground_truth_w = zip(*ground_truth_velocities)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot V over time
    # axs[0].plot(timestamps_ekf, ekf_v, label='/ekf/twist/twist/linear/x', color='blue', linestyle='-', linewidth=1)
    axs[0].plot(timestamps_real, real_v, label='/cmd_vel/linear/x', color='red', linestyle='-', linewidth=1)
    # axs[0].plot(timestamps_ground_truth, ground_truth_v, label='/ground_truth/twist/twist/linear/x', color='green', linestyle=':', linewidth=1)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='upper right')
    axs[0].set_title('V Position Over Time')
    axs[0].set_ylabel('V (m/s)')
    axs[0].set_xlabel('Time (s)')

    # Plot W position over time
    # axs[1].plot(timestamps_ekf, ekf_w, label='/ekf/twist/twist/angular/z', color='blue', linestyle='-', linewidth=1)
    axs[1].plot(timestamps_real, real_w, label='/cmd_vel/angular/z', color='red', linestyle='-', linewidth=1)
    # axs[1].plot(timestamps_ground_truth, ground_truth_w, label='/ground_truth/twist/twist/angular/z', color='green', linestyle=':', linewidth=1)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Angular velocity Over Time')
    axs[1].set_ylabel('W (rad/s)')
    axs[1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()


# Main execution
bag_path = "/home/nukharetrd/ros2_ws/src/pure_pursuit/ros_bag/rosbag2_2024_12_15-19_07_41"
# # New landmarks with IDs and coordinates
# # landmark_ids = [11, 12, 13, 21, 22, 23, 31, 32, 33]
# # landmarks_x = [-1.1, -1.1, -1.1, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1]
# # landmarks_y = [-1.1, 0.0, 1.1, -1.1, 0.0, 1.1, -1.1, 0.0, 1.1]
# landmark_ids = [0, 1, 2, 3, 4, 5]
# landmarks_x = [1.80, -0.45, -0.2, 1.20, 1.33, -0.09]
# landmarks_y =  [0.14, -0.11, 1.76, 1.18, -1.59, -1.64]
# lanmarks_z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

# Extract data for /odom, /ekf, and /ground_truth
timestamps_real, real_positions, real_velocities = extract_odom_positions_ros2(bag_path, "/odom")
#timestamps_ground_truth, ground_truth_positions, ground_truth_velocities = extract_odom_positions_ros2(bag_path, "/ground_truth")   #was groundtruth before
timestamps_cmdvel, _, cmd_vel_velocities = extract_odom_positions_ros2(bag_path, "/cmd_vel")
path_positions = extract_path_ros2(bag_path,"/global_path")
# print(path_positions)
interpolated_path_positions = interpolate_waypoints(path_positions, target_size= len(real_positions) )  # To ensure real_positions and path_positions have same dimensions

# Interpolate /ekf and /ground_truth to match /odom timestamps
if real_positions:
    #interpolated_ground_truth_positions = interpolate_data(timestamps_real, timestamps_ground_truth, ground_truth_positions)
    interpolated_cmd_velocities = interpolate_velocity_data(timestamps_real,timestamps_cmdvel, cmd_vel_velocities )
    #interpolated_ground_truth_velocities = interpolate_velocity_data(timestamps_real, timestamps_ground_truth, ground_truth_velocities)
    # # Compute errors
    # errors_ekf = compute_error(real_positions, interpolated_ekf_positions)
    errors_ground_truth = compute_error(real_positions, interpolated_path_positions)
    # errors_ekf_velocity = compute_error_velocities(interpolated_cmd_velocities, interpolated_ekf_velocities)

    # print("Errors between /odom and /ekf:")
    # for key, value in errors_ekf.items():
    #     print(f"{key}: {value}")

    print("\nErrors between /odom and /global_path:")
    for key, value in errors_ground_truth.items():
        print(f"{key}: {value}")

    # print("\n Velocity Errors between /cmd_vel and /ekf:")
    # for key, value in errors_ekf_velocity.items():
    #     print(f"{key}: {value}")
    # Plot the trajectories and X, Y, Theta over time
    plot_trajectories(real_positions, interpolated_path_positions)
    # plot_x_y_theta(timestamps_real, real_positions, timestamps_real, interpolated_ekf_positions, timestamps_real, interpolated_ground_truth_positions)
    plot_v_w(timestamps_real, interpolated_cmd_velocities)