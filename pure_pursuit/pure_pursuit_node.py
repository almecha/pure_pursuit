import rclpy
from rclpy.node import Node
from pure_pursuit.utils.pure_pursuit import PurePursuitController
from pure_pursuit.utils.utils import DifferentialDriveRobot, proportional_control, interpolate_waypoints
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
import numpy as np

class PurePursuit(Node):
    def __init__(self):
        super().__init__("PurePursuitNode")
        self.get_logger().info("PurePursuit is active!")
        # Necessary subscriptions
        self.path_sub = self.create_subscription(Path, "/global_path", self.path_sub_callback, 10)
        # Necessary publishers
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Controller timer
        self.dt = 1/15
        self.create_timer(self.dt, self.controller_step)
        # Initial pose and parameters for the robot object  
        init_pose = np.array([0.0, 0.0, 0.0])
        self.robot = DifferentialDriveRobot(init_pose)
        # self.robot.v = 0
        # self.robot.w = 1e-6
        # Pure Pursuit controller parameters
        self.pind = 0
        self.Lt = 0.2            # To be determined
        self.v_t = 0.14          # Target linear velocity in our path tracking
        self.path = None
        self.controller = None

    def path_sub_callback(self, msg : Path):
        # Convert ROS path message to numpy array for the controller
        self.path = interpolate_waypoints(np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses]))
        print(self.path[:])
        # Initialize the controller with the new path
        if self.path is not None:
            if self.controller is None:
                self.controller = PurePursuitController(self.robot, self.path, self.pind, self.Lt, self.v_t)
    
    def controller_step(self):
        msg = Twist()
        # Calculate target velocities
        if (self.controller is not None):
            acceleration = proportional_control(self.controller.target_velocity(), self.robot.v)
            self.w = self.controller.angular_velocity()
            self.robot.update_state([acceleration,self.w],self.dt)
        
        if (self.controller.vt == 0):
            msg.linear.x = 0
            msg.angular.z = 1e-6
        # Compose a message and send it
        msg.linear.x = self.robot.v
        msg.angular.z = self.robot.w
        print(f"Velocity is {self.robot.v}, angular is {self.robot.w}")
        self.twist_pub.publish(msg)


def main(args = None):
    rclpy.init(args = args)
    controller = PurePursuit()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()