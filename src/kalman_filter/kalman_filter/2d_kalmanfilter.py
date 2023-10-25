import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables
        #estimated location
        self.x_hat = np.zeros((2,1))
        #standard deviation = σ, variance = p, σ^2=p
        self.sigma_x = 1
        self.sigma_y = 1
        self.p = np.array([
        [self.sigma_x**2, 0],
        [0, self.sigma_y**2]])
        #H: Observation matrix
        self.h = np.array([
        [1, 0],
        [0, 1]])
        #Z: Measurment Matrix
        self.z = np.dot(self.h,self.x_hat)
        #Q: Covariance Matrix
        self.q_x = 0.01
        self.q_y = 0.01
        self.q = np.array([
        [self.q_x, 0],
        [0, self.q_y]])
        #F : State Transition Matrix
        self.f = np.array([
        [1, 0],
        [0, 1]])
        #R: Measurment Error
        self.r_x = 75
        self.r_y = 75
        self.r = np.array([
        [self.r_x, 0],
        [0, self.r_y]])
        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,
                                                 "/odom_estimated",1)
        

    def odom_callback(self, msg):
        print("Received Odometry message")
        # Extract the position measurements from the Odometry message
        z = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
        ])
        
        # Prediction step
        self.x_hat = np.dot(self.f, self.x_hat)
        self.p = np.dot(np.dot(self.f, self.p), self.f.T) + self.q
        
        # Update step
        K = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(np.dot(np.dot(self.h, self.p), self.h.T) + self.r))        
        self.x_hat = self.x_hat + np.dot(K, (z-self.x_hat))
        self.p = np.dot((np.eye(2) - np.dot(K, self.h)), self.p)        
        #publish the estimated reading
        estimated_msg = Odometry()
        estimated_msg.pose.pose.position.x = self.x_hat[0, 0]
        estimated_msg.pose.pose.position.y = self.x_hat[1, 0]
        self.estimated_pub.publish(estimated_msg)
        #pass    

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
