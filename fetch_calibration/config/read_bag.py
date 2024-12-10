import rosbag
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Open the bag file
bag = rosbag.Bag('/home/felipe/ft_ros/src/fetch_ros_IRVL/fetch_calibration/config/calibration_poses_ground.bag')
bag = rosbag.Bag('/home/felipe/ft_ros/src/fetch_ros_IRVL/fetch_calibration/config/calibration_poses.bag')

# Read messages from the bag file for specific topics
for topic, msg, t in bag.read_messages():
    print(f"Topic: {topic},{msg}, Timestamp: {t}")
    break
    
# Close the bag file
bag.close()
