import rospy
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Set up the ROS node
rospy.init_node("rosbag_to_video")

# Parameters
rosbag_path = "/home/aravindh/ETH/Thesis/vins_ws/euroc/MH_01_easy.bag"  # Path to your rosbag file
topic_name = "/cam0/image_raw"          # Topic to subscribe to
output_video_path = "MH_01_easy.mp4"    # Path to save the video
fps = 20                                   # Frames per second for the video

# Initialize CvBridge
bridge = CvBridge()

# Open the rosbag
bag = rosbag.Bag(rosbag_path, "r")

# Get the first message to determine image size
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    if isinstance(msg, Image):
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width, _ = cv_image.shape
        break

# Set up the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process the rosbag and write frames to the video
try:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if isinstance(msg, Image):
            # Convert the ROS Image message to a CV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Write the frame to the video
            video_writer.write(cv_image)

            # Optional: Display the frame (press 'q' to stop playback)
            cv2.imshow("Frame", cv_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Release resources
    video_writer.release()
    bag.close()
    cv2.destroyAllWindows()

print(f"Video saved to {output_video_path}")
