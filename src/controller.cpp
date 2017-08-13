#include "ros/ros.h"
#include "epos2/Torque.h"

bool print_request(epos2::Torque::Request &req, epos2::Torque::Response &res)
{
  // the force transform of the data type can cause problem
  ROS_INFO("request: position=%ld, torque=%ld", (long int)req.position, (long int)req.torque);
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "epos2");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("print_request", print_request);
  ROS_INFO("Ready to print_request.");
  ros::spin();

  return 0;
}