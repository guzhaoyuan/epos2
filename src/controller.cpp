/**
    controller.cpp
    Purpose: used for control the pendulum hardware

    @author Zhaoyuan Gu
    @version 0.1 08/23/17 
*/
#include "ros/ros.h"
#include "epos2/Torque.h" // service file

#include "wrap.h" // the head of epos control functions

#define position_offset 1000 
#define pulse_per_round 2000
#define PI 3.14159
#define V_LOW -8.0f
#define V_HIGH 8.0f
#define CURRENT_MAX 2
#define CURRENT_MIN -2 // res.torque = (-1,1)
#define TORQUE_AMP 1000 //torque applied = TORQUE_AMP * res.torque

/**
	define clockwise is minus, conterclockwise is positive
	define theta is the angle from upward axis to pumdulum, range (-PI , PI]
	state:[cos(theta) sin(thata) velocity]
	velocity range[-8.0 8.0]

	a round is 2000 trigs from the encoder
	when setup init, the pumdulum is downward and the position downward is 0,
	so the angle theta = (position+1000)%2000*2*PI
	velocity = (theta - theta_old)/dt
	reward = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
*/

ros::Time begin;
// ros::Duration interval(1.0); // 1s
ros::Duration interval(0,25000000); // 0s,25ms
ros::Time next;

int position_old, position_new; // for calc velocity
float angle_old, angle_new, pVelocityIs, pVelocityIs_old, reward;
short current, torque;

int random_init(){
	cout<<"init success, return first state"<<endl;
	// need to update angle while random init
}

bool applyTorque(epos2::Torque::Request &req, epos2::Torque::Response &res)
{
	if(req.init == 1){
		random_init();
		return true;
	}else{
		unsigned int ulErrorCode = 0;


		// ROS_INFO("now read position n current");
		get_position(g_pKeyHandle, g_usNodeId, &position_new, &ulErrorCode);
		// get_current(g_pKeyHandle, g_usNodeId, &current, &ulErrorCode);
		// get_velocity(g_pKeyHandle, g_usNodeId, &pVelocityIs, &ulErrorCode);
		
		//calc velocity before angle, using continuous position n angle, rad/s
		pVelocityIs = (float)(position_new - position_old)/pulse_per_round*2*PI/interval.toSec();
		pVelocityIs = min(V_HIGH,max(V_LOW,pVelocityIs)); // soft limit speed
		//calc angle
		angle_new = (float)((position_new+position_offset) % pulse_per_round)/pulse_per_round*2*PI;
		if(angle_new > PI)	angle_new-=2*PI; // angle range (-PI , PI]
	    //calc reward
	    torque = req.torque;
	    torque = min(CURRENT_MAX,max(CURRENT_MIN, (int)torque)); // soft limit torque
		reward = -(angle_old*angle_old + 0.1*pVelocityIs_old*pVelocityIs_old + 0.001*torque*torque);
		// cout<<angle_new<<",\t"<<pVelocityIs<<",\t"<<reward<<endl;

		// res.current = current;
		res.position_new = position_new;
		res.velocity = pVelocityIs;
		res.reward = reward;

		//update stored position and angle
		position_old = position_new;
		angle_old = angle_new;
		pVelocityIs_old = pVelocityIs;

		// the force transform of the data type can cause problem
		while((next - ros::Time::now()).toSec()<0){
			next += interval;
			ROS_INFO("");
		}
		(next - ros::Time::now()).sleep();

		ROS_INFO("now write: position=%ld, torque=%ld", (long int)req.position, (long int)req.torque);
		SetCurrentMust(g_pKeyHandle, g_usNodeId, TORQUE_AMP*torque, &ulErrorCode);

		// ROS_INFO("now return");
	}
	return true;
}



int main(int argc, char **argv)
{
	ros::init(argc, argv, "epos2");
	ros::NodeHandle n;

	begin = ros::Time::now();
	next = begin + interval;
	int lResult = MMC_FAILED;
	unsigned int ulErrorCode = 0;
	// Set parameter for usb IO operation
	SetDefaultParameters();
	// open device
	if((lResult = OpenDevice(&ulErrorCode))!=MMC_SUCCESS)
	{
		LogError("OpenDevice", lResult, ulErrorCode);
		return lResult;
	}
	
	SetEnableState(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	ActivateProfileCurrentMode(g_pKeyHandle, g_usNodeId, &ulErrorCode);

	ros::ServiceServer service = n.advertiseService("applyTorque", applyTorque);

	ROS_INFO("Ready to move.");
	ros::spin();

	//disable epos
	SetDisableState(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	//close device
	if((lResult = CloseDevice(&ulErrorCode))!=MMC_SUCCESS)
	{
		LogError("CloseDevice", lResult, ulErrorCode);
		return lResult;
	}
	return 0;
}