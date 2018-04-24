#pragma once

#define _USE_MATH_DEFINES
#include <QList>
#include <chrono>
#include <array>
#include "tracker/Types.h"

struct IMU_instance{
	//std::string device;
	//int sensor;
	float x, y, z;
	long long watch_time;
	std::chrono::milliseconds rec_time;
};

class Watch{
	bool is_assigned = false;
	std::string device_name;
	QList<IMU_instance> gyro_buffer;
	QList<IMU_instance> accel_buffer;
	//std::array<IMU_instance, 100 * 5 * 60> gyro_buffer;
	//std::array<IMU_instance, 100 * 5 * 60> accel_buffer;
	Handedness hand;


public:

	Watch();

	void set_name(std::string name);

	bool device_name_match(std::string name);

	void add_instance(float x, float y, float z, long long watch_time, int type);

	bool check_is_assigned();

	void save_data(std::string path);
};