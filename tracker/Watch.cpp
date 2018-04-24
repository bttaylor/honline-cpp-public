#include <tracker/Watch.h>
#include <qfile.h>
#include <fstream>
#include <iomanip>

Watch::Watch(){
	device_name = "";
	hand = LEFT_HAND;
}

void Watch::set_name(std::string name){
	device_name = name;
	is_assigned = true;
}

bool Watch::device_name_match(std::string name){
	if (device_name.compare(name) == 0)
		return true;
	return false;
}

void Watch::add_instance(float x, float y, float z, long long watch_time, int type){
	IMU_instance* instance = new IMU_instance();
	instance->x = x;
	instance->y = y;
	instance->z = z;
	instance->watch_time = watch_time;
	//instance.device = device_name;
	instance->rec_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	if (type == 1)  //accel
		accel_buffer.push_back(*instance);
	if (type == 2)
		gyro_buffer.push_back(*instance);

}

bool Watch::check_is_assigned(){
	return is_assigned;
}

void Watch::save_data(std::string path){
	std::ofstream outfile;
	std::string filepath = path + "watch_accel_data.csv";
	outfile.open(filepath, std::ofstream::app);

	for (size_t i = 0; i < accel_buffer.size(); i++) {
		
		outfile << std::to_string(accel_buffer.at(i).rec_time.count()) << " , " << accel_buffer.at(i).watch_time << " , ";
		outfile << accel_buffer.at(i).x << " , " << accel_buffer.at(i).y << " , " << accel_buffer.at(i).z << endl;

	}
	outfile.close();
	cout << "Saved " << accel_buffer.size() << " accel samples." << endl;

	filepath = path + "watch_gyro_data.csv";
	outfile.open(filepath, std::ofstream::app);

	for (size_t i = 0; i < gyro_buffer.size(); i++) {

		outfile << std::to_string(gyro_buffer.at(i).rec_time.count()) << " , " << gyro_buffer.at(i).watch_time << " , ";
		outfile << gyro_buffer.at(i).x << " , " << gyro_buffer.at(i).y << " , " << gyro_buffer.at(i).z << endl;

	}
	outfile.close();
	cout << "Saved " << gyro_buffer.size() << " gyro samples." << endl;
}