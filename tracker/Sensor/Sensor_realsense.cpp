#include "Sensor.h"
#include "tracker/Types.h"

#include "util/gl_wrapper.h"
#include "util/mylogger.h"
#include "util/Sleeper.h"
#include "util/tictoc.h"

#include <algorithm>


#ifndef HAS_REALSENSE
SensorRealSense::SensorRealSense(Camera *camera) : Sensor(camera){ LOG(INFO) << "Intel RealSense not available in your OS"; exit(0); }
int SensorRealSense::initialize(){ return 0; }
SensorRealSense::~SensorRealSense(){}
bool SensorRealSense::spin_wait_for_data(Scalar timeout_seconds){ return false; }
bool SensorRealSense::fetch_streams(DataFrame &frame){ return false; }
void SensorRealSense::start(){}
void SensorRealSense::stop(){}

#else

#include "Sensor.h"

#include <stdio.h>
#include <vector>
#include <exception>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>

#include <QElapsedTimer>
#include <QApplication>
#include <QMessageBox>

#include "pxcsensemanager.h"
#include "pxcsession.h"
#include "pxcprojection.h"
#include <opencv2/highgui/highgui.hpp>
#include "tracker/Data/DataFrame.h"
#include <iomanip>

#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

PXCImage::ImageData depth_buffer;
PXCImage::ImageData color_buffer;
PXCImage * sync_color_pxc;
PXCImage * sync_depth_pxc;

PXCSenseManager *sense_manager;
PXCProjection *projection;


int D_width = 640;
int D_height = 480;
const int BACK_BUFFER = 1;
const int FRONT_BUFFER = 0;
cv::Mat color_array[2];
cv::Mat full_color_array[2];
cv::Mat depth_array[2];

//std::vector<int> sensor_indicator_array[2];

int num_sensor_points_buffer;
std::vector<int> sensor_indicator_buffer;
bool wristband_found_buffer;
Vector3 wristband_center_buffer;
Vector3 wristband_direction_buffer;
cv::Mat sensor_silhouette_buffer;
cv::Mat sensor_silhouette_wrist_buffer;

std::thread sensor_thread;
std::mutex swap_mutex;
std::condition_variable condition;
bool main_released = true;
bool thread_released = false;

int i = 1;

int sensor_frame = 0;
int tracker_frame = 0;

SensorRealSense::SensorRealSense(Camera *camera, bool real_color, int downsampling_factor, bool fit_wrist_separately, WristBandColor color) : Sensor(camera) {
	this->handfinder = new HandFinder(camera, downsampling_factor, fit_wrist_separately, color);
	this->real_color = real_color;
	this->downsampling_factor = downsampling_factor;
}

int SensorRealSense::initialize() {
	sense_manager = PXCSenseManager::CreateInstance();
	if (!sense_manager) {
		wprintf_s(L"Unable to create the PXCSenseManager\n");
		return false;
	}
	
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, D_width, D_height, 60);
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, D_width, D_height, 60);
	sense_manager->Init();

	PXCSession *session = PXCSession::CreateInstance();
	PXCSession::ImplDesc desc, desc1;
	memset(&desc, 0, sizeof(desc));
	desc.group = PXCSession::IMPL_GROUP_SENSOR;
	desc.subgroup = PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;
	if (session->QueryImpl(&desc, 0, &desc1) < PXC_STATUS_NO_ERROR) return false;

	PXCCapture * capture;
	pxcStatus status = session->CreateImpl<PXCCapture>(&desc1, &capture);
	if (status != PXC_STATUS_NO_ERROR) {
		QMessageBox::critical(NULL, "FATAL ERROR", "Intel RealSense device not plugged?\n(CreateImpl<PXCCapture> failed)");
		exit(0);
	}

	PXCCapture::Device* device;
	device = capture->CreateDevice(0);
	projection = device->CreateProjection();

	this->initialized = true;

	sensor_thread = std::thread(&SensorRealSense::run, this);
	sensor_thread.detach();
	return true;
}

SensorRealSense::~SensorRealSense() {	
	if (!initialized) return;
	delete handfinder;
}

void add_boarders_to_depth(cv::Mat & depth, cv::Mat & mask) {
	int boarder_radius = 1;
	cv::Mat new_mask = mask.clone();
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			if (mask.at<unsigned char>(row, col) == 0) continue;
			for (int u = -boarder_radius; u <= boarder_radius; u++) {
				for (int v = -boarder_radius; v <= boarder_radius; v++) {
					if (abs(u) + abs(v) > boarder_radius) continue;
					if (row + u < 0 || row + u >= mask.rows ) continue;
					if (col + v < 0 || col + v >= mask.cols) continue;
					if (depth.at<unsigned short>(row + u, col + v) != 0) continue;
					depth.at<unsigned short>(row + u, col + v) = depth.at<unsigned short>(row, col);
					new_mask.at<unsigned char>(row + u, col + v) = 255;
				}
			}
		}
	}	
	mask = new_mask.clone(); 
}

bool SensorRealSense::spin_wait_for_data(Scalar timeout_seconds) {

	DataFrame frame(-1);
	QElapsedTimer chrono;
	chrono.start();
	while (fetch_streams(frame) == false) {
		LOG(INFO) << "Waiting for data.. " << chrono.elapsed();
		Sleeper::msleep(500);
		QApplication::processEvents(QEventLoop::AllEvents);
		if (chrono.elapsed() > 1000 * timeout_seconds)
			return false;
	}
	return true;
}

bool SensorRealSense::fetch_streams(DataFrame &frame) {
	if (initialized == false) this->initialize();
	PXCCapture::Sample *sample;

	//TICTOC_BLOCK(allocation, "Allocation") 
	{
		if (frame.depth.empty())
			frame.depth = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_16UC1, cv::Scalar(0));
		if (frame.color.empty())
			frame.color = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC3, cv::Scalar(0, 0, 0));

		if (sense_manager->AcquireFrame(true) < PXC_STATUS_NO_ERROR) return false;
		return true;
		sample = sense_manager->QuerySample();
	}

	//TICTOC_BLOCK(depth, "Depth")
	{
		sample->depth->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_DEPTH, &depth_buffer);
		unsigned short* data = ((unsigned short *)depth_buffer.planes[0]);
		
		for (int y = 0, y_sub = 0; y_sub < camera->height(); y += downsampling_factor, y_sub++) {
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += downsampling_factor, x_sub++) {
				frame.depth.at<unsigned short>(y_sub, x_sub) = data[y*D_width + (D_width - x - 1)];
			}
		}
		sample->depth->ReleaseAccess(&depth_buffer);

		/// Postprocess
		// cv::medianBlur(depth_cv, depth_cv, 5);		
	}

	//TICTOC_BLOCK(color, "Color")
	{
		PXCImage * sync_color_pxc = projection->CreateColorImageMappedToDepth(sample->depth, sample->color);
		sync_color_pxc->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &color_buffer);
		for (int y = 0, y_sub = 0; y_sub < camera->height(); y += downsampling_factor, y_sub++) {
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += downsampling_factor, x_sub++) {
				unsigned char r = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 0];
				unsigned char g = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 1];
				unsigned char b = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 2];
				frame.color.at<cv::Vec3b>(y_sub, x_sub) = cv::Vec3b(b, g, r); ///< SWAP Channels
			}
		}
		sync_color_pxc->ReleaseAccess(&color_buffer);
	}

	sense_manager->ReleaseFrame();
	return true;
}

bool SensorRealSense::fetch_streams_concurrently(DataFrame &frame, HandFinder & other_handfinder, cv::Mat & full_color) {

	std::unique_lock<std::mutex> lock(swap_mutex);
	condition.wait(lock, [] {return thread_released; });
	main_released = false;

	frame.color = color_array[FRONT_BUFFER].clone();
	frame.depth = depth_array[FRONT_BUFFER].clone();
	if (real_color) full_color = full_color_array[FRONT_BUFFER].clone();

	other_handfinder.num_sensor_points = num_sensor_points_buffer;
	other_handfinder.sensor_silhouette = sensor_silhouette_buffer.clone();
	if (other_handfinder.settings->fit_wrist_separately) 
		other_handfinder.sensor_silhouette_wrist = sensor_silhouette_wrist_buffer.clone();
	other_handfinder._wristband_found = wristband_found_buffer;
	other_handfinder._wband_center = wristband_center_buffer;
	other_handfinder._wband_dir = wristband_direction_buffer;

	for (size_t i = 0; i < other_handfinder.num_sensor_points; i++) 
		other_handfinder.sensor_indicator[i] = sensor_indicator_buffer[i];	

	//cout << "tracker_frame = " << tracker_frame << endl;

	main_released = true;
	lock.unlock();
	condition.notify_one();
	return true;
}

bool SensorRealSense::run() {
	PXCCapture::Sample *sample;
	for (;;) {
		if (initialized == false) this->initialize();

		//TICTOC_BLOCK(allocation, "Allocation") 		
		{
			if (sensor_indicator_buffer.empty()) 
				sensor_indicator_buffer = std::vector<int>(upper_bound_num_sensor_points, 0);
			if (depth_array[FRONT_BUFFER].empty()) {
				depth_array[FRONT_BUFFER] = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_16UC1, cv::Scalar(0));
			}
			if (depth_array[BACK_BUFFER].empty())
				depth_array[BACK_BUFFER] = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_16UC1, cv::Scalar(0));
			if (color_array[FRONT_BUFFER].empty())
				color_array[FRONT_BUFFER] = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC3, cv::Scalar(0, 0, 0));
			if (color_array[BACK_BUFFER].empty())
				color_array[BACK_BUFFER] = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC3, cv::Scalar(0, 0, 0));
			if (real_color) {
				if (full_color_array[FRONT_BUFFER].empty())
					full_color_array[FRONT_BUFFER] = cv::Mat(cv::Size(D_width, D_height), CV_8UC3, cv::Scalar(0, 0, 0));
				if (full_color_array[BACK_BUFFER].empty())
					full_color_array[BACK_BUFFER] = cv::Mat(cv::Size(D_width, D_height), CV_8UC3, cv::Scalar(0, 0, 0));
			}

			if (sense_manager->AcquireFrame(true) < PXC_STATUS_NO_ERROR) continue;
		}
		sample = sense_manager->QuerySample();

		sample->depth->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_DEPTH, &depth_buffer);
		unsigned short* data = ((unsigned short *)depth_buffer.planes[0]);
		for (int y = 0, y_sub = 0; y_sub < camera->height(); y += downsampling_factor, y_sub++) {
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += downsampling_factor, x_sub++) {
				if (x == 0 || y == 0 ) {
					depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = data[y*D_width + (D_width - x - 1)];
					continue;
				}
				if (x == camera->width() || y == camera->height()){
					depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = data[y*D_width + (D_width - x - 1)];
					continue;
				}
				std::vector<int> neighbors = {
					data[(y - 1)* D_width + (D_width - (x - 1) - 1)],
					data[(y + 0)* D_width + (D_width - (x - 1) - 1)],
					data[(y + 1)* D_width + (D_width - (x - 1) - 1)],
					data[(y - 1)* D_width + (D_width - (x + 0) - 1)],
					data[(y + 0)* D_width + (D_width - (x + 0) - 1)],
					data[(y + 1)* D_width + (D_width - (x + 0) - 1)],
					data[(y - 1)* D_width + (D_width - (x + 1) - 1)],
					data[(y + 0)* D_width + (D_width - (x + 1) - 1)],
					data[(y + 1)* D_width + (D_width - (x + 1) - 1)],
				};
				std::sort(neighbors.begin(), neighbors.end());
				depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = neighbors[4];
				//depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = data[y* D_width + (D_width - x - 1)];
			}
		}

		sample->depth->ReleaseAccess(&depth_buffer);

		if (real_color) {
			sample->color->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &color_buffer);
			for (int y = 0; y < D_height; y++) {
				for (int x = 0; x < D_width; x++) {
					unsigned char r = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 0];
					unsigned char g = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 1];
					unsigned char b = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 2];
					full_color_array[BACK_BUFFER].at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
				}
			}
			sample->color->ReleaseAccess(&color_buffer);
		}

		
		sync_color_pxc = projection->CreateColorImageMappedToDepth(sample->depth, sample->color);
		sync_color_pxc->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &color_buffer);
		for (int y = 0, y_sub = 0; y_sub < camera->height(); y += downsampling_factor, y_sub++) {
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += downsampling_factor, x_sub++) {
				unsigned char r = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 0];
				unsigned char g = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 1];
				unsigned char b = color_buffer.planes[0][y * D_width * 3 + (D_width - x - 1) * 3 + 2];
				color_array[BACK_BUFFER].at<cv::Vec3b>(y_sub, x_sub) = cv::Vec3b(b, g, r); ///< SWAP Channels
			}
		}
		sync_color_pxc->ReleaseAccess(&color_buffer);
		sync_color_pxc->Release();

		sense_manager->ReleaseFrame();

		handfinder->binary_classification(depth_array[BACK_BUFFER], color_array[BACK_BUFFER]);
		// add_boarders_to_depth(depth_array[BACK_BUFFER], handfinder->sensor_silhouette);
		handfinder->binary_classification_wrist(depth_array[BACK_BUFFER]);
		handfinder->get_sensor_indicator();

		// Lock the mutex and swap the buffers
		{
			std::unique_lock<std::mutex> lock(swap_mutex);
			condition.wait(lock, [] {return main_released; });
			thread_released = false;

			depth_array[FRONT_BUFFER] = depth_array[BACK_BUFFER].clone();
			color_array[FRONT_BUFFER] = color_array[BACK_BUFFER].clone();
			if (real_color) full_color_array[FRONT_BUFFER] = full_color_array[BACK_BUFFER].clone();

			num_sensor_points_buffer = handfinder->num_sensor_points;
			sensor_silhouette_buffer = handfinder->sensor_silhouette.clone();
			if (handfinder->settings->fit_wrist_separately) 
				sensor_silhouette_wrist_buffer = handfinder->sensor_silhouette_wrist.clone();
			wristband_found_buffer = handfinder->_wristband_found;
			wristband_center_buffer = handfinder->_wband_center;
			wristband_direction_buffer = handfinder->_wband_dir;

			for (size_t i = 0; i < handfinder->num_sensor_points; i++) sensor_indicator_buffer[i] = handfinder->sensor_indicator[i];
						
			tracker_frame = sensor_frame;
			sensor_frame++;

			thread_released = true;
			lock.unlock();
			condition.notify_one();
		}
	}
}

void SensorRealSense::start() {
	//if (!initialized)
	//this->initialize();
}

void SensorRealSense::stop() {
}
#endif
