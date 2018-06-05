#pragma once
#include <GL/glew.h>
#include <QGLWidget>
#include <QOpenGLWidget>
#include <QtNetwork\qudpsocket.h>
#include "tracker/ForwardDeclarations.h"
#include "tracker/OpenGL/KinectDataRenderer/KinectDataRenderer.h"
#include "tracker/OpenGL/ConvolutionRenderer/ConvolutionRenderer.h"
#include "tracker/OpenGL/FaceRenderer/FaceRenderer.h"
#include "tracker/Watch.h"
#include <QObject>

//class GLWidget : public QOpenGLWidget {
class GLWidget : public QGLWidget {
	Q_OBJECT
public:
	Worker * worker;
	DataStream * const datastream;
	SolutionStream * const solutions;

	Camera * const camera;
	FaceRenderer face_renderer;
	KinectDataRenderer kinect_renderer;
	ConvolutionRenderer convolution_renderer;

	bool playback;
	bool real_color;
	bool display_sensor_data = true;
	bool display_hand_model = true;
	bool display_model_outline = false;
	bool display_data_outline = false;
	bool display_data_correspondences = false;
	bool display_silhouette_correspondences = false;
	bool display_ground_truth_marker_positions = false; 
	bool display_ground_truth_reinit_constraints = false;
	bool display_model_marker_positions = false;
	bool display_fingertips = false;

	bool apply_estimated_shape = false;

	std::string data_path;
	std::string sequence_path;

public:

	GLWidget(Worker* worker, DataStream * datastream, SolutionStream * solutions, bool playback, bool real_color, std::string data_path, std::string sequence_path);

	~GLWidget();

	void initializeGL();

	void paintGL();

private:
	Eigen::Vector3f camera_center = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f image_center = Eigen::Vector3f(0, 0, 375);
	Eigen::Vector3f camera_up = Eigen::Vector3f(0, 1, 0);
	Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

	bool mouse_button_pressed = true;
	Eigen::Vector2f cursor_position = Eigen::Vector2f(640, 480);
	Eigen::Vector2f euler_angles = Eigen::Vector2f(-6.411, -1.8);
	Eigen::Vector2f initial_euler_angles = Eigen::Vector2f(-6.411, -1.8);
	float cursor_sensitivity = 0.003f;
	float wheel_sensitivity = 0.03f;

	void process_mouse_movement(GLfloat cursor_x, GLfloat cursor_y);

	void process_mouse_button_pressed(GLfloat cursor_x, GLfloat cursor_y);

	void process_mouse_button_released();

	void process_mouse_wheel(int delta);
	
	void mouseMoveEvent(QMouseEvent *event);

	void mousePressEvent(QMouseEvent *event);

	void wheelEvent(QWheelEvent * event);

	void keyPressEvent(QKeyEvent *event);

	std::vector<std::pair<Vector3, Vector3>> GLWidget::prepare_data_correspondences_for_degub_renderer();

	std::vector<std::pair<Vector3, Vector3>> GLWidget::prepare_silhouette_correspondences_for_degub_renderer();

public:
	Watch* watch;
	QUdpSocket *udpSocket = nullptr;
	public slots:
	void processPendingDatagrams(){
		std::cout << "Data Pending in GLWidget" << std::endl;
		QByteArray datagram;
		while (udpSocket->hasPendingDatagrams()){
			datagram.resize(int(udpSocket->pendingDatagramSize()));
			udpSocket->readDatagram(datagram.data(), datagram.size());
			std::cout << datagram.size() << " Bytes : " << datagram.constData() << std::endl;
			char* array = datagram.data();
			String device(array, 8);

			char float_v[4];
			float x, y, z;
			float_v[0] = array[23]; float_v[1] = array[22]; float_v[2] = array[21]; float_v[3] = array[20];
			memcpy(&x, &float_v, sizeof(x));
			float_v[0] = array[27]; float_v[1] = array[26]; float_v[2] = array[25]; float_v[3] = array[24];
			memcpy(&y, &float_v, sizeof(y));
			float_v[0] = array[31]; float_v[1] = array[30]; float_v[2] = array[29]; float_v[3] = array[28];
			memcpy(&z, &float_v, sizeof(z));

			char int_v[4];
			int type_num;
			int_v[0] = array[11]; int_v[1] = array[10]; int_v[2] = array[9]; int_v[3] = array[8];
			memcpy(&type_num, &int_v, sizeof(type_num));

			char long_v[8];
			long long time;
			long_v[0] = array[19]; long_v[1] = array[18]; long_v[2] = array[17]; long_v[3] = array[16];
			long_v[4] = array[15]; long_v[5] = array[14]; long_v[6] = array[13]; long_v[5] = array[12];
			memcpy(&time, &long_v, sizeof(time));


			if (!watch->check_is_assigned()){
				watch->set_name(device);
				std::cout << "Watch Found: " << device << endl;
				watch->add_instance(x, y, z, time, type_num);
			}
			else if(watch->device_name_match(device)){
				watch->add_instance(x, y, z, time, type_num);
			}
			else{
				std::cout << "Unrecognized Device Data: " << endl;
				std::cout << type_num << " " << time << " " << x << " " << y << " " << z << " " << device << std::endl;
			}
		}
	};
};
