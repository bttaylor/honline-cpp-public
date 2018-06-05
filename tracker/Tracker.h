#pragma once
#include <QTimer>
#include <QObject>
#include "util/mylogger.h"
#include "util/tictoc.h"
#include "tracker/ForwardDeclarations.h"
#include "tracker/Sensor/Sensor.h"
#include "tracker/Data/DataStream.h"
#include "tracker/HModel/GroundTruthLoader.h"
#include "tracker/Worker.h"
#include "tracker/Data/SolutionStream.h"
#include "tracker/Detection/QianDetection.h"
#include "tracker/Data/TextureColor8UC3.h"
#include "tracker/Data/TextureDepth16UC1.h"
#include "tracker/TwSettings.h"
#include "tracker/HModel/Model.h"
#include "tracker/OpenGL/CustomFrameBuffer.h"
#include "tracker/HModel/JacobianVerifier.h"

#include "tracker/Energy/Fitting/OnlinePerformanceMetrics.h"
#include "tracker/HModel/SyntheticDatasetGenerator.h"

#include <ctime>
#include <math.h>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include "tracker/HandFinder/connectedComponents.h" ///< only declared in opencv3
#include <numeric> ///< std::iota
#include <random>

#include "tracker/Detection/FindFingers.h"
#include "tracker/Detection/DetectionStream.h"
#include "tracker/HModel/FingertipsDetector.h"

#include "tracker/HModel/BatchSolver.h"
#include "tracker/HModel/KalmanFilter.h"

#include "GLWidget.h"

#include <FaceTracker/Tracker.h>


class Tracker : public QTimer {
public:
	Mode mode = LIVE;
	Sensor* sensor;
	DataStream* datastream;
	SolutionStream* solutions;
	Worker * const worker = NULL;

	OnlinePeformanceMetrics online_performance_metrics;
	SyntheticDatasetGenerator synthetic_dataset_generator;
	bool real_color;

	std::string sequence_path;

	ofstream tracking_metrics_file;
	ofstream weighted_metrics_file;
	ofstream solutions_file;

	float current_fps = 0;
	int first_frame_lag = 0;

	bool tracking_failed = false;
	bool initialization_enabled = true;
	bool tracking_enabled = false;

	int sequence_length;

	std::vector<std::vector<float>> certainties_vector;
	GLWidget * glwidget;

	//Brandon Mods
	std::vector<std::vector<double>> faceShapes;
	FACETRACKER::Tracker faceModel;
	cv::Mat tri;
	cv::Mat con;
	bool showFaceTrack = true;

	bool last_was_pause = false;
	int reset_frame = -101;
	int user_num;
	std::string prompt;
	int prompt_i = 0;
	std::vector<std::string> prompts;
	std::string current_prompt;
	bool handshape = false;
	int hs_set_num = 1;
	bool fingerspelling = false; //Fingerspelling Mod
	bool all_sets_done = false;
	int set_frame = 54;


	Tracker(Worker*worker, GLWidget * glwidget, double FPS, std::string sequence_path, bool real_color) : worker(worker), glwidget(glwidget), synthetic_dataset_generator(worker, sequence_path),
		faceModel("C:/Developer/FaceTracker-opencv2/model/face2.tracker"){
		setSingleShot(false);
		setInterval((1.0 / FPS)*1000.0);
		this->sequence_path = sequence_path;
		this->real_color = real_color;

		//Brandon Added
		load_prompts_list();

		if (worker->settings->sequence_length >= 0) {
			sequence_length = worker->settings->sequence_length;
			return;
		}

		if (worker->settings->dataset_type == TOMPSON) {
			sequence_length = 2440;
		}
		if (worker->settings->dataset_type == SRIDHAR) {
			sequence_length = 3155;
		}
		if (worker->settings->dataset_type == TKACH || BRANDON) {
			std::ifstream sequence_length_file(sequence_path + "sequence_length.txt");
			if (!sequence_length_file.is_open()) {
				cout << "cannot open sequence length file" << endl;
				sequence_length = 171 + 292;
				synthetic_dataset_generator.set_sequence_length(sequence_length);
			}
			else {
				std::string line;
				std::getline(sequence_length_file, line);
				sequence_length = std::stoi(line);
				sequence_length_file.close();
				synthetic_dataset_generator.set_sequence_length(sequence_length);
			}
		}

		//FaceTracker

		if (showFaceTrack) {
			cvNamedWindow("Face Tracker", 1);
			worker->glarea->activateWindow();
			//worker->set_focus();
		}

		tri = FACETRACKER::IO::LoadTri("C:/Developer/FaceTracker-opencv2/model/face.tri");
		con = FACETRACKER::IO::LoadCon("C:/Developer/FaceTracker-opencv2/model/face.con");

		faceShapes = std::vector<std::vector<double>>();
	}

	void toggle_tracking(bool on) {
		if (on == false) return;
		mode = LIVE;
		if (sensor->spin_wait_for_data(5) == false) LOG(INFO) << "no sensor data";
		solutions->reserve(30 * 60 * 5); // fps * sec * min
		start();
	}
	void toggle_benchmark(bool on) {
		if (on == false) return;

		setInterval((1.0 / 60)*1000.0);// 10
		mode = BENCHMARK;
		start();
	}
	void toggle_playback(bool on) {
		if (on == false) return;
		setInterval((1.0 / 1000)*1000.0);
		mode = PLAYBACK;
		start();
	}
private:
	void timerEvent(QTimerEvent*) {
		process_track();
		//compute_initial_transformations();
		process_face();
	}

public:

	int speedup = 1;
	int start_frame = 0;
	std::string sharp_sequence_name = "globalSubjectA";
	//std::string current_word = "aberdeen_1_";
	int frame_setback = 0;

	void print() {
		//cout << worker->current_frame.id << endl;
		/*cout << worker->model->get_beta()[2] + worker->model->get_beta()[43] << endl;
		float thumb_top_length = worker->model->beta[worker->model->shape_dofs_name_to_id_map["thumb_top_length"]] + worker->model->beta[worker->model->shape_dofs_name_to_id_map["thumb_additional_y"]];
		float template_thumb_top_length = worker->model->beta_template[worker->model->shape_dofs_name_to_id_map["thumb_top_length"]] + worker->model->beta_template[worker->model->shape_dofs_name_to_id_map["thumb_additional_y"]];
		cout << worker->model->beta[worker->model->shape_dofs_name_to_id_map["thumb_bottom_length"]] / worker->model->beta[worker->model->shape_dofs_name_to_id_map["thumb_middle_length"]] << " ";
		cout << worker->model->beta[worker->model->shape_dofs_name_to_id_map["thumb_middle_length"]] / thumb_top_length << endl;
		cout << worker->model->beta_template[worker->model->shape_dofs_name_to_id_map["thumb_bottom_length"]] / worker->model->beta_template[worker->model->shape_dofs_name_to_id_map["thumb_middle_length"]] << " ";
		cout << worker->model->beta_template[worker->model->shape_dofs_name_to_id_map["thumb_middle_length"]] / template_thumb_top_length << endl;*/

		//cout << worker->model->beta[worker->model->shape_dofs_name_to_id_map["wrist_width_left"]] + 2.0 * worker->model->beta[worker->model->shape_dofs_name_to_id_map["wrist_top_left_radius"]]  << " < "
		//	<< worker->model->beta[worker->model->shape_dofs_name_to_id_map["wrist_width"]] + 2.0 * worker->model->beta[worker->model->shape_dofs_name_to_id_map["wrist_bottom_left_radius"]] << endl;

	}

	void process_track() {
		//if (mode == PLAYBACK) { synthetic_dataset_generator.rotate_and_render_main(); return; }
		//compute_rendered_metrics_for_recorded_rendered_sequence(); return;
		//compare(); return; //worker->E_pose.explore_pose_space(1); return;
		//perturb_model_while_tracking();
		//cout << "1" << endl;
		if (worker->settings->pause_tracking == true) {
			if (!last_was_pause){
				cout << "Paused" << endl;
				last_was_pause = true;
			}
			worker->offscreen_renderer.render_offscreen(true, false, false);
			worker->updateGL(); return;
		}
		else{
			last_was_pause = false;
		}
		if (worker->current_frame.id < 0) { worker->updateGL(); Sleep(1500); worker->current_frame.id = start_frame; }

		//cout << "2" << endl;
		//Brandon
		if (worker->current_frame.id == 50){
			cout << "Write faces triggered" << endl;
			std::ofstream outfile;
			
			std::string filepath = "C:/Data/Faces.csv";
			outfile.open(filepath, std::ofstream::app);

			for (size_t i = 0; i < faceShapes.size(); i++) {
				std::vector<double> points = faceShapes.at(i);
				int n = points.size();
				for (int j = 0; j < n; ++j){
					outfile << points.at(j) << ", " ;
				}
				outfile << endl;
				
			}
			outfile.close();
			
		}
		if (set_frame == reset_frame + 31){

			//std::cout << "Enabling calibration. Setting termination _max_iters to 15." << endl;
			//worker->set_calibration_type(FULL);
			//worker->settings->termination_max_iters = 15;
			//tracking_enabled = true;
		}

		

		reinitialize();
		print();

		//cout << "3" << endl;
		static std::clock_t tracking_start_time = std::clock();
		if (worker->settings->restart_clock) { tracking_start_time = std::clock();  worker->current_frame.id = 0; worker->settings->restart_clock = false; }

		float frame_start_time = std::clock() - tracking_start_time; if (worker->settings->report_times) std::cout << endl;
		float sensor_fetching_time, sensor_processing_time, model_outline_time, tracking_time, rendering_time;

		{ /// Fetching sensor data
			if (mode == LIVE) {
				sensor->fetch_streams_concurrently(worker->current_frame, *worker->handfinder, worker->current_frame.full_color);
			}
			if (mode == BENCHMARK) {
				load_recorded_frame();
				//cout << "4" << endl;
				fetch_recorded_depth();
				//cout << "5" << endl;
				//process_new_sequence();
			}
			sensor_fetching_time = std::clock() - tracking_start_time; if (worker->settings->report_times) std::cout << "fetching = " << sensor_fetching_time - frame_start_time << endl;
		}

		/*
		cv::Mat dummy_region = worker->current_frame.depth.clone();
		dummy_region = dummy_region(cv::Range(0, dummy_region.rows / 2), cv::Range(0, dummy_region.cols));
		//cv::Mat dummy_region = worker->current_frame.depth(cv::Range(51, 60), cv::Range(51, 60));
		//cout << "dummy region: " <<  endl;
		unsigned short min_v = 9999;
		unsigned short max_v = 0;
		for (int i = 0; i < dummy_region.cols; i++){
			for (int j = 0; j < dummy_region.rows; j++){
				if (j > dummy_region.rows / 2) dummy_region.at<DepthPixel>(j, i) = 0;
				unsigned short val = dummy_region.at<DepthPixel>(j, i);
				if (val < min_v) min_v = val;
				if (val > max_v) max_v = val;
			}
		}
		cout << "dummy region: height: " << dummy_region.rows << endl;
	//	worker->face_texture->load(dummy_region.data, worker->current_frame.id);
		*/
		{ /// Processing sensor data
			if (worker->settings->stop_tracking_without_wristband && !worker->handfinder->wristband_found()) {
				datastream->add_frame(worker->current_frame.color.data, worker->current_frame.depth.data, worker->current_frame.full_color.data);
				worker->sensor_depth_texture->load(worker->current_frame.depth.data, worker->current_frame.id);				
				write_solutions_and_tracking_metrics();
				worker->offscreen_renderer.render_offscreen(true, false, false); worker->updateGL();
				worker->current_frame.id++;
				set_frame++;
				return;
			}
			//cout << "6" << endl;

			for (int row = 0; row < worker->handfinder->sensor_silhouette.rows; row++) {
				for (int col = 0; col < worker->handfinder->sensor_silhouette.cols; col++) {
					if (worker->handfinder->sensor_silhouette.at<uchar>(row, col) > 0) continue;
					worker->current_frame.depth.at<unsigned short>(row, col) = 0;
				}
			}

			if (worker->E_fitting.settings->fit2D_outline_enable) compute_data_outline();

			//cout << "7" << endl;
			datastream->add_frame(worker->current_frame.color.data, worker->current_frame.depth.data, worker->current_frame.full_color.data);
			worker->sensor_depth_texture->load(worker->current_frame.depth.data, worker->current_frame.id);

			sensor_processing_time = std::clock() - tracking_start_time; if (worker->settings->report_times) std::cout << "processing = " << sensor_processing_time - sensor_fetching_time << endl;
		}

		{ /// Tracking
			if (worker->E_fingertips._settings.enable_fingertips_prior) {
				static DetectionStream temp_detection = DetectionStream(worker->model);
				static FindFingers find_fingers = FindFingers(&temp_detection, worker->camera, worker->handfinder, &worker->current_frame);

				static FingertipsDetector fingertips_detector = FingertipsDetector(&find_fingers);
				fingertips_detector.find_fingers_main(false);
				worker->E_fingertips.data_fingertips_2d = fingertips_detector.find_fingers->xy_finger_tips;

				if (worker->current_frame.id >= 2647 && worker->settings->dataset_type == SRIDHAR) {
					find_fingers.find_fingers_main(false);
					worker->E_fingertips.data_fingertips_2d = find_fingers.z_finger_tips;
					worker->E_fingertips.data_fingertips_2d.insert(std::end(worker->E_fingertips.data_fingertips_2d), std::begin(find_fingers.xy_finger_tips), std::end(find_fingers.xy_finger_tips));
				}

			}

			if (set_frame == 0){
				std::vector<float> theta = worker->model->get_theta();
				theta[3] = 3.0;
				theta[4] = -3.0;
				theta[5] = 3.0;
				worker->model->update_theta(theta); worker->model->update_centers();
			}

			//cout << "8" << endl;
			if (tracking_enabled) {
				tracking_failed = worker->track_till_convergence();
			}
			
			if (initialization_enabled && tracking_failed) {
				static QianDetection detection(worker);
				if (detection.can_reinitialize()) {
					detection.reinitialize();
				}
			}
			tracking_time = std::clock() - tracking_start_time; if (worker->settings->report_times) std::cout << "tracking = " << tracking_time - sensor_processing_time << endl;
		}

		{ /// Rendering
			if (tracking_enabled) worker->offscreen_renderer.render_offscreen(true, false, false);
			if (!worker->settings->run_experiments) worker->updateGL();
			if (worker->settings->run_batch_solver) glFinish();
			save_rendered_frame();

			//if (mode == BENCHMARK && real_color) display_color_and_depth_input();		
			if (real_color) display_color_and_depth_input();
			rendering_time = std::clock() - tracking_start_time; if (worker->settings->report_times) std::cout << "rendering = " << rendering_time - tracking_time << endl;
		}

		{ /// Evaluating
			write_solutions_and_tracking_metrics();
			if (worker->settings->compute_rendered_metrics || worker->settings->write_synthetic_depth)
				write_synthetic_depth_and_rendered_metrics();
			if (worker->settings->write_marker_positions || worker->settings->compute_marker_based_metrics) {
				std::vector<Vector3> model_marker_positions; std::vector<size_t> model_marker_block_indices;
				worker->ground_truth_loader->get_marker_positions(model_marker_positions, model_marker_block_indices);
				if (worker->settings->write_marker_positions) {
					worker->ground_truth_loader->write_model_marker_positions(model_marker_positions);
				}
				if (worker->settings->compute_marker_based_metrics) {
					std::string marker_based_metrics_filename = sequence_path + "marker_based_metrics.txt";
					if (worker->settings->run_experiments) {
						marker_based_metrics_filename = sequence_path + "marker_based_metrics_exp.txt";
					}
					worker->ground_truth_loader->write_marker_based_metrics(model_marker_positions, marker_based_metrics_filename, worker->current_frame.id);
				}
			}
		}

		if (worker->current_frame.id == 0) first_frame_lag = std::clock() - tracking_start_time;
		else if (worker->settings->report_times) std::cout << "average = " << (std::clock() - (tracking_start_time + first_frame_lag)) / worker->current_frame.id << endl;

		if (!worker->settings->pause_current_frame) worker->current_frame.id++;
		//if (!worker->settings->pause_current_frame) set_frame++;

		if (!handshape && !fingerspelling)	{
			if (worker->current_frame.id == sequence_length && mode == BENCHMARK) process_end_of_sequence();
		}
		else{
			set_frame++;
			if (all_sets_done){
				worker->model->set_initial_pose();
				worker->model->write_model(worker->settings->calibrated_model_path + "second/");
				process_end_of_sequence();
			}
		}
		//write_current_frame();
	}

	void reinitialize() {
		std::vector<size_t> reinitialization_frame_ids;
		if (worker->settings->dataset_type == BRANDON) {
			//reinitialization_frame_ids = { 1, 424, 939, 1279, 1671, 2112, 2646, 3154 };
			//reinitialization_frame_ids = { 1, 172 };


		}
		if (worker->settings->dataset_type == SRIDHAR) {
			//reinitialization_frame_ids = { 1, 424, 939, 1279, 1671, 2112, 2646, 3154 };
			reinitialization_frame_ids = { 1, 424, 866, 1258, 1773, 2113, 2647,
			2826, 3011, 3064,
			2719, 2861 };

		}
		if (worker->settings->dataset_type == TOMPSON) {
			reinitialization_frame_ids = { 1, 3137, 3371, 4691, 4861, 4884, 5406, 5723, 6052, 6197, 6558, 6563, 6735, 7055, 7172, 7223, 7376,
				2980, 3365, 3625, 3875, 3920, 4150, 4230, 4565, 4800, 5020, 5640, 6400, 6700 };
		}

		for (size_t i = 0; i < reinitialization_frame_ids.size(); i++) {
			if (worker->current_frame.id == reinitialization_frame_ids[i]) {
				if (worker->settings->dataset_type == BRANDON){
					current_prompt = "afghanistan_1_";
					frame_setback += reinitialization_frame_ids[i];
					std::cout << "Reinitialize at frame: " << worker->current_frame.id << "  Read from: " + sequence_path + current_prompt + "depth-" + "_______" + ".png" << std::endl;
				}
				cout << worker->current_frame.id << endl;
				worker->ground_truth_loader->enable_ground_truth_reinit = true;
				cout << "reinitialize() setting termination_max_iters to 150" << endl;
				worker->settings->termination_max_iters = 150;
			}
			if (worker->current_frame.id == reinitialization_frame_ids[i] + 3) {
				worker->ground_truth_loader->enable_ground_truth_reinit = false;
				cout << "reinitialize() setting termination_max_iters to 15" << endl;
				worker->settings->termination_max_iters = 15;
			}
		}
	}

	void process_end_of_sequence() {
		if (worker->settings->run_batch_solver || worker->settings->run_kalman_filter || worker->settings->use_online_beta) {
			if (worker->settings->run_batch_solver) {

				size_t num_iterations;
				if (worker->settings->use_online_betas_for_batch_solver)
					num_iterations = 15;
				else num_iterations = 100;

				if (worker->model->calibration_type == NONE) worker->set_calibration_type(FULL);
				worker->E_shape._settings.enable_semantic_limits = false;
				worker->E_shape._settings.enable_shape_dofs_blockers = false;

				worker->batch_solver->batch_solve(num_iterations);
			}

			{ /// Apply estimated pose

				if (worker->settings->run_batch_solver) worker->model->update_beta(worker->batch_solver->get_beta());
				if (worker->settings->run_kalman_filter) worker->model->update_beta(worker->kalman_filter->get_estimate());

				//for (size_t i = 0; i < worker->model->num_betas; i++) cout << worker->model->beta[i] << " "; cout << endl;
				cout << "betas_estimated_path = " << sequence_path + "betas_estimated.txt" << endl;
				static ofstream estimated_betas_file(sequence_path + "betas_estimated.txt");
				for (size_t i = 0; i < worker->model->num_betas; i++) estimated_betas_file << worker->model->beta[i] << " "; cout << endl;
				estimated_betas_file.close();

				cout << "writing model" << endl;
				worker->model->set_initial_pose();
				if (worker->settings->load_calibrated_model)
					worker->model->write_model(worker->model->calibrated_model_path);  //Doesn't get set if not loaded
				else
					worker->model->write_model(worker->settings->calibrated_model_path);
				
			}
		}

		this->stop();
		worker->glarea->close();
		if (tracking_metrics_file.is_open()) tracking_metrics_file.close();
		if (weighted_metrics_file.is_open()) weighted_metrics_file.close();
		if (solutions_file.is_open()) solutions_file.close();
		worker->ground_truth_loader->marker_based_metrics_file.close();

		cout << "the end of sequence was reached, sequence length was " << worker->current_frame.id << " = " << sequence_length << endl;
	}

	void fetch_recorded_depth() {
		if (worker->settings->show_iteration_calls) cout << "fetch recorded depth" << endl;

		if (!worker->settings->use_synthetic_depth) {
			if (worker->settings->dataset_type == TKACH || BRANDON) {
				worker->handfinder->binary_classification(worker->current_frame.depth, worker->current_frame.color);
			}
			if (worker->settings->dataset_type == TOMPSON) {
				worker->ground_truth_loader->get_sensor_silhouette(worker->current_frame, worker->camera, worker->handfinder->sensor_silhouette, worker->handfinder->sensor_silhouette_wrist);
			}
			if (worker->settings->dataset_type == SHARP) {
				/// Load labels
				std::string dataset_path = "C:/Data/datasets/FingerPaint/";
				std::ostringstream stringstream;
				stringstream << std::setw(4) << std::setfill('0') << worker->current_frame.id + 1;
				cv::Mat labels = cv::imread(dataset_path + sharp_sequence_name + "_labels/" + sharp_sequence_name + "_labels_" + stringstream.str() + ".png");
				cv::flip(labels, labels, 1);
				worker->handfinder->sensor_silhouette = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
				if (worker->settings->fit_wrist_separately)
					worker->handfinder->sensor_silhouette_wrist = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));

				/// Find palm center
				Vector3 palm_center = Vector3(0, 0, 0);
				int counter = 0;
				for (int row = 0; row < labels.rows; row++) {
					for (int col = 0; col < labels.cols; col++) {
						if (labels.at<cv::Vec<unsigned char, 3>>(row, col) == cv::Vec<unsigned char, 3>(255, 255, 255)) continue;
						if (labels.at<cv::Vec<unsigned char, 3>>(row, col) == cv::Vec<unsigned char, 3>(0, 0, 0)) continue;

						unsigned short value = worker->current_frame.depth.at<unsigned short>(row, col);
						Vector3 p = value * worker->camera->inv_projection_matrix() * Vector3(col, row, 1);
						palm_center += p;
						counter++;
					}
				}
				palm_center /= counter;

				/// Set sensor_silhouette to zero outside of hand sphere
				Scalar crop_radius_sq = worker->handfinder->settings->crop_radius * worker->handfinder->settings->crop_radius;
				for (int row = 0; row < labels.rows; ++row) {
					for (int col = 0; col < labels.cols; ++col) {

						if (labels.at<cv::Vec<unsigned char, 3>>(row, col) == cv::Vec<unsigned char, 3>(255, 255, 255)) continue;

						if (labels.at<cv::Vec<unsigned char, 3>>(row, col) != cv::Vec<unsigned char, 3>(0, 0, 0)) {
							worker->handfinder->sensor_silhouette.at<unsigned char>(row, col) = 255;
						}
						else {
							unsigned short value = worker->current_frame.depth.at<unsigned short>(row, col);
							Vector3 p = value * worker->camera->inv_projection_matrix() * Vector3(col, row, 1);

							if ((p - palm_center).squaredNorm() < crop_radius_sq) {
								worker->handfinder->sensor_silhouette.at<unsigned char>(row, col) = 255;

								if (worker->settings->fit_wrist_separately) {
									worker->handfinder->sensor_silhouette_wrist.at<unsigned char>(row, col) = 255;
								}
							}
						}
					}
				}
			}
			if (worker->settings->dataset_type == SRIDHAR) {

				worker->handfinder->sensor_silhouette = cv::Mat::zeros(worker->current_frame.depth.rows, worker->current_frame.depth.cols, CV_8UC1);
				for (int row = 0; row < worker->current_frame.depth.rows; ++row) {
					for (int col = 0; col < worker->current_frame.depth.cols; ++col) {
						if (worker->current_frame.depth.at<unsigned short>(row, col) < 32001)
							worker->handfinder->sensor_silhouette.at<uchar>(row, col) = 255;
					}
				}

				cv::Mat labels, stats, centroids;
				int num_components = cv::connectedComponentsWithStats(worker->handfinder->sensor_silhouette, labels, stats, centroids, 4);

				std::vector< int > components_indices(num_components);
				std::iota(components_indices.begin(), components_indices.end(), 0);
				auto compare_areas = [stats](int i1, int i2) {
					int area1 = stats.at<int>(i1, cv::CC_STAT_AREA);
					int area2 = stats.at<int>(i2, cv::CC_STAT_AREA);
					return area1 > area2;
				};
				std::sort(components_indices.begin(), components_indices.end(), compare_areas);
				worker->handfinder->sensor_silhouette = (labels == components_indices[1]);

				/// Extract wristband average depth
				float average_depth = 0;
				int average_depth_count = 0;
				for (int row = 0; row < worker->current_frame.depth.rows; ++row) {
					for (int col = 0; col < worker->current_frame.depth.cols; ++col) {
						float depth = worker->current_frame.depth.at<unsigned short>(row, col);
						if (worker->handfinder->sensor_silhouette.at<unsigned char>(row, col) == 255) {
							average_depth += depth;
							average_depth_count++;
						}
					}
				}
				average_depth = average_depth / average_depth_count;
				cv::inRange(worker->current_frame.depth, average_depth - worker->handfinder->settings->depth_range / 2,
					average_depth + worker->handfinder->settings->depth_range / 2, worker->handfinder->sensor_silhouette);

				if (worker->settings->fit_wrist_separately)
					worker->handfinder->sensor_silhouette_wrist = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
			}
		}
		else {
			worker->handfinder->sensor_silhouette = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
			for (int row = 0; row < worker->current_frame.depth.rows; ++row) {
				for (int col = 0; col < worker->current_frame.depth.cols; ++col) {
					if (worker->current_frame.depth.at<unsigned short>(row, col) < RAND_MAX)
						worker->handfinder->sensor_silhouette.at<uchar>(row, col) = 255;
				}
			}
			if (worker->settings->fit_wrist_separately) {
				ostringstream stringstream;
				stringstream << std::setw(7) << std::setfill('0') << worker->current_frame.id;
				cv::Mat mask = cv::imread(sequence_path + "mask-" + stringstream.str() + ".png", cv::IMREAD_ANYCOLOR);
				worker->handfinder->sensor_silhouette_wrist = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
				for (int row = 0; row < worker->current_frame.depth.rows; ++row) {
					for (int col = 0; col < worker->current_frame.depth.cols; ++col) {
						if (worker->current_frame.depth.at<unsigned short>(row, col) < RAND_MAX) {
							if (mask.at<unsigned char>(row, col) == 28 || mask.at<unsigned char>(row, col) == 29)
								worker->handfinder->sensor_silhouette_wrist.at<uchar>(row, col) = 255;
						}
					}
				}
			}
		}

		//cv::imshow("sensor_silhouette", worker->handfinder->sensor_silhouette); cv::waitKey(3);
		if ((worker->settings->dataset_type == TKACH || worker->settings->dataset_type == BRANDON) && !worker->settings->use_synthetic_depth) {
			worker->handfinder->binary_classification_wrist(worker->current_frame.depth);
		}

		if (worker->settings->show_iteration_calls) cout << "handinder: get sensor indicator" << endl;
		worker->handfinder->get_sensor_indicator();

		if (!worker->current_frame.depth.data) {
			std::cout << "no depth data"; return;
		}
	}

	void write_solutions_and_tracking_metrics() {
		solutions->resize(datastream->size());
		if (!worker->settings->calibrate)
			solutions->set(datastream->size() - 1, worker->model->num_parameters, worker->model->get_theta());
		else
			solutions->set(datastream->size() - 1, worker->model->num_parameters, worker->model->get_theta(), worker->model->get_beta());

		if (mode == BENCHMARK) {
			std::string tracking_metrics_filename("");// = sequence_path + "online_continuous_metrics.txt";
			std::string weighted_metrics_filename("");// = sequence_path + "online_weighted_metrics.txt";
			if (handshape){
				tracking_metrics_filename = worker->settings->calibrated_model_path + "set" + std::to_string(hs_set_num) + "/" + current_prompt + "online_continuous_metrics.txt";
				weighted_metrics_filename = worker->settings->calibrated_model_path + "set" + std::to_string(hs_set_num) + "/" + current_prompt + "online_weighted_metrics.txt";
			}
			else{
				tracking_metrics_filename = worker->settings->calibrated_model_path + current_prompt + "online_continuous_metrics.txt";
				weighted_metrics_filename = worker->settings->calibrated_model_path + current_prompt + "online_weighted_metrics.txt";
			}
			if (worker->settings->run_experiments) {
				tracking_metrics_filename = sequence_path + "online_continuous_metrics_exp.txt";
				weighted_metrics_filename = sequence_path + "online_weighted_metrics_exp.txt";
			}

			if (!tracking_metrics_file.is_open()) tracking_metrics_file = ofstream(tracking_metrics_filename);
			if (tracking_metrics_file.is_open()) tracking_metrics_file << worker->tracking_error.pull_error << " " << worker->tracking_error.push_error << endl;

			if (!weighted_metrics_file.is_open()) weighted_metrics_file = ofstream(weighted_metrics_filename);
			if (weighted_metrics_file.is_open()) weighted_metrics_file << worker->tracking_error.weighted_error << endl;
		}
		if (handshape){
			if (!solutions_file.is_open()) solutions_file = ofstream(worker->settings->calibrated_model_path + "set" + std::to_string(hs_set_num) + "/" + current_prompt + "solutions.txt");
		}
		else{
			if (!solutions_file.is_open()) solutions_file = ofstream(worker->settings->calibrated_model_path + current_prompt + "solutions.txt");
		}
		//if (!solutions_file.is_open()) solutions_file = ofstream(sequence_path + "solutions.txt");
		if (solutions_file.is_open()) solutions_file << solutions->frames[datastream->size() - 1].transpose() << endl;

		if (worker->settings->write_estimated_certainties && worker->settings->run_kalman_filter) {
			static ofstream estimated_certainties_file(sequence_path + "estimated_certainties.txt");
			for (size_t i = 0; i < worker->model->num_betas; i++) estimated_certainties_file << worker->kalman_filter->estimated_hessian(i, i) << " ";
			estimated_certainties_file << endl;

			static ofstream measured_certainties_file(sequence_path + "measured_certainties.txt");
			for (size_t i = 0; i < worker->model->num_betas; i++) measured_certainties_file << worker->kalman_filter->measured_hessian(i, i) << " ";
			measured_certainties_file << endl;

			static ofstream estimated_values_file(sequence_path + "estimated_values.txt");
			for (size_t i = 0; i < worker->model->num_betas; i++) estimated_values_file << worker->kalman_filter->estimated_values(i) << " ";
			estimated_values_file << endl;

			static ofstream measured_values_file(sequence_path + "measured_values.txt");
			for (size_t i = 0; i < worker->model->num_betas; i++) measured_values_file << worker->kalman_filter->measured_values(i) << " ";
			measured_values_file << endl;
		}
	}

	void write_synthetic_depth_and_rendered_metrics() {
		static cv::Mat rendered_model, rendered_silhouette;
		{ // fetch texture
			if (rendered_model.empty()) rendered_model = cv::Mat(worker->camera->height(), worker->camera->width(), CV_16UC1, cv::Scalar(0));
			glBindTexture(GL_TEXTURE_2D, worker->offscreen_renderer.frame_buffer->depth_texture_id());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, rendered_model.data);
			glBindTexture(GL_TEXTURE_2D, 0);
			cv::flip(rendered_model, rendered_model, 0);

			if (rendered_silhouette.empty()) rendered_silhouette = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
			glBindTexture(GL_TEXTURE_2D, worker->offscreen_renderer.frame_buffer->silhouette_texture_id());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, rendered_silhouette.data);
			glBindTexture(GL_TEXTURE_2D, 0);
			cv::flip(rendered_silhouette, rendered_silhouette, 0);
		}

		if (worker->settings->write_synthetic_depth) {
			std::string path = "C:/Data/sensor-sequences/experiment/synthetic/";
			CreateDirectory(path.c_str(), NULL);
			online_performance_metrics.write_rendered_model(rendered_model, rendered_silhouette, worker->current_frame.id, path);
		}

		if (worker->settings->compute_rendered_metrics) {
			float pull_error = online_performance_metrics.compute_rastorized_3D_metric(
				rendered_model, worker->current_frame.depth, worker->handfinder->sensor_silhouette, worker->camera->inv_projection_matrix());
			float push_error = online_performance_metrics.compute_rastorized_2D_metric(
				rendered_model, worker->handfinder->sensor_silhouette, worker->E_fitting.distance_transform.idxs_image());

			static ofstream rendered_metrics_file(sequence_path + "online_rendered_metrics.txt");
			if (rendered_metrics_file.is_open()) {
				rendered_metrics_file << pull_error << " " << push_error << endl;
			}
		}
	}

	void load_recorded_parameters(std::string solution_path) {
		std::cout << "loading solutions function" << endl;
		std::ifstream solutions_file(solution_path);
		if (!solutions_file.is_open()) { std::cout << "cannot open solution file at " + solution_path << endl; exit(0); }
		int row = 0;
		for (std::string line; std::getline(solutions_file, line); row++) {
			//cout << row << endl;
			solutions->frames.push_back(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(num_thetas + worker->model->num_betas, 1));
			stringstream string_stream(line);
			std::string elem; size_t col = 0;

			while (string_stream >> elem) {
				if (col >= num_thetas + worker->model->num_betas) cout << "col = " << col << endl;
				solutions->frames[row](col++) = std::stof(elem);
			}
		}
		cout << "number of loaded solutions = " << solutions->frames.size() << endl;
		solutions_file.close();
	}

	void load_certainties(std::string path) {

		std::ifstream input_file(path);
		if (!input_file.is_open()) { std::cout << "cannot open estimated certainties file" << endl; exit(0); }
		int row = 0;
		for (std::string line; std::getline(input_file, line); row++) {
			certainties_vector.push_back(std::vector<float>(worker->model->num_betas, 0));
			stringstream string_stream(line);
			std::string elem; size_t col = 0;

			while (string_stream >> elem) {
				certainties_vector[row][col++] = std::stof(elem);
			}
		}
		input_file.close();
	}

	void load_recorded_frame() {
		if (worker->settings->show_iteration_calls) cout << "load recorded frame" << endl;

		std::ostringstream stringstream;
		if (worker->settings->dataset_type == TKACH) {
			if (handshape){
				std::string frame_str = std::string(7 - std::to_string(set_frame).length(), '0') + std::to_string(set_frame);
				//stringstream << std::setw(7) << std::setfill('0') << set_frame; // (worker->current_frame.id - frame_setback);
				std::string set_string = "set" + std::to_string(hs_set_num) + "/";
				std::string checkFile = sequence_path + set_string + current_prompt + "/depth-" + frame_str + ".png";
				//std::cout << "Checking: " << checkFile << endl;

				ifstream ftest(checkFile.c_str());
				while (ftest.fail() && !all_sets_done){
					std::cout << "Failed to Load: " << checkFile << endl;					
					if (prompt_i >= prompts.size()){
						hs_set_num++;
						if (hs_set_num > 3){
							std::cout << "End of all sets " << std::endl;
							all_sets_done = true;
						}
					}
					else{
						current_prompt = prompts[prompt_i++];
						set_frame = 0; 
						cout << "Next prompt: " << current_prompt << ". prompt_i: " << prompt_i << " of prompts.size(): " << prompts.size() << endl;
						//worker->model->set_initial_pose();
						cout << " model reset to initial pose" << endl;
						frame_str = std::string(7 - std::to_string(set_frame).length(), '0') + std::to_string(set_frame);
						stringstream << std::setw(7) << std::setfill('0') << set_frame; // (worker->current_frame.id - frame_setback);
						checkFile = sequence_path + set_string + current_prompt + "/depth-" + frame_str + ".png";
						ftest.open(checkFile.c_str());
					}
				}

				if (!all_sets_done){
					//cout << "reading: " << sequence_path + current_prompt + "/depth-" + frame_str + ".png" << endl;
					worker->current_frame.depth = cv::imread(sequence_path + set_string + current_prompt + "/depth-" + frame_str + ".png", cv::IMREAD_ANYDEPTH);
					if (!worker->settings->use_synthetic_depth) {
						//cout << "reading: " << sequence_path + current_prompt + "/color-" + frame_str + ".png" << endl;;
						worker->current_frame.color = cv::imread(sequence_path + set_string + current_prompt + "/color-" + frame_str + ".png");
						//worker->model->real_color = cv::imread(sequence_path + "full_color-" + stringstream.str() + ".png");
					}
				}

			}
			else if (fingerspelling){

				std::string frame_str = std::string(7 - std::to_string(set_frame).length(), '0') + std::to_string(set_frame);
				//stringstream << std::setw(7) << std::setfill('0') << set_frame; // (worker->current_frame.id - frame_setback);
				std::string checkFile = sequence_path + current_prompt + "_1_depth-" + frame_str + ".png";
				//std::cout << "Checking: " << checkFile << endl;

				ifstream ftest(checkFile.c_str());
				while (ftest.fail() && !all_sets_done){
					std::cout << "Failed to Load: " << checkFile << endl;					
					if (prompt_i >= prompts.size()){
						std::cout << "End of all sets " << std::endl;
						all_sets_done = true;
					}
					else{
						current_prompt = prompts[prompt_i++];
						set_frame = 0;
						cout << "Next prompt: " << current_prompt << ". prompt_i: " << prompt_i << " of prompts.size(): " << prompts.size() << endl;
						reset_hand();
						//worker->model->set_initial_pose();
						//cout << " model reset to initial pose" << endl;
						frame_str = std::string(7 - std::to_string(set_frame).length(), '0') + std::to_string(set_frame);
						stringstream << std::setw(7) << std::setfill('0') << set_frame; // (worker->current_frame.id - frame_setback);
						checkFile = sequence_path + current_prompt + "_1_depth-" + frame_str + ".png";
						ftest.open(checkFile.c_str());
					}
				}

				if (!all_sets_done){
					//cout << std::to_string(set_frame) << " ";
					//cout << "reading: " << sequence_path + current_prompt + "_1_depth-" + frame_str + ".png" << endl;
					if (worker->settings->handedness == LEFT_HAND){
						worker->current_frame.depth = cv::imread(sequence_path + current_prompt + "_1_depth-" + frame_str + ".png", cv::IMREAD_ANYDEPTH);
					}
					else{
						cv::Mat temp = cv::imread(sequence_path + current_prompt + "_1_depth-" + frame_str + ".png", cv::IMREAD_ANYDEPTH);
						cv::flip(temp, temp, 1);
						worker->current_frame.depth = temp; // cv::imread(sequence_path + current_prompt + "_1_depth-" + frame_str + ".png", cv::IMREAD_ANYDEPTH);
					}
					if (!worker->settings->use_synthetic_depth) {
						//cout << "reading: " << sequence_path + current_prompt + "/color-" + frame_str + ".png" << endl;;
						if (worker->settings->handedness == LEFT_HAND){
							worker->current_frame.color = cv::imread(sequence_path + current_prompt + "_1_color-" + frame_str + ".png");
						}
						else{
							cv::Mat temp = cv::imread(sequence_path + current_prompt + "_1_color-" + frame_str + ".png");
							cv::flip(temp, temp, 1);
							worker->current_frame.color = temp;
						}
						//worker->model->real_color = cv::imread(sequence_path + "full_color-" + stringstream.str() + ".png");
					}
				}
			}
			else{
				stringstream << std::setw(7) << std::setfill('0') << (worker->current_frame.id - frame_setback);
				//std::cout << sequence_path + "sensor_data/" + "depth-" + stringstream.str() + ".png" << endl;
				worker->current_frame.depth = cv::imread(sequence_path + "sensor_data/" + "depth-" + stringstream.str() + ".png", cv::IMREAD_ANYDEPTH);
				if (!worker->settings->use_synthetic_depth) {
					worker->current_frame.color = cv::imread(sequence_path + "sensor_data/" + "color-" + stringstream.str() + ".png");
					//worker->model->real_color = cv::imread(sequence_path + "full_color-" + stringstream.str() + ".png");
				}
			}
		}
		/*
		if (worker->settings->dataset_type == TKACH) {
			stringstream << std::setw(7) << std::setfill('0') << worker->current_frame.id;
			//std::cout << sequence_path + "depth-" + stringstream.str() + ".png" << endl;
			worker->current_frame.depth = cv::imread(sequence_path + "depth-" + stringstream.str() + ".png", cv::IMREAD_ANYDEPTH);
			if (!worker->settings->use_synthetic_depth) {
				worker->current_frame.color = cv::imread(sequence_path + "color-" + stringstream.str() + ".png");
				//worker->model->real_color = cv::imread(sequence_path + "full_color-" + stringstream.str() + ".png");
			}
		}
		*/
		if (worker->settings->dataset_type == TOMPSON) {
			std::string dataset_path = "C:/Data/datasets/nyu_hand_dataset_v2/test/";
			stringstream << std::setw(7) << std::setfill('0') << worker->current_frame.id + 1;
			//in each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue
			cv::Mat input_image = cv::imread(dataset_path + "depth_1_" + stringstream.str() + ".png", cv::IMREAD_COLOR);
			worker->current_frame.depth = cv::Mat(worker->camera->height(), worker->camera->width(), CV_16UC1, cv::Scalar(0));
			for (int y = 0, y_sub = 0; y_sub < worker->camera->height(); y += 2, y_sub++) {
				for (int x = 0, x_sub = 0; x_sub < worker->camera->width(); x += 2, x_sub++) {
					cv::Vec3b pixel = input_image.at<cv::Vec3b>(y, x);
					uchar b = pixel[0]; uchar g = pixel[1];
					unsigned short result = (g << 8) + b;
					if (result == 2001) result = 0;
					worker->current_frame.depth.at<unsigned short>(y_sub, x_sub) = result;
				}
			}
		}
		if (worker->settings->dataset_type == SHARP) {
			std::string dataset_path = "C:/Data/datasets/FingerPaint/";
			stringstream << std::setw(4) << std::setfill('0') << worker->current_frame.id + 1;
			worker->current_frame.depth = 0.1 * cv::imread(dataset_path + sharp_sequence_name + "_depth/" + sharp_sequence_name + "_depth_" + stringstream.str() + ".png", cv::IMREAD_ANYDEPTH);
			cv::flip(worker->current_frame.depth, worker->current_frame.depth, 1);
		}
		if (worker->settings->dataset_type == SRIDHAR) {
			std::string dataset_path = "C:/Data/datasets/dexter1/data/ALL/depth/";
			stringstream << std::setw(4) << std::setfill('0') << worker->current_frame.id + 1;
			worker->current_frame.depth = cv::imread(dataset_path + stringstream.str() + ".png", cv::IMREAD_ANYDEPTH);
		}
		bool display = false;
		if (display) {
			cv::Mat normalized_depth;
			cv::normalize(worker->current_frame.depth, normalized_depth, 0, std::numeric_limits<unsigned short>::max(), cv::NORM_MINMAX);
			cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);
			cv::imshow("depth", normalized_depth); cv::waitKey(3);
		}
	}

	void display_color_and_depth_input() {
		/*cv::Mat normalized_depth = worker->current_frame.depth.clone();
		cv::inRange(normalized_depth, worker->camera->zNear(), worker->camera->zFar(), normalized_depth);
		cv::normalize(normalized_depth, normalized_depth, 127, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::resize(normalized_depth, normalized_depth, cv::Size(2 * normalized_depth.cols, 2 * normalized_depth.rows), cv::INTER_CUBIC);//resize image
		cv::moveWindow("DEPTH", 592, 855); cv::imshow("DEPTH", normalized_depth);*/
		if (true){
		//if (glwidget->display_model_outline) {
			cv::Mat real_color = worker->current_frame.full_color; //cv::flip(worker->current_frame.full_color, real_color, 0);
			//cv::namedWindow("RGB");	cv::moveWindow("RGB", 592, 375); 
			cv::imshow("RGB", real_color);
		}
	}

	void save_rendered_frame(){
		bool write_image = true;
		if (write_image) {
			std::ostringstream stringstream; stringstream << std::setw(4) << std::setfill('0') << set_frame;
			std::string img_folder = worker->settings->calibrated_model_path + "images/" + current_prompt;
			if (handshape)
				img_folder = worker->settings->calibrated_model_path + "set" + std::to_string(hs_set_num) + "/images/" + current_prompt;

			if (GetFileAttributesA(img_folder.c_str()) == INVALID_FILE_ATTRIBUTES)
				CreateDirectory(img_folder.c_str(), NULL);

			std::string image_name = worker->settings->calibrated_model_path + "images/" + current_prompt + "/" + current_prompt + stringstream.str() + ".jpg";
			saveRendering(image_name);
		}
	}


	void saveRendering(std::string filename) {

		int width = 640 * 2;
		int height = 480 * 2;
		unsigned char* imageData = (unsigned char *)malloc((int)(width*height*(3)));
		glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, imageData);

		FILE *filePtr = fopen(filename.c_str(), "wb");
		if (!filePtr)
			return;

		BITMAPFILEHEADER bitmapFileHeader;
		BITMAPINFOHEADER bitmapInfoHeader;

		bitmapFileHeader.bfType = 0x4D42; //"BM"
		bitmapFileHeader.bfSize = width*height * 3;
		bitmapFileHeader.bfReserved1 = 0;
		bitmapFileHeader.bfReserved2 = 0;
		bitmapFileHeader.bfOffBits =
			sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

		bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
		bitmapInfoHeader.biWidth = width;
		bitmapInfoHeader.biHeight = height;
		bitmapInfoHeader.biPlanes = 1;
		bitmapInfoHeader.biBitCount = 24;
		bitmapInfoHeader.biCompression = BI_RGB;
		bitmapInfoHeader.biSizeImage = 0;
		bitmapInfoHeader.biXPelsPerMeter = 0; // ?
		bitmapInfoHeader.biYPelsPerMeter = 0; // ?
		bitmapInfoHeader.biClrUsed = 0;
		bitmapInfoHeader.biClrImportant = 0;

		fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);
		fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);
		fwrite(imageData, width*height * 3, 1, filePtr);
		fclose(filePtr);

		free(imageData);
	}

	void write_current_frame() {
		size_t frame_index = datastream->get_last_frame_index();
		cout << "writing frame " << frame_index << std::endl;
		string writing_path = sequence_path;

		std::ostringstream stringstream;
		stringstream << std::setw(7) << std::setfill('0') << frame_index + 1;
		std::string filename = writing_path + "mask-" + stringstream.str() + ".png";
		cv::imwrite(filename, worker->handfinder->sensor_silhouette);

		//worker->model->write_model(writing_path, frame_index);
	}

	void compute_data_outline() {

		//add_boarders_to_depth();

		if (worker->settings->fit_wrist_separately) {
			for (size_t i = 0; i < worker->handfinder->sensor_silhouette.rows; i++) {
				for (size_t j = 0; j < worker->handfinder->sensor_silhouette.cols; j++) {
					if (worker->handfinder->sensor_silhouette.at<unsigned char>(i, j) != 255) continue;
					if (worker->handfinder->sensor_silhouette_wrist.at<unsigned char>(i, j) == 255) {
						worker->handfinder->sensor_silhouette.at<unsigned char>(i, j) = 0;
					}
				}
			}
		}

		cv::Mat depth = worker->current_frame.depth.clone();

		/// > Close hand mask
		cv::Mat mask;
		int closing_size = 2;
		cv::Mat closing_element = cv::getStructuringElement(2, cv::Size(2 * closing_size + 1, 2 * closing_size + 1), cv::Point(closing_size, closing_size));
		cv::morphologyEx(worker->handfinder->sensor_silhouette, mask, cv::MORPH_CLOSE, closing_element);

		cv::Mat labels, stats, centroids;
		cv::Mat inverted_mask;
		bitwise_not(mask, inverted_mask);
		int num_components = cv::connectedComponentsWithStats(inverted_mask, labels, stats, centroids, 4);

		std::vector< int > components_indices(num_components);
		std::iota(components_indices.begin(), components_indices.end(), 0);
		auto compare_areas = [stats](int i1, int i2) {
			int area1 = stats.at<int>(i1, cv::CC_STAT_AREA);
			int area2 = stats.at<int>(i2, cv::CC_STAT_AREA);
			return area1 > area2;
		};
		std::sort(components_indices.begin(), components_indices.end(), compare_areas);
		mask = (labels != components_indices[0]);

		worker->handfinder->sensor_silhouette_closed = mask;

		/// > Inpaint the depth image
		std::unordered_map<int, int> parent_locations;
		cv::Mat visited = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
		struct node {
			int row; int col;
			int parent_row; int parent_col;
			unsigned short parent_value;
			node(int row, int col, unsigned short parent_value, int parent_row, int parent_col) {
				this->row = row; this->col = col;
				this->parent_value = parent_value;
				this->parent_row = parent_row; this->parent_col = parent_col;
			}
		};
		auto push_gap_neighbours = [=](int row, int col, std::queue<node> & neighbours, int parent_row, int parent_col) {
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (u == 0 && v == 0) continue;
					if (mask.at<uchar>(row + u, col + u) == 0) continue;
					if (visited.at<uchar>(row + u, col + v) == 1) continue;
					if (row + u < 0 || row + u >= worker->camera->height()) continue;
					if (col + v < 0 || col + v >= worker->camera->width()) continue;
					if (depth.at<unsigned short>(row + u, col + v) != 0) continue;
					if (abs(row - parent_row) > 0 || abs(col - parent_col) > 0) continue;
					neighbours.push(node(row + u, col + v, depth.at<unsigned short>(parent_row, parent_col), parent_row, parent_col));

				}
			}
		};
		std::queue<node> neighbours;
		for (int row = 0; row < mask.rows; row++) {
			for (int col = 0; col < mask.cols; col++) {
				if (visited.at<uchar>(row, col) == 1) continue;
				if (mask.at<uchar>(row, col) == 0) continue;
				if (depth.at<unsigned short>(row, col) == 0) continue;
				if (worker->current_frame.depth.at<unsigned short>(row, col) == 0)
					cout << "error in entry" << endl;
				push_gap_neighbours(row, col, neighbours, row, col);

				while (!neighbours.empty()) {
					node current = neighbours.front(); neighbours.pop();
					if (visited.at<uchar>(current.row, current.col) == 1) continue;
					depth.at<unsigned short>(current.row, current.col) = current.parent_value;
					parent_locations[current.col * worker->camera->height() + current.row] = current.parent_col * worker->camera->height() + current.parent_row;
					visited.at<uchar>(current.row, current.col) = 1;

					push_gap_neighbours(current.row, current.col, neighbours, current.parent_row, current.parent_col);
				}
			}
		}

		/// > Find outline
		worker->handfinder->sensor_outline = cv::Mat(worker->camera->height(), worker->camera->width(), CV_8UC1, cv::Scalar(0));
		for (int row = 0; row < mask.rows; row++) {
			for (int col = 0; col < mask.cols; col++) {
				if (mask.at<unsigned char>(row, col) == 0) continue;
				for (int u = -1; u <= 1; u++) {
					for (int v = -1; v <= 1; v++) {
						if (u == 0 && v == 0) continue;
						if (row + u < 0 || row + u >= worker->camera->height()) continue;
						if (col + v < 0 || col + v >= worker->camera->width()) continue;
						//if (mask.at<uchar>(row + u, col + v) == 0) continue;
						if (mask.at<uchar>(row + u, col + v) == 0) worker->handfinder->sensor_outline.at<unsigned char>(row, col) = 255;
						if (depth.at<unsigned short>(row, col) < depth.at<unsigned short>(row + u, col + v) - 5) {
							int parent_hash = parent_locations[col * worker->camera->height() + row];
							int parent_row = parent_hash % worker->camera->height();
							int parent_col = parent_hash / worker->camera->height();

							if (worker->current_frame.depth.at<unsigned short>(parent_row, parent_col) == 0) continue;

							worker->handfinder->sensor_outline.at<unsigned char>(parent_row, parent_col) = 255;
						}
					}
				}
			}
		}
	}

	void compute_rendered_metrics_for_recorded_rendered_sequence() {
		if (worker->current_frame.id < 0) worker->current_frame.id = 0;
		std::string input_path = "C:/Data/msr-comparison/";
		std::ostringstream stringstream; stringstream << std::setfill('0') << std::setw(7) << worker->current_frame.id;

		//std::string model_filename = input_path + "Taylor/model-" + stringstream.str() + ".png";
		//std::string model_filename = input_path + "Tkach_2016/model-" + stringstream.str() + ".png";
		//std::string model_filename = input_path + "\Sharp_2015/" + std::to_string(worker->current_frame.id) + "-Rendered depth---image.png";
		//std::string model_filename = "C:/Data/msr-comparison/guess_who_Taylor16/" + worker->settings->sequence_name + std::to_string(worker->current_frame.id) + "-Rendered depth---image.png";
		std::string model_filename = "C:/Data/sensor-sequences/experiment/synthetic/depth-" + stringstream.str() + ".png";
		//std::string model_filename = "C:/Data/honline-data/guess_who_dataset_rendered/" + worker->settings->sequence_name + "/hadjust_rendered/depth-" + stringstream.str() + ".png";
		std::cout << model_filename << endl;

		cv::Mat model_image = cv::imread(model_filename, CV_LOAD_IMAGE_UNCHANGED);

		load_recorded_frame();
		worker->handfinder->binary_classification(worker->current_frame.depth, worker->current_frame.color);
		cv::Mat data_image = worker->current_frame.depth;
		cv::Mat mask_image = worker->handfinder->sensor_silhouette;

		/// Crop data
		for (int row = 0; row < mask_image.rows; row++) {
			for (int col = 0; col < mask_image.cols; col++) {
				if (mask_image.at<uchar>(row, col) > 0) continue;
				data_image.at<unsigned short>(row, col) = 0;
			}
		}

		/// Crop rendered model
		Scalar crop_radius_sq = worker->handfinder->settings->crop_radius *  worker->handfinder->settings->crop_radius;
		Vector3 crop_center = worker->handfinder->wristband_center() + worker->handfinder->wristband_direction() * (worker->handfinder->settings->crop_radius - worker->handfinder->settings->wrist_band_length);

		for (int row = 0; row < model_image.rows; ++row) {
			for (int col = 0; col < model_image.cols; ++col) {
				int z = model_image.at<unsigned short>(row, col);
				Vector3 p_pixel = worker->camera->depth_to_world(col, row, z);
				if ((p_pixel - crop_center).squaredNorm() >= crop_radius_sq)
					model_image.at<unsigned short>(row, col) = 0;
			}
		}

		{ /// Compute rendered metrics

			float pull_error = online_performance_metrics.compute_rastorized_3D_metric(model_image, worker->current_frame.depth, worker->handfinder->sensor_silhouette, worker->camera->inv_projection_matrix());

			static cv::Mat sensor_silhouette_flipped;
			cv::flip(worker->handfinder->sensor_silhouette, sensor_silhouette_flipped, 0);
			worker->E_fitting.distance_transform.exec(sensor_silhouette_flipped.data, 125);
			float push_error = online_performance_metrics.compute_rastorized_2D_metric(model_image, worker->handfinder->sensor_silhouette, worker->E_fitting.distance_transform.idxs_image());

			static ofstream rendered_metrics_file(sequence_path + "rendered_metrics_honline_new.txt");
			if (rendered_metrics_file.is_open()) {
				rendered_metrics_file << pull_error << " " << push_error << endl;
			}
		}

		/// Display
		{
			std::vector<cv::Mat> merge_channels_vector;

			/// First channel
			cv::Mat empty_channel = cv::Mat::zeros(worker->camera->height(), worker->camera->width(), CV_8UC1); empty_channel.setTo(255);
			merge_channels_vector.push_back(empty_channel);

			/// Second channel
			cv::Mat normalized_data_image;
			cv::normalize(data_image, normalized_data_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			merge_channels_vector.push_back(normalized_data_image);

			/// Third channel
			cv::Mat normalized_model_image;
			cv::normalize(model_image, normalized_model_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			merge_channels_vector.push_back(normalized_model_image);

			/// Merge the channels
			cv::Mat merge_channels_image;
			cv::merge(merge_channels_vector, merge_channels_image);
			cv::imshow("merge_channels_image", merge_channels_image); cv::waitKey(3);
		}
		worker->current_frame.id++;
	}

	void perturb_model_while_tracking() {
		static std::vector<float> beta_perturbation = worker->model->get_beta();
		static int frame_index = -1;
		std::ostringstream stringstream;
		stringstream << std::setw(7) << std::setfill('0') << frame_index;
		std::string synthetic_path = "C:/Data/sensor-sequences/synthetic/";
		std::string filename = synthetic_path + "depth-" + stringstream.str() + ".png";
		if (frame_index > 0) {
			cv::imwrite(filename, worker->current_frame.depth);
		}
		frame_index++;

		static bool pass_throught = true;
		static int pause_frames_count = 0;

		if (worker->current_frame.id % worker->settings->frames_interval_between_measurements == 0 && worker->settings->pause_tracking == false && (pass_throught = !pass_throught)) {
			worker->model->perturb_parameters(frame_index * 1000 % RAND_MAX, worker->settings->uniform_scaling_mean, 0.5, 0.0);
			beta_perturbation = worker->model->get_beta();
			worker->offscreen_renderer.render_offscreen(true, false, false);
			worker->updateGL(); pause_frames_count = 15;
		}

		worker->settings->pause_tracking = (pause_frames_count--) > 0;

		if (worker->settings->pause_tracking == true) {
			for (size_t i = 0; i < num_thetas; i++) solutions_file << worker->model->theta[i] << " ";
			for (size_t i = 0; i < worker->model->num_betas; i++) solutions_file << worker->model->beta[i] << " ";
			solutions_file << endl;

			if (weighted_metrics_file.is_open()) weighted_metrics_file << worker->tracking_error.weighted_error << endl;

			worker->model->radii[34] = worker->model->radii_template[34] * worker->model->beta_latent[0];
			worker->model->radii[35] = worker->model->radii_template[35] * worker->model->beta_latent[0];
			worker->model->radii[36] = worker->model->radii_template[36] * worker->model->beta_latent[0];
			worker->model->radii[37] = worker->model->radii_template[37] * worker->model->beta_latent[0];
		}

		static ofstream beta_perturbation_file(synthetic_path + "beta_perturbation.txt");
		for (size_t i = 0; i < num_thetas; i++) beta_perturbation_file << worker->model->theta[i] << " ";
		for (size_t i = 0; i < worker->model->num_betas; i++) beta_perturbation_file << beta_perturbation[i] << " ";
		beta_perturbation_file << endl;

	}

	void playback() {
		if (worker->settings->calibration_type == NONE) worker->calibration_finished = true;
		cout << worker->current_frame.id << endl;
		bool beta_only = false;
		static std::vector<float> theta_restpose;
		if (beta_only) glwidget->display_sensor_data = false;

		if (worker->settings->pause_tracking == true) { worker->offscreen_renderer.render_offscreen(true, false, false); worker->updateGL(); return; }

		if (worker->current_frame.id < 0) {

			theta_restpose = std::vector<float>(num_thetas, 0);
			theta_restpose[1] = -40; theta_restpose[2] = 320; theta_restpose[9] = 0.7; theta_restpose[10] = 0.6; theta_restpose[12] = -0.3f;
			theta_restpose[3] = -0.2f; theta_restpose[7] = 0.12f;

			worker->updateGL(); Sleep(1000); worker->current_frame.id = start_frame;
			if (worker->settings->show_iteration_calls) cout << "loading solutions" << endl;
			load_recorded_parameters(sequence_path + "solutions.txt");
			if (worker->settings->display_estimated_certainty) load_certainties(sequence_path + "estimated_certainties.txt");
			if (worker->settings->display_measured_certainty) load_certainties(sequence_path + "measured_certainties.txt");
		}

		/// Loading data
		{
			if (worker->settings->show_iteration_calls) cout << "loading data" << endl;
			load_recorded_frame();
			worker->sensor_depth_texture->load(worker->current_frame.depth.data, worker->current_frame.id);
		}

		/// Loading solutions 
		{
			Eigen::Matrix<Scalar, Eigen::Dynamic, 1> parameters = solutions->frames[worker->current_frame.id - 1];
			//Eigen::Matrix<Scalar, Eigen::Dynamic, 1> parameters = solutions->frames[worker->current_frame.id];
			std::vector<float> theta = std::vector<float>(parameters.data(), parameters.data() + num_thetas);
			std::vector<float> beta = std::vector<float>(parameters.data() + num_thetas, parameters.data() + num_thetas + worker->model->num_betas);

			if (std::accumulate(beta.begin(), beta.end(), 0.0f) > 0) {
				worker->model->update_beta(beta);
			}
			else {
				if (worker->calibration_finished == false) {
					cout << worker->current_frame.id << endl;
					worker->set_calibration_type(NONE);
					worker->model->load_model_from_file(true, sequence_path);
					worker->model->update_parameters(std::vector<float>(num_thetas, 0));
					worker->model->initialize_offsets(false);
					worker->model->update_theta(theta);
					worker->model->update_centers();
					worker->calibration_finished = true;
				}
			}
			if (beta_only) {
				worker->model->update_theta(theta_restpose);
			}
			else {
				worker->model->update_theta(theta);
			}
			worker->model->update_centers();
			worker->model->compute_outline();
		}

		/// Rendering
		{
			if (worker->settings->show_iteration_calls) cout << "rendering" << endl;
			if (real_color) display_color_and_depth_input();

			if (worker->settings->display_estimated_certainty || worker->settings->display_measured_certainty) {
				for (size_t i = 0; i < worker->model->num_betas; i++) {
					worker->model->beta_certainty[i] = certainties_vector[worker->current_frame.id][i];
				}
			}
			worker->offscreen_renderer.render_offscreen(true, false, false);

			{ /// Render				
				worker->updateGL();
				if (real_color) {
					cv::namedWindow("RGB");	cv::moveWindow("RGB", 592, 375);
					cv::imshow("RGB", worker->model->real_color);
				}
				/*
				bool write_image = true;
				if (write_image) {
					QImage image = glwidget->grabFramebuffer();
					std::ostringstream stringstream; stringstream << std::setw(4) << std::setfill('0') << worker->current_frame.id;
					std::string image_name;
					if (beta_only)
						image_name = "C:/Data/honline-video/FRAMES/" + worker->settings->sequence_name + "/beta-only/" + stringstream.str() + ".jpg";
					else
						image_name = "C:/Data/honline-video/FRAMES/" + worker->settings->sequence_name + "/beta-theta/" + stringstream.str() + ".jpg";
					image.save(image_name.c_str());
				}*/
			}
		}

		worker->current_frame.id++;
	}

	//Brandon Function for loading prompts
	void load_prompts_list() {
		if (worker->settings->calibration_type == 0)
			cout << "calibration_type == 0" << endl;
		else
			cout << "calibration type != 0" << endl;

		prompts = std::vector<std::string>();
		if (handshape) {
			cout << "Loading ABCs" << endl;
			ifstream fp("C:/Projects/MyoFaceVersion/hmodel/data/prompts/handshapes.txt"); // non_abcList.txt");
			std::string word;
			while (getline(fp, word)) {
				prompts.push_back(word);
			}
			fp.close();
			prompt_i = 0;
		}
		else {
			ifstream fp("C:/Projects/MyoFaceVersion/hmodel/data/prompts/hgjpqz_words.txt"); // randomizedLists.txt");
			//ifstream fp("C:/Projects/MyoFaceVersion/hmodel/data/prompts/no_ghjpqz_words.txt"); // randomizedLists.txt");
			std::string word;
			while (getline(fp, word)) {
				if (worker->settings->calibration_type == NONE){  //0 == NONE
					prompts.push_back(word);
				}else{
					if (prompts.size() < 12)
						prompts.push_back(word);
				}
				//cout << "loaded: " << word << endl;
			}
			fp.close();
			prompt_i = 0;
		}
		current_prompt = prompts[prompt_i++];
		std::cout << "Loaded " << prompts.size() << " words. Current Word: " << current_prompt << std::endl;
	}

	void reset_hand(){
		//worker->model->set_initial_pose();
		//cout << "Disabling calibration. Setting termination_max_iters to 150." << endl;
		//worker->settings->termination_max_iters = 150;
		//worker->set_calibration_type(NONE);
		reset_frame = set_frame;
		//initialization_enabled = true;
		//tracking_failed = true;
		//tracking_enabled = false;

		solutions_file.close();
		tracking_metrics_file.close();
		weighted_metrics_file.close();
	}



	void process_face(){
		cv::Mat im, gray,frame;
		if (real_color)
			frame = worker->current_frame.full_color;
			//frame = worker->current_frame.color;
		else
			frame = worker->current_frame.color;
		float scale = 1;
		int64 t1, t0 = cvGetTickCount(); int fnum = 0;
		double fps = 0;
		char sss[256]; std::string text;
		//bool showFaceTrack = true;
		std::vector<int> wSize1(1); wSize1[0] = 7;
		std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
		int nIter = 5; double clamp = 3, fTol = 0.01;
		bool failed = true;
		int fpd = -1;
		bool fcheck = false;

		//IplImage* I = cvQueryFrame(worker->camera); if (!I)continue; frame = I;
		if (scale == 1)im = frame;
		else cv::resize(frame, im, cv::Size(scale*frame.cols, scale*frame.rows));
		cv::flip(im, im, 1);
		cv::cvtColor(im, gray, CV_BGR2GRAY);

		//track this image

		std::vector<int> wSize;
		if (failed)
			wSize = wSize2;
		else
			wSize = wSize1;
		if (faceModel.Track(gray, wSize, fpd, nIter, clamp, fTol, fcheck) == 0){
			int idx = faceModel._clm.GetViewIdx(); failed = false;
			DrawFaceTracker(im, faceModel._shape, con, tri, faceModel._clm._visi[idx]);
			FacePointsForRenderer(im, faceModel._shape, faceModel._clm._visi[idx]);
		}
		else{
			if (showFaceTrack){
				cv::Mat R(im, cvRect(0, 0, 150, 50));
				R = cv::Scalar(0, 0, 255);
			}
			faceModel.FrameReset(); failed = true;
		}


		//draw framerate on display image 
		if (fnum >= 9){
			t1 = cvGetTickCount();
			fps = 10.0 / ((double(t1 - t0) / cvGetTickFrequency()) / 1e+6);
			t0 = t1; fnum = 0;
		}
		else fnum += 1;
		if (showFaceTrack) {
			sprintf(sss, "%d frames/sec", (int)round(fps)); text = sss;
			cv::putText(im, text, cv::Point(10, 20),
				CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255));
			//std::cout << text << " im.w: " << im.cols << " im.r: " << im.rows << std::endl;

			imshow("Face Tracker", im);
		}
		//int c = cvWaitKey(10);
		//if (c == 27)break; else if (char(c) == 'd')model.FrameReset();
		
	}

	void FacePointsForRenderer(cv::Mat &image, cv::Mat &shape, cv::Mat &visi){
		cv::Mat facepass = cv::Mat::zeros(image.rows, image.cols, CV_16UC1);
		int n = shape.rows / 2;
		cv::Point p1;
		//draw points
		std::vector<double> points = std::vector<double>(n * 2);
		for (int i = 0; i < n; i++){
			if (visi.at<int>(i, 0) == 0){
				points.push_back(-1);
				points.push_back(-1);
				continue;
			}
			points.push_back(shape.at<double>(i, 0));
			points.push_back(shape.at<double>(i + n, 0));
			p1 = cv::Point(image.cols - shape.at<double>(i, 0), shape.at<double>(i + n, 0));
			facepass.at<DepthPixel>(p1) = 500;
		}
		worker->face_texture->load(facepass.data, worker->current_frame.id);
	}

	void DrawFaceTracker(cv::Mat &image, cv::Mat &shape, cv::Mat &con, cv::Mat &tri, cv::Mat &visi)
	{
		int i, n = shape.rows / 2; cv::Point p1, p2; cv::Scalar c;
		//cout << "N = " << n << " Face points" << endl;
		//draw triangulation
		c = CV_RGB(0, 0, 0);
		for (i = 0; i < tri.rows; i++){
			if (visi.at<int>(tri.at<int>(i, 0), 0) == 0 ||
				visi.at<int>(tri.at<int>(i, 1), 0) == 0 ||
				visi.at<int>(tri.at<int>(i, 2), 0) == 0)continue;
			p1 = cv::Point(shape.at<double>(tri.at<int>(i, 0), 0),
				shape.at<double>(tri.at<int>(i, 0) + n, 0));
			p2 = cv::Point(shape.at<double>(tri.at<int>(i, 1), 0),
				shape.at<double>(tri.at<int>(i, 1) + n, 0));
			cv::line(image, p1, p2, c);
			p1 = cv::Point(shape.at<double>(tri.at<int>(i, 0), 0),
				shape.at<double>(tri.at<int>(i, 0) + n, 0));
			p2 = cv::Point(shape.at<double>(tri.at<int>(i, 2), 0),
				shape.at<double>(tri.at<int>(i, 2) + n, 0));
			cv::line(image, p1, p2, c);
			p1 = cv::Point(shape.at<double>(tri.at<int>(i, 2), 0),
				shape.at<double>(tri.at<int>(i, 2) + n, 0));
			p2 = cv::Point(shape.at<double>(tri.at<int>(i, 1), 0),
				shape.at<double>(tri.at<int>(i, 1) + n, 0));
			cv::line(image, p1, p2, c);
		}
		//draw connections
		c = CV_RGB(0, 0, 255);
		for (i = 0; i < con.cols; i++){
			if (visi.at<int>(con.at<int>(0, i), 0) == 0 ||
				visi.at<int>(con.at<int>(1, i), 0) == 0)continue;
			p1 = cv::Point(shape.at<double>(con.at<int>(0, i), 0),
				shape.at<double>(con.at<int>(0, i) + n, 0));
			p2 = cv::Point(shape.at<double>(con.at<int>(1, i), 0),
				shape.at<double>(con.at<int>(1, i) + n, 0));
			cv::line(image, p1, p2, c, 1);
		}
		//draw points
		std::vector<double> points = std::vector<double>(n * 2);
		for (i = 0; i < n; i++){
			if (visi.at<int>(i, 0) == 0){
				points.push_back(-1);
				points.push_back(-1);
				continue;
			}
			points.push_back(shape.at<double>(i, 0));
			points.push_back(shape.at<double>(i + n, 0));
			p1 = cv::Point(shape.at<double>(i, 0), shape.at<double>(i + n, 0));
			p2 = cv::Point(image.cols - shape.at<double>(i, 0), shape.at<double>(i + n, 0));
			c = CV_RGB(255, 0, 0); cv::circle(image, p1, 2, c);
			//facepass.at<DepthPixel>(p2) = 500;
		}
		faceShapes.push_back(points);

		//worker->face_texture->load(facepass.data, worker->current_frame.id);
		return;
	}


};


