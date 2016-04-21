#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "liblinear/linear.h"
#include <stdio.h>
#include <sys/time.h>
//#include <thread>
//#include <mutex>
// #include <atomic>
//using namespace std;
//using namespace cv;


#define MAXFINDTIMES 100000

//std::mutex m;
class BoundingBox {
public:
	float start_x;
	float start_y;
	float width;
	float height;
	float center_x;
	float center_y;
	BoundingBox(){
		start_x = 0;
		start_y = 0;
		width = 0;
		height = 0;
		center_x = 0;
		center_y = 0;
	}
};

class FeatureLocations
{
public:
    int lmark1;
    int lmark2;
	cv::Point2d start;
	cv::Point2d end;
	FeatureLocations(int landmark1, int landmark2, cv::Point2d a, cv::Point2d b){
        lmark1 = landmark1;
        lmark2 = landmark2;
		start = a;
		end = b;
	}
	FeatureLocations(){
        lmark1 = 0;
        lmark2 = 0;
		start = cv::Point2d(0.0, 0.0);
		end = cv::Point2d(0.0, 0.0);
	};
};

class Parameters {
	//private:
public:
	int local_features_num_;
	int landmarks_num_per_face_;
	int regressor_stages_;
	int tree_depth_;
	int trees_num_per_forest_;
//    int group_num_;
//    std::vector<std::vector<int>> groups_;
	std::vector<float> local_radius_by_stage_;
    std::vector<float> detect_factor_by_stage_;
	int initial_guess_;
	cv::Mat_<float> mean_shape_;
    int predict_regressor_stages_;
//    std::set<int> predict_group_;
};

cv::Mat_<float> ProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bbox);
cv::Mat_<float> ReProjection(const cv::Mat_<float>& shape, const BoundingBox& bbox);
cv::Mat_<float> GetMeanShape(const std::vector<cv::Mat_<float> >& all_shapes, std::vector<int>& ground_truth_faces,
	const std::vector<BoundingBox>& all_bboxes);
void getSimilarityTransform(const cv::Mat_<float>& shape_to,
	const cv::Mat_<float>& shape_from,
	cv::Mat_<float>& rotation, float& scale);

//cv::Mat_<float> LoadGroundTruthShape(std::string& name);
cv::Mat_<float> LoadGroundTruthShape(const char* name);

int LoadImages(std::vector<cv::Mat_<uchar> >& images, std::vector<cv::Mat_<float> >& ground_truth_shapes, std::vector<int> & ground_truth_faces,
	std::vector<BoundingBox>& bboxes, std::string file_names);

bool ShapeInRect(cv::Mat_<float>& ground_truth_shape, cv::Rect&);

std::vector<cv::Rect_<int> > DetectFaces(cv::Mat_<uchar>& image);
std::vector<cv::Rect> DetectFaces(cv::Mat_<uchar>& image, cv::CascadeClassifier& classifier);

float CalculateError(cv::Mat_<float>& ground_truth_shape, cv::Mat_<float>& predicted_shape);

void DrawPredictImage(cv::Mat_<uchar>& image, cv::Mat_<float>& shapes);

BoundingBox GetBoundingBox(cv::Mat_<float>& shape, int width, int height);

#endif
