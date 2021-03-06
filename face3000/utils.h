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
//#include <Accelerate/Accelerate.h>

#define MAXFINDTIMES 16*256*256*256

#define DETECT_ADD_DEPTH 0

extern int NUM_LANDMARKS;
extern int debug_on_;

enum
{
    CASCADE_ORIENT_TOP_LEFT = 0,
    CASCADE_ORIENT_TOP_RIGHT,
    CASCADE_ORIENT_BOTTOM_LEFT
};

enum
{
    CASCADE_CATEGORY_FRONT = 1,
    CASCADE_CATEGORY_LEFT = 2,
    CASCADE_CATEGORY_RIGHT = 4,
    CASCADE_CATEGORY_OPEN_MOUTH = 8
};

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
    int category_num_;
    std::vector<cv::Mat_<float>> category_mean_shapes_;
};

cv::Mat_<float> ProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bbox);
cv::Mat_<float> ReProjection(const cv::Mat_<float>& shape, const BoundingBox& bbox);
cv::Mat_<float> GetMeanShape(const std::vector<cv::Mat_<float> >& all_shapes, std::vector<int>& ground_truth_faces,
	const std::vector<BoundingBox>& all_bboxes);
std::vector<cv::Mat_<float>> GetCategoryMeanShapes(std::vector<cv::Mat_<float> >& all_shapes, std::vector<int>& ground_truth_faces, std::vector<int> & ground_truth_categorys,
                                                   std::vector<BoundingBox>& all_bboxes);
void getSimilarityTransformAcc(const cv::Mat_<float>& shape_to,
                               const cv::Mat_<float>& shape_from,
                               cv::Mat_<float>& rotation, float& scale);
void getSimilarityTransform(const cv::Mat_<float>& shape_to,
	const cv::Mat_<float>& shape_from,
	cv::Mat_<float>& rotation, float& scale);

//cv::Mat_<float> LoadGroundTruthShape(std::string& name);
cv::Mat_<float> LoadGroundTruthShape(const char* name, int& gender);
BoundingBox CalculateBoundingBox(cv::Mat_<float>& shape);
BoundingBox CalculateBoundingBoxRotation(cv::Mat_<float>& shape, cv::Mat_<float>& rotation);
int LoadImages(std::vector<cv::Mat_<uchar> >& images, std::vector<cv::Mat_<float> >& ground_truth_shapes, std::vector<int> & ground_truth_faces, std::vector<int>& ground_truth_genders,
	std::vector<BoundingBox>& bboxes, std::string file_names, std::string neg_file_names);

bool ShapeInRect(cv::Mat_<float>& ground_truth_shape, cv::Rect&);

std::vector<cv::Rect_<int> > DetectFaces(cv::Mat_<uchar>& image);
std::vector<cv::Rect> DetectFaces(cv::Mat_<uchar>& image, cv::CascadeClassifier& classifier);

float CalculateError(cv::Mat_<float>& ground_truth_shape, cv::Mat_<float>& predicted_shape);
float CalculateError2(cv::Mat_<float>& ground_truth_shape, cv::Mat_<float>& predicted_shape, int stage, int landmark);

void DrawImage(cv::Mat_<uchar> image, cv::Mat_<float>& ishape);
void DrawPredictImage(cv::Mat_<uchar>& image, cv::Mat_<float>& shapes);
void DrawImageNoShow(cv::Mat image, cv::Mat_<float>& ishape);
void DrawImageNoShowOrientation(cv::Mat image, cv::Mat_<float>& ishape, int orient);
BoundingBox GetBoundingBox(cv::Mat_<float>& shape, int width, int height);
//int colorDistance(uchar p1, uchar p2);
cv::Mat_<float> convertShape(cv::Mat_<float> shape);
cv::Mat_<float> reConvertShape(cv::Mat_<float> shape);

int symmetricPoint(int p);
int adjointPoint(int p);

#endif
