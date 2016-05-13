#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "utils.h"
#include "randomforest.h"

class CascadeRegressor;

class Regressor {
public:
	int stage_;
	Parameters params_;
	std::vector<RandomForest> rd_forests_;
    std::vector<struct model*> linear_model_x_;
    std::vector<struct model*> linear_model_y_;
    float **modreg;

    struct feature_node* tmp_binary_features;

//    cv::Mat_<uchar> tmp_image;
//    cv::Mat_<float> tmp_current_shape;
//    BoundingBox tmp_bbox;
//    cv::Mat_<float> tmp_rotation;
//    float tmp_scale;
//    int leaf_index_count[68]; //modi by xujj
//    int feature_node_index[68];
    // std::atomic<int> cur_landmark {0};

public:
	Regressor();
	~Regressor();
    Regressor(const Regressor&);
	std::vector<cv::Mat_<float> > Train(std::vector<cv::Mat_<uchar> >& images,
		std::vector<int>& augmented_images_index,
		std::vector<cv::Mat_<float> >& augmented_ground_truth_shapes,
        std::vector<int> & augmented_ground_truth_faces,
        std::vector<BoundingBox>& augmented_bboxes,
        std::vector<cv::Mat_<float> >& augmented_current_shapes,
        std::vector<float>& current_fi,
        std::vector<float>& current_weight,
        std::vector<int>& find_times,
		const Parameters& params,
		const int stage,
        const int pos_num,
        CascadeRegressor *casRegressor);
    struct feature_node* GetGlobalBinaryFeatures(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, bool& is_face);
    struct feature_node* NegMineGetGlobalBinaryFeatures(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, bool& is_face, int stage, int currentStage, int landmark, int tree, bool& stop);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape,
		BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float &score, bool& is_face);
    cv::Mat_<float> NegMinePredict(cv::Mat_<uchar>& image,
                                              cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, bool& is_face, int stage, int currentStage, int landmark, int tree);
	void LoadRegressor(std::string ModelName, int stage);
	void SaveRegressor(std::string ModelName, int stage);
//    void ConstructLeafCount();
    // struct feature_node* GetGlobalBinaryFeaturesThread(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale);
    struct feature_node* GetGlobalBinaryFeaturesMP(cv::Mat_<uchar>& image,
        cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale);
    // void GetFeaThread();
};

enum
{
    CASCADE_FLAG_BIGGEST_ONLY = 1,
    CASCADE_FLAG_SEARCH_MAX_TO_MIN = 2,
    CASCADE_FLAG_TRACK_MODE = 4  //跟踪模式，根据上次的检测结果在周围检索
};

class CascadeRegressor {
public:
	Parameters params_;
	std::vector<cv::Mat_<uchar> > images_;
	std::vector<cv::Mat_<float> > ground_truth_shapes_;
	std::vector<BoundingBox> bboxes_;
	std::vector<Regressor> regressors_;
    cv::Mat_<float> lastRes;
    int antiJitter;
public:
	CascadeRegressor();
	void Train(std::vector<cv::Mat_<uchar> >& images,
		std::vector<cv::Mat_<float> >& ground_truth_shapes,
        std::vector<int> ground_truth_faces,
		std::vector<BoundingBox>& bboxes,
		Parameters& params,
        int pos_num);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& ground_truth_shape);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, bool &is_face, float &score);
    cv::Mat_<float> NegMinePredict(cv::Mat_<uchar>& image,
                                   cv::Mat_<float>& current_shape, BoundingBox& bbox, bool& is_face, float& fi, int stage, int landmark, int tree);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);
    std::vector<cv::Rect> detectMultiScale(cv::Mat_<uchar>& image,
                                                             std::vector<cv::Mat_<float>>& shapes, float scaleFactor, int minNeighbors=2, int flags=0,
                                                             int minSize=100 );
};

#endif
