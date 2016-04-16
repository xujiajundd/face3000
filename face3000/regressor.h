#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "utils.h"
#include "randomforest.h"


class Regressor {
public:
	int stage_;
	Parameters params_;
	std::vector<RandomForest> rd_forests_;
    std::vector<struct model*> linear_model_x_;
    std::vector<struct model*> linear_model_y_;

    struct feature_node* tmp_binary_features;
    cv::Mat_<uchar> tmp_image;
    cv::Mat_<float> tmp_current_shape;
    BoundingBox tmp_bbox;
    cv::Mat_<float> tmp_rotation;
    float tmp_scale;
    int leaf_index_count[68]; //modi by xujj
    int feature_node_index[68];
    // std::atomic<int> cur_landmark {0};

public:
	Regressor();
	~Regressor();
    Regressor(const Regressor&);
	std::vector<cv::Mat_<float> > Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		const std::vector<cv::Mat_<float> >& augmented_ground_truth_shapes,
        const std::vector<int> & augmented_ground_truth_faces,
        const std::vector<BoundingBox>& augmented_bboxes,
        const std::vector<cv::Mat_<float> >& augmented_current_shapes,
        std::vector<float>& current_fi,
        std::vector<float>& current_weight,
		const Parameters& params,
		const int stage);
    struct feature_node* GetGlobalBinaryFeatures(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, int groupNum, float &score, bool &is_face);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape,
		BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float &score, bool &is_face);
	void LoadRegressor(std::string ModelName, int stage);
	void SaveRegressor(std::string ModelName, int stage);
    void ConstructLeafCount();
    // struct feature_node* GetGlobalBinaryFeaturesThread(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale);
    struct feature_node* GetGlobalBinaryFeaturesMP(cv::Mat_<uchar>& image,
        cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale);
    // void GetFeaThread();
};

class CascadeRegressor {
public:
	Parameters params_;
	std::vector<cv::Mat_<uchar> > images_;
	std::vector<cv::Mat_<float> > ground_truth_shapes_;

    //std::vector<struct model*> linear_model_x_;
    //std::vector<struct model*> linear_model_y_;
    //std::vector<cv::Mat_<float> > current_shapes_;
	std::vector<BoundingBox> bboxes_;
	//cv::Mat_<float> mean_shape_;
	std::vector<Regressor> regressors_;
//    std::vector<float> stage_delta_;
//    float alignment_confidence_;
    cv::Mat_<float> lastRes;
    int antiJitter;
public:
	CascadeRegressor();
	void Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<cv::Mat_<float> >& ground_truth_shapes,
        const std::vector<int> ground_truth_faces,
		//const std::vector<cv::Mat_<float> >& current_shapes,
		const std::vector<BoundingBox>& bboxes,
		Parameters& params,
        int pos_num);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& ground_truth_shape);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, bool &is_face);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);

};

#endif
