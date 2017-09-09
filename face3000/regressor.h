#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "utils.h"
#include "randomforest.h"
#include <Accelerate/Accelerate.h>

class CascadeRegressor;

class Regressor {
public:
	int stage_;
	Parameters params_;
	std::vector<RandomForest> rd_forests_;
    std::vector<struct model*> linear_model_x_;
    std::vector<struct model*> linear_model_y_;
    float **modreg;
    int cameraOrient;

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
    void TrainGender(std::vector<cv::Mat_<uchar> >& images,
                                        std::vector<int>& augmented_images_index,
                                        std::vector<cv::Mat_<float> >& augmented_ground_truth_shapes,
                                        std::vector<int> & augmented_ground_truth_genders,
                                        std::vector<BoundingBox>& augmented_bboxes,
                                        std::vector<cv::Mat_<float> >& augmented_current_shapes,
                                        std::vector<float>& current_fi,
                                        std::vector<float>& current_weight,
                                        std::vector<int>& find_times,
                                        const Parameters& params,
                                        const int stage,
                                        const int pos_num,
                                        CascadeRegressor *casRegressor);
    struct feature_node* GetGlobalBinaryFeatures(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold);
    struct feature_node* GetGlobalBinaryFeaturesOld(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold);
    
    struct feature_node* NegMineGetGlobalBinaryFeatures(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, int stage, int currentStage, int landmark, int tree, bool& stop);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape,
		BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold);
    cv::Mat_<float> NegMinePredict(cv::Mat_<uchar>& image,
                                              cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, int stage, int currentStage, int landmark, int tree);
	void LoadRegressor(std::string ModelName, int stage);
	void SaveRegressor(std::string ModelName, int stage);
//    void ConstructLeafCount();
    // struct feature_node* GetGlobalBinaryFeaturesThread(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale);
    struct feature_node* GetGlobalBinaryFeaturesMP(cv::Mat_<uchar>& image,
        cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale);
    void GetGlobalBinaryFeaturesShort(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold, feature_node_short * fnode);
    void GetGlobalBinaryFeaturesGender(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_male, float& lastThreshold);
    cv::Mat_<float> PredictShort( cv::Mat_<float>& current_shape, feature_node_short* fnode, cv::Mat_<float>& rotation, float scale );
    cv::Mat_<float> PredictPos(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape,
                            BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold);
    // void GetFeaThread();
};

enum
{
    CASCADE_FLAG_BIGGEST_ONLY = 1,
    CASCADE_FLAG_TRACK_MODE = 2  //跟踪模式，根据上次的检测结果在周围检索
};

enum{
    CASCADE_PRIORITY_PERFORMANCE = 1,
    CASCADE_PRIORITY_NORMAL = 2,
    CASCADE_PRIORITY_ACCURACY = 3
};

class Tracker{
public:
    int trackCount;
    int maxTrackId;
    struct trackItem{
        int trackId;
        struct timeval previousTime;
        cv::Mat_<float> previousShape;
        cv::Mat_<float> previousRotation;
        BoundingBox     previousBox;
    };
    std::vector<struct trackItem> trackItems;
    std::vector<int> lostTrackIdsLeft;
    std::vector<int> lostTrackIdsRight;
public:
    Tracker();
    ~Tracker();
    bool isTracking(int i);
    void deTracking(int i, cv::Mat_<uchar>& image);
    bool isBoxInTrackingArea(BoundingBox box);
    int track(int trackId, cv::Mat_<float>& shape, cv::Mat_<float>& rotation, BoundingBox& box, cv::Mat_<uchar> &image);
};


class CascadeRegressor {
public:
	Parameters params_;
	std::vector<cv::Mat_<uchar> > images_;
	std::vector<cv::Mat_<float> > ground_truth_shapes_;
	std::vector<BoundingBox> bboxes_;
	std::vector<Regressor> regressors_;
    cv::Mat_<float> lastRes;
    struct timeval previousFrameTime;
    struct timeval previousScanTime;
    std::vector<cv::Mat_<float>> previousFrameRotations;
    std::vector<cv::Mat_<float>> previousFrameShapes;
    int antiJitter;
    bool isLoaded;
    cv::Mat_<float> previousFrameShape;
    cv::Mat_<float> previousFrameRotation;
    int trimNum;
    float trimFactor;
    float scaleFactor;
    int flags;
//    int defaultMinSize;
    int minSizeFactor;
    float shuffle;
    int searchPriority;
    int cameraOrient;
    bool multiOrientSupport;
    struct feature_node_short **f_nodes;
    Tracker tracker;
    
public:
	CascadeRegressor();
    ~CascadeRegressor();
	void Train(std::vector<cv::Mat_<uchar> >& images,
		std::vector<cv::Mat_<float> >& ground_truth_shapes,
        std::vector<int> ground_truth_faces,
		std::vector<BoundingBox>& bboxes,
		Parameters& params,
        int pos_num);
    void TrainGender(std::vector<cv::Mat_<uchar> >& images,
               std::vector<cv::Mat_<float> >& ground_truth_shapes,
               std::vector<int> ground_truth_genders,
               std::vector<BoundingBox>& bboxes,
               Parameters& params,
               int pos_num);
	cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& ground_truth_shape);
    cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float& score);
    cv::Mat_<float> Predict(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float& score, cv::Mat_<float>& rot);
    cv::Mat_<float> PredictPos(cv::Mat_<uchar>& image, cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float& score, int stage);
    cv::Mat_<float> NegMinePredict(cv::Mat_<uchar>& image,
                                   cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float& fi, int stage, int landmark, int tree);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);
    std::vector<cv::Rect> detectMultiScale(cv::Mat_<uchar>& image,
                                                             std::vector<cv::Mat_<float>>& shapes, float scaleFactor, int minNeighbors=2, int flags=0,
                                                             int minSize=100 );
    void unload();
    bool detectOne(cv::Mat_<uchar>& image, cv::Rect& rect, cv::Mat_<float>& shape, int flags = 0, int cameraOrient = CASCADE_ORIENT_TOP_LEFT);
    int detectGender(cv::Mat_<uchar>& image, cv::Mat_<float>& shape, int cameraOrient = CASCADE_ORIENT_TOP_LEFT);
};

#endif
