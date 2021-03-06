#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "utils.h"
#include <set>
//#include "regressor.h"
class CascadeRegressor;

class Node {
public:
	int leaf_identity; // used only when it is leaf node, and is unique among the tree
	Node* left_child_;
	Node* right_child_;
//	int samples_;
	bool is_leaf_;
//    bool is_leaf_a;
	int depth_; // recording current depth
	float threshold_;
//	bool thre_changed_;
	FeatureLocations feature_locations_;
    float score_; //这个score，在叶子节点存放score，在根节点存放实际是threshold
	Node(Node* left, Node* right, float thres, bool leaf);
	Node(Node* left, Node* right, float thres);
	Node();
    ~Node();
};

class RandomForest {
public:
    Parameters param_;
	int stage_;
	int local_features_num_;
    int landmark_num_;
	int landmark_index_;
	int tree_depth_;
	int trees_num_per_forest_;
	float local_radius_;
    float detect_factor_;
	int all_leaf_nodes_;
    int true_pos_num_;
	cv::Mat_<float> mean_shape_;
	std::vector<Node*> trees_;
	std::vector<FeatureLocations> local_position_; // size = param_.local_features_num
	std::vector<cv::Mat_<float> >* regression_targets_;
//    std::vector<int> augmented_ground_truth_faces_;
//    std::vector<float> current_weight_;
//    std::vector<cv::Mat_<float> > augmented_ground_truth_shapes_;
//    std::vector<cv::Mat_<float> > augmented_current_shapes_;
    CascadeRegressor *casRegressor_;
    cv::RNG rd;
    
	bool TrainForest(//std::vector<cv::Mat_<float> >& regression_targets, 
		std::vector<cv::Mat_<uchar> >& images,
		std::vector<int>& augmented_images_index,
		std::vector<cv::Mat_<float> >& augmented_ground_truth_shapes,
		std::vector<BoundingBox>& augmented_bboxes,
		std::vector<cv::Mat_<float> >& augmented_current_shapes,
        std::vector<int> & augmented_ground_truth_faces,
        std::vector<float> & current_fi,
        std::vector<float> & current_weight,
        std::vector<int> & find_times,
		std::vector<cv::Mat_<float> >& rotations,
		std::vector<float>& scales);
    
    bool TrainForestGender(//std::vector<cv::Mat_<float> >& regression_targets,
                     std::vector<cv::Mat_<uchar> >& images,
                     std::vector<int>& augmented_images_index,
                     std::vector<cv::Mat_<float> >& augmented_ground_truth_shapes,
                     std::vector<BoundingBox>& augmented_bboxes,
                     std::vector<cv::Mat_<float> >& augmented_current_shapes,
                     std::vector<int> & augmented_ground_truth_genders,
                     std::vector<float> & current_fi,
                     std::vector<float> & current_weight,
                     std::vector<int> & find_times,
                     std::vector<cv::Mat_<float> >& rotations,
                     std::vector<float>& scales);
    
    Node* BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int> & augmented_ground_truth_faces,
                    std::vector<float> & current_weight, int current_depth);
	int FindSplitFeature(Node* node, std::set<int>& selected_indexes,
                         cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int> & augmented_ground_truth_faces,
                         std::vector<float> & current_weight, std::vector<int>& left_indexes, std::vector<int>& right_indexes);
	cv::Mat_<float> GetBinaryFeatures(const cv::Mat_<float>& image,
		const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale);
//	int MarkLeafIdentity(Node* node, int count);
//	int GetNodeOutput(Node* node, const cv::Mat_<float>& image,
//		const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale);
	//predict()
    int GetBinaryFeatureIndex(int tree_index, const cv::Mat_<uchar>& image,
	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale, float& score);
	RandomForest();
    ~RandomForest();
	RandomForest(Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<float> >& regression_targets, CascadeRegressor *casRegressor, int true_pos_num);
	void WriteTree(Node* p, std::ofstream& fout);
	Node* ReadTree(std::ifstream& fin);
	void SaveRandomForest(std::ofstream& fout);
	void LoadRandomForest(std::ifstream& fin);
};

#endif
