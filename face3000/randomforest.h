#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H
#include "utils.h"
#include <set>
class Node {
public:
	int leaf_identity; // used only when it is leaf node, and is unique among the tree
	Node* left_child_;
	Node* right_child_;
	int samples_;
	bool is_leaf_;
	int depth_; // recording current depth
	float threshold_;
	bool thre_changed_;
	FeatureLocations feature_locations_;
    float score_; //这个score，在叶子节点存放score，在根节点存放实际是threshold
	Node(Node* left, Node* right, float thres, bool leaf);
	Node(Node* left, Node* right, float thres);
	Node();
};

class RandomForest {
public:
	int stage_;
	int local_features_num_;
	int landmark_index_;
	int tree_depth_;
	int trees_num_per_forest_;
	float local_radius_;
	int all_leaf_nodes_;
	cv::Mat_<float> mean_shape_;
	std::vector<Node*> trees_;
	std::vector<FeatureLocations> local_position_; // size = param_.local_features_num
	std::vector<cv::Mat_<float> >* regression_targets_;
    std::vector<int> augmented_ground_truth_faces_;
    std::vector<float> current_weight_;
    
	bool TrainForest(//std::vector<cv::Mat_<float> >& regression_targets, 
		const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		//const std::vector<cv::Mat_<float> >& augmented_ground_truth_shapes,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<float> >& augmented_current_shapes,
        const std::vector<int> & augmented_ground_truth_faces,
        std::vector<float> & current_fi,
        std::vector<float> & current_weight,
		const std::vector<cv::Mat_<float> >& rotations,
		const std::vector<float>& scales);
	Node* BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, int current_depth);
	int FindSplitFeature(Node* node, std::set<int>& selected_indexes,
		cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes);
	cv::Mat_<float> GetBinaryFeatures(const cv::Mat_<float>& image,
		const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale);
	int MarkLeafIdentity(Node* node, int count);
	int GetNodeOutput(Node* node, const cv::Mat_<float>& image,
		const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale);
	//predict()
	int GetBinaryFeatureIndex(int tree_index, const cv::Mat_<float>& image,
	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale, float *score);
	RandomForest();
	RandomForest(Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<float> >& regression_targets);
	void WriteTree(Node* p, std::ofstream& fout);
	Node* ReadTree(std::ifstream& fin);
	void SaveRandomForest(std::ofstream& fout);
	void LoadRandomForest(std::ifstream& fin);
};

#endif
