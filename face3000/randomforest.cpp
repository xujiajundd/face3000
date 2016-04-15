#include "randomforest.h"
#include <time.h>
#include <algorithm>
#include <stack>

Node::Node(){
	left_child_ = NULL;
	right_child_ = NULL;
	is_leaf_ = false;
	threshold_ = 0.0;
	leaf_identity = -1;
	samples_ = -1;
	thre_changed_ = false;
    score_ = 0.0; //TODO:模型保存和读取时也要加上
}

Node::Node(Node* left, Node* right, float thres){
	Node(left, right, thres, false);
}

Node::Node(Node* left, Node* right, float thres, bool leaf){
	left_child_ = left;
	right_child_ = right;
	is_leaf_ = leaf;
	threshold_ = thres;
	//offset_ = cv::Point2f(0, 0);
}

int my_cmp(std::pair<float,int> p1, std::pair<float,int> p2)
{
    return p1.first < p2.first;
};

bool RandomForest::TrainForest(//std::vector<cv::Mat_<float>>& regression_targets,
	const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<int>& augmented_images_index,
	//const std::vector<cv::Mat_<float>>& augmented_ground_truth_shapes,
	const std::vector<BoundingBox>& augmented_bboxes,
	const std::vector<cv::Mat_<float> >& augmented_current_shapes,
    const std::vector<int> & augmented_ground_truth_faces,
    std::vector<float> & current_fi,
    std::vector<float> & current_weight,
	const std::vector<cv::Mat_<float> >& rotations,
	const std::vector<float>& scales){
    
    augmented_ground_truth_faces_ = augmented_ground_truth_faces;
    current_weight_ = current_weight;
    //std::cout << "build forest of landmark: " << landmark_index_ << " of stage: " << stage_ << std::endl;
	//regression_targets_ = &regression_targets;
	time_t current_time;
	current_time = time(0);
	cv::RNG rd(current_time);
	// random generate feature locations
	//std::cout << "generate feature locations" << std::endl;
	local_position_.clear();
	local_position_.resize(local_features_num_);
	for (int i = 0; i < local_features_num_; i++){
		float x, y;
		do{
			x = rd.uniform(-local_radius_, local_radius_); 
			y = rd.uniform(-local_radius_, local_radius_);
//            if ( i < 4 && x < -local_radius_/5.0) continue; //add by xujj, 避免脸部外侧的pixel
//            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
//            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
		} while (x*x + y*y > local_radius_*local_radius_);
		cv::Point2f a(x, y);

		do{
			x = rd.uniform(-local_radius_, local_radius_);
			y = rd.uniform(-local_radius_, local_radius_);
//            if ( i < 4 && x < -local_radius_/5.0) continue;
//            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
//            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
		} while (x*x + y*y > local_radius_*local_radius_);
		cv::Point2f b(x, y);

		local_position_[i] = FeatureLocations(a, b);
	}
	//std::cout << "get pixel differences" << std::endl;
	cv::Mat_<int> pixel_differences(local_features_num_, augmented_images_index.size()); // matrix: features*images
	
	for (int i = 0; i < augmented_images_index.size(); i++){
		
		cv::Mat_<float> rotation = rotations[i];
		float scale = scales[i];
		//getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]),mean_shape_, rotation, scale);
		
		for (int j = 0; j < local_features_num_; j++){
			FeatureLocations pos = local_position_[j];
			float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
			float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			int real_x = delta_x + augmented_current_shapes[i](landmark_index_, 0);
			int real_y = delta_y + augmented_current_shapes[i](landmark_index_, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			int tmp = (int)images[augmented_images_index[i]](real_y, real_x); //real_y at first

			delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
			delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			real_x = delta_x + augmented_current_shapes[i](landmark_index_, 0);
			real_y = delta_y + augmented_current_shapes[i](landmark_index_, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			pixel_differences(j, i) = tmp - (int)images[augmented_images_index[i]](real_y, real_x); 
		}
	}
	// train Random Forest
	// construct each tree in the forest
	
	float overlap = 0.4;
	int step = floor(((float)augmented_images_index.size())*overlap / (trees_num_per_forest_ - 1));
	trees_.clear();
	all_leaf_nodes_ = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
		int start_index = i*step;
		int end_index = augmented_images_index.size() - (trees_num_per_forest_ - i - 1)*step;
		//cv::Mat_<int> data = pixel_differences(cv::Range(0, local_features_num_), cv::Range(start_index, end_index));
		//cv::Mat_<int> sorted_data;
		//cv::sortIdx(data, sorted_data, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
		std::set<int> selected_indexes; //这个是用来表示那个feature已经被用过了
		std::vector<int> images_indexes;
		for (int j = start_index; j < end_index; j++){
			images_indexes.push_back(j);
		}
        //计算每个训练实例的weight
        for(int k=0;k<current_weight.size();++k)
        {
            current_weight[k] = exp(0.0-augmented_ground_truth_faces[k]*current_fi[k]);
            //current_weight[k]=1;
            if ( current_weight_[k] > 5000.0 ) current_weight_[k] = 5000.0;
        }
        
		Node* root = BuildTree(selected_indexes, pixel_differences, images_indexes, 0);
		trees_.push_back(root);
        
        //计算每个训练实例的fi
        for ( int n=0; n<augmented_images_index.size(); n++){
            float score = 0;
            //用训练实例去遍历此树得出叶子节点的score
//            cv::Mat_<float> rotation;
//            float scale;
//            getSimilarityTransform(ProjectShape(augmented_current_shapes[n],augmented_bboxes[n]),mean_shape_,rotation,scale);
            GetBinaryFeatureIndex(i, images[augmented_images_index[n]], augmented_bboxes[n], augmented_current_shapes[n], rotations[n] , scales[n], &score);
            current_fi[n] += score;
        }
        //开始计算这棵树的detection threshold
        std::vector<std::pair<float,int>> fiSort;
        fiSort.clear();
        for(int n=0;n<current_fi.size();++n)
        {
//            if (find_times[augmented_images[n]]<=MAXFINDTIMES)
            fiSort.push_back(std::pair<float,int>(current_fi[n],n));
        }
        // ascent , small fi means false sample
        sort(fiSort.begin(),fiSort.end(),my_cmp);
        // compute recall
        // set threshold
        float max_recall=0,min_error=1;
//        int idx_tmp=-1;
        
//        std::vector<std::pair<float,float>> precise_recall;
        for (int n=0;n<fiSort.size();++n)
        {
            int true_pos=0;int false_neg=0;
            int true_neg=0;int false_pos=0;
            for(int m=0;m<fiSort.size();++m)
            {
                int isFace = augmented_ground_truth_faces[fiSort[m].second];
                // below the threshold as non-face
                if (m<n)
                {
                    if (isFace==1)
                    {
                        false_neg++;
                    }
                    else
                    {
                        true_neg++;
                    }
                }
                // up the threshold as face
                else
                {
                    if (isFace==1)
                    {
                        true_pos++;
                    }
                    else
                    {
                        false_pos++;
                    }
                }
            }
            
            if (true_pos/(true_pos+false_neg+FLT_MIN)>=max_recall)
            {
                max_recall=true_pos/(true_pos+false_neg+FLT_MIN);
//                precise_recall.push_back(pair<float,float>(true_pos/(true_pos+false_neg+FLT_MIN),false_pos/(false_pos+true_neg+FLT_MIN)));
                root->score_ =fiSort[n].first; //跟节点的score作为threshold使用
                //idx_tmp=n;
            }
            else
                break;
        }
        
	}
	/*int count = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
		Node* root = trees_[i];
		count = MarkLeafIdentity(root, count);
	}
	all_leaf_nodes_ = count;*/
	return true;
}


Node* RandomForest::BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, int current_depth){
	if (images_indexes.size() > 0){ // the node may not split under some cases
		Node* node = new Node();
		node->depth_ = current_depth;
		node->samples_ = images_indexes.size();
		std::vector<int> left_indexes, right_indexes;
		if (current_depth == tree_depth_){ // the node reaches max depth
			node->is_leaf_ = true;
			node->leaf_identity = all_leaf_nodes_;
			all_leaf_nodes_++;
            //计算叶子节点的score
            float leaf_pos_weight = 0;
            float leaf_neg_weight = 0;
            for ( int i=0; i<images_indexes.size(); i++){
                if ( augmented_ground_truth_faces_[images_indexes[i]] == 1){
                    leaf_pos_weight += current_weight_[images_indexes[i]];
                }
                else{
                    leaf_neg_weight += current_weight_[images_indexes[i]];
                }
            }
            node->score_ = 0.5*(((leaf_pos_weight-0.0)<FLT_EPSILON)?0:log(leaf_pos_weight))-0.5*(((leaf_neg_weight-0.0)<FLT_EPSILON)?0:log(leaf_neg_weight))/*/log(2.0)*/;
			return node;
		}

		int ret = FindSplitFeature(node, selected_indexes, pixel_differences, images_indexes, left_indexes, right_indexes);
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1){ // the current node contain all sample when reaches max variance reduction, it is leaf node
			node->is_leaf_ = true;
			node->leaf_identity = all_leaf_nodes_;
			all_leaf_nodes_++;
            //计算叶子节点的score, 同上
            float leaf_pos_weight = 0;
            float leaf_neg_weight = 0;
            for ( int i=0; i<images_indexes.size(); i++){
                if ( augmented_ground_truth_faces_[images_indexes[i]] == 1){
                    leaf_pos_weight += current_weight_[images_indexes[i]];
                }
                else{
                    leaf_neg_weight += current_weight_[images_indexes[i]];
                }
            }
            node->score_ = 0.5*(((leaf_pos_weight-0.0)<FLT_EPSILON)?0:log(leaf_pos_weight))-0.5*(((leaf_neg_weight-0.0)<FLT_EPSILON)?0:log(leaf_neg_weight))/*/log(2.0)*/;
			return node;
		}

		//if (current_depth + 1 < tree_depth_){
		node->left_child_ = BuildTree(selected_indexes, pixel_differences, left_indexes, current_depth + 1);
		node->right_child_ = BuildTree(selected_indexes, pixel_differences, right_indexes, current_depth + 1);
		//}
		return node;
	}
	else{ // this case is not possible in this data structure
		return NULL;
	}
}

int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_indexes, 
	cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes){
	std::vector<int> val;
	//cv::Mat_<int> sorted_fea;
	time_t current_time;
	current_time = time(0);
	cv::RNG rd(current_time);
	int threshold;
//	float var = -1000000000000.0; // use -DBL_MAX will be better
	int feature_index = -1;
//	std::vector<int> tmp_left_indexes, tmp_right_indexes;
    
    std::vector<float> vars;
    std::vector<float> entropys;
    std::vector<std::pair<int,int>> thresholds;

	//int j = 0, tmp_index;
	for (int j = 0; j < local_features_num_; j++){
		if (selected_indexes.find(j) == selected_indexes.end()){
//			tmp_left_indexes.clear();
//			tmp_right_indexes.clear();
            
            int num_l_shapes = 0, num_r_shapes = 0;
			float var_lc = 0.0, var_rc = 0.0, var_red = 0.0;
			float Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;
			float Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;
            
            int num_l_pos_faces = 0, num_l_neg_faces = 0, num_r_pos_faces = 0, num_r_neg_faces;
            float total_l_pos_weight = 0.0, total_l_neg_weight = 0.0;
            float total_r_pos_weight = 0.0, total_r_neg_weight = 0.0;
            
			// random generate threshold
			std::vector<int> data;
			data.reserve(images_indexes.size());
			for (int i = 0; i < images_indexes.size(); i++){
				data.push_back(pixel_differences(j, images_indexes[i]));
			}
			std::sort(data.begin(), data.end());
			int tmp_index = floor((int)(images_indexes.size()*(0.5 + 0.9*(rd.uniform(0.0, 1.0) - 0.5))));
			int tmp_threshold = data[tmp_index];
			for (int i = 0; i < images_indexes.size(); i++){
				int index = images_indexes[i];
				if (pixel_differences(j, index) < tmp_threshold){
//					tmp_left_indexes.push_back(index);
                    if ( augmented_ground_truth_faces_[index] == 1){
                        // do with regression target
                        num_l_shapes++;
                        float value = regression_targets_->at(index)(landmark_index_, 0);
                        Ex_2_lc += pow(value, 2);
                        Ex_lc += value;
                        value = regression_targets_->at(index)(landmark_index_, 1);
                        Ey_2_lc += pow(value, 2);
                        Ey_lc += value;
                        
                        num_l_pos_faces++;
                        total_l_pos_weight += current_weight_[index];
                    }
                    else{ //负样本
                        num_l_neg_faces++;
                        total_l_neg_weight += current_weight_[index];
                    }
				}
				else{
//					tmp_right_indexes.push_back(index);
                    if ( augmented_ground_truth_faces_[index] == 1){
                        num_r_shapes++;
                        float value = regression_targets_->at(index)(landmark_index_, 0);
                        Ex_2_rc += pow(value, 2);
                        Ex_rc += value;
                        value = regression_targets_->at(index)(landmark_index_, 1);
                        Ey_2_rc += pow(value, 2);
                        Ey_rc += value;
                        
                        num_r_pos_faces++;
                        total_r_pos_weight += current_weight_[index];
                    }
                    else{ //负样本
                        num_r_neg_faces++;
                        total_r_neg_weight += current_weight_[index];
                    }
				}
			}
			if (num_l_shapes == 0){
				var_lc = 0.0;
			} else{
				var_lc = Ex_2_lc / num_l_shapes - pow(Ex_lc / num_l_shapes, 2)
					+ Ey_2_lc / num_l_shapes - pow(Ey_lc / num_l_shapes, 2);
			}
			if (num_r_shapes == 0){
				var_rc = 0.0;
			} else{
				var_rc = Ex_2_rc / num_r_shapes - pow(Ex_rc / num_r_shapes, 2)
					+ Ey_2_rc / num_r_shapes - pow(Ey_rc / num_r_shapes, 2);
			}
//			var_red = -var_lc*num_l_shapes - var_rc*num_r_shapes;
            var_red = var_lc*num_l_shapes + var_rc*num_r_shapes;
            thresholds.push_back(std::pair<int,int>(tmp_threshold, j));
            vars.push_back(var_red);
            
            
            int left_sample_num = num_l_pos_faces + num_l_neg_faces;
            int right_sample_num = num_r_pos_faces + num_r_neg_faces;
//            int total_sample_num =  left_sample_num + right_sample_num;
            
            float total_l_weight = total_l_pos_weight + total_l_neg_weight;
            float total_r_weight = total_r_pos_weight + total_r_neg_weight;
            float total_weight = total_l_weight + total_r_weight;
            
            float entropy = 0.0, entropy_lc = 0.0, entropy_rc = 0.0;
            
            if ( left_sample_num == 0 ){
                entropy_lc = 0.0;
            }
            else{
                float entropy_tmp = total_l_pos_weight / ( total_l_weight + FLT_MIN );
                if ( (entropy_tmp-0.0) < FLT_EPSILON){
                    entropy_lc = 0.0;
                }
                else{
                    entropy_lc = - ( total_l_weight / ( total_weight + FLT_MIN)) * ((entropy_tmp + FLT_MIN) * log(entropy_tmp + FLT_MIN)/log(2.0) + ( 1 - entropy_tmp + FLT_MIN) * log(1-entropy_tmp + FLT_MIN)/log(2.0));
                }
            }
            
            if ( right_sample_num == 0 ){
                entropy_rc = 0.0;
            }
            else{
                float entropy_tmp = total_r_pos_weight / ( total_r_weight + FLT_MIN );
                if ( (entropy_tmp-0.0) < FLT_EPSILON){
                    entropy_rc = 0.0;
                }
                else{
                    entropy_rc = - ( total_r_weight / ( total_weight + FLT_MIN)) * ((entropy_tmp + FLT_MIN) * log(entropy_tmp + FLT_MIN)/log(2.0) + ( 1 - entropy_tmp + FLT_MIN) * log(1-entropy_tmp + FLT_MIN)/log(2.0));
                }
            }
            
            entropy = entropy_lc + entropy_rc;
            entropys.push_back(entropy);
            
//			if (var_red > var){
//				var = var_red;
//				threshold = tmp_threshold;
//				feature_index = j;
//				left_indexes = tmp_left_indexes;
//				right_indexes = tmp_right_indexes;
//			}
		}
        else{
            //这个feature已经被用掉了
        }
	}
    
    
    //这里把var和entropy做归一化，然后取其和的最小值，这样可以做到分类和回归在同一个feature上都做到较优。
    //这个方法和原论文的方法不同，原论文按照概率值来选择偏向分类还是回归树
    
    float minvar = *std::min_element(std::begin(vars), std::end(vars));
    float maxvar = *std::max_element(std::begin(vars), std::end(vars));
    float minent = *std::min_element(std::begin(entropys), std::end(entropys));
    float maxent = *std::max_element(std::begin(entropys), std::end(entropys));
    
    float summin = FLT_MAX;
    float indexmin = 0;
    for ( int i=0; i<vars.size(); i++){
        float tmpvar = ( vars[i] - minvar ) / (maxvar - minvar);
        float tmpent = ( entropys[i] - minent ) / (maxent - minent);
        float tmpsum = tmpvar + tmpent; //这个可以根据stage用不同的系数
        if ( tmpsum < summin ){
            summin = tmpsum;
            indexmin = i;
        }
    }
    
    feature_index = thresholds[indexmin].second;
    threshold = thresholds[indexmin].first;
    
    for ( int i=0; i < images_indexes.size(); i++ ){
        int index = images_indexes[i];
        if (pixel_differences(feature_index, index) < threshold){
            left_indexes.push_back(index);
        }
        else{
            right_indexes.push_back(index);
        }
    }
    
	if (feature_index != -1) // actually feature_index will never be -1 
	{
		if (left_indexes.size() == 0 || right_indexes.size() == 0){
			node->is_leaf_ = true; // the node can contain all the samples
			return 1;
		}
		node->threshold_ = threshold;
		node->thre_changed_ = true;
		node->feature_locations_ = local_position_[feature_index];
		selected_indexes.insert(feature_index);
		return 0;
	}
	
	return -1;
}

int RandomForest::MarkLeafIdentity(Node* node, int count){
	std::stack<Node*> s;
	Node* p_current = node; 
	
	if (node == NULL){
		return count;
	}
	// the node in the tree is either leaf node or internal node that has both left and right children
	while (1)//p_current || !s.empty())
	{
		
		if (p_current->is_leaf_){
			p_current->leaf_identity = count;
			count++;
			if (s.empty()){
				return count;
			}
			p_current = s.top()->right_child_;
			s.pop();
		}
		else{
			s.push(p_current);
			p_current = p_current->left_child_;
		}
		
		/*while (!p_current && !s.empty()){
			p_current = s.top();
			s.pop();
			p_current = p_current->right_child_; 
		}*/
	}
	
}

cv::Mat_<float> RandomForest::GetBinaryFeatures(const cv::Mat_<float>& image,
	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale){
	cv::Mat_<float> res(1, all_leaf_nodes_, 0.0);
	for (int i = 0; i < trees_num_per_forest_; i++){
		Node* node = trees_[i];
		while (!node->is_leaf_){
			int direction = GetNodeOutput(node, image, bbox, current_shape, rotation, scale);
			if (direction == -1){
				node = node->left_child_;
			}
			else{
				node = node->right_child_;
			}
		}
		res(0, node->leaf_identity) = 1.0;
	}
	return res;
}

int RandomForest::GetBinaryFeatureIndex(int tree_index, const cv::Mat_<float>& image,
	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale, float *score){
	Node* node = trees_[tree_index];
	while (!node->is_leaf_){
		FeatureLocations& pos = node->feature_locations_;
		float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
		float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		int real_x = delta_x + current_shape(landmark_index_, 0);
		int real_y = delta_y + current_shape(landmark_index_, 1);
		real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
		real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
		int tmp = (int)image(real_y, real_x); //real_y at first

		delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
		delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		real_x = delta_x + current_shape(landmark_index_, 0);
		real_y = delta_y + current_shape(landmark_index_, 1);
		real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
		real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
		if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
			node = node->left_child_;// go left
		}
		else{
			node = node->right_child_;// go right
		}
	}
    *score = node->score_;
	return node->leaf_identity;
}


int RandomForest::GetNodeOutput(Node* node, const cv::Mat_<float>& image,
	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale){
	
	FeatureLocations& pos = node->feature_locations_;
	float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
	float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
	delta_x = scale*delta_x*bbox.width / 2.0;
	delta_y = scale*delta_y*bbox.height / 2.0;
	int real_x = delta_x + current_shape(landmark_index_, 0);
	int real_y = delta_y + current_shape(landmark_index_, 1);
	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
	int tmp = (int)image(real_y, real_x); //real_y at first

	delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
	delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
	delta_x = scale*delta_x*bbox.width / 2.0;
	delta_y = scale*delta_y*bbox.height / 2.0;
	real_x = delta_x + current_shape(landmark_index_, 0);
	real_y = delta_y + current_shape(landmark_index_, 1);
	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
	if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
		return -1; // go left
	}
	else{
		return 1; // go right
	}

}

RandomForest::RandomForest(Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<float> >& regression_targets){
	stage_ = stage;
	local_features_num_ = param.local_features_num_;
	landmark_index_ = landmark_index;
	tree_depth_ = param.tree_depth_;
	trees_num_per_forest_ = param.trees_num_per_forest_;
	local_radius_ = param.local_radius_by_stage_[stage_];
	mean_shape_ = param.mean_shape_;
	regression_targets_ = &regression_targets; // get the address pointer, not reference
}

RandomForest::RandomForest(){
	
}

void RandomForest::SaveRandomForest(std::ofstream& fout){
//    fout.setf(std::ios::fixed, std::ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
    fout.precision(3);  // 设置精度 2
	fout << stage_ << " "
		<< local_features_num_ << " "
		<< landmark_index_ << " "
		<< tree_depth_ << " "
		<< trees_num_per_forest_ << " "
		<< local_radius_ << " "
		<< all_leaf_nodes_ << " "
		<< trees_.size() << std::endl;
	for (int i = 0; i < trees_.size(); i++){
		Node* root = trees_[i];
		WriteTree(root, fout);
	}
}

void RandomForest::WriteTree(Node* p, std::ofstream& fout){
	if (!p){
		fout << "#" << std::endl;
	}
	else{
		fout <<"Y" << " "
			<< p->threshold_ << " " 
			<< p->is_leaf_ << " "
			<< p->leaf_identity << " "
			<< p->depth_ << " "
            << p->score_ << " "
			<< p->feature_locations_.start.x << " "
			<< p->feature_locations_.start.y << " "
			<< p->feature_locations_.end.x << " "
			<< p->feature_locations_.end.y << std::endl;
		WriteTree(p->left_child_, fout);
		WriteTree(p->right_child_, fout);
	}
}

Node* RandomForest::ReadTree(std::ifstream& fin){
	std::string flag;
	fin >> flag;
	if (flag == "Y"){
		Node* p = new Node();
		fin >> p->threshold_
			>> p->is_leaf_
			>> p->leaf_identity
			>> p->depth_
            >> p->score_
			>> p->feature_locations_.start.x
			>> p->feature_locations_.start.y
			>> p->feature_locations_.end.x
			>> p->feature_locations_.end.y;
		p->left_child_ = ReadTree(fin);
		p->right_child_ = ReadTree(fin);
		return p;
	}
	else{
		return NULL;
	}
}

void RandomForest::LoadRandomForest(std::ifstream& fin){
	
	int tree_size;
	fin >> stage_
		>> local_features_num_
		>> landmark_index_
		>> tree_depth_
		>> trees_num_per_forest_
		>> local_radius_
		>> all_leaf_nodes_
		>> tree_size;
	std::string start_flag;
	trees_.clear();
	for (int i = 0; i < tree_size; i++){
		Node* root = ReadTree(fin);
		trees_.push_back(root);
	}
}
