#include "randomforest.h"
#include <time.h>
#include <algorithm>
#include <stack>
#include "regressor.h"

Node::Node(){
	left_child_ = NULL;
	right_child_ = NULL;
	is_leaf_ = false;
//    is_leaf_a = false;
	threshold_ = 0.0;
	leaf_identity = -1;
//	samples_ = -1;
//	thre_changed_ = false;
    score_ = 0.0;
}

Node::Node(Node* left, Node* right, float thres){
	Node(left, right, thres, false);
}

Node::Node(Node* left, Node* right, float thres, bool leaf){
	left_child_ = left;
	right_child_ = right;
	is_leaf_ = leaf;
//    is_leaf_a = leaf;
	threshold_ = thres;
	//offset_ = cv::Point2f(0, 0);
}

Node::~Node(){
    //    std::cout << "~Node()" << std::endl;
    if ( left_child_ != NULL ) delete left_child_;
    if ( right_child_ != NULL ) delete right_child_;
}

int my_cmp(std::pair<float,int> p1, std::pair<float,int> p2)
{
    return p1.first < p2.first;
};

bool RandomForest::TrainForest(//std::vector<cv::Mat_<float>>& regression_targets,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<int>& augmented_images_index,
	std::vector<cv::Mat_<float>>& augmented_ground_truth_shapes,
	std::vector<BoundingBox>& augmented_bboxes,
	std::vector<cv::Mat_<float> >& augmented_current_shapes,
    std::vector<int> & augmented_ground_truth_faces,
    std::vector<float> & current_fi,
    std::vector<float> & current_weight,
    std::vector<int> & find_times,
	std::vector<cv::Mat_<float> >& rotations,
	std::vector<float>& scales){
    
//    augmented_ground_truth_shapes_ = augmented_ground_truth_shapes;
//    augmented_current_shapes_ = augmented_current_shapes;
//    current_weight_ = current_weight;
    //std::cout << "build forest of landmark: " << landmark_index_ << " of stage: " << stage_ << std::endl;
	//regression_targets_ = &regression_targets;
//	time_t current_time;
//	current_time = time(0);
//	cv::RNG rd(current_time);
//	 random generate feature locations
	std::cout << "generate feature locations" << std::endl;
//	local_position_.clear();
//	local_position_.resize(local_features_num_);
//	for (int i = 0; i < local_features_num_; i++){
//		float x, y;
//		do{
//			x = rd.uniform(-local_radius_, local_radius_); 
//			y = rd.uniform(-local_radius_, local_radius_);
////            if ( i < 4 && x < -local_radius_/5.0) continue; //add by xujj, 避免脸部外侧的pixel
////            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
////            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
//		} while (x*x + y*y > local_radius_*local_radius_);
//		cv::Point2f a(x, y);
//
//		do{
//			x = rd.uniform(-local_radius_, local_radius_);
//			y = rd.uniform(-local_radius_, local_radius_);
////            if ( i < 4 && x < -local_radius_/5.0) continue;
////            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
////            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
//		} while (x*x + y*y > local_radius_*local_radius_);
//		cv::Point2f b(x, y);
//
//        //TODO，这个地方可以试多个策略：1）自己，2）自己和随机一个，3）随机两个
//        int landmark1 = (int)rd.uniform(0, landmark_num_);
//        int landmark2 = (int)rd.uniform(0, landmark_num_);
//		local_position_[i] = FeatureLocations(landmark1, landmark2, a, b);
//	}
//	//std::cout << "get pixel differences" << std::endl;
//	cv::Mat_<int> pixel_differences(local_features_num_, augmented_images_index.size()); // matrix: features*images
//	
//#pragma omp parallel for
//	for (int i = 0; i < augmented_images_index.size(); i++){
//		
//		cv::Mat_<float> rotation = rotations[i];
//		float scale = scales[i];
//		//getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]),mean_shape_, rotation, scale);
//		
//		for (int j = 0; j < local_features_num_; j++){
//			FeatureLocations pos = local_position_[j];
//			float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
//			float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
//			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
//			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
//			int real_x = delta_x + augmented_current_shapes[i](pos.lmark1, 0);
//			int real_y = delta_y + augmented_current_shapes[i](pos.lmark1, 1);
//			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
//			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
//			int tmp = (int)images[augmented_images_index[i]](real_y, real_x); //real_y at first
//
//			delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
//			delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
//			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
//			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
//			real_x = delta_x + augmented_current_shapes[i](pos.lmark2, 0);
//			real_y = delta_y + augmented_current_shapes[i](pos.lmark2, 1);
//			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
//			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
//			pixel_differences(j, i) = tmp - (int)images[augmented_images_index[i]](real_y, real_x); 
//		}
//	}
	// train Random Forest
	// construct each tree in the forest
	
	float overlap = 0.4;
	int step = floor(((float)augmented_images_index.size())*overlap / (trees_num_per_forest_ - 1));
	trees_.clear();
	all_leaf_nodes_ = 0;
    int accuDeleteNum = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
        //为每棵树都重新生成一次local feature，否则每个森林树太多时，特征不够用。如果16棵树，4层，需要31*16个不同特征
        local_position_.clear();
        local_position_.resize(local_features_num_);
        for (int n = 0; n < local_features_num_; n++){
            float x, y;
            do{
                x = rd.uniform(-local_radius_, local_radius_);
                y = rd.uniform(-local_radius_, local_radius_);
//                            if ( i < 4 && x < -local_radius_/5.0) continue; //add by xujj, 避免脸部外侧的pixel
//                            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
//                            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
            } while (x*x + y*y > local_radius_*local_radius_);
            cv::Point2f a(x, y);

            do{
                x = rd.uniform(-local_radius_, local_radius_);
                y = rd.uniform(-local_radius_, local_radius_);
//                            if ( i < 4 && x < -local_radius_/5.0) continue;
//                            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
//                            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
            } while (x*x + y*y > local_radius_*local_radius_);
            cv::Point2f b(x, y);

            //TODO，这个地方可以试多个策略：1）自己，2）自己和随机一个，3）随机两个
            //TODO，在前2个stage，用3，第三stage用2，后续stage，用1？如何？
            int landmark1, landmark2;
            if ( stage_ == 0 && landmark_index_ < 17 /*&& landmark_index_ < 50*/ ){
                landmark1 = (int)rd.uniform(0, landmark_num_);
                landmark2 = (int)rd.uniform(0, landmark_num_);
            }
            else{
                landmark1 = landmark_index_;
//                landmark2 = (int)rd.uniform(0, landmark_num_);
                landmark2 = landmark_index_;
            }
//            //test
//            landmark1 = landmark_index_;
//            landmark2 = landmark_index_;
            
//            else if (  n % (stage_ + 3) == 0 ){
//                landmark1 = landmark_index_;
//                landmark2 = landmark_index_;
//            }
//            else if ( n % (stage_ + 3) == 1 ){
//                landmark1 = landmark_index_;
//                landmark2 = (int)rd.uniform(0, landmark_num_);
//            }
//            else {
//                landmark1 = landmark_index_;
//                landmark2 = adjointPoint(landmark1);
//            }
            local_position_[n] = FeatureLocations(landmark1, landmark2, a, b);
        }
        //std::cout << "get pixel differences" << std::endl;
        cv::Mat_<int> pixel_differences(local_features_num_, augmented_images_index.size()); // matrix: features*images

//#pragma omp parallel for
        for (int n = 0; n < augmented_images_index.size(); n++){

            cv::Mat_<float> rotation = rotations[n];
            float scale = scales[n];
            int gauss = (int) augmented_bboxes[n].width / 300;
            gauss = std::min(2, gauss);
            for (int j = 0; j < local_features_num_; j++){
                FeatureLocations pos = local_position_[j];
                float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = scale*delta_x*augmented_bboxes[n].width / 2.0;
                delta_y = scale*delta_y*augmented_bboxes[n].height / 2.0;
                int real_x = delta_x + augmented_current_shapes[n](pos.lmark1, 0);
                int real_y = delta_y + augmented_current_shapes[n](pos.lmark1, 1);
                real_x = std::max(gauss, std::min(real_x, images[augmented_images_index[n]].cols - 1 - gauss)); // which cols
                real_y = std::max(gauss, std::min(real_y, images[augmented_images_index[n]].rows - 1 - gauss)); // which rows
                int tmp = (int)(2*images[augmented_images_index[n]](real_y, real_x) + images[augmented_images_index[n]](real_y-gauss, real_x) + images[augmented_images_index[n]](real_y+gauss, real_x) + images[augmented_images_index[n]](real_y, real_x-gauss) +images[augmented_images_index[n]](real_y, real_x+gauss)) / 6 ; //real_y at first
                //int tmp = images[augmented_images_index[n]](real_y, real_x);
                
                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = scale*delta_x*augmented_bboxes[n].width / 2.0;
                delta_y = scale*delta_y*augmented_bboxes[n].height / 2.0;
                real_x = delta_x + augmented_current_shapes[n](pos.lmark2, 0);
                real_y = delta_y + augmented_current_shapes[n](pos.lmark2, 1);
                real_x = std::max(gauss, std::min(real_x, images[augmented_images_index[n]].cols - 1 - gauss)); // which cols
                real_y = std::max(gauss, std::min(real_y, images[augmented_images_index[n]].rows - 1 - gauss)); // which rows
                int tmp2 = (int)(2*images[augmented_images_index[n]](real_y, real_x) + images[augmented_images_index[n]](real_y-gauss, real_x) + images[augmented_images_index[n]](real_y+gauss, real_x) + images[augmented_images_index[n]](real_y, real_x-gauss) +images[augmented_images_index[n]](real_y, real_x+gauss)) / 6 ;
                //int tmp2 = images[augmented_images_index[n]](real_y, real_x);
                if ( i % 2 == 0 ){
                    pixel_differences(j, n) = abs(tmp - tmp2);
                }
                else{
                    pixel_differences(j, n) = tmp - tmp2;
                }
            }
        }

        //计算每个训练实例的weight
        int drop_pos_count = 0;
        int drop_neg_count = 0;
        for(int k=0;k<current_weight.size();++k)
        {
            current_weight[k] = exp(0.0-augmented_ground_truth_faces[k]*current_fi[k]);
            if ( current_weight[k] > 10000000000.0 && find_times[k] < MAXFINDTIMES){ //试验一下去掉这个的效果
                find_times[k] = MAXFINDTIMES+4;
                if ( augmented_ground_truth_faces[k] == 1 ){
                    drop_pos_count++;
                }
                else{
                    drop_neg_count++;
                }
                augmented_ground_truth_faces[k] = -1;
                if ( debug_on_ ){
                    DrawPredictImage(images[augmented_images_index[k]], augmented_current_shapes[k]);
                    std::cout << "fi:" << current_fi[k] << std::endl;
                }
            }
        }
        if ( drop_pos_count > 0 || drop_neg_count > 0 )
            std::cout << "drop too high weight, pos:" << drop_pos_count << " neg:" << drop_neg_count << std::endl;
        
        //这个地方这么做不对了，因为负例都在后面，这么一分，后面的树都是负例。先全部实例都拿去训练算了，所以start和end都搞成全量吧。。。
        int start_index = 0;//i*step;
        int end_index = augmented_images_index.size();//augmented_images_index.size() - (trees_num_per_forest_ - i - 1)*step;
		//cv::Mat_<int> data = pixel_differences(cv::Range(0, local_features_num_), cv::Range(start_index, end_index));
		//cv::Mat_<int> sorted_data;
		//cv::sortIdx(data, sorted_data, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
		std::set<int> selected_feature_indexes; //这个是用来表示那个feature已经被用过了
        selected_feature_indexes.clear();

//        current_time = time(0);
//        cv::RNG rd(current_time);
		std::vector<int> images_indexes;
		for (int j = start_index; j < end_index; j++){
            if ( find_times[j] < MAXFINDTIMES){
                //让每棵树的样本有点不一样
                int r = rd.uniform(0, trees_num_per_forest_);
                if ( r != 0 ){
                    images_indexes.push_back(j);
                }
            }
		}
        
		Node* root = BuildTree(selected_feature_indexes, pixel_differences, images_indexes, augmented_ground_truth_faces, current_weight, 0);
		trees_.push_back(root);
        
        //计算每个训练实例的fi
        for ( int n=0; n<augmented_images_index.size(); n++){
            if ( find_times[n] >= MAXFINDTIMES ) continue;
            float score = 0;
            //用训练实例去遍历此树得出叶子节点的score
//            cv::Mat_<float> rotation;
//            float scale;
//            getSimilarityTransform(ProjectShape(augmented_current_shapes[n],augmented_bboxes[n]),mean_shape_,rotation,scale);
            GetBinaryFeatureIndex(i, images[augmented_images_index[n]], augmented_bboxes[n], augmented_current_shapes[n], rotations[n] , scales[n], score);
            current_fi[n] += score;
        }
        
        //开始计算这棵树的detection threshold
        std::vector<std::pair<float,int>> fiSort;
        fiSort.clear();
        for(int n=0;n<current_fi.size();++n)
        {
            if ( find_times[n] >= MAXFINDTIMES ) continue;
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
            //TODO：这个地方这样貌似取了最大召回率，应该再核对一下错误率。
            //
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
        
        //TODO:这个地方要开始删除fi分值小于阀值的负例
        //TODO:如果负例不够了，要挖掘一些。挖掘应该指的的找一些图片，按训练到目前为止的模型判定为face但实际非face的东东
        int deleteNumber = 0;
        int mineHardNegNumber = 0;
        int mineNormalNegNumber = 0;
//        int maxFindRepeat = 2;
        
        for ( int n=0; n<fiSort.size(); ++n){
            if ( fiSort[n].first < root->score_ ){
                deleteNumber++;
            }
            else{
                break;
            }
        }
        //TODO：后面有else break这个要去掉就可能可以并行来
//#pragma omp parallel for
        for ( int n=0; n<deleteNumber; ++n){
            if ( fiSort[n].first < root->score_ ){
                int idx = fiSort[n].second;
                bool faceFound = false;
                
                //接下来开始挖掘hard neg example
                if ( augmented_ground_truth_faces[idx] == -1 && find_times[idx] < MAXFINDTIMES ){
                    
//                    BoundingBox max_fi_box;
//                    cv::Mat_<float> max_fi_shape;
//                    float max_fi = -100000000.0;
                    int p = idx % true_pos_num_;
                    int so = (find_times[idx] & 0x7f000000) >> 24;
                    int ss = (find_times[idx] & 0x00ff0000) >> 16;
                    int sx = (find_times[idx] & 0x0000ff00) >> 8;
                    int sy = (find_times[idx] & 0x000000ff);
                    if ( so < 4 ){ //先普通挖掘
                        //int ss = find_times[idx] / ( MAXFINDTIMES / 32 );
    //                    for ( int sw_size = 50 * std::pow(1.1, ss); sw_size < std::min(cols, rows); sw_size = 50 * std::pow(1.1, ss++)){
    //                    int sw_size = 50 + 10 * ss;
                        for ( int orient = so; orient < 4; orient++ ){
                            float cols = images[augmented_images_index[idx]].cols;
                            float rows = images[augmented_images_index[idx]].rows;
                            for ( int sw_size_o = 64 * std::pow(1.06, ss); sw_size_o < std::min(cols, rows); sw_size_o = 64 * std::pow(1.06, ss)){
                                int sw_size;
                                //把一部分负例从大到小来用
                                if ( idx % 2 == 0 ){
                                    sw_size = sw_size_o;
                                }
                                else{
                                    sw_size = std::min(cols, rows) + 63 - sw_size_o;
                                }
                                ss++;
                                int p = (idx+ss+so) % true_pos_num_;
                                float shuffle_size = sw_size * 0.06;
                                if ( shuffle_size > 5 ) shuffle_size = 5;
                                for ( int sw_x = shuffle_size * sx; sw_x<cols - sw_size && sx < 256; sw_x+=shuffle_size){
                                    sx++;
                                    for ( int sw_y = shuffle_size * sy; sw_y<rows - sw_size && sy < 256; sw_y+=shuffle_size){
                                        sy++;
                                        BoundingBox new_box;
                                        new_box.start_x=sw_x;
                                        new_box.start_y=sw_y;
                                        new_box.width= sw_size;
                                        new_box.height=new_box.width;
                                        new_box.center_x=new_box.start_x + new_box.width/2.0;
                                        new_box.center_y=new_box.start_y + new_box.height/2.0;
//                                        cv::Mat_<float> temp1 = ProjectShape(augmented_ground_truth_shapes[idx], augmented_bboxes[idx]);
//                                        augmented_ground_truth_shapes[idx] = ReProjection(temp1, new_box);
                                        cv::Mat_<float> temp2 = ProjectShape(augmented_ground_truth_shapes[p], augmented_bboxes[p]);
                                        augmented_current_shapes[idx]=ReProjection(temp2, new_box);
                                        augmented_bboxes[idx]=new_box;

                                        int tmp_isface=1;
                                        float tmp_fi=0;
                                        
                                        //这个时候，自己在第stage_, landmark_index_的i树上
                                        //cv::Mat_<float> shape = augmented_current_shapes[idx].clone();
                                        casRegressor_->NegMinePredict(images[augmented_images_index[idx]],
                                                                      augmented_current_shapes[idx], new_box, tmp_isface, tmp_fi, stage_, landmark_index_, i);
                                        if ( tmp_isface){
                                            current_fi[idx] = tmp_fi;
                                            current_weight[idx] = exp(0.0-augmented_ground_truth_faces[idx]*current_fi[idx]);
                                            if ( current_weight[idx] > 10000000 ) continue;
                                            faceFound = true;
                                            //augmented_current_shapes[idx] = shape;
                                            //augmented_bboxes[idx]=new_box;
                                            find_times[idx] = 256*256*256*orient + 256*256*ss + 256*sx + sy;
                                            
                                            cv::Mat_<float> rotation;
                                            float scale;
                                            getSimilarityTransform(ProjectShape(augmented_current_shapes[idx], augmented_bboxes[idx]), mean_shape_, rotation, scale);
                                            rotations[idx] = rotation;
                                            scales[idx] = scale;
                                            
                                            //std::cout << tmp_fi << " so:" << so << " ss:" << ss << " sx:" << sx << " sy:" << sy << " idx:" << idx << std::endl;;
    //                                        cv::Rect rect;
    //                                        rect.x = new_box.start_x;
    //                                        rect.y = new_box.start_y;
    //                                        rect.width = new_box.width;
    //                                        rect.height = new_box.height;
    //                                        std::cout << rect << std::endl;
                                            //测试看看
    //                                        if ( landmark_index_ > 25 ){
    //                                            cv::Mat_<uchar> image = images[augmented_images_index[idx]].clone();
    //                                            cv::Rect rect;
    //                                            rect.x = new_box.start_x;
    //                                            rect.y = new_box.start_y;
    //                                            rect.width = new_box.width;
    //                                            rect.height = new_box.height;
    //                                            std::cout << rect << std::endl;
    //                                            cv::rectangle(image, rect, cv::Scalar(100));
    //                                            cv::imshow("test", image);
    //                                            cv::waitKey(0);
    //                                        }
                                            break;
                                        }
                                    }
                                    if ( faceFound ){
                                        break;
                                    }
                                    sy = 0;
                                }
                                if ( faceFound){
                                    break;
                                }
                                sx = 0; sy = 0;
                            }
                            if ( faceFound){
                                break;
                            }
                            ss = 0; sx = 0; sy = 0;
//                            //把图片旋转90度
                              //这个地方如果索引了同一张图片，一个挖掘完了改索引为正例图片，另一个还没挖掘玩给旋转了，然后就悲剧了，正例被旋转。。。
//                            cv::Mat_<uchar> temp = images[augmented_images_index[idx]];
////                            cv::imshow("origin", temp);
////                            cv::waitKey(0);
//                            cv::transpose(temp, temp);
////                            cv::imshow("transpose", temp);
////                            cv::waitKey(0);
//                            cv::flip(temp, temp, 1);
////                            cv::imshow("flip", temp);
////                            cv::waitKey(0);
//                            images[augmented_images_index[idx]] = temp;
                        }
                        if ( !faceFound ){
                            find_times[idx] = 256*256*256*4;
                            so = 4;
                            ss = 0;
                            sx = 0;
                            sy = 0;
                        }
                    }
                    if ( !faceFound ){
                        //再从正例中找特别负例，以后都得从这儿找，不能再执行上面那部分
//                        std::cout<<"得从正例中找负例:" << idx << std::endl;
                        int p = augmented_images_index[idx] - true_pos_num_;
                        if ( p < true_pos_num_ ){
//                            augmented_images_index[idx] = p;
                            images[augmented_images_index[idx]] = images[p];
                            float cols = images[augmented_images_index[idx]].cols;
                            float rows = images[augmented_images_index[idx]].rows;
                            
                            BoundingBox pos_box = augmented_bboxes[p*(param_.initial_guess_+1)]; //这个地方坑死了。。。
                            BoundingBox search_box;
                            search_box.start_x = pos_box.start_x - 0.9 * pos_box.width;
                            search_box.start_y = pos_box.start_y - 0.9 * pos_box.height;
                            search_box.width = 2.8 * pos_box.width;
                            search_box.height = 2.8 * pos_box.height;
                            if ( search_box.start_x < 0 ) search_box.start_x = 0;
                            if ( search_box.start_y < 0 ) search_box.start_y = 0;
                            if (( search_box.start_x + search_box.width ) > cols ) search_box.width = cols - search_box.start_x;
                            if (( search_box.start_y + search_box.height) > rows ) search_box.height = rows - search_box.start_y;
                            
                            
                            for ( int sw_size = 50 * std::pow(1.2, ss); sw_size < std::min(search_box.width, search_box.height); sw_size = 50 * std::pow(1.2, ss)){
                                ss++;
                                float shuffle_size = sw_size * 0.2;
                                //if ( shuffle_size > 15 ) shuffle_size = 15;
                                for ( int sw_x = shuffle_size * sx; sw_x<search_box.width - sw_size && sx < 256; sw_x+=shuffle_size){
                                    sx++;
                                    for ( int sw_y = shuffle_size * sy; sw_y<search_box.height - sw_size && sy < 256; sw_y+=shuffle_size){
                                        sy++;
                                        BoundingBox new_box;
                                        new_box.start_x= search_box.start_x + sw_x;
                                        new_box.start_y= search_box.start_y + sw_y;
                                        new_box.width= sw_size;
                                        new_box.height=new_box.width;
                                        new_box.center_x=new_box.start_x + new_box.width/2.0;
                                        new_box.center_y=new_box.start_y + new_box.height/2.0;
                                        
                                        float delta_start = sqrtf(powf((new_box.start_x - pos_box.start_x), 2.0) + powf((new_box.start_y - pos_box.start_y), 2.0));
                                        float delta_end = sqrtf(powf((new_box.start_x + new_box.width - pos_box.start_x - pos_box.width), 2.0) + powf((new_box.start_y + new_box.height - pos_box.start_y - pos_box.height), 2.0));
                                        if ( delta_start < 0.4 * pos_box.width && delta_end < 0.4 * pos_box.width ) continue; //判断与正例的位置接近则不采用
                                        if ( (delta_start + delta_end) < 0.5 * pos_box.width  ) continue;
                                        
//                                        cv::Mat_<float> temp1 = ProjectShape(augmented_ground_truth_shapes[p], augmented_bboxes[p]);
//                                        augmented_ground_truth_shapes[idx] = ReProjection(temp1, new_box);
                                        //cv::Mat_<float> temp2 = ProjectShape(augmented_current_shapes[p], augmented_bboxes[p]);
                                        cv::Mat_<float> temp2 = mean_shape_;
                                        augmented_current_shapes[idx]=ReProjection(temp2, new_box);
                                        augmented_bboxes[idx]=new_box;
                                        
                                        int tmp_isface=1;
                                        float tmp_fi=0;
                                        
                                        //cv::Mat_<float> shape = augmented_current_shapes[idx].clone();
                                        casRegressor_->NegMinePredict(images[augmented_images_index[idx]],
                                                                      augmented_current_shapes[idx], new_box, tmp_isface, tmp_fi, stage_, landmark_index_, i);
                                        
                                        if ( tmp_isface){
                                            float error = CalculateError2(augmented_ground_truth_shapes[p*(param_.initial_guess_+1)], augmented_current_shapes[idx], stage_, landmark_index_);
                                            //float error = CalculateError(augmented_ground_truth_shapes[p*(param_.initial_guess_+1)], augmented_current_shapes[idx]);
                                            
                                            //FOR DEBUG
                                            if ( debug_on_ ){
                                                std::cout << " error:" << error << " fi:" << tmp_fi << " stage:" << stage_ << " landmark:" << landmark_index_ << " tree:" << i << " idx:" << idx << " image index:" << augmented_images_index[idx] << " p:" << p << std::endl;
                                                cv::Mat_<uchar> tempImage = images[augmented_images_index[idx]].clone();
                                                cv::Mat_<float> tempShape = reConvertShape(augmented_current_shapes[idx]);
                                                for (int iii = 0; iii < tempShape.rows; iii++){
                                                    cv::circle(tempImage, cv::Point2f(tempShape(iii, 0), tempShape(iii, 1)), 1, cv::Scalar(255,255,255));
                                                    if ( iii != 0 && iii != 17 && iii != 22 && iii != 27 && iii!= 36 && iii != 42 && iii!= 48 && iii!= 60 && iii!=68 && iii!=69)
                                                        cv::line(tempImage, cv::Point2f(tempShape(iii-1, 0), tempShape(iii-1, 1)), cv::Point2f(tempShape(iii, 0), tempShape(iii, 1)), cv::Scalar(0,255,0));
                                                }
                                                cv::line(tempImage, cv::Point2f(tempShape(36, 0), tempShape(36, 1)), cv::Point2f(tempShape(41, 0), tempShape(41, 1)), cv::Scalar(0,255,0));
                                                cv::line(tempImage, cv::Point2f(tempShape(42, 0), tempShape(42, 1)), cv::Point2f(tempShape(47, 0), tempShape(47, 1)), cv::Scalar(0,255,0));
                                                cv::line(tempImage, cv::Point2f(tempShape(30, 0), tempShape(30, 1)), cv::Point2f(tempShape(35, 0), tempShape(35, 1)), cv::Scalar(0,255,0));
                                                cv::line(tempImage, cv::Point2f(tempShape(48, 0), tempShape(48, 1)), cv::Point2f(tempShape(59, 0), tempShape(59, 1)), cv::Scalar(0,255,0));
                                                cv::line(tempImage, cv::Point2f(tempShape(60, 0), tempShape(60, 1)), cv::Point2f(tempShape(67, 0), tempShape(67, 1)), cv::Scalar(0,255,0));
                                                
                                                cv::Rect rect;
                                                rect.x = new_box.start_x;
                                                rect.y = new_box.start_y;
                                                rect.width = new_box.width;
                                                rect.height = new_box.height;
                                                cv::rectangle(tempImage, rect, cv::Scalar(255,0,0), 1);
                                                
                                                rect.x = search_box.start_x;
                                                rect.y = search_box.start_y;
                                                rect.width = search_box.width;
                                                rect.height = search_box.height;
                                                cv::rectangle(tempImage, rect, cv::Scalar(0,0,255), 1);
                                                
                                                rect.x = pos_box.start_x;
                                                rect.y = pos_box.start_y;
                                                rect.width = pos_box.width;
                                                rect.height = pos_box.height;
                                                cv::rectangle(tempImage, rect, cv::Scalar(0,255,255), 1);
                                                
                                                cv::imshow("negmine", tempImage);
                                                cv::waitKey(0);
                                                
                                            }
                                            
                                            
                                            if ( ( error >= 0.3 || ( error > (0.2 && + (5-stage_)*0.04 ) ) )/* && tmp_fi < 45.0 */ ){
//                                                if ( tmp_fi > 0 ) tmp_fi /= 5.0;
                                                current_fi[idx] = tmp_fi;
                                                current_weight[idx] = exp(0.0-augmented_ground_truth_faces[idx]*current_fi[idx]);
                                                if ( current_weight[idx] > 10000000 ) continue;
                                                faceFound = true;
//                                                augmented_current_shapes[idx] = shape;
//                                                augmented_bboxes[idx]=new_box;
                                                find_times[idx] = 256*256*256*4 + 256*256*ss + 256*sx + sy;
                                                //std::cout << "hard:" << tmp_fi << " so:" << so << " ss:" << ss << " sx:" << sx << " sy:" << sy << " idx:" << idx << std::endl;
                                                cv::Mat_<float> rotation;
                                                float scale;
                                                getSimilarityTransform(ProjectShape(augmented_current_shapes[idx], augmented_bboxes[idx]), mean_shape_, rotation, scale);
                                                rotations[idx] = rotation;
                                                scales[idx] = scale;
                                                break;
                                            }
                                        }
                                    }
                                    if ( faceFound ){
                                        break;
                                    }
                                    sy = 0;
                                }
                                if ( faceFound){
                                    break;
                                }
                                sx = 0; sy = 0;
                            }
                            if ( faceFound ){
                                mineHardNegNumber++;
                                //std::cout<< "从正例中找到一个" << std::endl;
                            }
                            else{
                                find_times[idx] = MAXFINDTIMES;
//                                if ( find_times[idx] >= MAXFINDTIMES - maxFindRepeat ){
//                                    find_times[idx]++;
//                                }
//                                else{
//                                    find_times[idx] = MAXFINDTIMES - maxFindRepeat;
//                                }
                            }
                        } //没有对应的正例可以用
                        else{
                            find_times[idx] = MAXFINDTIMES;
                        }
                    }
                    else{
                        mineNormalNegNumber++;
                    }
                }
                else{
                    std::cout << "no mine for idx:" << idx << " find_times:" << find_times[idx] << " " << augmented_ground_truth_faces[idx] <<std::endl;
                }
            }
        }
        accuDeleteNum += deleteNumber;
        std::cout << "fi<threshold delete number:" << deleteNumber << " threshold:" << root->score_  << " normal mine:" << mineNormalNegNumber <<" hard mine:" << mineHardNegNumber  << std::endl;
	}
	/*int count = 0;
	for (int i = 0; i < trees_num_per_forest_; i++){
		Node* root = trees_[i];
		count = MarkLeafIdentity(root, count);
	}
	all_leaf_nodes_ = count;*/
    std::cout << "accumulated delete number:" << accuDeleteNum << std::endl;
    
	return true;
}


Node* RandomForest::BuildTree(std::set<int>& selected_feature_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int> & augmented_ground_truth_faces,std::vector<float> & current_weight, int current_depth){
	if (images_indexes.size() > 0){ // the node may not split under some cases
		Node* node = new Node();
		node->depth_ = current_depth;
//		node->samples_ = images_indexes.size();
		std::vector<int> left_indexes, right_indexes;
//        if ( current_depth == tree_depth_ - DETECT_ADD_DEPTH ){
//            node->is_leaf_a = true;
//            node->leaf_identity = all_leaf_nodes_;
//            all_leaf_nodes_++;
//          
//        }
		if (current_depth == tree_depth_){ // the node reaches max depth
			node->is_leaf_ = true;
			node->leaf_identity = all_leaf_nodes_;
			all_leaf_nodes_++;
            //计算叶子节点的score
            float leaf_pos_weight = FLT_MIN;
            float leaf_neg_weight = FLT_MIN;
            for ( int i=0; i<images_indexes.size(); i++){
                int index = images_indexes[i];
                if ( augmented_ground_truth_faces[index] == 1){
                    leaf_pos_weight += current_weight[index];
                }
                else{
                    leaf_neg_weight += current_weight[index];
                }
            }
            
             //加上一个shape alignment的情况来影响score试试，让shape不好的正例score降低
//            node->score_ = 0.5*(((leaf_pos_weight-0.0)<FLT_EPSILON)?0:log(leaf_pos_weight))-0.5*(((leaf_neg_weight-0.0)<FLT_EPSILON)?0:log(leaf_neg_weight))/*/log(2.0)*/;
            node->score_ = 0.5*(log(leaf_pos_weight)- log(leaf_neg_weight)) /*/log(2.0)*/;
			return node;
		}

        int ret = FindSplitFeature(node, selected_feature_indexes, pixel_differences, images_indexes, augmented_ground_truth_faces,
                                   current_weight, left_indexes, right_indexes);
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1){ // the current node contain all sample when reaches max variance reduction, it is leaf node
            std::cout<< "the current node contain all sample when reaches max variance reduction, it is leaf node" << " images:" << images_indexes.size() << std::endl;
			node->is_leaf_ = true;
//            if (current_depth < tree_depth_ - DETECT_ADD_DEPTH ) {
//                node->is_leaf_a = true;
                node->leaf_identity = all_leaf_nodes_;
                all_leaf_nodes_++;
//            }
            //计算叶子节点的score, 同上
            float leaf_pos_weight = FLT_MIN;
            float leaf_neg_weight = FLT_MIN;
            for ( int i=0; i<images_indexes.size(); i++){
                int index = images_indexes[i];
                if ( augmented_ground_truth_faces[index] == 1){
                    leaf_pos_weight += current_weight[index];
                }
                else{
                    leaf_neg_weight += current_weight[index];
                }
            }
            
//            node->score_ = 0.5*(((leaf_pos_weight-0.0)<FLT_EPSILON)?0:log(leaf_pos_weight))-0.5*(((leaf_neg_weight-0.0)<FLT_EPSILON)?0:log(leaf_neg_weight))/*/log(2.0)*/;
            node->score_ = 0.5*(log(leaf_pos_weight)- log(leaf_neg_weight));
			return node;
		}

		//if (current_depth + 1 < tree_depth_){
        node->left_child_ = BuildTree(selected_feature_indexes, pixel_differences, left_indexes, augmented_ground_truth_faces,
                                      current_weight, current_depth + 1);
        node->right_child_ = BuildTree(selected_feature_indexes, pixel_differences, right_indexes, augmented_ground_truth_faces,
                                       current_weight,  current_depth + 1);
		//}
		return node;
	}
	else{ // this case is not possible in this data structure
		return NULL;
	}
}

int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_feature_indexes, 
	cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int> & augmented_ground_truth_faces,std::vector<float> & current_weight, std::vector<int>& left_indexes, std::vector<int>& right_indexes){
//	std::vector<int> val;
	//cv::Mat_<int> sorted_fea;
//	time_t current_time;
//	current_time = time(0);
//	cv::RNG rd(current_time);
	int threshold;
//	float var = -1000000000000.0; // use -DBL_MAX will be better
	int feature_index = -1;
//	std::vector<int> tmp_left_indexes, tmp_right_indexes;
    
    std::vector<double> vars;
    std::vector<double> entropys;
    std::vector<std::pair<int,int>> thresholds;
    vars.clear();
    entropys.clear();
    thresholds.clear();
	//int j = 0, tmp_index;
//#pragma omp parallel for
	for (int j = 0; j < local_features_num_; j++){
		if (selected_feature_indexes.find(j) == selected_feature_indexes.end()){
            std::vector<int> data;
            data.reserve(images_indexes.size());
            for (int i = 0; i < images_indexes.size(); i++){
                data.push_back(pixel_differences(j, images_indexes[i]));
            }
            std::sort(data.begin(), data.end());
            
            //重复循环t次，尝试取不同的threshhold，总共的次数为t*local_feature_num。看看能否使得分类和回归的同时最优可能增加一些
            for ( int t=0; t<1; t++ ){
                int num_l_shapes = 0, num_r_shapes = 0;
                double var_lc = 0.0, var_rc = 0.0, var_red = 0.0;
                double Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;
                double Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;
                
                int num_l_pos_faces = 0, num_l_neg_faces = 0, num_r_pos_faces = 0, num_r_neg_faces = 0;
                double total_l_pos_weight = 0.0, total_l_neg_weight = 0.0;
                double total_r_pos_weight = 0.0, total_r_neg_weight = 0.0;
                
                // random generate threshold
                int tmp_index = floor((int)(images_indexes.size()*(0.5 + 0.8*(rd.uniform(0.0, 1.0) - 0.5))));
                int tmp_threshold = data[tmp_index];
                for (int i = 0; i < images_indexes.size(); i++){
                    int index = images_indexes[i];
                    if (pixel_differences(j, index) < tmp_threshold){
    //					tmp_left_indexes.push_back(index);
                        if ( augmented_ground_truth_faces[index] == 1){
                            // do with regression target
                            num_l_shapes++;
                            float value = regression_targets_->at(index)(landmark_index_, 0);
                            Ex_2_lc += pow(value, 2);
                            Ex_lc += value;
                            value = regression_targets_->at(index)(landmark_index_, 1);
                            Ey_2_lc += pow(value, 2);
                            Ey_lc += value;
                            
                            num_l_pos_faces++;
                            total_l_pos_weight += current_weight[index];
                        }
                        else{ //负样本
                            num_l_neg_faces++;
                            total_l_neg_weight += current_weight[index];
                        }
                    }
                    else{
    //					tmp_right_indexes.push_back(index);
                        if ( augmented_ground_truth_faces[index] == 1){
                            num_r_shapes++;
                            float value = regression_targets_->at(index)(landmark_index_, 0);
                            Ex_2_rc += pow(value, 2);
                            Ex_rc += value;
                            value = regression_targets_->at(index)(landmark_index_, 1);
                            Ey_2_rc += pow(value, 2);
                            Ey_rc += value;
                            
                            num_r_pos_faces++;
                            total_r_pos_weight += current_weight[index];
                        }
                        else{ //负样本
                            num_r_neg_faces++;
                            total_r_neg_weight += current_weight[index];
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
                if ( var_red != var_red){
                    std::cout<<"var_red:"<<var_red << " var_lc:" << var_lc << " var_rc:" << var_rc << std::endl;
                }
                thresholds.push_back(std::pair<int,int>(tmp_threshold, j));
                vars.push_back(var_red);
                
                
                int left_sample_num = num_l_pos_faces + num_l_neg_faces;
                int right_sample_num = num_r_pos_faces + num_r_neg_faces;
    //            int total_sample_num =  left_sample_num + right_sample_num;
                
                double total_l_weight = total_l_pos_weight + total_l_neg_weight;
                double total_r_weight = total_r_pos_weight + total_r_neg_weight;
                double total_weight = total_l_weight + total_r_weight;
                
                double entropy = 0.0, entropy_lc = 0.0, entropy_rc = 0.0;
                double entropy_tmp = 0.0;
                
                if ( left_sample_num == 0 ){
                    entropy_lc = 0.0;
                }
                else{
                    entropy_tmp = total_l_pos_weight / ( total_l_weight + DBL_MIN);
                    if ( (entropy_tmp-0.0) < DBL_EPSILON || (1.0-entropy_tmp) < DBL_EPSILON){
                        entropy_lc = 0.0;
                    }
                    else{
                        entropy_lc = - ( total_l_weight / ( total_weight + DBL_MIN)) * ((entropy_tmp) * log(entropy_tmp)/log(2.0) + ( 1.0 - entropy_tmp) * log(1.0-entropy_tmp)/log(2.0));
                    }
                }
                if ( entropy_lc != entropy_lc ){ //判断是不是Nan
                    std::cout<<"entropy_lc:"<<entropy_lc<< " tw:"<<total_weight<< " tlw:"<<total_l_weight<< " tlpw:" << total_l_pos_weight << " tmp" << entropy_tmp << std::endl;
                    entropy_lc = 0;
                }

                if ( right_sample_num == 0 ){
                    entropy_rc = 0.0;
                }
                else{
                    entropy_tmp = total_r_pos_weight / ( total_r_weight + DBL_MIN);
                    if ( (entropy_tmp-0.0) < DBL_EPSILON || (1.0-entropy_tmp) < DBL_EPSILON){
                        entropy_rc = 0.0;
                    }
                    else{
                        entropy_rc = - ( total_r_weight / ( total_weight + DBL_MIN)) * ((entropy_tmp ) * log(entropy_tmp)/log(2.0) + ( 1.0 - entropy_tmp) * log(1.0-entropy_tmp)/log(2.0));
                    }
                }

                if ( entropy_rc != entropy_rc ){ //判断是不是Nan
                    std::cout<<"entropy_rc:"<<entropy_rc<< " tw:"<<total_weight<< " trw:"<<total_r_weight<<  " trpw:" << total_r_pos_weight << " tmp" << entropy_tmp << std::endl;
                    entropy_rc = 0;
                }

                entropy = entropy_lc + entropy_rc;

                entropys.push_back(entropy);
            }
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
    
    
    //这里把var和entropy做归一化，然后取其和的最小值，这样可以做到分类和回归在同一个feature上都做到较优。用一个factor来控制影响比例
    //这个方法和原论文的方法不同，原论文按照概率值来选择偏向分类还是回归树
    
    double minvar = *std::min_element(std::begin(vars), std::end(vars));
    double maxvar = *std::max_element(std::begin(vars), std::end(vars));
    double minent = *std::min_element(std::begin(entropys), std::end(entropys));
    double maxent = *std::max_element(std::begin(entropys), std::end(entropys));
    
    double summin = DBL_MAX;
    double indexmin = 0;
    float df = detect_factor_;
//    if ( landmark_index_ > 50 ){
//        if ( stage_ == 0 ){
//            df = 0.3; //脸的外轮廓多alignement，少detect
//        }
//        else if ( stage_ == 1 ){
//            df = 0.4;
//        }
//        else if ( stage_ == 2 ){
//            df = 0.7;
//        }
//        else if ( stage_ == 3 ){
//            df = 0.8;
//        }
//        else {
//            df = 0.9;
//        }
//    }
    if ( node->depth_ >= tree_depth_ - DETECT_ADD_DEPTH){
        df = 1.0; //全部用于detect分类
    }
//    else if ( landmark_index_ < 17 ){
//        df = detect_factor_ + 0.1;
//    }
//    if ( stage_ == 0 && landmark_index_ < 10 ) df = 0.9;
//    if ( stage_ == 1 && landmark_index_ > 26 && landmark_index_ < 35 ) df = 0.8;
//    if ( stage_ == 2 && landmark_index_ > 35 && landmark_index_ < 48 ) df = 0.7;
//    if ( stage_ == 3 && landmark_index_ > 26 && landmark_index_ < 35 ) df = 0.5;
//    if ( stage_ == 4 && landmark_index_ > 35 && landmark_index_ < 48 ) df = 0.5;
    

    
    for ( int i=0; i<vars.size(); i++){
        double tmpvar = ( vars[i] - minvar ) / (maxvar - minvar + DBL_MIN);
        double tmpent = ( entropys[i] - minent ) / (maxent - minent + DBL_MIN);
        double tmpsum = (1.0 - df) * tmpvar + df * tmpent; //这个可以根据stage用不同的系数，TODO:要不要根据depth也调整？
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
            std::cout << "minvar:" << minvar << " maxvar:" << maxvar << " minent:" << minent << " maxent:" << maxent << " threshhold:" << threshold <<" index:" << indexmin << " size:" << images_indexes.size() << std::endl;
//            if (threshold == 0 ){
//                for ( int i=0; i<vars.size(); i++){
//                    std::cout << i << ": var:" << vars[i] << " entropy:" << entropys[i] << " threshold:" << thresholds[i].first << " index:" << thresholds[i].second << std::endl;
//                }
//            }
			return 1;
		}
		node->threshold_ = threshold;
//		node->thre_changed_ = true;
		node->feature_locations_ = local_position_[feature_index];
		selected_feature_indexes.insert(feature_index);
		return 0;
	}
	
	return -1;
}

//int RandomForest::MarkLeafIdentity(Node* node, int count){
//	std::stack<Node*> s;
//	Node* p_current = node; 
//	
//	if (node == NULL){
//		return count;
//	}
//	// the node in the tree is either leaf node or internal node that has both left and right children
//	while (1)//p_current || !s.empty())
//	{
//		
//		if (p_current->is_leaf_){
//			p_current->leaf_identity = count;
//			count++;
//			if (s.empty()){
//				return count;
//			}
//			p_current = s.top()->right_child_;
//			s.pop();
//		}
//		else{
//			s.push(p_current);
//			p_current = p_current->left_child_;
//		}
//		
//		/*while (!p_current && !s.empty()){
//			p_current = s.top();
//			s.pop();
//			p_current = p_current->right_child_; 
//		}*/
//	}
//	
//}

//cv::Mat_<float> RandomForest::GetBinaryFeatures(const cv::Mat_<float>& image,
//	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale){
//	cv::Mat_<float> res(1, all_leaf_nodes_, 0.0);
//	for (int i = 0; i < trees_num_per_forest_; i++){
//		Node* node = trees_[i];
//		while (!node->is_leaf_){
//			int direction = GetNodeOutput(node, image, bbox, current_shape, rotation, scale);
//			if (direction == -1){
//				node = node->left_child_;
//			}
//			else{
//				node = node->right_child_;
//			}
//		}
//		res(0, node->leaf_identity) = 1.0;
//	}
//	return res;
//}

int RandomForest::GetBinaryFeatureIndex(int tree_index, const cv::Mat_<uchar>& image,
	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale, float& score){
	Node* node = trees_[tree_index];
    int gauss = (int) bbox.width / 300;
    gauss = std::min(2, gauss);
	while (!node->is_leaf_){
		FeatureLocations& pos = node->feature_locations_;
		float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
		float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		int real_x = delta_x + current_shape(pos.lmark1, 0);
		int real_y = delta_y + current_shape(pos.lmark1, 1);
		real_x = std::max(gauss, std::min(real_x, image.cols - 1 - gauss)); // which cols
		real_y = std::max(gauss, std::min(real_y, image.rows - 1 - gauss)); // which rows
		int tmp = (int)(2*image(real_y, real_x) + image(real_y-gauss, real_x) + image(real_y+gauss, real_x) + image(real_y, real_x-gauss) +image(real_y, real_x+gauss)) / 6 ; //real_y at first
        //int tmp = image(real_y, real_x);
        
		delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
		delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
		delta_x = scale*delta_x*bbox.width / 2.0;
		delta_y = scale*delta_y*bbox.height / 2.0;
		real_x = delta_x + current_shape(pos.lmark2, 0);
		real_y = delta_y + current_shape(pos.lmark2, 1);
		real_x = std::max(gauss, std::min(real_x, image.cols - 1 - gauss)); // which cols
		real_y = std::max(gauss, std::min(real_y, image.rows - 1 - gauss)); // which rows
        int tmp2 = (int)(2*image(real_y, real_x) + image(real_y-gauss, real_x) + image(real_y+gauss, real_x) + image(real_y, real_x-gauss) +image(real_y, real_x+gauss)) / 6 ;
		//int tmp2 = image(real_y, real_x);
        if ( tree_index % 2 == 0 ){
            if ( abs(tmp - tmp2) < node->threshold_){
                node = node->left_child_;// go left
            }
            else{
                node = node->right_child_;// go right
            }
        }
        else{
            if ( (tmp - tmp2) < node->threshold_){
                node = node->left_child_;// go left
            }
            else{
                node = node->right_child_;// go right
            }
        }
	}
    score = node->score_;
	return node->leaf_identity;
}


//int RandomForest::GetNodeOutput(Node* node, const cv::Mat_<float>& image,
//	const BoundingBox& bbox, const cv::Mat_<float>& current_shape, const cv::Mat_<float>& rotation, const float& scale){
//	
//	FeatureLocations& pos = node->feature_locations_;
//	float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
//	float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
//	delta_x = scale*delta_x*bbox.width / 2.0;
//	delta_y = scale*delta_y*bbox.height / 2.0;
//	int real_x = delta_x + current_shape(landmark_index_, 0);
//	int real_y = delta_y + current_shape(landmark_index_, 1);
//	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
//	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
//	int tmp = (int)image(real_y, real_x); //real_y at first
//
//	delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
//	delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
//	delta_x = scale*delta_x*bbox.width / 2.0;
//	delta_y = scale*delta_y*bbox.height / 2.0;
//	real_x = delta_x + current_shape(landmark_index_, 0);
//	real_y = delta_y + current_shape(landmark_index_, 1);
//	real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
//	real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
//	if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
//		return -1; // go left
//	}
//	else{
//		return 1; // go right
//	}
//
//}



bool RandomForest::TrainForestGender(//std::vector<cv::Mat_<float>>& regression_targets,
                               std::vector<cv::Mat_<uchar> >& images,
                               std::vector<int>& augmented_images_index,
                               std::vector<cv::Mat_<float>>& augmented_ground_truth_shapes,
                               std::vector<BoundingBox>& augmented_bboxes,
                               std::vector<cv::Mat_<float> >& augmented_current_shapes,
                               std::vector<int> & augmented_ground_truth_genders,
                               std::vector<float> & current_fi,
                               std::vector<float> & current_weight,
                               std::vector<int> & find_times,
                               std::vector<cv::Mat_<float> >& rotations,
                               std::vector<float>& scales){
    
//    time_t current_time;
//    current_time = time(0);
//    cv::RNG rd(current_time);
    //	 random generate feature locations
    std::cout << "generate feature locations" << std::endl;

    float overlap = 0.4;
    int step = floor(((float)augmented_images_index.size())*overlap / (trees_num_per_forest_ - 1));
    trees_.clear();
    all_leaf_nodes_ = 0;
    int accuDeleteNum = 0;
    for (int i = 0; i < trees_num_per_forest_; i++){
        //为每棵树都重新生成一次local feature，否则每个森林树太多时，特征不够用。如果16棵树，4层，需要31*16个不同特征
        local_position_.clear();
        local_position_.resize(local_features_num_);
        for (int n = 0; n < local_features_num_; n++){
            float x, y;
            do{
                x = rd.uniform(-local_radius_, local_radius_);
                y = rd.uniform(-local_radius_, local_radius_);
                //                            if ( i < 4 && x < -local_radius_/5.0) continue; //add by xujj, 避免脸部外侧的pixel
                //                            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
                //                            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
            } while (x*x + y*y > local_radius_*local_radius_);
            cv::Point2f a(x, y);
            
            do{
                x = rd.uniform(-local_radius_, local_radius_);
                y = rd.uniform(-local_radius_, local_radius_);
                //                            if ( i < 4 && x < -local_radius_/5.0) continue;
                //                            if ( (i>13 && i<17) && x > local_radius_/5.0) continue;
                //                            if ( (i>6 && i<10 ) && y > local_radius_/5.0) continue;
            } while (x*x + y*y > local_radius_*local_radius_);
            cv::Point2f b(x, y);
            
            //TODO，这个地方可以试多个策略：1）自己，2）自己和随机一个，3）随机两个
            //TODO，在前2个stage，用3，第三stage用2，后续stage，用1？如何？
            int landmark1, landmark2;
            if ( stage_ == 0 ){
                landmark1 = (int)rd.uniform(0, landmark_num_);
                landmark2 = (int)rd.uniform(0, landmark_num_);
            }
            else{
                landmark1 = landmark_index_;
                //                landmark2 = (int)rd.uniform(0, landmark_num_);
                landmark2 = landmark_index_;
            }
            //            //test
            //            landmark1 = landmark_index_;
            //            landmark2 = landmark_index_;
            
            //            else if (  n % (stage_ + 3) == 0 ){
            //                landmark1 = landmark_index_;
            //                landmark2 = landmark_index_;
            //            }
            //            else if ( n % (stage_ + 3) == 1 ){
            //                landmark1 = landmark_index_;
            //                landmark2 = (int)rd.uniform(0, landmark_num_);
            //            }
            //            else {
            //                landmark1 = landmark_index_;
            //                landmark2 = adjointPoint(landmark1);
            //            }
            local_position_[n] = FeatureLocations(landmark1, landmark2, a, b);
        }
        //std::cout << "get pixel differences" << std::endl;
        cv::Mat_<int> pixel_differences(local_features_num_, augmented_images_index.size()); // matrix: features*images
        
        //#pragma omp parallel for
        for (int n = 0; n < augmented_images_index.size(); n++){
            
            cv::Mat_<float> rotation = rotations[n];
            float scale = scales[n];
            int gauss = (int) augmented_bboxes[n].width / 200;
            gauss = std::min(2, gauss);
            for (int j = 0; j < local_features_num_; j++){
                FeatureLocations pos = local_position_[j];
                float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = scale*delta_x*augmented_bboxes[n].width / 2.0;
                delta_y = scale*delta_y*augmented_bboxes[n].height / 2.0;
                int real_x = delta_x + augmented_current_shapes[n](pos.lmark1, 0);
                int real_y = delta_y + augmented_current_shapes[n](pos.lmark1, 1);
                real_x = std::max(gauss, std::min(real_x, images[augmented_images_index[n]].cols - 1 - gauss)); // which cols
                real_y = std::max(gauss, std::min(real_y, images[augmented_images_index[n]].rows - 1 - gauss)); // which rows
                int tmp = (int)(2*images[augmented_images_index[n]](real_y, real_x) + images[augmented_images_index[n]](real_y-gauss, real_x) + images[augmented_images_index[n]](real_y+gauss, real_x) + images[augmented_images_index[n]](real_y, real_x-gauss) +images[augmented_images_index[n]](real_y, real_x+gauss)) / 6 ; //real_y at first
                //int tmp = images[augmented_images_index[n]](real_y, real_x);
                
                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = scale*delta_x*augmented_bboxes[n].width / 2.0;
                delta_y = scale*delta_y*augmented_bboxes[n].height / 2.0;
                real_x = delta_x + augmented_current_shapes[n](pos.lmark2, 0);
                real_y = delta_y + augmented_current_shapes[n](pos.lmark2, 1);
                real_x = std::max(gauss, std::min(real_x, images[augmented_images_index[n]].cols - 1 - gauss)); // which cols
                real_y = std::max(gauss, std::min(real_y, images[augmented_images_index[n]].rows - 1 - gauss)); // which rows
                int tmp2 = (int)(2*images[augmented_images_index[n]](real_y, real_x) + images[augmented_images_index[n]](real_y-gauss, real_x) + images[augmented_images_index[n]](real_y+gauss, real_x) + images[augmented_images_index[n]](real_y, real_x-gauss) +images[augmented_images_index[n]](real_y, real_x+gauss)) / 6 ;
                //int tmp2 = images[augmented_images_index[n]](real_y, real_x);
                if ( i % 2 == 0 ){
                    pixel_differences(j, n) = abs(tmp - tmp2);
                }
                else{
                    pixel_differences(j, n) = tmp - tmp2;
                }
            }
        }
        
        //计算每个训练实例的weight
        for(int k=0;k<current_weight.size();++k)
        {
            current_weight[k] = exp(0.0-augmented_ground_truth_genders[k]*current_fi[k]);
        }
        
        int start_index = 0;//i*step;
        int end_index = augmented_images_index.size();//augmented_images_index.size() - (trees_num_per_forest_ - i - 1)*step;
        //cv::Mat_<int> data = pixel_differences(cv::Range(0, local_features_num_), cv::Range(start_index, end_index));
        //cv::Mat_<int> sorted_data;
        //cv::sortIdx(data, sorted_data, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
        std::set<int> selected_feature_indexes; //这个是用来表示那个feature已经被用过了
        selected_feature_indexes.clear();
        
//        current_time = time(0);
//        cv::RNG rd(current_time);
        std::vector<int> images_indexes;
        for (int j = start_index; j < end_index; j++){
            if ( find_times[j] < MAXFINDTIMES){
                //让每棵树的样本有点不一样
                int r = rd.uniform(0, trees_num_per_forest_);
                if ( r != 0 ){
                    images_indexes.push_back(j);
                }
            }
        }
        
        Node* root = BuildTree(selected_feature_indexes, pixel_differences, images_indexes, augmented_ground_truth_genders, current_weight, 0);
        trees_.push_back(root);
        
        //计算每个训练实例的fi
        for ( int n=0; n<augmented_images_index.size(); n++){
            if ( find_times[n] >= MAXFINDTIMES ) continue;
            float score = 0;
            //用训练实例去遍历此树得出叶子节点的score
            //            cv::Mat_<float> rotation;
            //            float scale;
            //            getSimilarityTransform(ProjectShape(augmented_current_shapes[n],augmented_bboxes[n]),mean_shape_,rotation,scale);
            GetBinaryFeatureIndex(i, images[augmented_images_index[n]], augmented_bboxes[n], augmented_current_shapes[n], rotations[n] , scales[n], score);
            current_fi[n] += score;
        }
        
        //开始计算这棵树的detection threshold
        std::vector<std::pair<float,int>> fiSort;
        fiSort.clear();
        for(int n=0;n<current_fi.size();++n)
        {
            if ( find_times[n] >= MAXFINDTIMES ) continue;
            fiSort.push_back(std::pair<float,int>(current_fi[n],n));
        }
        // ascent , small fi means false sample
        sort(fiSort.begin(),fiSort.end(),my_cmp);
        // compute recall
        // set threshold
        float max_recall=0;
        int min_error = fiSort.size();
        int nn;
        float cscore=0, pscore=0, nscore=0;
        
        //        int idx_tmp=-1;
        
        //        std::vector<std::pair<float,float>> precise_recall;
        for (int n=0;n<fiSort.size();++n)
        {
            int true_pos=0;int false_neg=0;
            int true_neg=0;int false_pos=0;
            for(int m=0;m<fiSort.size();++m)
            {
                int isFace = augmented_ground_truth_genders[fiSort[m].second];
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
            //TODO：这个地方这样貌似取了最大召回率，应该再核对一下错误率。
            //
            if ( min_error > (false_neg + false_pos) /*true_pos/(true_pos+false_neg+FLT_MIN)>=max_recall*/)
            {
//                max_recall=true_pos/(true_pos+false_neg+FLT_MIN);
                min_error = false_neg + false_pos;
                //                precise_recall.push_back(pair<float,float>(true_pos/(true_pos+false_neg+FLT_MIN),false_pos/(false_pos+true_neg+FLT_MIN)));
                                nn = n;
                cscore = fiSort[n].first;
                if ( n > 0 ) pscore = fiSort[n-1].first;
                if ( n < fiSort.size() - 1 ) nscore = fiSort[n+1].first;
                root->score_ = (pscore + cscore ) / 2; //跟节点的score作为threshold使用

                //idx_tmp=n;
            }
        }
        std::cout << "min_error " << min_error << " nn:" << nn << " pscore:" << pscore << " cscore:" << cscore << " nscore:" << nscore <<  std::endl;
    }
    
    return true;
}


RandomForest::RandomForest(Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<float> >& regression_targets, CascadeRegressor *casRegressor, int true_pos_num){
	stage_ = stage;
    param_ = param;
    local_features_num_ = param.local_features_num_; // 200 + param.local_features_num_ / ( stage_ + 1 );
    if ( stage == 0 && landmark_index < 17 ){
        local_features_num_ = param.local_features_num_;
    }
//    else if ( stage >= 3 ){
//        local_features_num_ = param.local_features_num_ / 2;
//    }
    else{
        local_features_num_ = param.local_features_num_ / 4;
    }

	landmark_index_ = landmark_index;
    landmark_num_ = param.landmarks_num_per_face_;
	tree_depth_ = param.tree_depth_;
	trees_num_per_forest_ = param.trees_num_per_forest_;
	local_radius_ = param.local_radius_by_stage_[stage_];
	mean_shape_ = param.mean_shape_;
    detect_factor_ = param.detect_factor_by_stage_[stage_];
	regression_targets_ = &regression_targets; // get the address pointer, not reference
    true_pos_num_ = true_pos_num;
    casRegressor_ = casRegressor;
    
    time_t current_time;
    current_time = time(0);
    rd = rd(current_time);
}

RandomForest::RandomForest(){
	
}

RandomForest::~RandomForest(){
    //    std::cout << "~RandomForest()" << std::endl;
    for ( int i=0; i<trees_.size(); i++ ){
        Node *node = trees_[i];
        delete node;
    }
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
//            << p->is_leaf_a << " "
			<< p->leaf_identity << " "
			<< p->depth_ << " "
            << p->score_ << " "
            << p->feature_locations_.lmark1 << " "
            << p->feature_locations_.lmark2 << " "
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
//            >> p->is_leaf_a
			>> p->leaf_identity
			>> p->depth_
            >> p->score_
            >> p->feature_locations_.lmark1
            >> p->feature_locations_.lmark2
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
