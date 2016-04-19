#include "regressor.h"
#include <time.h>
#include <assert.h>
//SYSTEM MACORS LISTS: http://sourceforge.net/p/predef/wiki/OperatingSystems/
//#ifdef _WIN32 // can be used under 32 and 64 bits both
//#include <direct.h>
//#elif __linux__
#include <sys/types.h>
#include <sys/stat.h>

//#endif
CascadeRegressor::CascadeRegressor(){
    lastRes = cv::Mat_<float>(68,2,0.0);
}

void CascadeRegressor::Train(std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<float> >& ground_truth_shapes,
    std::vector<int> ground_truth_faces,
	std::vector<BoundingBox>& bboxes,
	Parameters& params,
    int pos_num ){

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	std::cout << "Start training..." << std::endl;
	images_ = images;
	params_ = params;
	bboxes_ = bboxes;
	ground_truth_shapes_ = ground_truth_shapes;

	std::vector<int> augmented_images_index; // just index in images_
	std::vector<BoundingBox> augmented_bboxes;
	std::vector<cv::Mat_<float> > augmented_ground_truth_shapes;
    std::vector<int> augmented_ground_truth_faces;
	std::vector<cv::Mat_<float> > augmented_current_shapes; //
    std::vector<float> current_fi;
    std::vector<float> current_weight;
    std::vector<int> find_times;
    
	time_t current_time;
	current_time = time(0);
	//cv::RNG *random_generator = new cv::RNG();
	std::cout << "augment data sets" << std::endl;
	cv::RNG random_generator(current_time);
    for (int i = 0; i < pos_num; i++){
        augmented_images_index.push_back(i);
        augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
        augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
        augmented_bboxes.push_back(bboxes_[i]);
        augmented_current_shapes.push_back(ReProjection(params_.mean_shape_, bboxes_[i]));
        current_fi.push_back(0);
        current_weight.push_back(1);
        find_times.push_back(0);
        
		for (int j = 0; j < params_.initial_guess_; j++)
		{
			int index = 0;
			do {
				index = random_generator.uniform(0, pos_num);
			}while(index == i);

            if ( ground_truth_faces[i] == -1 ){
                std::cout << "Error..." << std::endl;
            }
            else{
                augmented_images_index.push_back(i);
                augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
                augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
                augmented_bboxes.push_back(bboxes_[i]);
                cv::Mat_<float> temp = ground_truth_shapes_[index];
                temp = ProjectShape(temp, bboxes_[index]);
                temp = ReProjection(temp, bboxes_[i]);
                augmented_current_shapes.push_back(temp);
                current_fi.push_back(0);
                current_weight.push_back(1);
                find_times.push_back(0);
            }
		}

	}

    for ( int i = pos_num; i < images_.size(); i++ ){
        augmented_images_index.push_back(i);
        augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
        augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
        augmented_bboxes.push_back(bboxes_[i]);
        augmented_current_shapes.push_back(ReProjection(params_.mean_shape_, bboxes_[i]));
        current_fi.push_back(0);
        current_weight.push_back(1);
        find_times.push_back(0);

        for (int j = 0; j < params_.initial_guess_; j++)
        {
            int index = 0;
            do {
                index = random_generator.uniform(pos_num, images_.size());
            }while(index == i);

            if ( ground_truth_faces[i] == -1 ){
                //TODO:对于不是face的image，是不是可以循环多生成一些负例？需要生成一些随机的box，待做
                for ( int k=0; k<1; k++){
                    augmented_images_index.push_back(i);
                    augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
                    augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
                    augmented_bboxes.push_back(bboxes_[i]);
                    cv::Mat_<float> temp = ground_truth_shapes_[index];
                    temp = ProjectShape(temp, bboxes_[index]);
                    temp = ReProjection(temp, bboxes_[i]);
                    augmented_current_shapes.push_back(temp);
                    current_fi.push_back(0);
                    current_weight.push_back(1);
                    find_times.push_back(0);
                }
            }
            else{
                std::cout << "Error..." << std::endl;
            }
        }
    }

    pos_num += pos_num * params_.initial_guess_;
    
    std::cout << "augmented size: " << augmented_current_shapes.size() << std::endl;

	std::vector<cv::Mat_<float> > shape_increaments;

    regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
        gettimeofday(&t1, NULL);
		std::cout << "training stage: " << i << " of " << params_.regressor_stages_  << std::endl;
		shape_increaments = regressors_[i].Train(images_,
											augmented_images_index,
											augmented_ground_truth_shapes,
                                            augmented_ground_truth_faces,
											augmented_bboxes,
											augmented_current_shapes,
                                            current_fi,
                                            current_weight,
                                            find_times,
											params_,
											i,
                                            pos_num,
                                            this);
		std::cout << "update current shapes" << std::endl;
		float error = 0.0;
        int count = 0;
		for (int j = 0; j < shape_increaments.size(); j++){
			augmented_current_shapes[j] = shape_increaments[j] + ProjectShape(augmented_current_shapes[j], augmented_bboxes[j]);
			augmented_current_shapes[j] = ReProjection(augmented_current_shapes[j], augmented_bboxes[j]);
            if ( augmented_ground_truth_faces[j] == 1){ //pos example才计算误差
			    error += CalculateError(augmented_ground_truth_shapes[j], augmented_current_shapes[j]);
                count++;
            }
		}

        gettimeofday(&t1, NULL);
        std::cout << "regression error: " <<  error << ": " << error/count << " time:" << t1.tv_sec << std::endl;
	}
}

std::vector<cv::Mat_<float> > Regressor::Train(std::vector<cv::Mat_<uchar> >& images,
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
    CascadeRegressor *casRegressor){

	stage_ = stage;
	params_ = params;

	std::vector<cv::Mat_<float> > regression_targets;
	std::vector<cv::Mat_<float> > rotations_;
	std::vector<float> scales_;
	regression_targets.resize(augmented_current_shapes.size());
	rotations_.resize(augmented_current_shapes.size());
	scales_.resize(augmented_current_shapes.size());

	// calculate the regression targets
	std::cout << "calculate regression targets" << std::endl;
    #pragma omp parallel for
	for (int i = 0; i < augmented_current_shapes.size(); i++){
		regression_targets[i] = ProjectShape(augmented_ground_truth_shapes[i], augmented_bboxes[i])
			- ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]);
		cv::Mat_<float> rotation;
		float scale;
		getSimilarityTransform(params_.mean_shape_, ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), rotation, scale);
		cv::transpose(rotation, rotation);
		regression_targets[i] = scale * regression_targets[i] * rotation;
		getSimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), params_.mean_shape_, rotation, scale);
		rotations_[i] = rotation;
		scales_[i] = scale;
	}

	std::cout << "train forest of stage:" << stage_ << std::endl;
	rd_forests_.resize(params_.landmarks_num_per_face_);
//    #pragma omp parallel for
    
    std::vector<int> landmarks;
    for (int g=0; g<params_.group_num_; g++){
        for ( int j=0; j<params_.groups_[g].size(); j++ ){
            if ( params_.groups_[g][j] >= 0 ){
                landmarks.push_back(params_.groups_[g][j]);
            }
        }
    }
	for (int ii = 0; ii < landmarks.size(); ++ii){
        int i = landmarks[ii];
        std::cout << "landmark: " << i << std::endl;
		rd_forests_[i] = RandomForest(params_, i, stage_, regression_targets, casRegressor);
        rd_forests_[i].TrainForest(
			images,augmented_images_index, augmented_ground_truth_shapes, augmented_bboxes, augmented_current_shapes,
            augmented_ground_truth_faces, current_fi, current_weight, find_times,
			rotations_, scales_);
	}
	std::cout << "Get Global Binary Features" << std::endl;
    linear_model_x_.resize(params_.landmarks_num_per_face_);
    linear_model_y_.resize(params_.landmarks_num_per_face_);
    
    struct feature_node ***global_binary_features;
    global_binary_features = new struct feature_node** [params_.group_num_];
    
    for (int g=0; g < params_.group_num_; g++){ //分组来处理部分全局关系
        global_binary_features[g] = new struct feature_node* [augmented_current_shapes.size()];
        for(int i = 0; i < augmented_current_shapes.size(); ++i){
            global_binary_features[g][i] = new feature_node[params_.trees_num_per_forest_*params_.groups_[g].size()+1];
        }
        int num_feature = 0;
        for (int i=0; i < params_.groups_[g].size(); ++i){
            num_feature += rd_forests_[abs(params_.groups_[g][i])].all_leaf_nodes_;
        }
        #pragma omp parallel for
        for (int i = 0; i < augmented_current_shapes.size(); ++i){
            int index = 1;
            int ind = 0;
            const cv::Mat_<float>& rotation = rotations_[i];
            const float scale = scales_[i];
            const cv::Mat_<uchar>& image = images[augmented_images_index[i]];
            const BoundingBox& bbox = augmented_bboxes[i];
            const cv::Mat_<float>& current_shape = augmented_current_shapes[i];
            for (int jj = 0; jj < params_.groups_[g].size(); ++jj){
                int j = abs(params_.groups_[g][jj]);
                for (int k = 0; k < params_.trees_num_per_forest_; ++k){
                    Node* node = rd_forests_[j].trees_[k];
                    while (!node->is_leaf_){
                        FeatureLocations& pos = node->feature_locations_;
                        float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                        float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                        delta_x = scale*delta_x*bbox.width / 2.0;
                        delta_y = scale*delta_y*bbox.height / 2.0;
//                        int real_x = delta_x + current_shape(j, 0);
//                        int real_y = delta_y + current_shape(j, 1);
                        int real_x = delta_x + current_shape(pos.lmark1, 0);
                        int real_y = delta_y + current_shape(pos.lmark1, 1);
                        real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                        real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                        int tmp = (int)image(real_y, real_x); //real_y at first

                        delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                        delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                        delta_x = scale*delta_x*bbox.width / 2.0;
                        delta_y = scale*delta_y*bbox.height / 2.0;
                        real_x = delta_x + current_shape(pos.lmark2, 0);
                        real_y = delta_y + current_shape(pos.lmark2, 1);
                        real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                        real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                        if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
                            node = node->left_child_;// go left
                        }
                        else{
                            node = node->right_child_;// go right
                        }
                    }
                    global_binary_features[g][i][ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k, images[augmented_images_index[i]], augmented_bboxes[i], augmented_current_shapes[i], rotations_[i], scales_[i]);
                    global_binary_features[g][i][ind].value = 1.0;
                    ind++;
                    //std::cout << global_binary_features[i][ind].index << " ";
                }
                index += rd_forests_[j].all_leaf_nodes_;
            }
            if (i%500 == 0 && i > 0){
//                std::cout << "extracted " << i << " images" << std::endl;
            }
            global_binary_features[g][i][params_.trees_num_per_forest_*params_.groups_[g].size()].index = -1;
            global_binary_features[g][i][params_.trees_num_per_forest_*params_.groups_[g].size()].value = -1.0;
        }
        std::cout << "\n";

        //TODO:这里回归的时候，不要回归negtive face实例！
        //重新生成回归用的binary feature和targets
        struct problem* prob = new struct problem;
        prob->l = pos_num; //augmented_current_shapes.size();
        prob->n = num_feature;
        prob->x = global_binary_features[g];
        prob->bias = -1;

        struct parameter* regression_params = new struct parameter;
        regression_params-> solver_type = L2R_L2LOSS_SVR_DUAL;
        regression_params->C = 1.0/pos_num; //augmented_current_shapes.size();
        regression_params->p = 0;

        std::cout << "Global Regression of stage " << stage_ << std::endl;

        float** targets = new float*[params_.groups_[g].size()];
        for (int i = 0; i < params_.groups_[g].size(); ++i){
            targets[i] = new float[pos_num];
        }
        #pragma omp parallel for
        for (int ii = 0; ii < params_.groups_[g].size(); ++ii){
            int i = params_.groups_[g][ii];
            if ( i < 0 ) continue;
            std::cout << "regress landmark " << i << std::endl;
            for(int j = 0; j< pos_num;j++){
                targets[ii][j] = regression_targets[j](i, 0);
            }
            prob->y = targets[ii];
            check_parameter(prob, regression_params);
            struct model* regression_model = train(prob, regression_params);
            linear_model_x_[i] = regression_model;
            for(int j = 0; j < pos_num; j++){
                targets[ii][j] = regression_targets[j](i, 1);
            }
            prob->y = targets[ii];
            check_parameter(prob, regression_params);
            regression_model = train(prob, regression_params);
            linear_model_y_[i] = regression_model;

        }
        for (int i = 0; i < params_.groups_[g].size(); ++i){
            delete[] targets[i];// = new float[augmented_current_shapes.size()];
        }
        delete[] targets;
    }

	std::cout << "predict regression targets" << std::endl;

    std::vector<cv::Mat_<float> > predict_regression_targets;
    predict_regression_targets.resize(augmented_current_shapes.size());
    
    #pragma omp parallel for
    //TODO：如果是negtive sample，直接返回0或? 者允许negtive计算，但在算误差时去掉
    for (int i = 0; i < augmented_current_shapes.size(); i++){
        cv::Mat_<float> a(params_.landmarks_num_per_face_, 2, 0.0);
        for (int g = 0; g < params_.group_num_; g++){
            for (int jj = 0; jj < params_.groups_[g].size(); jj++){
                int j = params_.groups_[g][jj];
                if ( j < 0 ) continue;
                a(j, 0) = predict(linear_model_x_[j], global_binary_features[g][i]);
                a(j, 1) = predict(linear_model_y_[j], global_binary_features[g][i]);
            }
        }
        cv::Mat_<float> rot;
        cv::transpose(rotations_[i], rot);
        predict_regression_targets[i] = scales_[i] * a * rot;
        if (i%500 == 0 && i > 0){
//             std::cout << "predict " << i << " images" << std::endl;
        }
    }
    std::cout << "\n";

    for (int g = 0; g < params_.group_num_; g++){
        for (int i = 0; i< augmented_current_shapes.size(); i++){
            delete[] global_binary_features[g][i];
        }
        delete[] global_binary_features[g];
    }
    delete[] global_binary_features;

	return predict_regression_targets;
}


//cv::Mat_<float> CascadeRegressor::Predict(cv::Mat_<uchar>& image,
//	cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& ground_truth_shape){
//
//	cv::Mat_<uchar> tmp;
//	image.copyTo(tmp);
//
//	for (int j = 0; j < current_shape.rows; j++){
//		cv::circle(tmp, cv::Point2f(current_shape(j, 0), current_shape(j, 1)), 2, (255));
//	}
//	cv::imshow("show image", tmp);
//	cv::waitKey(0);
//
//	for (int i = 0; i < params_.regressor_stages_; i++){
//
//		cv::Mat_<float> rotation;
//		float scale;
//		if(i==0){
//			getSimilarityTransform(ProjectShape(ground_truth_shape, bbox), params_.mean_shape_, rotation, scale);
//		}else{
//			getSimilarityTransform(ProjectShape(current_shape, bbox), params_.mean_shape_, rotation, scale);
//		}
//
//		cv::Mat_<float> shape_increaments = regressors_[i].Predict(image, current_shape, bbox, rotation, scale);
//		current_shape = shape_increaments + ProjectShape(current_shape, bbox);
//		current_shape = ReProjection(current_shape, bbox);
//		image.copyTo(tmp);
//		for (int j = 0; j < current_shape.rows; j++){
//			cv::circle(tmp, cv::Point2f(current_shape(j, 0), current_shape(j, 1)), 2, (255));
//		}
//		cv::imshow("show image", tmp);
//		cv::waitKey(0);
//	}
//	cv::Mat_<float> res = current_shape;
//	return res;
//}


cv::Mat_<float> CascadeRegressor::Predict(cv::Mat_<uchar>& image,
	cv::Mat_<float>& current_shape, BoundingBox& bbox, bool &is_face){
//    cv::Mat_<float> rshape = ProjectShape( current_shape.clone(), bbox);

    float score = 0;
	for (int i = 0; i < params_.predict_regressor_stages_; i++){
        cv::Mat_<float> rotation;
		float scale;
		getSimilarityTransform(ProjectShape(current_shape, bbox), params_.mean_shape_, rotation, scale);
		cv::Mat_<float> shape_increaments = regressors_[i].Predict(image, current_shape, bbox, rotation, scale, score, is_face);
        if ( !is_face ){
            std::cout << "检测不是face!!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
            return current_shape;
        }
		current_shape = shape_increaments + ProjectShape(current_shape, bbox);
		current_shape = ReProjection(current_shape, bbox);
        
//        for (int ii = 0; ii < current_shape.rows; ii++){
//            cv::circle(image, cv::Point2f(current_shape(ii, 0), current_shape(ii, 1)), 2, (255));
//            if ( ii > 0 && ii != 17 && ii != 22 && ii != 27 && ii!= 36 && ii != 42 && ii!= 48 )
//                cv::line(image, cv::Point2f(current_shape(ii-1, 0), current_shape(ii-1, 1)), cv::Point2f(current_shape(ii, 0), current_shape(ii, 1)), cvScalar(30*i,255,0));
//        }
//        cv::imshow("show image", image);
//        cv::waitKey(0);
        
//        cv::Mat_<float> temp = current_shape.rowRange(36, 41)-current_shape.rowRange(42, 47);
//        float x =mean(temp.col(0))[0];
//        float y = mean(temp.col(1))[1];
//        float interocular_distance = sqrt(x*x+y*y);
//        float delta = norm(shape_increaments)/(current_shape.rows*interocular_distance * params_.local_radius_by_stage_[i]);
//        stage_delta_.push_back(delta);
	}
    
    //最小二乘法算出斜率，方差等
//    float xiyi, yy, xi2;
//    for ( int i=0; i<params_.predict_regressor_stages_; i++ ){
//        xiyi += i*stage_delta_[i];
//        yy += stage_delta_[i];
//    }
//    xi2 = 30.0; yy = yy / params_.predict_regressor_stages_;
//    float b = ( xiyi - params_.predict_regressor_stages_ * 2 * yy ) / (30 - params_.predict_regressor_stages_ * 4);
//    float a = yy - b * 2;
//    float c;
//    for ( int i=0; i<params_.predict_regressor_stages_; i++){
//        c += ( stage_delta_[i] - a - b*i ) * ( stage_delta_[i] - a - b*i );
//    }
//    std::cout << a << " " << b << " " << c << std::endl;

    cv::Mat_<float> res = current_shape;
    
    //add by xujj, 做一个小幅抖动滤波
    if ( antiJitter == 1 && params_.landmarks_num_per_face_ == 68 ){
        for ( int j=0; j<68; j++){
            float alphax = fabs(res(j,0)-lastRes(j,0)) / 10.0;
            float alphay = fabs(res(j,1)-lastRes(j,1)) / 10.0;
            if (alphax > 1.0 ) alphax = 1.0;
            if (alphay > 1.0 ) alphay = 1.0;
            
            lastRes(j,0) = ((1.0-alphax)*lastRes(j,0) + alphax*res(j,0));
            lastRes(j,1) = ((1.0-alphay)*lastRes(j,1) + alphay*res(j,1));
        }
        return lastRes;
    }

	return res;
}

Regressor::Regressor(){
    tmp_binary_features = NULL;
}

Regressor::Regressor(const Regressor &a){
}

Regressor::~Regressor(){
    if ( tmp_binary_features != NULL ){
        delete[] tmp_binary_features;
    }
}
/*
struct feature_node* Regressor::GetGlobalBinaryFeaturesThread(cv::Mat_<uchar>& image,
    cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale){
    struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
    tmp_binary_features = binary_features;
    tmp_image = image;
    tmp_current_shape = current_shape;
    tmp_bbox = bbox;
    tmp_rotation = rotation;
    tmp_scale = scale;
    // cur_landmark.store(0);


    int num_threads = 2;
    std::thread t1, t2;
    std::vector<std::thread> pool;
    //struct timeval tt1, tt2;
    //gettimeofday(&tt1, NULL);
    for(int i = 0; i < num_threads; i++){
        //t1 = std::thread(&Regressor::GetFeaThread, this);
        pool.push_back(std::thread(&Regressor::GetFeaThread, this));
    }
    //gettimeofday(&tt2, NULL);
    //std::cout << "threads: " << tt2.tv_sec - tt1.tv_sec + (tt2.tv_usec - tt1.tv_usec)/1000000.0 << std::endl;

    for(int i = 0; i < num_threads; i++){
        pool[i].join();
    }

    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;

    return binary_features;
}
*/
/*
void Regressor::GetFeaThread(){
    int cur = -1;
    while(1){
        cur = cur_landmark.fetch_add(1);
        if(cur >= params_.landmarks_num_per_face_){
            return;
        }
        //std::cout << stage_ << ": " << cur << std::endl;
        int ind = cur*params_.trees_num_per_forest_;
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            Node* node = rd_forests_[cur].trees_[k];
            while (!node->is_leaf_){
                FeatureLocations& pos = node->feature_locations_;
                float delta_x = tmp_rotation(0, 0)*pos.start.x + tmp_rotation(0, 1)*pos.start.y;
                float delta_y = tmp_rotation(1, 0)*pos.start.x + tmp_rotation(1, 1)*pos.start.y;
                delta_x = tmp_scale*delta_x*tmp_bbox.width / 2.0;
                delta_y = tmp_scale*delta_y*tmp_bbox.height / 2.0;
                int real_x = delta_x + tmp_current_shape(cur, 0);
                int real_y = delta_y + tmp_current_shape(cur, 1);
                real_x = std::max(0, std::min(real_x, tmp_image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, tmp_image.rows - 1)); // which rows
                int tmp = (int)tmp_image(real_y, real_x); //real_y at first

                delta_x = tmp_rotation(0, 0)*pos.end.x + tmp_rotation(0, 1)*pos.end.y;
                delta_y = tmp_rotation(1, 0)*pos.end.x + tmp_rotation(1, 1)*pos.end.y;
                delta_x = tmp_scale*delta_x*tmp_bbox.width / 2.0;
                delta_y = tmp_scale*delta_y*tmp_bbox.height / 2.0;
                real_x = delta_x + tmp_current_shape(cur, 0);
                real_y = delta_y + tmp_current_shape(cur, 1);
                real_x = std::max(0, std::min(real_x, tmp_image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, tmp_image.rows - 1)); // which rows
                if ((tmp - (int)tmp_image(real_y, real_x)) < node->threshold_){
                    node = node->left_child_;// go left
                }
                else{
                    node = node->right_child_;// go right
                }
            }

            //int ind = j*params_.trees_num_per_forest_ + k;
            tmp_binary_features[ind].index = leaf_index_count[cur] + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
            tmp_binary_features[ind].value = 1.0;
            ind++;
            //std::cout << binary_features[ind].index << " ";
        }
    }
}
*/

//del by xujj
//struct feature_node* Regressor::GetGlobalBinaryFeaturesMP(cv::Mat_<uchar>& image,
//    cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale){
//    int index = 1;
//
//    struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
//    //int ind = 0;
//#pragma omp parallel for
//    for (int j = 0; j < params_.landmarks_num_per_face_; ++j)
//    {
//        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
//        {
//            Node* node = rd_forests_[j].trees_[k];
//            while (!node->is_leaf_){
//                FeatureLocations& pos = node->feature_locations_;
//                float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
//                float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
//                delta_x = scale*delta_x*bbox.width / 2.0;
//                delta_y = scale*delta_y*bbox.height / 2.0;
//                int real_x = delta_x + current_shape(j, 0);
//                int real_y = delta_y + current_shape(j, 1);
//                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
//                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
//                int tmp = (int)image(real_y, real_x); //real_y at first
//
//                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
//                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
//                delta_x = scale*delta_x*bbox.width / 2.0;
//                delta_y = scale*delta_y*bbox.height / 2.0;
//                real_x = delta_x + current_shape(j, 0);
//                real_y = delta_y + current_shape(j, 1);
//                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
//                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
//                if ((tmp - (int)image(real_y, real_x)) < node->threshold_){
//                    node = node->left_child_;// go left
//                }
//                else{
//                    node = node->right_child_;// go right
//                }
//            }
//
//            //int ind = j*params_.trees_num_per_forest_ + k;
//            int ind = feature_node_index[j] + k;
//            binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
//            //binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
//            binary_features[ind].value = 1.0;
//            //ind++;
//            //std::cout << binary_features[ind].index << " ";
//        }
//
//        //index += rd_forests_[j].all_leaf_nodes_;
//    }
//    //std::cout << "\n";
//    //std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
//    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
//    binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
//    return binary_features;
//}

struct feature_node* Regressor::GetGlobalBinaryFeatures(cv::Mat_<uchar>& image,
    cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, int groupNum, float &score, bool &is_face){
    int index = 1;
    if ( tmp_binary_features == NULL ){
//        struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.groups_[groupNum].size()+1]; //这条性能可能可以优化
        tmp_binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_ +1];
    }
    int ind = 0;
    float ss = scale * bbox.width / 2.0; //add by xujj
    for (int jj = 0; jj < params_.groups_[groupNum].size(); ++jj)
    {
        int j = abs(params_.groups_[groupNum][jj]);
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            Node* node = rd_forests_[j].trees_[k];
            while (!node->is_leaf_){
                FeatureLocations& pos = node->feature_locations_;
                float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = ss * delta_x; //scale*delta_x*bbox.width / 2.0;
                delta_y = ss * delta_y; //scale*delta_y*bbox.height / 2.0;
                int real_x = delta_x + current_shape(pos.lmark1, 0);
                int real_y = delta_y + current_shape(pos.lmark1, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                int tmp = (int)image(real_y, real_x); //real_y at first

                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = ss * delta_x; //scale*delta_x*bbox.width / 2.0;
                delta_y = ss * delta_y; //scale*delta_y*bbox.height / 2.0;
                real_x = delta_x + current_shape(pos.lmark2, 0);
                real_y = delta_y + current_shape(pos.lmark2, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                if ( (tmp - (int)image(real_y, real_x)) < node->threshold_){
                    node = node->left_child_;// go left
                }
                else{
                    node = node->right_child_;// go right
                }
            }
            if ( params_.groups_[groupNum][jj] >= 0 ){ //如果是负值节点，只是为了训练相关性，所以不做face score累加和判断。
                score += node->score_;
                if ( score < rd_forests_[j].trees_[k]->score_ ){
                    is_face = false;
                    std::cout <<"stage:"<<stage_ << "j=" << j << " k=" <<  k << " score:" << score << " threshold:" << rd_forests_[j].trees_[k]->score_<< std::endl;
                    return tmp_binary_features;
                }
            }
            //int ind = j*params_.trees_num_per_forest_ + k;
            //int ind = feature_node_index[j] + k;
            //binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
            tmp_binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
            tmp_binary_features[ind].value = 1.0;
            ind++;
            //std::cout << binary_features[ind].index << " ";
        }

        index += rd_forests_[j].all_leaf_nodes_;
    }
    //std::cout << "\n";
    //std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
    tmp_binary_features[params_.trees_num_per_forest_*params_.groups_[groupNum].size()].index = -1;
    tmp_binary_features[params_.trees_num_per_forest_*params_.groups_[groupNum].size()].value = -1.0;
    return tmp_binary_features;
}

cv::Mat_<float> Regressor::Predict(cv::Mat_<uchar>& image,
	cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float &score, bool &is_face){

	cv::Mat_<float> predict_result(current_shape.rows, current_shape.cols, 0.0);
//    cv::Mat predict_result(current_shape.rows, current_shape.cols, CV_32F, 0.0);
    
    //struct timeval t1, t2;

    for (int g=0; g<params_.group_num_; g++ ){
    //    gettimeofday(&t1, NULL);
        //feature_node* binary_features = GetGlobalBinaryFeaturesThread(image, current_shape, bbox, rotation, scale);
        feature_node* binary_features = GetGlobalBinaryFeatures(image, current_shape, bbox, rotation, scale, g, score, is_face);
        if ( !is_face ){
            return predict_result;
        }
    //    gettimeofday(&t2, NULL);
    //    std::cout << "getBinaryFeatures " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << std::endl;
    //    feature_node* tmp_binary_features = GetGlobalBinaryFeaturesMP(image, current_shape, bbox, rotation, scale);
    //    for (int i = 0; i < params_.trees_num_per_forest_*params_.landmarks_num_per_face_; i++){
    //        std::cout << binary_features[i].index << " ";
    //    }
    //    std::cout << "ha\n";
    //    for (int i = 0; i < params_.trees_num_per_forest_*params_.landmarks_num_per_face_; i++){
    //        std::cout << tmp_binary_features[i].index << " ";
    //    }
    //    std::cout << "ha2\n";
    //    gettimeofday(&t1, NULL);

        for (int ii = 0; ii < params_.groups_[g].size(); ii++){
            int i = params_.groups_[g][ii];
            if ( i < 0 || !params_.predict_group_.count(i) ) continue;
    //		predict_result(i, 0) = predict(linear_model_x_[i], binary_features);
    //        predict_result(i, 1) = predict(linear_model_y_[i], binary_features);
            
            int idx;
            const feature_node *lx=binary_features;
            float *wx =linear_model_x_[i]->w;
            float *wy = linear_model_y_[i]->w;
    //        float resultx = 0.0, resulty = 0.0;
            for(; (idx=lx->index)!=-1 && idx < linear_model_x_[i]->nr_feature; lx++){
                idx--;
                predict_result(i,0) += wx[idx]; 
                predict_result(i,1) += wy[idx];
            }
    //        predict_result(i,0) = resultx;
    //        predict_result(i,1) = resulty;

        }
     //   gettimeofday(&t2, NULL);
     //   std::cout << "linear " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << std::endl;
     //   delete[] binary_features;
    }

	cv::Mat_<float> rot;
	cv::transpose(rotation, rot);
    
    //delete[] tmp_binary_features;
	return scale*predict_result*rot;
}

void CascadeRegressor::LoadCascadeRegressor(std::string ModelName){
	std::ifstream fin;
    fin.open((ModelName + "_params.txt").c_str(), std::fstream::in);
	params_ = Parameters();
	fin >> params_.local_features_num_
		>> params_.landmarks_num_per_face_
		>> params_.regressor_stages_
		>> params_.tree_depth_
		>> params_.trees_num_per_forest_
		>> params_.initial_guess_
        >> params_.group_num_;

	std::vector<float> local_radius_by_stage;
	local_radius_by_stage.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
		fin >> local_radius_by_stage[i];
	}
	params_.local_radius_by_stage_ = local_radius_by_stage;

	cv::Mat_<float> mean_shape(params_.landmarks_num_per_face_, 2, 0.0);
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fin >> mean_shape(i, 0) >> mean_shape(i, 1);
	}
	params_.mean_shape_ = mean_shape;
    
    std::vector<std::vector<int>> groups;
    groups.resize(params_.group_num_);
    for (int i=0; i<params_.group_num_; i++ ){
        std::vector<int> group;
        int groupSize;
        fin >> groupSize;
        group.resize(groupSize);
        for ( int j=0; j<groupSize; j++){
            fin >> group[j];
        }
        groups[i] = group;
    }
    params_.groups_ = groups;
    fin.close();
    
    params_.predict_regressor_stages_ = params_.regressor_stages_;
    params_.predict_group_.clear();
    for ( int i=0; i<params_.landmarks_num_per_face_; i++ ){
        params_.predict_group_.insert(i);
    }
    
	regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
        regressors_[i].params_ = params_;
		regressors_[i].LoadRegressor(ModelName, i);
        regressors_[i].ConstructLeafCount();
	}
}


void CascadeRegressor::SaveCascadeRegressor(std::string ModelName){
	std::ofstream fout;
    fout.open((ModelName + "_params.txt").c_str(), std::fstream::out);
	fout << params_.local_features_num_ << " "
		<< params_.landmarks_num_per_face_ << " "
		<< params_.regressor_stages_ << " "
		<< params_.tree_depth_ << " "
		<< params_.trees_num_per_forest_ << " "
		<< params_.initial_guess_ << " "
        << params_.group_num_ << std::endl;
    
	for (int i = 0; i < params_.regressor_stages_; i++){
		fout << params_.local_radius_by_stage_[i] << std::endl;
	}
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fout << params_.mean_shape_(i, 0) << " " << params_.mean_shape_(i, 1) << std::endl;
	}
    for (int i = 0; i<params_.group_num_; i++){
        fout << params_.groups_[i].size() << " " << std::endl;
        for (int j = 0; j<params_.groups_[i].size(); j++){
            fout << params_.groups_[i][j] << " ";
        }
        fout << std::endl;
    }

	fout.close();

    for (int i = 0; i < params_.regressor_stages_; i++){
		//regressors_[i].SaveRegressor(fout);
        regressors_[i].SaveRegressor(ModelName, i);
		//regressors_[i].params_ = params_;
	}

}


void Regressor::LoadRegressor(std::string ModelName, int stage){
	char buffer[50];
    sprintf(buffer, "%s_%d_regressor.txt", ModelName.c_str(), stage);
	std::ifstream fin;
	fin.open(buffer, std::fstream::in);
	int rd_size, linear_size;
	fin >> stage_ >> rd_size >> linear_size;
	rd_forests_.resize(rd_size);
	for (int i = 0; i < rd_size; i++){
		rd_forests_[i].LoadRandomForest(fin);
	}
	linear_model_x_.clear();
	linear_model_y_.clear();
	for (int i = 0; i < linear_size; i++){
        sprintf(buffer, "%s_%d/%d_linear_x.txt", ModelName.c_str(), stage_, i);
        fin.close();
        fin.open(buffer, std::fstream::in);
		linear_model_x_.push_back(load_model_bin(fin));
        fin.close();
        sprintf(buffer, "%s_%d/%d_linear_y.txt", ModelName.c_str(), stage_, i);
        fin.open(buffer, std::fstream::in);
		linear_model_y_.push_back(load_model_bin(fin));
        fin.close();
	}
}

void Regressor::ConstructLeafCount(){
    int index = 1;
    int ind = params_.trees_num_per_forest_;
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        leaf_index_count[i] = index;
        index += rd_forests_[i].all_leaf_nodes_;
        feature_node_index[i] = ind*i;
    }
}

void Regressor::SaveRegressor(std::string ModelName, int stage){
	char buffer[50];
	//strcpy(buffer, ModelName.c_str());
	assert(stage == stage_);
    sprintf(buffer, "%s_%d_regressor.txt", ModelName.c_str(), stage);

	std::ofstream fout;
	fout.open(buffer, std::fstream::out);
	fout << stage_ << " "
		<< rd_forests_.size() << " "
        << linear_model_x_.size() << std::endl;

	for (int i = 0; i < rd_forests_.size(); i++){
		rd_forests_[i].SaveRandomForest(fout);
	}

    for (
         int i = 0; i < linear_model_x_.size(); i++){
        sprintf(buffer, "%s_%d", ModelName.c_str(), stage_);
//#ifdef _WIN32 // can be used under 32 and 64 bits
//        _mkdir(buffer);
//#elif __linux__
        struct stat st = {0};
        if (stat(buffer, &st) == -1) {
            mkdir(buffer, 0777);
        }
//#endif
		//_mkdir(buffer);
        sprintf(buffer, "%s_%d/%d_linear_x.txt", ModelName.c_str(), stage_, i);
        fout.close();
        fout.open(buffer, std::fstream::out);
		save_model_bin(fout, linear_model_x_[i]);
        fout.close();
        sprintf(buffer, "%s_%d/%d_linear_y.txt", ModelName.c_str(), stage_, i);
        fout.open(buffer, std::fstream::out);
		save_model_bin(fout, linear_model_y_[i]);
        fout.close();
	}
}
