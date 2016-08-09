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
//    lastRes = cv::Mat_<float>(68,2,0.0);
    previousScanTime.tv_sec = 0;
    previousScanTime.tv_usec = 0;
    previousFrameTime.tv_sec = 0;
    previousFrameTime.tv_usec = 0;
    previousFrameRotations.clear();
    previousFrameShapes.clear();
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
        
        if ( debug_on_){
            DrawPredictImage(images[i], ground_truth_shapes[i]);
        }
        
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
                BoundingBox ibox = bboxes_[i];
                float minor = random_generator.uniform(-ibox.width, ibox.width);
                minor = 0.01 * minor;
                ibox.start_x -= minor/2.0;
                ibox.start_y -= minor/2.0;
                ibox.width += minor;
                ibox.height += minor;
                if ( ibox.start_x < 0 ) ibox.start_x = 0;
                if ( ibox.start_y < 0 ) ibox.start_y = 0;
                if ( (ibox.start_x + ibox.width) > images[i].cols ) ibox.width = images[i].cols - ibox.start_x;
                if ( (ibox.start_y + ibox.height) > images[i].rows ) ibox.height = images[i].rows - ibox.start_y;
                ibox.center_x = ibox.start_x + ibox.width / 2.0;
                ibox.center_y = ibox.start_y + ibox.height / 2.0;
                //这个地方对box也做了一点小小的扰动
                augmented_images_index.push_back(i);
                augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);// ReProjection(ProjectShape(ground_truth_shapes_[i], bboxes_[i]), ibox));
                augmented_ground_truth_faces.push_back(ground_truth_faces[i]);
                augmented_bboxes.push_back(ibox);
                cv::Mat_<float> temp; // = ground_truth_shapes_[index];
//                temp = ProjectShape(temp, bboxes_[index]);
//                temp = ReProjection(temp, ibox);
                do {
                    index = random_generator.uniform(0, pos_num);
                    while ( index == i ){
                        index = random_generator.uniform(0, pos_num);
                    }
//                    do {
//                        index = random_generator.uniform(0, pos_num);
//                    }while(index == i);
                    temp = ground_truth_shapes_[index];
                    temp = ProjectShape(temp, bboxes_[index]);
                    temp = ReProjection(temp, ibox);
                    if ( debug_on_){
                        if ( CalculateError(ground_truth_shapes[i], temp) > 0.5 ){
                            DrawPredictImage(images[i], ground_truth_shapes[i]);
                            DrawPredictImage(images[i], temp);
                        }
                    }
                } while ( CalculateError(ground_truth_shapes[i], temp) > 0.5 ); //这个地方可能会死循环的
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

        for (int j = 0; j < params_.initial_guess_ && j < 1; j++)
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
    float error = 0.0;
    int count = 0;
    for (int j = 0; j < augmented_ground_truth_shapes.size(); j++){
        if ( augmented_ground_truth_faces[j] == 1){ //pos example才计算误差
            error += CalculateError(augmented_ground_truth_shapes[j], augmented_current_shapes[j]);
            count++;
        }
    }
    std::cout << "initial error: " <<  error << ": " << error/count << std::endl;
    

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
                float e = CalculateError(augmented_ground_truth_shapes[j], augmented_current_shapes[j]);
                if ( e * (3+i) > 1.5){
                    //表示本阶段alignment的结果比较差，取消作为正例
                    find_times[j] = MAXFINDTIMES+8;
                    augmented_ground_truth_faces[j] = -1;
                    std::cout << "Alignment error:" << e << " for:"<< j << " image index:" << augmented_images_index[j] << std::endl;
                    if ( debug_on_ ){
                        DrawPredictImage(images[augmented_images_index[j]], augmented_current_shapes[j]);
                    }
                }
                error += e;
                count++;
            }
		}

        gettimeofday(&t1, NULL);
        std::cout << std::endl;
        std::cout << "regression error: " <<  error << ": " << error/count << " time:" << t1.tv_sec << std::endl;
	}
    if ( debug_on_){
        for ( int n=0; n<augmented_ground_truth_faces.size(); n++){
            if ( find_times[n] < MAXFINDTIMES && augmented_ground_truth_faces[n] == -1 ){
                DrawPredictImage(images[augmented_images_index[n]], augmented_current_shapes[n]);
            }
        }
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
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        std::cout << "landmark: " << i << std::endl;
        int true_pos_num = pos_num / ( params.initial_guess_ + 1 );
		rd_forests_[i] = RandomForest(params_, i, stage_, regression_targets, casRegressor, true_pos_num);
        rd_forests_[i].TrainForest(
			images,augmented_images_index, augmented_ground_truth_shapes, augmented_bboxes, augmented_current_shapes,
            augmented_ground_truth_faces, current_fi, current_weight, find_times,
			rotations_, scales_);
        int pos_examples_num = 0;
        int neg_examples_num = 0;
        for ( int n=0; n<augmented_ground_truth_faces.size(); n++){
            if ( find_times[n] < MAXFINDTIMES && augmented_ground_truth_faces[n] == 1 ){
                pos_examples_num++;
            }
            if ( find_times[n] < MAXFINDTIMES && augmented_ground_truth_faces[n] == -1 ){
                neg_examples_num++;
            }
        }
        std::cout<< "positive example left:" << pos_examples_num << " negative example left:" << neg_examples_num << std::endl;
	}
	std::cout << "Get Global Binary Features" << std::endl;

    struct feature_node **global_binary_features;
    global_binary_features = new struct feature_node* [augmented_current_shapes.size()];

    for(int i = 0; i < augmented_current_shapes.size(); ++i){
        global_binary_features[i] = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
    }
    int num_feature = 0;
    for (int i=0; i < params_.landmarks_num_per_face_; ++i){
        num_feature += rd_forests_[i].all_leaf_nodes_;
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
        for (int j = 0; j < params_.landmarks_num_per_face_; ++j){
            for (int k = 0; k < params_.trees_num_per_forest_; ++k){
                Node* node = rd_forests_[j].trees_[k];
                while (!node->is_leaf_){
                    if ( node->is_leaf_a ){
                        global_binary_features[i][ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k, images[augmented_images_index[i]], augmented_bboxes[i], augmented_current_shapes[i], rotations_[i], scales_[i]);
                        global_binary_features[i][ind].value = 1.0;
                        ind++;
                    }
                    FeatureLocations& pos = node->feature_locations_;
                    float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                    float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                    delta_x = scale*delta_x*bbox.width / 2.0;
                    delta_y = scale*delta_y*bbox.height / 2.0;
                    int real_x = delta_x + current_shape(pos.lmark1, 0);
                    int real_y = delta_y + current_shape(pos.lmark1, 1);
                    real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                    real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows//////
                    // int tmp = (int)(2*image(real_y, real_x) + image(real_y-1, real_x) + image(real_y+1, real_x) + image(real_y, real_x-1) +image(real_y, real_x+1)) / 6 ; //real_y at first
                    int tmp = image(real_y, real_x);
                    
                    delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                    delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                    delta_x = scale*delta_x*bbox.width / 2.0;
                    delta_y = scale*delta_y*bbox.height / 2.0;
                    real_x = delta_x + current_shape(pos.lmark2, 0);
                    real_y = delta_y + current_shape(pos.lmark2, 1);
                    real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                    real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                    //int tmp2 = (int)(2*image(real_y, real_x) + image(real_y-1, real_x) + image(real_y+1, real_x) + image(real_y, real_x-1) +image(real_y, real_x+1)) / 6 ;
                    int tmp2 = image(real_y, real_x);

                    if ( k % 2 == 0 ){
                        if (abs(tmp-tmp2) < node->threshold_){
                            node = node->left_child_;// go left
                        }
                        else{
                            node = node->right_child_;// go right
                        }
                    }
                    else{
                        if ( (tmp-tmp2) < node->threshold_){
                            node = node->left_child_;// go left
                        }
                        else{
                            node = node->right_child_;// go right
                        }
                    }
                }
                if ( node->is_leaf_a ){
                    global_binary_features[i][ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k, images[augmented_images_index[i]], augmented_bboxes[i], augmented_current_shapes[i], rotations_[i], scales_[i]);
                    global_binary_features[i][ind].value = 1.0;
                    ind++;
                }
                //std::cout << global_binary_features[i][ind].index << " ";
            }
            index += rd_forests_[j].all_leaf_nodes_;
        }
        if (i%500 == 0 && i > 0){
//                std::cout << "extracted " << i << " images" << std::endl;
        }
        global_binary_features[i][params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
        global_binary_features[i][params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
    }
    std::cout << "\n";

    struct problem* prob = new struct problem;
    prob->l = pos_num; //augmented_current_shapes.size();
    prob->n = num_feature;
    prob->x = global_binary_features;
    prob->bias = -1;

    struct parameter* regression_params = new struct parameter;
    regression_params-> solver_type = L2R_L2LOSS_SVR_DUAL;
    regression_params->C = 1.0/pos_num; //augmented_current_shapes.size();
    regression_params->p = 0;

    std::cout << "Global Regression of stage " << stage_ << std::endl;
    linear_model_x_.resize(params_.landmarks_num_per_face_);
    linear_model_y_.resize(params_.landmarks_num_per_face_);
    float** targets = new float*[params_.landmarks_num_per_face_];
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        targets[i] = new float[pos_num];
    }
    #pragma omp parallel for
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){

        std::cout << "regress landmark " << i << std::endl;
        for(int j = 0; j< pos_num;j++){
            targets[i][j] = regression_targets[j](i, 0);
        }
        prob->y = targets[i];
        check_parameter(prob, regression_params);
        struct model* regression_model = train(prob, regression_params);
        linear_model_x_[i] = regression_model;
        for(int j = 0; j < pos_num; j++){
            targets[i][j] = regression_targets[j](i, 1);
        }
        prob->y = targets[i];
        check_parameter(prob, regression_params);
        regression_model = train(prob, regression_params);
        linear_model_y_[i] = regression_model;

    }
    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
        delete[] targets[i];// = new float[augmented_current_shapes.size()];
    }
    delete[] targets;
	std::cout << "predict regression targets" << std::endl;

    std::vector<cv::Mat_<float> > predict_regression_targets;
    predict_regression_targets.resize(augmented_current_shapes.size());
    #pragma omp parallel for
    for (int i = 0; i < augmented_current_shapes.size(); i++){
        cv::Mat_<float> a(params_.landmarks_num_per_face_, 2, 0.0);
        for (int j = 0; j < params_.landmarks_num_per_face_; j++){
            a(j, 0) = predict(linear_model_x_[j], global_binary_features[i]);
            a(j, 1) = predict(linear_model_y_[j], global_binary_features[i]);
        }
        cv::Mat_<float> rot;
        cv::transpose(rotations_[i], rot);
        predict_regression_targets[i] = scales_[i] * a * rot;
        if (i%500 == 0 && i > 0){
//             std::cout << "predict " << i << " images" << std::endl;
        }
    }
    std::cout << "\n";


    for (int i = 0; i< augmented_current_shapes.size(); i++){
        delete[] global_binary_features[i];
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
                                          cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float& score ){
    cv::Mat_<float> rot(2,2,0.0);
    return Predict(image, current_shape, bbox, is_face, score, rot);
}

cv::Mat_<float> CascadeRegressor::Predict(cv::Mat_<uchar>& image,
                                          cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float& score, cv::Mat_<float>& rot){

    cv::Mat_<float> rotation;
    float scale;
    float lastThreshold = -1.0;
	for (int i = 0; i < params_.predict_regressor_stages_; i++){

//        struct timeval t1, t2;
//        gettimeofday(&t1, NULL);
        //这个耗时百分之一毫秒左右，
		getSimilarityTransform(/*ProjectShape(current_shape, bbox)*/current_shape, params_.mean_shape_, rotation, scale);
//        gettimeofday(&t2, NULL);
//        std::cout << "transform: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << std::endl;
		cv::Mat_<float> shape_increaments = regressors_[i].Predict(image, current_shape, bbox, rotation, scale, score, is_face, lastThreshold);
        if ( is_face != 1){
            //std::cout << "检测不是face!!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
            return current_shape;
        }
        current_shape = shape_increaments + /*ProjectShape(current_shape, bbox)*/current_shape;
		//current_shape = ReProjection(current_shape, bbox);
	}
    cv::Mat_<float> res = ReProjection(current_shape, bbox); //current_shape;
    
    //add by xujj, 做一个小幅抖动滤波，这个在detect时无效了。。。//TODO：可放到检测合并结束的时候再做
//    if ( antiJitter == 1 && params_.landmarks_num_per_face_ == 68 ){
//        for ( int j=0; j<68; j++){
//            float alphax = fabs(res(j,0)-lastRes(j,0)) / 10.0;
//            float alphay = fabs(res(j,1)-lastRes(j,1)) / 10.0;
//            if (alphax > 1.0 ) alphax = 1.0;
//            if (alphay > 1.0 ) alphay = 1.0;
//            
//            lastRes(j,0) = ((1.0-alphax)*lastRes(j,0) + alphax*res(j,0));
//            lastRes(j,1) = ((1.0-alphay)*lastRes(j,1) + alphay*res(j,1));
//        }
//        return lastRes;
//    }
    rot = rotation;
	return res;
}

/*
 * \breif nms Non-maximum suppression
 *  the algorithm is from https://github.com/ShaoqingRen/SPP_net/blob/master/nms%2Fnms_mex.cpp
 *
 * \param rects     area of faces
 * \param scores    score of faces
 * \param overlap   overlap threshold
 * \return          picked index
*/
/*
static vector<int> nms(const vector<Rect>& rects, const vector<double>& scores, \
                       double overlap) {
    const int n = rects.size();
    vector<double> areas(n);
    
    typedef std::multimap<double, int> ScoreMapper;
    ScoreMapper map;
    for (int i = 0; i < n; i++) {
        map.insert(ScoreMapper::value_type(scores[i], i));
        areas[i] = rects[i].width*rects[i].height;
    }
    
    int picked_n = 0;
    vector<int> picked(n);
    while (map.size() != 0) {
        int last = map.rbegin()->second; // get the index of maximum score value
        picked[picked_n] = last;
        picked_n++;
        
        for (ScoreMapper::iterator it = map.begin(); it != map.end();) {
            int idx = it->second;
            double x1 = std::max(rects[idx].x, rects[last].x);
            double y1 = std::max(rects[idx].y, rects[last].y);
            double x2 = std::min(rects[idx].x + rects[idx].width, rects[last].x + rects[last].width);
            double y2 = std::min(rects[idx].y + rects[idx].height, rects[last].y + rects[last].height);
            double w = std::max(0., x2 - x1);
            double h = std::max(0., y2 - y1);
            double ov = w*h / (areas[idx] + areas[last] - w*h);
            if (ov > overlap) {
                ScoreMapper::iterator tmp = it;
                tmp++;
                map.erase(it);
                it = tmp;
            }
            else{
                it++;
            }
        }
    }
    
    picked.resize(picked_n);
    return picked;
}
 */

bool box_overlap(BoundingBox box1, BoundingBox box2){
    float sx = std::max(box1.start_x, box2.start_x);
    float sy = std::max(box1.start_y, box2.start_y);
    float ex = std::min(box1.start_x+box1.width, box2.start_x+box2.width);
    float ey = std::min(box1.start_y+box1.height, box2.start_y+box2.height);
    float oversquare = 0;
    if ( ex > sx && ey > sy ){
        oversquare = (ex - sx)*(ey - sy);
    }

    float square1 = box1.width * box1.height;
    float square2 = box2.width * box2.height;

    if ( oversquare > 0.5 * std::min(square1, square2)) {
        return true;
    }

    return false;
}

std::vector<cv::Rect> CascadeRegressor::detectMultiScale(cv::Mat_<uchar>& image,
                                                         std::vector<cv::Mat_<float>>& shapes, float scaleFactor, int minNeighbors, int flags,
                                                         int minSize){
    std::vector<cv::Rect> faces;
    shapes.clear();
    faces.clear();
    float shuffle = 0.1;
    int biggest = flags & CASCADE_FLAG_BIGGEST_ONLY;
    int track_mode = flags & CASCADE_FLAG_TRACK_MODE;
    int currentSize;

    int faceFound = 0;
    int nonface = 0;

    struct candidate{
        float score;
        BoundingBox box;
        cv::Mat_<float> shape;
        cv::Mat_<float> rotation;
        int neighbors;
    };

    std::vector<struct candidate> candidates;

    std::vector<cv::Rect> searchRects;
    std::vector<int> minSizes;
    std::vector<int> maxSizes;
    searchRects.clear();
    minSizes.clear();
    maxSizes.clear();

    struct timeval current_time;
    gettimeofday(&current_time, NULL);

    //测试一下初始位置旋转过来的效果，可用于下一步做跟踪
    cv::Mat_<float> default_shape;//(params_.landmarks_num_per_face_, 2, 0.0);
//    for ( int i=0; i<params_.landmarks_num_per_face_; i++){
//        float cos = 0.0;
//        float sin = 1.0;
//        default_shape(i,0)=  cos*params_.mean_shape_(i,0) + sin*params_.mean_shape_(i,1);
//        default_shape(i,1)= -sin*params_.mean_shape_(i,0) + cos*params_.mean_shape_(i,1);
//    }

    if ( track_mode ){
        //跟踪模式，首先是时间与上次帧在100ms之内，然后，计算defaultshape和框。
        if ((current_time.tv_sec*1000000 - previousFrameTime.tv_sec*1000000 + current_time.tv_usec - previousFrameTime.tv_usec) > 200000 || previousFrameShapes.size() == 0){
            //没有有效的上次识别
            if ( current_time.tv_sec - previousScanTime.tv_sec > 1 ){
                //做一次全局扫描
                minSizes.push_back(minSize);
                maxSizes.push_back(std::min(image.cols, image.rows));
                cv::Rect sRect;
                sRect.x = 0; sRect.y = 0; sRect.width = image.cols; sRect.height = image.rows;
                searchRects.push_back(sRect);
                default_shape = params_.mean_shape_;
            }
            else{
                //这次轮空即可
                //std::cout << "current_time " << current_time.tv_sec << " " << current_time.tv_usec << " " << previousScanTime.tv_sec <<  std::endl;
                return faces;
            }
        }
        else{
            for ( int i=0; i < previousFrameShapes.size(); i++){
                BoundingBox sbox = CalculateBoundingBox(previousFrameShapes[i]);
                minSizes.push_back(sbox.width*0.9);
                maxSizes.push_back(sbox.width*1.1);
                cv::Rect sRect;
                sRect.x = sbox.start_x - sbox.width/4.0;
                sRect.y = sbox.start_y - sbox.height/4.0;
                sRect.width = 1.5 * sbox.width;
                sRect.height = 1.5 * sbox.height;
                if ( sRect.x < -sbox.width/2 ) sRect.x = -sbox.width/2;
                if ( sRect.y < -sbox.height/2 ) sRect.y = - sbox.height/2;
                if ( sRect.x + sRect.width >  image.cols + sbox.width/2 ) sRect.width = image.cols + sbox.width /2 - sRect.x;
                if ( sRect.y + sRect.height > image.rows + sbox.height/2 ) sRect.height = image.rows + sbox.height/2 - sRect.y;
                searchRects.push_back(sRect);
                cv::Mat_<float> rot;
                cv::transpose(previousFrameRotations[i], rot);
                default_shape =  params_.mean_shape_ * rot;
            }
        }
    }
    else{
        minSizes.push_back(minSize);
        maxSizes.push_back(std::min(image.cols, image.rows));
        cv::Rect sRect;
        sRect.x = 0; sRect.y = 0; sRect.width = image.cols; sRect.height = image.rows;
        searchRects.push_back(sRect);
        default_shape = params_.mean_shape_;
    }
    //TODO:可以根据返回的score或者stage，改变shuffle及scale的尺度？不同尺度下，也可以利用改信息？
    int scan_count=0;
    //struct timeval t1, t2;
    for( int s=0; s<searchRects.size(); s++ ){
        cv::Rect sRect = searchRects[s];
        currentSize = maxSizes[s];
        while ( currentSize >= minSizes[s] && currentSize <= maxSizes[s]){
            BoundingBox box;
            box.width = currentSize;
            box.height = currentSize;
            for ( int i=sRect.x; i<=sRect.x+sRect.width-currentSize; i+= currentSize*shuffle){
                box.start_x = i;
                box.center_x = box.start_x + box.width/2.0;
                for ( int j=sRect.y; j<=sRect.y+sRect.height-currentSize; j+=currentSize*shuffle){
                    //gettimeofday(&t1, NULL);
                    scan_count++;
                    box.start_y = j;
                    box.center_y = box.start_y + box.height/2.0;
                    int is_face = 1;
                    cv::Mat_<float> current_shape = default_shape.clone(); //ReProjection(default_shape, box);
                    float score = 0;
                    cv::Mat_<float> rotation(2,2,0.0);
                    cv::Mat_<float> res = Predict(image, current_shape, box, is_face, score, rotation);
                    //gettimeofday(&t2, NULL);
                    //std::cout << is_face << " time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " score:" <<  std::endl;
                    if ( is_face == 1){
                        faceFound++;
                        bool has_overlap = false;
                        for ( int c=0; c<candidates.size(); c++){
                            if ( box_overlap(box, candidates[c].box)){ //TODO:这个算法这样有漏洞的
                                if ( score > candidates[c].score){
                                    candidates[c].score = score;
                                    candidates[c].box = box;
                                    candidates[c].shape = res;
                                    candidates[c].neighbors++;
                                    candidates[c].rotation = rotation;
                                    //if ( candidates[c].neighbors > 6 ) goto _destfor;
                                }
                                else{
                                    candidates[c].neighbors++;
                                }
                                //这个结果处理完
                                has_overlap = true;
                                break;
                            }
                            else{

                            }
                        }
                        if ( !has_overlap ){
                            struct candidate cand;
                            cand.score = score;
                            cand.box = box;
                            cand.shape = res;
                            cand.neighbors = 0;
                            cand.rotation = rotation;
                            candidates.push_back(cand);
                        }
                    }
                    else{
                        nonface++;
                    }
                }
            }
            //std::cout<<"count:"<<scan_count<<" face found:"<<faceFound<< std::endl;
            currentSize /= scaleFactor;
        }
    }

_destfor:
    for ( int c=0; c<candidates.size(); c++){
        if ( candidates[c].neighbors >= minNeighbors  ){
            previousFrameRotations.clear();
            previousFrameShapes.clear();
            gettimeofday(&previousFrameTime, NULL);
            break;
        }
    }

    for ( int c=0; c<candidates.size(); c++){
        if ( candidates[c].neighbors >= minNeighbors  ){
            cv::Rect rect;
            rect.x = candidates[c].box.start_x;
            rect.width = candidates[c].box.width;
            rect.y = candidates[c].box.start_y;
            rect.height = candidates[c].box.height;
            faces.push_back(rect);
            shapes.push_back(candidates[c].shape);
            previousFrameRotations.push_back(candidates[c].rotation);
            previousFrameShapes.push_back(candidates[c].shape);
        }
    }
    if ( faces.size() == 0 )
        gettimeofday(&previousScanTime, NULL);
//    std::cout<<"count:"<<scan_count<<std::endl;
//    std::cout<<"count:"<<scan_count<<" face found:"<<faceFound<< std::endl;
    return faces;
}



cv::Mat_<float> CascadeRegressor::NegMinePredict(cv::Mat_<uchar>& image,
                                          cv::Mat_<float>& current_shape, BoundingBox& bbox, int& is_face, float &fi, int stage, int landmark, int tree){
    //    cv::Mat_<float> rshape = ProjectShape( current_shape.clone(), bbox);
    
//    float score = 0;
    for (int i = 0; i <= stage; i++){
        cv::Mat_<float> rotation;
        float scale;
        getSimilarityTransform(ProjectShape(current_shape, bbox), params_.mean_shape_, rotation, scale);
        cv::Mat_<float> shape_increaments = regressors_[i].NegMinePredict(image, current_shape, bbox, rotation, scale, fi, is_face, stage, i, landmark, tree);
        if ( is_face != 1 || shape_increaments.empty()){
//            std::cout << "挖掘负例，不是face!!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
//            fi = score;
            return current_shape;
        }
        current_shape = shape_increaments + ProjectShape(current_shape, bbox);
        current_shape = ReProjection(current_shape, bbox);
    }
    cv::Mat_<float> res = current_shape;
//    fi = score;
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


struct feature_node* Regressor::GetGlobalBinaryFeatures(cv::Mat_<uchar>& image,
    cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold){
    int index = 1;
    if ( tmp_binary_features == NULL ){
//        struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.groups_[groupNum].size()+1]; //这条性能可能可以优化
        tmp_binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_ +1];
    }

    int ind = 0;
    float ss = scale * bbox.width / 2.0; //add by xujj
    cv::Mat_<float> current_shape_re = ReProjection(current_shape, bbox);
    for (int j = 0; j < params_.landmarks_num_per_face_; ++j)
    {
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            int outBound = 0;
            Node* node = rd_forests_[j].trees_[k];
            while (!node->is_leaf_){
                if ( node->is_leaf_a ){
                    tmp_binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
                    tmp_binary_features[ind].value = 1.0;
                    ind++;
                }
                FeatureLocations& pos = node->feature_locations_;
                float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = ss * delta_x; //scale*delta_x*bbox.width / 2.0;
                delta_y = ss * delta_y; //scale*delta_y*bbox.height / 2.0;
                int real_x = delta_x + current_shape_re(pos.lmark1, 0);
                int real_y = delta_y + current_shape_re(pos.lmark1, 1);
                if ( real_x < 0 || real_y < 0 || real_x >= image.cols || real_y >= image.rows ){
                    outBound++;
                }
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                //int tmp = (int)(2*image(real_y, real_x) + image(real_y-1, real_x) + image(real_y+1, real_x) + image(real_y, real_x-1) +image(real_y, real_x+1)) / 6 ; //real_y at first
                int tmp = image(real_y, real_x);
                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = ss * delta_x; //scale*delta_x*bbox.width / 2.0;
                delta_y = ss * delta_y; //scale*delta_y*bbox.height / 2.0;
                real_x = delta_x + current_shape_re(pos.lmark2, 0);
                real_y = delta_y + current_shape_re(pos.lmark2, 1);
                if ( real_x < 0 || real_y < 0 || real_x >= image.cols || real_y >= image.rows ){
                    outBound++;
                }
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                //int tmp2 = (int)(2*image(real_y, real_x) + image(real_y-1, real_x) + image(real_y+1, real_x) + image(real_y, real_x-1) +image(real_y, real_x+1)) / 6 ;
                int tmp2 = image(real_y, real_x);
                if ( k % 2 == 0 ){
                    if ( abs(tmp - tmp2 ) < node->threshold_){
                        node = node->left_child_;// go left
                    }
                    else{
                        node = node->right_child_;// go right
                    }
                }
                else{
                    if ( (tmp - tmp2 ) < node->threshold_){
                        node = node->left_child_;// go left
                    }
                    else{
                        node = node->right_child_;// go right
                    }
                }
            }
            if ( outBound > 6 ) {
                if ( k == 0 ){
                    if ( j == 0 ){
                        score += rd_forests_[j].trees_[k]->score_ - lastThreshold;
                    }
                    else{
                        score += rd_forests_[j].trees_[k]->score_ - rd_forests_[j-1].trees_[params_.trees_num_per_forest_-1]->score_;
                    }
                }
                else{
                    score += rd_forests_[j].trees_[k]->score_ - rd_forests_[j].trees_[k-1]->score_;
                }
            }
            else{
                score += node->score_;
            }
            if ( score < rd_forests_[j].trees_[k]->score_ ){
                is_face = - stage_;
                //std::cout <<"stage:"<<stage_ << "lmark=" << j << " tree=" <<  k << " score:" << score << " threshold:" << rd_forests_[j].trees_[k]->score_<< std::endl;
                return tmp_binary_features;
            }
            //int ind = j*params_.trees_num_per_forest_ + k;
            //int ind = feature_node_index[j] + k;
            //binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
            if ( node->is_leaf_a ){
                tmp_binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
                tmp_binary_features[ind].value = 1.0;
                ind++;
            }
            //std::cout << binary_features[ind].index << " ";
        }

        index += rd_forests_[j].all_leaf_nodes_;
    }
    //std::cout << "\n";
    //std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
    tmp_binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
    tmp_binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
    lastThreshold = rd_forests_[params_.landmarks_num_per_face_-1].trees_[params_.trees_num_per_forest_-1]->score_;
    return tmp_binary_features;
}

struct feature_node* Regressor::NegMineGetGlobalBinaryFeatures(cv::Mat_<uchar>& image,
                                                        cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, int stage, int currentStage, int landmark, int tree, bool &stop){
    int index = 1;
    if ( tmp_binary_features == NULL ){
        //        struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.groups_[groupNum].size()+1]; //这条性能可能可以优化
        tmp_binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_ +1];
    }
    int ind = 0;
    float ss = scale * bbox.width / 2.0; //add by xujj
    for (int j = 0; j < params_.landmarks_num_per_face_; ++j)
    {
        for (int k = 0; k < params_.trees_num_per_forest_; ++k)
        {
            Node* node = rd_forests_[j].trees_[k];
            while (!node->is_leaf_){
                if ( node->is_leaf_a ){
                    tmp_binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
                    tmp_binary_features[ind].value = 1.0;
                    ind++;
                }
                FeatureLocations& pos = node->feature_locations_;
                float delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
                float delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
                delta_x = ss * delta_x; //scale*delta_x*bbox.width / 2.0;
                delta_y = ss * delta_y; //scale*delta_y*bbox.height / 2.0;
                int real_x = delta_x + current_shape(pos.lmark1, 0);
                int real_y = delta_y + current_shape(pos.lmark1, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                //int tmp = (int)(2*image(real_y, real_x) + image(real_y-1, real_x) + image(real_y+1, real_x) + image(real_y, real_x-1) +image(real_y, real_x+1)) / 6 ; //real_y at first
                int tmp = image(real_y, real_x);
                
                delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
                delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
                delta_x = ss * delta_x; //scale*delta_x*bbox.width / 2.0;
                delta_y = ss * delta_y; //scale*delta_y*bbox.height / 2.0;
                real_x = delta_x + current_shape(pos.lmark2, 0);
                real_y = delta_y + current_shape(pos.lmark2, 1);
                real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
                real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
                //int tmp2 = (int)(2*image(real_y, real_x) + image(real_y-1, real_x) + image(real_y+1, real_x) + image(real_y, real_x-1) +image(real_y, real_x+1)) / 6 ;
                int tmp2 = image(real_y, real_x);
                if ( k % 2 == 0 ){
                    if ( abs(tmp-tmp2) < node->threshold_){
                        node = node->left_child_;// go left
                    }
                    else{
                        node = node->right_child_;// go right
                    }
                }
                else{
                    if ( (tmp-tmp2) < node->threshold_){
                        node = node->left_child_;// go left
                    }
                    else{
                        node = node->right_child_;// go right
                    }
                }
            }
            score += node->score_;
            if ( score < rd_forests_[j].trees_[k]->score_ ){
                is_face = - currentStage;
                //std::cout <<"stage:"<<stage_ << "j=" << j << " k=" <<  k << " score:" << score << " threshold:" << rd_forests_[j].trees_[k]->score_<< std::endl;
                return tmp_binary_features;
            }

            //int ind = j*params_.trees_num_per_forest_ + k;
            //int ind = feature_node_index[j] + k;
            //binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
            if ( node->is_leaf_a ){
                tmp_binary_features[ind].index = index + node->leaf_identity;//rd_forests_[j].GetBinaryFeatureIndex(k,image, bbox, current_shape, rotation, scale);
                tmp_binary_features[ind].value = 1.0;
                ind++;
            }
            //std::cout << binary_features[ind].index << " ";
            if ( stage == currentStage && landmark == j && tree == k){
                is_face = 1;
                stop = true;
                return tmp_binary_features;
            }
        }
        index += rd_forests_[j].all_leaf_nodes_;
        
    }
    //std::cout << "\n";
    //std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
    tmp_binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
    tmp_binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
    return tmp_binary_features;
}

cv::Mat_<float> Regressor::Predict(cv::Mat_<uchar>& image,
	cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face, float& lastThreshold){

    cv::Mat_<float> predict_result;
    feature_node* binary_features = GetGlobalBinaryFeatures(image, current_shape, bbox, rotation, scale, score, is_face, lastThreshold);
    if ( is_face != 1 ){
        return predict_result;
    }
//    struct timeval t1, t2;
//    gettimeofday(&t1, NULL);
    predict_result = cv::Mat_<float>(current_shape.rows, current_shape.cols);
//    for (int i = 0; i < params_.landmarks_num_per_face_; i++){
////		predict_result(i, 0) = predict(linear_model_x_[i], binary_features);
////        predict_result(i, 1) = predict(linear_model_y_[i], binary_features);
//        
//        int idx;
//        const feature_node *lx=binary_features;
//        float *wx =linear_model_x_[i]->w;
//        float *wy = linear_model_y_[i]->w;
////        float resultx = 0.0, resulty = 0.0;
//        for(; (idx=lx->index)!=-1 && idx < linear_model_x_[i]->nr_feature; lx++){
//            idx--;
//            predict_result(i,0) += wx[idx]; 
//            predict_result(i,1) += wy[idx];
//        }
//
//    }

    //performance test
    //模型数据结构改为：m[idx][2*landmarks], idx为binary_feature的长度
    int idx;
    const feature_node *lx=binary_features;
    float sum[2*params_.landmarks_num_per_face_];
    for ( int i=0; i<2*params_.landmarks_num_per_face_; i++) sum[i] = 0;
    
    for(; (idx=lx->index)!=-1 && idx < linear_model_x_[0]->nr_feature; lx++){
        idx--;
//        cblas_saxpy(2*params_.landmarks_num_per_face_, 1.0, &modreg[idx][0], 1, sum, 1); //用这个速度跟后面的循环差不多
        for (int i = 0; i < 2*params_.landmarks_num_per_face_; i++){
            sum[i] += modreg[idx][i];
        }
    }
    for ( int i=0; i<params_.landmarks_num_per_face_; i++){
        predict_result(i,0) = sum[2*i];
        predict_result(i,1) = sum[2*i+1];
    }

     //   gettimeofday(&t2, NULL);
     //   std::cout << "linear " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << std::endl;
     //   delete[] binary_features;

	cv::Mat_<float> rot;
	cv::transpose(rotation, rot);
    
    //delete[] tmp_binary_features;
	return scale*predict_result*rot;
}

cv::Mat_<float> Regressor::NegMinePredict(cv::Mat_<uchar>& image,
                                   cv::Mat_<float>& current_shape, BoundingBox& bbox, cv::Mat_<float>& rotation, float scale, float& score, int& is_face,  int stage, int currentStage, int landmark, int tree){
    
    cv::Mat_<float> predict_result;

        bool stop = false;
        feature_node* binary_features = NegMineGetGlobalBinaryFeatures(image, current_shape, bbox, rotation, scale, score, is_face, stage, currentStage, landmark, tree, stop);
        if ( is_face != 1 ){
            return predict_result;
        }
        if ( stop ){
            return predict_result;
        }
        if ( stage == currentStage ) return predict_result; //如果已经测试到当前的stage，不用做shape回归了。

        predict_result = cv::Mat_<float>(current_shape.rows, current_shape.cols, 0.0);

        for (int i = 0; i < params_.landmarks_num_per_face_; i++){
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
		>> params_.initial_guess_;

    NUM_LANDMARKS = params_.landmarks_num_per_face_;
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

    fin.close();
    
    params_.predict_regressor_stages_ = params_.regressor_stages_;
    
	regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
        regressors_[i].params_ = params_;
		regressors_[i].LoadRegressor(ModelName, i);
//        regressors_[i].ConstructLeafCount();
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
		<< params_.initial_guess_ << std::endl;
    
	for (int i = 0; i < params_.regressor_stages_; i++){
		fout << params_.local_radius_by_stage_[i] << std::endl;
	}
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fout << params_.mean_shape_(i, 0) << " " << params_.mean_shape_(i, 1) << std::endl;
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
    float max_linear = -9999999.0;
    float min_linear = 9999999.0;
    modreg = new float *[linear_model_x_[0]->nr_feature];
    for ( int i = 0; i < linear_model_x_[0]->nr_feature; i++){
        modreg[i] = new float[2*params_.landmarks_num_per_face_];
        for ( int j = 0; j<params_.landmarks_num_per_face_; j++ ){
            float *wx =linear_model_x_[j]->w; 
            float *wy = linear_model_y_[j]->w;
            modreg[i][2*j] = wx[i];
            modreg[i][2*j+1] = wy[i];
            if ( wx[i] > max_linear ) max_linear = wx[i];
            if ( wx[i] < min_linear ) min_linear = wx[i];
            if ( wy[i] > max_linear ) max_linear = wy[i];
            if ( wy[i] < min_linear ) min_linear = wy[i];
        }
    }
//    std::cout<<"regression value max:" << max_linear << " min:" << min_linear << std::endl;
}

//void Regressor::ConstructLeafCount(){
//    int index = 1;
//    int ind = params_.trees_num_per_forest_;
//    for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
//        leaf_index_count[i] = index;
//        index += rd_forests_[i].all_leaf_nodes_;
//        feature_node_index[i] = ind*i;
//    }
//}

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
