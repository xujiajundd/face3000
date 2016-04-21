//#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
//#include <Windows.h>
#include "headers.h"
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <omp.h>
#endif
#include <Accelerate/Accelerate.h>

#ifndef DLIB_NO_GUI_SUPPORT
#define DLIB_NO_GUI_SUPPORT
#endif
//#include <dlib/config.h>
//#include <dlib/opencv.h>
//#include <dlib/image_processing/frontal_face_detector.h>
////#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>
////#include <dlib/gui_widgets.h>
//#include <dlib/all/source.cpp>

//#include <sys/time.h>
//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")
using namespace cv;
using namespace std;
//using namespace dlib;

//frontal_face_detector detector;
//correlation_tracker tracker;
bool faceDetected = false;

void DrawPredictedImage(cv::Mat_<uchar> image, cv::Mat_<float>& shape){
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), (255));
    }
    cv::imshow("show image", image);
    cv::waitKey(0);
}

void DrawPredictedImageContinue(cv::Mat image, cv::Mat_<float>& shape){
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, Scalar(255,255,255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!= 48 && i!=68 && i!=69)
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), Scalar(0,255,0));
    }
    cv::imshow("show image", image);
    char c = cv::waitKey( 10);
    
}


void Test(const char* ModelName){
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<float> > ground_truth_shapes;
    std::vector<int> ground_truth_faces;
	std::vector<BoundingBox> bboxes;
	std::string file_names = "/Users/xujiajun/developer/dataset/helen/test_jpgs.txt"; //"./../dataset/helen/train_jpgs.txt";
    LoadImages(images, ground_truth_shapes, ground_truth_faces, bboxes, file_names);
    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	for (int i = 0; i < images.size(); i++){
		cv::Mat_<float> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
        //struct timeval t1, t2;
        //gettimeofday(&t1, NULL);
        bool is_face = true;
        float score = 0;
        cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, bboxes[i], is_face, score);//, ground_truth_shapes[i]);

        //cout << res << std::endl;
        //cout << res - ground_truth_shapes[i] << std::endl;
        //float err = CalculateError(grodund_truth_shapes[i], res);
        //cout << "error: " << err << std::endl;
        cv::Rect faceRec;
        faceRec.x = bboxes[i].start_x;
        faceRec.y = bboxes[i].start_y;
        faceRec.width = bboxes[i].width;
        faceRec.height = bboxes[i].height;
        cv::rectangle(images[i], faceRec, (255), 1);
        
        DrawPredictedImage(images[i], res);

        
        float scale = 1.2;
        float shuffle = 0.15;
        int minSize = 100;
        int order = -1;
        int currentSize;
        bool biggest_only;
        
        if ( order == 1 ){
            currentSize = minSize;
        }
        else{
            currentSize = std::min(images[i].cols, images[i].rows);
        }
        
        while ( currentSize >= minSize && currentSize <= std::min(images[i].cols, images[i].rows)){
            for ( int ix=0; ix<images[i].cols-currentSize; ix+= currentSize*shuffle){
                for ( int jy=0; jy<images[i].rows-currentSize; jy+=currentSize*shuffle){
                    BoundingBox box;
                    box.start_x = ix;
                    box.start_y = jy;
                    box.width = currentSize;
                    box.height = currentSize;
                    box.center_x = box.start_x + box.width/2.0;
                    box.center_y = box.start_y + box.width/2.0;
                    bool is_face = true;
                    cv::Mat_<float> current_shape = ReProjection(cas_load.params_.mean_shape_, box);
                    float score = 0;
                    cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, box, is_face, score);
                    if ( is_face){
                        cv::Mat_<uchar> img = images[i].clone();
                        cv::Rect rect;
                        rect.x = box.start_x;
                        rect.y = box.start_y;
                        rect.width = box.width;
                        rect.height = box.height;
                        cv::rectangle(img, rect, (255), 1);
                        DrawPredictedImage(img, res);
                    }
                }
            }
            
            if ( order == 1 ){
                currentSize *= scale;
            }
            else{
                currentSize /= scale;
            }
        }
        
        
        
		//if (i == 10) break;
	}
    gettimeofday(&t2, NULL);
    float time_full = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << "time full: " << time_full << " : " << time_full/images.size() << endl;
	return;
}

void TestVideo(const char* ModelName){
    CascadeRegressor rg;
    rg.LoadCascadeRegressor(ModelName);
    rg.antiJitter = -1;
//    rg.params_.predict_group_.erase(0);
//    for (int i = 0; i < rg.params_.regressor_stages_; i++){
//        rg.regressors_[i].params_ = rg.params_;
//    }
//    rg.params_.predict_regressor_stages_ = 3;
    std::string fn_haar = "/Users/xujiajun/developer/dataset/haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    bool yes = haar_cascade.load(fn_haar);
    std::cout << "detector: " << yes << std::endl;
    
    std::string fn_haar_eye = "/Users/xujiajun/developer/dataset/haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_eye_cascade;
    haar_eye_cascade.load(fn_haar_eye);
    
    const string WindowName = "Face Detection example";
    namedWindow(WindowName);
    VideoCapture VideoStream(0);
    VideoStream.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    VideoStream.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    
    if (!VideoStream.isOpened()){
        printf("Error: Cannot open video stream from camera\n");
        return;
    }
    cv::Mat frame;
    Mat_<uchar> image;
    
    cv::Mat_<float> last_shape;
    bool lastShaped = false;
    while (true){
        VideoStream >> frame;
        cvtColor(frame, image, COLOR_RGB2GRAY);

        
        struct timeval t1, t2;
        float timeuse;
        gettimeofday(&t1, NULL);
        std::vector<cv::Rect> faces, eyes;
        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0
                                      |cv::CASCADE_FIND_BIGGEST_OBJECT
//                                      |cv::CASCADE_DO_ROUGH_SEARCH
                                      , cv::Size(100, 100));
//        cv_image<uchar>cimg(image);
//        std::vector<dlib::rectangle> faces;
//        if ( faceDetected ){
//            double result = tracker.update(cimg);
//            if ( result > 5.0 ){
//                dlib::rectangle r = tracker.get_position();
//                faces.clear();
//                faces.push_back(r);
//            }
//            else{
//                faceDetected = false;
//            }
//        }
//        else{
//            faces = detector(cimg);
//            if ( faces.size()>0){
//                tracker.start_track(cimg, faces[0]);
//                //faceDetected = true;
//            }
//        }
        
        gettimeofday(&t2, NULL);
        cout << faces.size() << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0  << endl;
        
        for (int i = 0; i < faces.size() && i < 1; i++){
//            gettimeofday(&t1, NULL);
//            cv::Mat face = image(faces[i]);
//            haar_eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(80,80));
//            
//            if (eyes.size())
//            {
//                cv::Rect rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
////                tpl  = im(rect);
//            }
//            gettimeofday(&t2, NULL);
//            cout << eyes.size() << "eye detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0  << endl;
            
            cv::Rect faceRec = faces[i];
//            cout << faces[0] << endl;
//            cv::Rect faceRec;
//            faceRec.x = faces[i].left();
//            faceRec.y = faces[i].top();
//            faceRec.width = faces[i].right() - faces[i].left();
//            faceRec.height = faces[i].bottom() - faces[i].top();
            
            BoundingBox bbox;
            bbox.start_x = faceRec.x;
            bbox.start_y = faceRec.y;
            bbox.width = faceRec.width;
            bbox.height = faceRec.height;
            bbox.center_x = bbox.start_x + bbox.width / 2.0;
            bbox.center_y = bbox.start_y + bbox.height / 2.0;
            cv::Mat_<float> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
//            if ( lastShaped ){
//                current_shape = last_shape;
//            }
            //cv::Mat_<float> tmp = image.clone();
            //DrawPredictedImage(tmp, current_shape);
            gettimeofday(&t1, NULL);
            bool is_face = true;
            float score = 0;
            cv::Mat_<float> res = rg.Predict(image, current_shape, bbox, is_face, score);//, ground_truth_shapes[i]);
            gettimeofday(&t2, NULL);
//            last_shape = res.clone(); lastShaped = true;
            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
            
//            for ( int d=0; d<rg.params_.predict_regressor_stages_; d++){
//                cout << rg.stage_delta_[d] << " ";
//                cv::line(frame, Point2f(20*d+20, 450), Point2f(20*d+20, 450 - 200000*rg.stage_delta_[d]), (255));
//            }
//            cout << endl;
            
            cv::rectangle(image, faceRec, (255), 1);
            //cv::imshow("show image", image);
            //cv::waitKey(0);

            DrawPredictedImageContinue(frame, res);
//            imshow(WindowName, image);
        }
//        imshow(WindowName, image);
    }
    return;
}


void TestImage(const char* name, CascadeRegressor& rg){
	std::string fn_haar = "/Users/xujiajun/developer/dataset/haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(fn_haar);
	std::cout << "detector: " << yes << std::endl;
	cv::Mat_<uchar> image = cv::imread(name, 0);
		if (image.cols > 2000){
			cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
			//ground_truth_shape /= 3.0;
		}
		else if (image.cols > 1400 && image.cols <= 2000){
			cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
			//ground_truth_shape /= 2.0;
		}
    std::vector<cv::Rect> faces;

    struct timeval t1, t2;
    float timeuse;
    gettimeofday(&t1, NULL);
    haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(100, 100));
    gettimeofday(&t2, NULL);
    cout << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    for (int i = 0; i < faces.size(); i++){
        cv::Rect faceRec = faces[i];
        BoundingBox bbox;
        bbox.start_x = faceRec.x;
        bbox.start_y = faceRec.y;
        bbox.width = faceRec.width;
        bbox.height = faceRec.height;
        bbox.center_x = bbox.start_x + bbox.width / 2.0;
        bbox.center_y = bbox.start_y + bbox.height / 2.0;
        cv::Mat_<float> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
        //cv::Mat_<float> tmp = image.clone();
        //DrawPredictedImage(tmp, current_shape);
        gettimeofday(&t1, NULL);
        bool is_face = true;
        float score = 0;
        cv::Mat_<float> res = rg.Predict(image, current_shape, bbox, is_face, score);//, ground_truth_shapes[i]);
        gettimeofday(&t2, NULL);
        cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
        
        cv::Mat_<uchar> img = image.clone();
        cv::rectangle(img, faceRec, (255), 1);
        //cv::imshow("show image", image);
        //cv::waitKey(0);
        DrawPredictedImage(img, res);
        break;
    }
    
    float scale = 1.2;
    float shuffle = 0.15;
    int minSize = image.cols/4;
    int order = -1;
    int currentSize;
    bool biggest_only;
    int faceFound = 0;
    int nonface = 0;
    if ( order == 1 ){
        currentSize = minSize;
    }
    else{
        currentSize = std::min(image.cols, image.rows);
    }
    gettimeofday(&t1, NULL);
    while ( currentSize >= minSize && currentSize <= std::min(image.cols, image.rows)){
        for ( int i=0; i<image.cols-currentSize; i+= currentSize*shuffle){
            for ( int j=0; j<image.rows-currentSize; j+=currentSize*shuffle){
                BoundingBox box;
                box.start_x = i;
                box.start_y = j;
                box.width = currentSize;
                box.height = currentSize;
                box.center_x = box.start_x + box.width/2.0;
                box.center_y = box.start_y + box.width/2.0;
                bool is_face = true;
                cv::Mat_<float> current_shape = ReProjection(rg.params_.mean_shape_, box);
                float score = 0;
                cv::Mat_<float> res = rg.Predict(image, current_shape, box, is_face, score);
                if ( is_face){
                    faceFound++;
//                    std::cout << "score:" << score << std::endl;
//                    cv::Mat_<uchar> img = image.clone();
//                    cv::Rect rect;
//                    rect.x = box.start_x;
//                    rect.y = box.start_y;
//                    rect.width = box.width;
//                    rect.height = box.height;
//                    cv::rectangle(img, rect, (255), 1);
//                    DrawPredictedImage(img, res);
                }
                else{
                    nonface++;
                }
            }
        }
        
        if ( order == 1 ){
            currentSize *= scale;
        }
        else{
            currentSize /= scale;
        }
    }
    
    gettimeofday(&t2, NULL);
    cout << "jda face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " face found:" << faceFound << " nonface checked:" << nonface << endl;
    
	return;
}

void Test(const char* ModelName, const char* name){
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
	TestImage(name, cas_load);
    return;
}

//TODO: detect multiscale, regression performance improve

void Train(const char* ModelName){
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<float> > ground_truth_shapes;
    std::vector<int> ground_truth_faces;
	std::vector<BoundingBox> bboxes;
    std::string file_names = "/Users/xujiajun/developer/dataset/helen/train_jpgs.txt";
    Parameters params;
    // train_jpgs.txt contains all the paths for each image, one image per line
    // for example: in Linux you can use ls *.jpg > train_jpgs.txt to get the paths
    // the file looks like as below
    /*
    	1.jpg
    	2.jpg
    	3.jpg
    	...
    	1000.jpg
    */
    
	int pos_num = LoadImages(images, ground_truth_shapes, ground_truth_faces, bboxes, file_names);
	params.mean_shape_ = GetMeanShape(ground_truth_shapes, ground_truth_faces, bboxes);
    
    params.local_features_num_ = 2000;
	params.landmarks_num_per_face_ = 68;
    params.regressor_stages_ = 5;
//    params.local_radius_by_stage_.push_back(0.6);
//    params.local_radius_by_stage_.push_back(0.5);
	params.local_radius_by_stage_.push_back(0.45);
    params.local_radius_by_stage_.push_back(0.3);
    params.local_radius_by_stage_.push_back(0.2);
	params.local_radius_by_stage_.push_back(0.1);//0.1
    params.local_radius_by_stage_.push_back(0.08);//0.08
    params.local_radius_by_stage_.push_back(0.05);
    params.local_radius_by_stage_.push_back(0.04);
    params.local_radius_by_stage_.push_back(0.03);

//    params.local_radius_by_stage_.push_back(0.2);
//    params.local_radius_by_stage_.push_back(0.15);
//    params.local_radius_by_stage_.push_back(0.1);
//    params.local_radius_by_stage_.push_back(0.8);
//    params.local_radius_by_stage_.push_back(0.05);
//    params.local_radius_by_stage_.push_back(0.03);
    
    params.detect_factor_by_stage_.push_back(0.9);
    params.detect_factor_by_stage_.push_back(0.7);
    params.detect_factor_by_stage_.push_back(0.5);
    params.detect_factor_by_stage_.push_back(0.3);
    params.detect_factor_by_stage_.push_back(0.1);
    params.detect_factor_by_stage_.push_back(0.1);
    params.detect_factor_by_stage_.push_back(0.1);
    params.detect_factor_by_stage_.push_back(0.1);
    
    params.tree_depth_ = 5;
    params.trees_num_per_forest_ = 12;
    params.initial_guess_ = 1;
    
//    params.group_num_ = 6;
//    std::vector<int> group1, group2, group3, group4, group5, group6, group7;
//    
//    for ( int i=17; i<27; i++ ) group2.push_back(i);
//    group2.push_back(-36);
//    group2.push_back(-45);
//    group2.push_back(-39);
//    group2.push_back(-42);
//    params.groups_.push_back(group2);
//    
//    for ( int i=27; i<36; i++ ) group3.push_back(i);
//    group3.push_back(-39);
//    group3.push_back(-42);
////    group3.push_back(-21);
////    group3.push_back(-22);
//    group3.push_back(-48);
//    group3.push_back(-54);
//    group3.push_back(-2);
//    group3.push_back(-14);
//    params.groups_.push_back(group3);
//    
//    for ( int i=36; i<48; i++) group4.push_back(i);
//    params.groups_.push_back(group4);
//    
//    for ( int i=48; i<55; i++) group5.push_back(i);
//    for ( int i=60; i<65; i++) group5.push_back(i);
//    params.groups_.push_back(group5);
//    
//    for ( int i= 55; i<60; i++) group6.push_back(i);
//    for ( int i= 65; i<68; i++) group6.push_back(i);
//    group6.push_back(-48);
////    group6.push_back(-60);
//    group6.push_back(-54);
////    group6.push_back(-64);
//    params.groups_.push_back(group6);
//    
////    group7.push_back(68);
////    group7.push_back(69);
////    params.groups_.push_back(group7);
//
//    //调整数序，让face内部的点先计算
//    for ( int i=0; i<17; i++ ) group1.push_back(i);
//    //    group1.push_back(-36);
//    //    group1.push_back(-45);
//    params.groups_.push_back(group1);

    CascadeRegressor cas_reg;
    cas_reg.Train(images, ground_truth_shapes, ground_truth_faces, bboxes, params, pos_num);
    cas_reg.SaveCascadeRegressor(ModelName);
    
	return;
}
//TODO:研究下回归数据要不要用short型存放，节省模型空间。是否还是搞成全局回归，改数据结构，用数组提高性能。（研究运算cache优化）
//回归的模式中，有没有可能看出detect的特征规律？
//用其他的特征？
//

void Hello(){
    int modellen = 2000;
    int stride = 16;
    int dim = 68;
    float x[modellen][dim], y[dim];
    for ( int i=0; i<dim; i++ ){
        for (int j=0; j<modellen; j++ ){
            x[j][i] = i*j;
            y[i] = 0;
        }
    }
    int rn = 10000;
    float sum;
    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for ( int r=0; r<rn; r++){
        for ( int i=0; i<dim; i++){
            for (int j=0; j<modellen; j+= stride ){
                y[i] += x[j][i];
            }
        }
    }
    gettimeofday(&t2, NULL);
    sum = 0;
    for ( int i=0; i<dim; i++){
        sum += y[i];
    }
    cout << sum << endl;
    cout << "time1: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;


    float xx[modellen][dim], yy[dim];
    for (int j=0; j<modellen; j++){
        for ( int i=0; i<dim; i++ ){
            xx[j][i] = i*j;
            yy[i] = 0;
        }
    }

    float ssum;

    gettimeofday(&t1, NULL);

    for ( int r=0; r<rn; r++){
        for (int j=0; j<modellen; j+=stride){
            for ( int i=0; i<dim; i++){
                yy[i] += xx[j][i];
            }
        }
    }
    gettimeofday(&t2, NULL);
    ssum = 0;

    for ( int i=0; i<dim; i++){
        ssum += yy[i];
    }
    cout << ssum << endl;
    cout << "time2: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;


//    gettimeofday(&t1, NULL);
//    float done = 1.0;
//    int ione = stride;
//
//    for ( int r=0; r<rn; r++){
//        float *xx = &x[0], *yy=&y[0];
//        cblas_saxpy(dim, done, xx, ione, yy, ione);
////        vDSP_vadd(x, 1, y, 1, z, 1, dim);
//    }
//    gettimeofday(&t2, NULL);
//    sum = 0;
//    for ( int i=0; i<dim; i++){
//        sum += y[i];
//    }
//    cout << sum << endl;
//    cout << "time1: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;







/*
    for ( int i=0; i<68; i++ ){
        std::cout << y[i] << ", ";
    }
    std::cout << endl;
    
    int pixel = 0;
    cv::Mat_<int> img = cv::Mat_<int>(640, 640);
    for (int i=0; i<640; i++){
        for ( int j=0; j<640; j++){
            img(i,j) = i*j;
        }
    }
    gettimeofday(&t1, NULL);
    for (int k=0; k<10000; k++){
        for (int i=0; i<640; i++){
            for ( int j=0; j<640; j++){
                img(0,j) += img(i,j);
            }
        }
    }
    gettimeofday(&t2, NULL);
    cout << "time mat: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    
    gettimeofday(&t1, NULL);
    for (int k=0; k<10000; k++){
        for (int i=0; i<640; i++){
            for ( int j=0; j<640; j++){
                img(i,0) += img(i,j);
            }
        }
    }
    gettimeofday(&t2, NULL);
    cout << "time mat: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    
    
    int uimg[68][2048];
    for (int i=0; i<68; i++){
        for ( int j=0; j<2048; j++){
            uimg[i][j] = i*j;
        }
    }
    
    gettimeofday(&t1, NULL);
    for (int k=0; k<10000; k++){
        for (int i=0; i<68; i++){
            for ( int j=0; j<2048; j=j+3){
                uimg[0][j] += uimg[i][j];
            }
        }
    }
    gettimeofday(&t2, NULL);
    cout << "time array: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << "   " << pixel << endl;
    
    
    int uimg2[2048][68];
    for (int i=0; i<2048; i++){
        for ( int j=0; j<68; j++){
            uimg2[i][j] = i*j;
        }
    }
    gettimeofday(&t1, NULL);
    for (int k=0; k<10000; k++){
        for (int i=0; i<2048; i=i+3){
            for ( int j=0; j<68;j++){
                uimg2[i][0] += uimg2[i][j];
            }
        }
    }
    gettimeofday(&t2, NULL);
    cout << "time array: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << "   " << pixel << endl;
    */
//    time_t current_time;
//    current_time = time(0);
//    cv::RNG random_generator(current_time);
//    int r = random_generator.uniform(0, 10000);
//    random_generator.uniform(0.0, 10.0);
//    
//    struct feature_node* binary_features = new feature_node[1089];
//    std::vector<struct model*> linear;
//    linear.resize(68);
////    model  linear[68];
//
//    
//    for (int i=0; i<68; i++){
//        linear[i] = new model;
//        float *w = (float *)malloc(1088*8);
//        for ( int j=0; j<1088*8; j++){
//            w[j] = random_generator.uniform(0.0, 10.0);
//        }
//        linear[i]->w = w;
//    }
// 
//    gettimeofday(&t1, NULL);
//
//    float sumw[68];
//    for ( int k=0; k<10; k++){
//        for (int i=0; i<1088; i++){
//            for ( int j=0; j<5; j++){
//                binary_features[i].index = 8*i;
//                binary_features[i].value = 1;
//            }
//        }
//        cv::Mat_<float> predict_result(68,2, 0.0);
//        binary_features[1088].index = -1;
//        binary_features[1088].value = 0;
//        for ( int i=0; i<68; i++){
//            int idx;
//            const feature_node *lx = binary_features;
//            float *w=linear[i]->w;
//            float result = 0.0;
//            for(; (idx=lx->index)!=-1 && idx < 1088; lx++){
//                sumw[i] += w[idx]; //为了这儿减少一次减法，在getglobalfeature的地方改了index的初值为0;
//            }
//            predict_result(i,0) = sumw[i];
//            predict_result(i,1) = sumw[i];
////            for ( int j=0; j<1088; j++){
////                sumw[i] += linear[i].w[binary_features[j].index];
////            }
//        }
//        
//    }
//
//    gettimeofday(&t2, NULL);
//    cout << "time sumw1: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
//    

//    gettimeofday(&t1, NULL);
//
//    for ( int k=0; k<10; k++){
//        for (int i=0; i<1088; i++){
//            for ( int j=0; j<5; j++){
//                binary_features[i].index = 8*i;
//                binary_features[i].value = 1.0;
//            }
//        }
//        for ( int j=0; j<1088; j++){
//            for ( int i=0; i<68; i++){
//                sumw[i] += linear[i]->w[binary_features[j].index];
//            }
//        }
//    }
//    gettimeofday(&t2, NULL);
//    cout << "time sumw: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0  << endl;
}

int main(int argc, char* argv[])
{
//    detector = get_frontal_face_detector();
    
	if (argc >= 3)
	{
		if (strcmp(argv[1], "train") == 0)
		{
			std::cout << "enter train\n";
			Train(argv[2]);

            return 0;
		}
		if (strcmp(argv[1], "test") == 0)
		{
			std::cout << "enter test\n";
            if (argc == 3){
                Test(argv[2]);
            }
            else{
                Test(argv[2], argv[3]);
            }
            return 0;
		}
        if (strcmp(argv[1], "video") == 0)
        {
            std::cout << "enter video\n";
            if (argc == 3){
                TestVideo(argv[2]);
            }
            return 0;
        }
	}
    else if ( argc == 2){
        if (strcmp(argv[1], "hello") == 0)
        {
            Hello();
            return 0;
        }
    }

    std::cout << "use [./application train ModelName] or [./application test ModelName [image_name]] \n";
	return 0;
}

