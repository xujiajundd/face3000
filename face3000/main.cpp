//#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <string>
#include <time.h>
//#include <Windows.h>
#include "headers.h"
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <omp.h>
#endif


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

void DrawPredictedImage(cv::Mat_<uchar> image, cv::Mat_<float>& ishape){
    Mat_<float> shape = reConvertShape(ishape);
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, Scalar(255,255,255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!= 60 && i!=68 && i!=69)
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), Scalar(0,255,0));
    }
    cv::line(image, cv::Point2f(shape(36, 0), shape(36, 1)), cv::Point2f(shape(41, 0), shape(41, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(42, 0), shape(42, 1)), cv::Point2f(shape(47, 0), shape(47, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(30, 0), shape(30, 1)), cv::Point2f(shape(35, 0), shape(35, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(48, 0), shape(48, 1)), cv::Point2f(shape(59, 0), shape(59, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(60, 0), shape(60, 1)), cv::Point2f(shape(67, 0), shape(67, 1)), Scalar(0,255,0));
    cv::imshow("show image", image);
    cv::waitKey(0);
}

void DrawPredictedImageContinue(cv::Mat image, cv::Mat_<float>& ishape){
    if ( ishape.rows <= 5 ){
        for (int i = 0; i < ishape.rows; i++){
            cv::circle(image, cv::Point2f(ishape(i, 0), ishape(i, 1)), 5, Scalar(255,255,255));
        }
        cv::imshow("show image", image);
        char c = cv::waitKey( 10);
        return;
    }
    Mat_<float> shape = reConvertShape(ishape);
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, Scalar(255,255,255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!= 60 && i!=68 && i!=69)
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), Scalar(0,255,0));
    }
    cv::line(image, cv::Point2f(shape(36, 0), shape(36, 1)), cv::Point2f(shape(41, 0), shape(41, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(42, 0), shape(42, 1)), cv::Point2f(shape(47, 0), shape(47, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(30, 0), shape(30, 1)), cv::Point2f(shape(35, 0), shape(35, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(48, 0), shape(48, 1)), cv::Point2f(shape(59, 0), shape(59, 1)), Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(60, 0), shape(60, 1)), cv::Point2f(shape(67, 0), shape(67, 1)), Scalar(0,255,0));
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
    std::string neg_file_names = "/Users/xujiajun/developer/dataset/helen/test_jpgs.txt.no";
    LoadImages(images, ground_truth_shapes, ground_truth_faces, bboxes, file_names, neg_file_names);
    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	for (int i = 0; i < images.size(); i++){
        cv::Mat_<float> current_shape = cas_load.params_.mean_shape_.clone(); //ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
        //struct timeval t1, t2;
        //gettimeofday(&t1, NULL);
        int is_face = 1;
        float score = 0;
        float variance = 0;
        cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, bboxes[i], is_face, score);//, ground_truth_shapes[i]);

        //cout << res << std::endl;
        //cout << res - ground_truth_shapes[i] << std::endl;
        //float err = CalculateError(grodund_truth_shapes[i], res);
        cout << "first score: " << score  << std::endl;
        cv::Rect faceRec;
        faceRec.x = bboxes[i].start_x;
        faceRec.y = bboxes[i].start_y;
        faceRec.width = bboxes[i].width;
        faceRec.height = bboxes[i].height;
        cv::rectangle(images[i], faceRec, (255), 1);
        
        DrawPredictedImage(images[i], res);

        
        float scale = 1.1;
        float shuffle = 0.1;
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
                    int is_face = 1;
                    cv::Mat_<float> current_shape = cas_load.params_.mean_shape_.clone(); //ReProjection(cas_load.params_.mean_shape_, box);
                    float score = 0;
                    cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, box, is_face, score);
                    if ( is_face == 1){
                        std::cout << "score:" << score << std::endl;
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

void Test2(const char* ModelName){
    CascadeRegressor cas_load;
    cas_load.LoadCascadeRegressor(ModelName);
    std::vector<cv::Mat_<uchar> > images;
    std::vector<cv::Mat_<float> > ground_truth_shapes;
    std::vector<int> ground_truth_faces;
    std::vector<BoundingBox> bboxes;
    std::string file_names = "/Users/xujiajun/developer/dataset/helen/test2_jpgs.txt"; //"./../dataset/helen/train_jpgs.txt";
    std::string neg_file_names = "/Users/xujiajun/developer/dataset/helen/test_jpgs.txt.no";
    LoadImages(images, ground_truth_shapes, ground_truth_faces, bboxes, file_names, neg_file_names);
    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (int i = 0; i < images.size(); i++){
        cv::Mat_<float> current_shape = cas_load.params_.mean_shape_.clone(); //ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
        //struct timeval t1, t2;
        //gettimeofday(&t1, NULL);
        int is_face = 1;
        float score = 0;
        cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, bboxes[i], is_face, score);//, ground_truth_shapes[i]);
        
        //cout << res << std::endl;
        //cout << res - ground_truth_shapes[i] << std::endl;
        //float err = CalculateError(grodund_truth_shapes[i], res);
        cout << "first score: " << score  << std::endl;
        cv::Rect faceRec;
        faceRec.x = bboxes[i].start_x;
        faceRec.y = bboxes[i].start_y;
        faceRec.width = bboxes[i].width;
        faceRec.height = bboxes[i].height;
        cv::rectangle(images[i], faceRec, (255), 1);
        
        DrawPredictedImage(images[i], res);
        
        
        float scale = 1.1;
        float shuffle = 0.1;
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
                    int is_face = 1;
                    cv::Mat_<float> current_shape = cas_load.params_.mean_shape_.clone();// ReProjection(cas_load.params_.mean_shape_, box);
                    float score = 0;
                    cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, box, is_face, score);
                    if ( is_face == 1){
                        std::cout << "score:" << score << std::endl;
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
    rg.antiJitter = 1;
//    rg.params_.predict_group_.erase(0);
//    for (int i = 0; i < rg.params_.regressor_stages_; i++){
//        rg.regressors_[i].params_ = rg.params_;
//    }
    
    rg.params_.predict_regressor_stages_ = 4;
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
        cvtColor(frame, image, COLOR_BGR2GRAY);
        
        struct timeval t1, t2;
        float timeuse;
        if ( false ){
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
            
    //        gettimeofday(&t2, NULL);
    //        cout << faces.size() << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0  << endl;
    //        
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
                int is_face = 1;
                float score = 0;
                cv::Mat_<float> res = rg.Predict(image, current_shape, bbox, is_face, score);//, ground_truth_shapes[i]);
                gettimeofday(&t2, NULL);
                if ( is_face != 1 ) continue;
    //            last_shape = res.clone(); lastShaped = true;
                cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " isface" << is_face << " score:" << score << endl;
                
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
        else{ //用新的方法
//            gettimeofday(&t1, NULL);
//            std::vector<cv::Mat_<float>> shapes;
//            std::vector<cv::Rect> rects = rg.detectMultiScale(image, shapes, 1.1, 2, 0|CASCADE_FLAG_TRACK_MODE, 150);
//            gettimeofday(&t2, NULL);
//            //            last_shape = res.clone(); lastShaped = true;
//            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " faces:" << rects.size() <<  endl;
//            if ( rects.size() == 0 ) continue;
//            //            for ( int d=0; d<rg.params_.predict_regressor_stages_; d++){
//            //                cout << rg.stage_delta_[d] << " ";
//            //                cv::line(frame, Point2f(20*d+20, 450), Point2f(20*d+20, 450 - 200000*rg.stage_delta_[d]), (255));
//            //            }
//            //            cout << endl;
//            for ( int c=0; c<rects.size(); c++){
//                cv::rectangle(frame, rects[c], (255), 1);
//                DrawPredictedImageContinue(frame, shapes[c]);
//            }
            gettimeofday(&t1, NULL);
            cv::Mat_<float> shape;
            cv::Rect rect;
            bool ret = rg.detectOne(image, rect, shape);
            gettimeofday(&t2, NULL);
            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " faces:" << ret <<  endl;
            cv::rectangle(frame, rect, (255), 1);
            if ( ret ){
//                DrawImageNoShowOrientation(frame, shape, CASCADE_ORIENT_TOP_LEFT);
                DrawPredictedImageContinue(frame, shape);
            }
        }
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

//    cv::imshow("show image", image);
//    cv::waitKey(0);
//    cvtColor(image,image,COLOR_BGRA2BGR);
//    cv::imshow("show image", image);
//    cv::waitKey(0);
//    cv::Mat_<uchar> imageyuv;
//    cvtColor(image, imageyuv, COLOR_BGR2YUV_YV12);
//    std::cout << (int)imageyuv(0,2) << std::endl;
//    cv::imshow("show image", imageyuv);
//    cv::waitKey(0);

    struct timeval t1, t2;
    float timeuse;
    gettimeofday(&t1, NULL);
    haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(100, 100));
    gettimeofday(&t2, NULL);
    cout << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " faces:" << faces.size() << endl;
    for (int i = 0; i < faces.size(); i++){
        cv::Rect faceRec = faces[i];
        BoundingBox bbox;
        bbox.start_x = faceRec.x;
        bbox.start_y = faceRec.y+50;
        bbox.width = faceRec.width-50;
        bbox.height = faceRec.height;
        bbox.center_x = bbox.start_x + bbox.width / 2.0;
        bbox.center_y = bbox.start_y + bbox.height / 2.0;
        cv::Mat_<float> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
        //cv::Mat_<float> tmp = image.clone();
        //DrawPredictedImage(tmp, current_shape);
        gettimeofday(&t1, NULL);
        int is_face = 1;
        float score = 0;
        cv::Mat_<float> res = rg.Predict(image, current_shape, bbox, is_face, score);//, ground_truth_shapes[i]);
        gettimeofday(&t2, NULL);
        cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " isface:" << is_face << " score:" << score << endl;
        
        cv::Mat_<uchar> img = image.clone();
        cv::rectangle(img, faceRec, (255), 1);
        //cv::imshow("show image", image);
        //cv::waitKey(0);
        DrawPredictedImage(img, res);
        break;
    }
    

    float scale = 1.1;
    float shuffle = 0.1;
    int minSize = 50;
    int order = -1;
    int currentSize;
    bool biggest_only;
    int faceFound=0, nonface=0;

    if ( order == 1 ){
        currentSize = minSize;
    }
    else{
        currentSize = std::min(image.cols, image.rows);
    }

    while ( currentSize >= minSize && currentSize <= std::min(image.cols, image.rows)){
        for ( int ix=0; ix<image.cols-currentSize; ix+= currentSize*shuffle){
            for ( int jy=0; jy<image.rows-currentSize; jy+=currentSize*shuffle){
                BoundingBox box;
                box.start_x = ix;
                box.start_y = jy;
                box.width = currentSize;
                box.height = currentSize;
                box.center_x = box.start_x + box.width/2.0;
                box.center_y = box.start_y + box.width/2.0;
                int is_face = 1;
                cv::Mat_<float> current_shape = ReProjection(rg.params_.mean_shape_, box);
                float score = 0;
                cv::Mat_<float> res = rg.Predict(image, current_shape, box, is_face, score);
                if ( is_face == 1){
                    std::cout << "score:" << score << std::endl;
                    cv::Mat_<uchar> img = image.clone();
                    cv::Rect rect;
                    rect.x = box.start_x;
                    rect.y = box.start_y;
                    rect.width = box.width;
                    rect.height = box.height;
                    cv::rectangle(img, rect, (255), 1);
                    DrawPredictedImage(img, res);
                }
                else if ( is_face < 0 ){
                    std::cout<< "isface:" << is_face << std::endl;
                    cv::Mat_<uchar> img = image.clone();
                    cv::Rect rect;
                    rect.x = box.start_x;
                    rect.y = box.start_y;
                    rect.width = box.width;
                    rect.height = box.height;
                    cv::rectangle(img, rect, (255,0,0), 3 );
                    DrawPredictedImageContinue(img, res);
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
    std::string neg_file_names = "/Users/xujiajun/developer/dataset/helen/train_neg_jpgs.txt";
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
    NUM_LANDMARKS = 68;
    
	int pos_num = LoadImages(images, ground_truth_shapes, ground_truth_faces, bboxes, file_names, neg_file_names);
	params.mean_shape_ = GetMeanShape(ground_truth_shapes, ground_truth_faces, bboxes);
    
    //检查一下各种数算得对不对
//    for ( int i=0; i<images.size(); i++){
//        cv::Rect rect;
//        rect.x = bboxes[i].start_x;
//        rect.y = bboxes[i].start_y;
//        rect.width = bboxes[i].width;
//        rect.height = bboxes[i].height;
//        cv::rectangle(images[i], rect, Scalar(255));
//        cv::Mat_<float> shape = ReProjection(params.mean_shape_, bboxes[i]);
//        DrawPredictedImage(images[i], shape);
//        DrawPredictedImage(images[i], ground_truth_shapes[i]);
//    }
    
    params.local_features_num_ = 8000;
	params.landmarks_num_per_face_ = NUM_LANDMARKS;
    params.regressor_stages_ = 5;
//    params.local_radius_by_stage_.push_back(0.6);
//    params.local_radius_by_stage_.push_back(0.5);
	params.local_radius_by_stage_.push_back(0.45);
    params.local_radius_by_stage_.push_back(0.3);
    params.local_radius_by_stage_.push_back(0.2);
	params.local_radius_by_stage_.push_back(0.1);//0.1
    params.local_radius_by_stage_.push_back(0.08);//0.08
    params.local_radius_by_stage_.push_back(0.08);
    params.local_radius_by_stage_.push_back(0.04);
    params.local_radius_by_stage_.push_back(0.03);

//    params.local_radius_by_stage_.push_back(0.2);
//    params.local_radius_by_stage_.push_back(0.15);
//    params.local_radius_by_stage_.push_back(0.1);
//    params.local_radius_by_stage_.push_back(0.8);
//    params.local_radius_by_stage_.push_back(0.05);
//    params.local_radius_by_stage_.push_back(0.03);
    
    params.detect_factor_by_stage_.push_back(0.6);
    params.detect_factor_by_stage_.push_back(0.5);
    params.detect_factor_by_stage_.push_back(0.5);
    params.detect_factor_by_stage_.push_back(0.4);
    params.detect_factor_by_stage_.push_back(0.4);
    params.detect_factor_by_stage_.push_back(0.6);
    params.detect_factor_by_stage_.push_back(0.4);
    params.detect_factor_by_stage_.push_back(0.2);
    
    params.tree_depth_ = 3;
    params.trees_num_per_forest_ = 5;
    params.initial_guess_ = 2;

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
    struct timeval t1, t2;
    int length = 100000;
    short sss[length];
    int   iii[length];
    float fff[length];
    double ddd[length];
    short ssum=0;
    int isum=0;
    float fsum=0;
    double dsum=0;
    for  ( int i=0; i<length; i++ ){
        sss[i] = (short)i;
        iii[i] = i;
        fff[i] = (float)i;
        ddd[i] = (double)i;
    }
    
    gettimeofday(&t1, NULL);
    for ( int i=0; i<length; i++ ){
        ssum += sss[i];
    }
    gettimeofday(&t2, NULL);
    cout << "time1: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " sum:" << ssum << endl;
    
    gettimeofday(&t1, NULL);
    for ( int i=0; i<length; i++ ){
        isum += iii[i];
    }
    gettimeofday(&t2, NULL);
    cout << "time2: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " sum:" << isum << endl;
    
    gettimeofday(&t1, NULL);
    for ( int i=0; i<length; i++ ){
        fsum += fff[i];
    }
    gettimeofday(&t2, NULL);
    cout << "time3: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " sum:" << fsum << endl;
    
    gettimeofday(&t1, NULL);
    for ( int i=0; i<length; i++ ){
        dsum += ddd[i];
    }
    gettimeofday(&t2, NULL);
    cout << "time4: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " sum:" << dsum << endl;
    
    
    cv::Mat_<uchar> img = cv::imread("/Users/xujiajun/developer/face3000/face3000/tang.jpg", 0);
    int cols = img.cols;
    int rows = img.rows;
    uchar *idata = img.data;
    int step = img.step;
    short sum;
    sum = 0;
    gettimeofday(&t1, NULL);
    for ( int i=0; i<rows; i++ ){
        for ( int j=0; j<cols; j++){
            sum += img(i,j);
        }
    }
    
    gettimeofday(&t2, NULL);
    cout << "sum of image mat: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " sum:" << sum << endl;
    
    sum = 0;
    gettimeofday(&t1, NULL);
    for ( int i=0; i<rows; i++ ){
        int p = i*cols;
        uchar *pt = idata + p;
        for ( int j=0; j<cols; j++){
            sum += *(pt + j);
        }
    }
    
    gettimeofday(&t2, NULL);
    cout << "sum of image pointer: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " sum:" << sum << endl;
    
//    int modellen = 2000;
//    int stride = 16;
//    int dim = 68;
//    float x[modellen][dim], y[dim];
//    for ( int i=0; i<dim; i++ ){
//        for (int j=0; j<modellen; j++ ){
//            x[j][i] = i*j;
//            y[i] = 0;
//        }
//    }
//    int rn = 10000;
//    float sum;
//    
//
//    gettimeofday(&t1, NULL);
//    for ( int r=0; r<rn; r++){
//        for ( int i=0; i<dim; i++){
//            for (int j=0; j<modellen; j+= stride ){
//                y[i] += x[j][i];
//            }
//        }
//    }
//    gettimeofday(&t2, NULL);
//    sum = 0;
//    for ( int i=0; i<dim; i++){
//        sum += y[i];
//    }
//    cout << sum << endl;
//    cout << "time1: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
//
//
//    float xx[modellen][dim], yy[dim];
//    for (int j=0; j<modellen; j++){
//        for ( int i=0; i<dim; i++ ){
//            xx[j][i] = i*j;
//            yy[i] = 0;
//        }
//    }
//
//    float ssum;
//
//    gettimeofday(&t1, NULL);
//
//    for ( int r=0; r<rn; r++){
//        for (int j=0; j<modellen; j+=stride){
//            for ( int i=0; i<dim; i++){
//                yy[i] += xx[j][i];
//            }
//        }
//    }
//    gettimeofday(&t2, NULL);
//    ssum = 0;
//
//    for ( int i=0; i<dim; i++){
//        ssum += yy[i];
//    }
//    cout << ssum << endl;
//    cout << "time2: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
}



//给一个目录，循环处理子目录下的图片，文件名改为子目录-子目录-数字编号，用其他识别器，把人脸抹掉
std::vector<std::string>  List(const char *path) {
    std::vector<std::string> fileLists;
    struct dirent* ent = NULL;
    DIR *pDir;
    pDir = opendir(path);
    if (pDir == NULL) {
        //被当作目录，但是执行opendir后发现又不是目录，比如软链接就会发生这样的情况。
        return fileLists;
    }
    while (NULL != (ent = readdir(pDir))) {
        if (ent->d_type == 8) {
            //file
//            for (int i = 0; i < level; i++) {
//                printf("-");
//            }
            if ( ent->d_name[0] == '.' ) continue;
 //           printf("%s/%s\n", path, ent->d_name);
            std::string _path(path);
            std::string _fileName(ent->d_name);
            std::string fullFilePath = _path + "/" + _fileName;
//            std::cout << fullFilePath << std::endl;
            fileLists.push_back(fullFilePath);
        } else {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0 ) {
                continue;
            }
            //directory
            std::string _path(path);
            std::string _dirName(ent->d_name);
            std::string fullDirPath = _path + "/" + _dirName;
//            for (int i = 0; i < level; i++) {
//                printf(" ");
//            }
//            printf("%s\n", ent->d_name);
            vector<std::string> l = List(fullDirPath.c_str());
            for ( int i=0; i<l.size(); i++){
                fileLists.push_back(l[i]);
            }
        }
    }
    return fileLists;
}

void GenNeg(const char* path){
    std::string destDir = "/Users/xujiajun/developer/dataset/helen/negset/";
    std::string sourceFile = "/Users/xujiajun/developer/dataset/helen/train_jpgs.txt.full";
    std::string fn_haar = "/Users/xujiajun/developer/dataset/haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    bool yes = haar_cascade.load(fn_haar);
    std::cout << "detector: " << yes << std::endl;
    std::ifstream fin;
    fin.open(sourceFile.c_str(), std::ifstream::in);
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
    
    std::string name;
    //std::cout << name << std::endl;
    int count;
    while (fin >> name){
        //std::cout << "reading file: " << name << std::endl;
        std::cout << name << std::endl;
//        std::string pts = name.substr(0, name.length() - 3) + "pts";
        
        cv::Mat_<uchar> image = cv::imread(("/Users/xujiajun/developer/dataset/helen/" + name).c_str(), 0);
//        cv::Mat_<uchar> image = cv::imread(files[i], 1);
        if (image.cols > 2000){
            cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
        }
        else if (image.cols > 1400 && image.cols <= 2000){
            cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
        }

        std::vector<cv::Rect> faces;
        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(50, 50));
        std::cout << faces.size() << endl;
        for (int i = 0; i < faces.size(); i++){
            cv::Rect faceRec = faces[i];
            cv::rectangle(image, faceRec, Scalar(image(faceRec.y, faceRec.x)), CV_FILLED);
        }
        if ( faces.size() > 0 ){
            std:string dd = "/";
            int pos = name.find(dd);
            while (pos != -1){
                name.replace(pos, dd.length(), std::string("-"));
                pos = name.find(dd);
            }
            cv::imwrite((destDir+name).c_str(), image);
        }
    }
    fin.close();
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
        if (strcmp(argv[1], "test2") == 0)
        {
            std::cout << "enter test\n";
            if (argc == 3){
                Test2(argv[2]);
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
        if (strcmp(argv[1], "genneg") == 0){
            std::cout << "generate negative image\n";
            if ( argc == 3){
                GenNeg(argv[2]);
            }
            return 0;
        }
        if (strcmp(argv[1], "annotate") == 0){
            std::cout << "annotate image\n";
            if ( argc == 3){
                annotate_main(argv[2]);
            }
        }
        if (strcmp(argv[1], "annotate_filter") == 0){
            std::cout << "annotate filter image\n";
            if ( argc == 3){
                annotate_filter(argv[2]);
            }
        }
	}
    else if ( argc == 2){
        if (strcmp(argv[1], "hello") == 0)
        {
            Hello();
            return 0;
        }
        if (strcmp(argv[1], "annotate") == 0){
            std::cout << "annotate image\n";
            annotate_main("");
        }
    }

//    std::cout << "use [./application train ModelName] or [./application test ModelName [image_name]] \n";
	return 0;
}





//TODO LIST
/*
 1、做一个标注工具，自动先对齐，人工校正来标注
 2、做一个生成负例的工具，自动从图片中扣除人脸
 3、看f3000的最新代码，关于validataion那部分的改进
 4、性能调试，做time profile
 5、如何提高训练速度？
 6、如何压缩模型：把回归部分的float改为short？千分之一的精度应该够了。观察一下模型的最大最小值
 7、做成手机上predict用的包
 8、
 
 */
