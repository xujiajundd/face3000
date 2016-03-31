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
//#include <Accelerate/Accelerate.h>

#ifndef DLIB_NO_GUI_SUPPORT
#define DLIB_NO_GUI_SUPPORT
#endif
#include <dlib/config.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include <dlib/all/source.cpp>

//#include <sys/time.h>
//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")
using namespace cv;
using namespace std;
using namespace dlib;

frontal_face_detector detector;
correlation_tracker tracker;
bool faceDetected = false;

void DrawPredictedImage(cv::Mat_<uchar> image, cv::Mat_<float>& shape){
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 )
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), (255));
    }
    cv::imshow("show image", image);
    cv::waitKey(0);
}

void DrawPredictedImageContinue(cv::Mat image, cv::Mat_<float>& shape){
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 )
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), Scalar(0,255,0));
    }
    cv::imshow("show image", image);
    cv::waitKey( 10);
}


void Test(const char* ModelName){
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<float> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
	std::string file_names = "/Users/xujiajun/developer/dataset/helen/test_jpgs.txt"; //"./../dataset/helen/train_jpgs.txt";
	LoadImages(images, ground_truth_shapes, bboxes, file_names);
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	for (int i = 0; i < images.size(); i++){
		cv::Mat_<float> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
        //struct timeval t1, t2;
        //gettimeofday(&t1, NULL);
        cv::Mat_<float> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);

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
    
    std::string fn_haar = "/Users/xujiajun/developer/dataset/haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    bool yes = haar_cascade.load(fn_haar);
    std::cout << "detector: " << yes << std::endl;
    
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
    
//    cv::Mat_<float> last_shape;
//    bool lastShaped = false;
    while (true){
        VideoStream >> frame;
        cvtColor(frame, image, COLOR_RGB2GRAY);

        
        struct timeval t1, t2;
        float timeuse;
        gettimeofday(&t1, NULL);
        std::vector<cv::Rect> faces;
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
            cv::Mat_<float> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
            gettimeofday(&t2, NULL);
//            last_shape = res; lastShaped = true;
            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
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
        cv::Mat_<float> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
        gettimeofday(&t2, NULL);
        cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
        cv::rectangle(image, faceRec, (255), 1);
        //cv::imshow("show image", image);
        //cv::waitKey(0);
        DrawPredictedImage(image, res);
        break;
    }
	return;
}

void Test(const char* ModelName, const char* name){
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
	TestImage(name, cas_load);
    return;
}


void Train(const char* ModelName){
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<float> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
    std::string file_names = "/Users/xujiajun/developer/dataset/helen/train_jpgs.txt";
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
    
	LoadImages(images, ground_truth_shapes, bboxes, file_names);

	Parameters params;
    params.local_features_num_ = 500;
	params.landmarks_num_per_face_ = 17;
    params.regressor_stages_ = 5;
//    params.local_radius_by_stage_.push_back(0.6);
//    params.local_radius_by_stage_.push_back(0.5);
	params.local_radius_by_stage_.push_back(0.4);
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
    
    params.tree_depth_ = 3;
    params.trees_num_per_forest_ = 8;
    params.initial_guess_ = 10;

	params.mean_shape_ = GetMeanShape(ground_truth_shapes, bboxes);
    
//    for ( int i = 1; i < 7; i++ ){
//        char buffer[50];
//        sprintf(buffer, "%s_d%d_n%d", ModelName, i, 1<<(6-i));
//        params.tree_depth_ = i;
//        params.trees_num_per_forest_ = 1<<(6-i);
        CascadeRegressor cas_reg;
        cas_reg.Train(images, ground_truth_shapes, bboxes, params);
        cas_reg.SaveCascadeRegressor(ModelName);
        
//        cout << buffer << endl;
//        cout << "***********************************************" << endl << endl;
//    }
    
	return;
}

void Hello(){
    int dim = 68000;
    float x[dim], y[dim], z[dim];
    for ( int i=0; i<dim; i++ ){
        x[i] = i;
        y[i] = (float)i/2.0;
    }
    int rn = 10;
    
    struct timeval t1, t2;
    /*
    gettimeofday(&t1, NULL);
    float done = 1.0;
    int ione = 1;
    for ( int r=0; r<rn; r++){
//        cblas_saxpy(dim, done, x, ione, y, ione);
        vDSP_vadd(x, 1, y, 1, z, 1, dim);
    }
    gettimeofday(&t2, NULL);
    cout << "time: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    
    gettimeofday(&t1, NULL);
    for ( int r=0; r<rn; r++){
        for ( int i=0; i<dim; i++){
            y[i] += x[i];
        }
    }
    gettimeofday(&t2, NULL);
    cout << "time: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    
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
    time_t current_time;
    current_time = time(0);
    cv::RNG random_generator(current_time);
    int r = random_generator.uniform(0, 10000);
    random_generator.uniform(0.0, 10.0);
    
    struct feature_node* binary_features = new feature_node[1089];
    std::vector<struct model*> linear;
    linear.resize(68);
//    model  linear[68];

    
    for (int i=0; i<68; i++){
        linear[i] = new model;
        float *w = (float *)malloc(1088*8);
        for ( int j=0; j<1088*8; j++){
            w[j] = random_generator.uniform(0.0, 10.0);
        }
        linear[i]->w = w;
    }
 
    gettimeofday(&t1, NULL);

    float sumw[68], sum;
    for ( int k=0; k<10; k++){
        for (int i=0; i<1088; i++){
            for ( int j=0; j<5; j++){
                binary_features[i].index = 8*i;
                binary_features[i].value = 1;
            }
        }
        cv::Mat_<float> predict_result(68,2, 0.0);
        binary_features[1088].index = -1;
        binary_features[1088].value = 0;
        for ( int i=0; i<68; i++){
            int idx;
            const feature_node *lx = binary_features;
            float *w=linear[i]->w;
            float result = 0.0;
            for(; (idx=lx->index)!=-1 && idx < 1088; lx++){
                sumw[i] += w[idx]; //为了这儿减少一次减法，在getglobalfeature的地方改了index的初值为0;
            }
            predict_result(i,0) = sumw[i];
            predict_result(i,1) = sumw[i];
//            for ( int j=0; j<1088; j++){
//                sumw[i] += linear[i].w[binary_features[j].index];
//            }
        }
        
    }

    gettimeofday(&t2, NULL);
    cout << "time sumw1: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    
    
    gettimeofday(&t1, NULL);

    for ( int k=0; k<10; k++){
        for (int i=0; i<1088; i++){
            for ( int j=0; j<5; j++){
                binary_features[i].index = 8*i;
                binary_features[i].value = 1.0;
            }
        }
        for ( int j=0; j<1088; j++){
            for ( int i=0; i<68; i++){
                sumw[i] += linear[i]->w[binary_features[j].index];
            }
        }
    }
    gettimeofday(&t2, NULL);
    cout << "time sumw: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0  << endl;
}

int main(int argc, char* argv[])
{
    detector = get_frontal_face_detector();
    
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
            std::cout << "enter test\n";
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



/*
string fn_haar = "D:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
cv::CascadeClassifier haar_cascade;
bool yes = haar_cascade.load(fn_haar);
cv::Mat img = cv::imread("helen/trainset/103770709_1.jpg");// "helen/trainset/232194_1.jpg");
cv::Mat gray;
float scale = 1.3f;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
Mat smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
equalizeHist(smallImg, smallImg);

std::vector<cv::Rect_<int> > faces;
haar_cascade.detectMultiScale(gray, faces,
1.1, 2, 0, Size(30,30));
printf_s("number of faces: %d\n", faces.size());
for (int i = 0; i < faces.size(); i++)
{
cv::Rect face_i = faces[i];
cv::Rect ret;
ret.x = face_i.x*scale;
ret.y = face_i.y*scale;
ret.width = (face_i.width - 1)*scale;
ret.height = (face_i.height - 1)*scale;
cv::Mat face = gray(face_i);
rectangle(img, face_i, (255, 255, 255), 1);
//rectangle(img, ret, (0, 0, 255), 1);
}
//imshow("ppµÄö¦ÕÕ", img);
//waitKey();

//return 0;

int * pResults = NULL;
pResults = facedetect_frontal((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
1.2f, 2, 24);
printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));
//print the detection results
for (int i = 0; i < (pResults ? *pResults : 0); i++)
{
short * p = ((short*)(pResults + 1)) + 6 * i;
int x = p[0];
int y = p[1];
int w = p[2];
int h = p[3];
int neighbors = p[4];

printf("face_rect=[%d, %d, %d, %d], neighbors=%d\n", x, y, w, h, neighbors);
cv::Rect faceRec(x,y,w,h);
cv::rectangle(img, faceRec, (0, 255, 0), 1);
}


cv::imshow("ppµÄö¦ÕÕ", img);
cv::waitKey();

*/
