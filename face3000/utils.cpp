#include "utils.h"
//#include <dlib/config.h>
//#include <dlib/opencv.h>
//#include <dlib/image_processing/frontal_face_detector.h>
////#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>

//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")

// project the global shape coordinates to [-1, 1]x[-1, 1]

//dlib::frontal_face_detector fdetector = dlib::get_frontal_face_detector();;

cv::Mat_<float> ProjectShape(const cv::Mat_<float>& shape, const BoundingBox& bbox){
	cv::Mat_<float> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = (shape(i, 0) - bbox.center_x) / (bbox.width / 2.0);
		results(i, 1) = (shape(i, 1) - bbox.center_y) / (bbox.height / 2.0);
	}
	return results;
}

// reproject the shape to global coordinates
cv::Mat_<float> ReProjection(const cv::Mat_<float>& shape, const BoundingBox& bbox){
	cv::Mat_<float> results(shape.rows, 2);
	for (int i = 0; i < shape.rows; i++){
		results(i, 0) = shape(i, 0)*bbox.width / 2.0 + bbox.center_x;
		results(i, 1) = shape(i, 1)*bbox.height / 2.0 + bbox.center_y;
	}
	return results;
}

// get the mean shape, [-1, 1]x[-1, 1]
cv::Mat_<float> GetMeanShape(const std::vector<cv::Mat_<float> >& all_shapes, std::vector<int>& ground_truth_faces,
	const std::vector<BoundingBox>& all_bboxes) {

	cv::Mat_<float> mean_shape = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
    int count=0;
	for (int i = 0; i < all_shapes.size(); i++)
	{
        if ( ground_truth_faces[i] == 1 ){
		    mean_shape += ProjectShape(all_shapes[i], all_bboxes[i]);
            count++;
        }
	}
	mean_shape = 1.0 / count*mean_shape;
	return mean_shape;
}

// get the rotation and scale parameters by transferring shape_from to shape_to, shape_to = M*shape_from
void getSimilarityTransform(const cv::Mat_<float>& shape_to,
	const cv::Mat_<float>& shape_from,
	cv::Mat_<float>& rotation, float& scale){
	rotation = cv::Mat(2, 2, 0.0);
	scale = 0;

	// center the data
	float center_x_1 = 0.0;
	float center_y_1 = 0.0;
	float center_x_2 = 0.0;
	float center_y_2 = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		center_x_1 += shape_to(i, 0);
		center_y_1 += shape_to(i, 1);
		center_x_2 += shape_from(i, 0);
		center_y_2 += shape_from(i, 1);
	}
	center_x_1 /= shape_to.rows;
	center_y_1 /= shape_to.rows;
	center_x_2 /= shape_from.rows;
	center_y_2 /= shape_from.rows;

	cv::Mat_<float> temp1 = shape_to.clone();
	cv::Mat_<float> temp2 = shape_from.clone();
	for (int i = 0; i < shape_to.rows; i++){
		temp1(i, 0) -= center_x_1;
		temp1(i, 1) -= center_y_1;
		temp2(i, 0) -= center_x_2;
		temp2(i, 1) -= center_y_2;
	}


	cv::Mat_<float> covariance1, covariance2;
	cv::Mat_<float> mean1, mean2;
	// calculate covariance matrix
    cv::calcCovarMatrix(temp1, covariance1, mean1, cv::COVAR_COLS, CV_32F); //CV_COVAR_COLS
    cv::calcCovarMatrix(temp2, covariance2, mean2, cv::COVAR_COLS, CV_32F);

	float s1 = sqrt(norm(covariance1));
	float s2 = sqrt(norm(covariance2));
	scale = s1 / s2;
	temp1 = 1.0 / s1 * temp1;
	temp2 = 1.0 / s2 * temp2;

	float num = 0.0;
	float den = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
		den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
	}

	float norm = sqrt(num*num + den*den);
	float sin_theta = num / norm;
	float cos_theta = den / norm;
	rotation(0, 0) = cos_theta;
	rotation(0, 1) = -sin_theta;
	rotation(1, 0) = sin_theta;
	rotation(1, 1) = cos_theta;
}

cv::Mat_<float> LoadGroundTruthShape(const char* name){
	int landmarks = 0;
	std::ifstream fin;
	std::string temp;
	fin.open(name, std::fstream::in);
	getline(fin, temp);// read first line
	fin >> temp >> landmarks;
//    landmarks = 17; //add by xujj
	cv::Mat_<float> shape(landmarks, 2);
	getline(fin, temp); // read '\n' of the second line
	getline(fin, temp); // read third line
	for (int i = 0; i<landmarks; i++){
		fin >> shape(i, 0) >> shape(i, 1);
	}
    //add by xujj, 用于校验的点
//    shape(68,0) = (shape(37,0) + shape(38,0) + shape(40, 0) + shape(41, 0))/4.0;
//    shape(68,1) = (shape(37,1) + shape(38,1) + shape(40, 1) + shape(41, 1))/4.0;
//    shape(69,0) = (shape(43,0) + shape(44,0) + shape(46, 0) + shape(47, 0))/4.0;
//    shape(69,1) = (shape(43,1) + shape(44,1) + shape(46, 1) + shape(47, 1))/4.0;
	fin.close();
	return shape;
}

cv::Mat_<float> LoadGroundTruthShapeForDetect(const char* name){
    int landmarks = 0;
    std::ifstream fin;
    std::string temp;
    fin.open(name, std::fstream::in);
    getline(fin, temp);// read first line
    fin >> temp >> landmarks;
    //    landmarks = 17; //add by xujj
    cv::Mat_<float> shape(landmarks, 2);
    getline(fin, temp); // read '\n' of the second line
    getline(fin, temp); // read third line
    for (int i = 0; i<landmarks; i++){
        fin >> shape(i, 0) >> shape(i, 1);
    }
    fin.close();
    cv::Mat_<float> shapeDetect(2,2);
    shapeDetect(0,0) = shape(39,0);
    shapeDetect(0,1) = shape(39,1);
    shapeDetect(1,0) = shape(42,0);
    shapeDetect(1,1) = shape(42,1);
    return shapeDetect;
}

bool ShapeInRect(cv::Mat_<float>& shape, cv::Rect& ret){
	float sum_x = 0.0, sum_y = 0.0;
	float max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;

	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
    if (std::abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 2.0) return false;
    if (std::abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 2.0) return false;
	return true;
}

std::vector<cv::Rect> DetectFaces(cv::Mat_<uchar>& image, cv::CascadeClassifier& classifier){
	std::vector<cv::Rect_<int> > faces;
	classifier.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
	return faces;
}


int LoadImages(std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<float> >& ground_truth_shapes,
    std::vector<int>& ground_truth_faces,
	//const std::vector<cv::Mat_<float> >& current_shapes,
	std::vector<BoundingBox>& bboxes,
	std::string file_names){
	
	// change this function to satisfy your own needs
	// for .box files I just use another program before this LoadImage() function
	// the contents in .box is just the bounding box of a face, including the center point of the box
	// you can just use the face rectangle detected by opencv with a little effort calculating the center point's position yourself.
	// you may use some utils function is this utils.cpp file
	// delete unnecessary lines below, my codes are just an example
	
	std::string fn_haar = "/Users/xujiajun/developer/dataset/haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(fn_haar);
	std::cout << "detector: " << yes << std::endl;
	std::cout << "loading images\n";
	std::ifstream fin;
	fin.open(file_names.c_str(), std::ifstream::in);
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
	int count = 0;
    int pos_num = 0;

	//std::cout << name << std::endl;
	while (fin >> name){
		//std::cout << "reading file: " << name << std::endl;
		std::cout << name << std::endl;
		std::string pts = name.substr(0, name.length() - 3) + "pts";
        
        cv::Mat_<uchar> image = cv::imread(("/Users/xujiajun/developer/dataset/helen/" + name).c_str(), 0);
        cv::Mat_<float> ground_truth_shape = LoadGroundTruthShape(("/Users/xujiajun/developer/dataset/helen/" + pts).c_str());
        

		if (image.cols > 2000){
			cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape /= 3.0;
		}
		else if (image.cols > 1400 && image.cols <= 2000){  
			cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape /= 2.0;
		}

        std::vector<cv::Rect> faces;
//        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(100, 100)); //原来是30
        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0
                                      |cv::CASCADE_FIND_BIGGEST_OBJECT
                                      //                                      |cv::CASCADE_DO_ROUGH_SEARCH
                                      , cv::Size(60, 60));
//        dlib::cv_image<uchar>cimg(image);
//        std::vector<dlib::rectangle> faces;
//        faces = fdetector(cimg);
        
        
        for (int i = 0; i < faces.size(); i++){
            cv::Rect faceRec = faces[i];
//            cv::Rect faceRec;
//            faceRec.x = faces[i].left();
//            faceRec.y = faces[i].top();
//            faceRec.width = faces[i].right() - faces[i].left();
//            faceRec.height = faces[i].bottom() - faces[i].top();
            
            
            if (ShapeInRect(ground_truth_shape, faceRec)){ 
            	// check if the detected face rectangle is in the ground_truth_shape
                images.push_back(image);
                ground_truth_shapes.push_back(ground_truth_shape);
                ground_truth_faces.push_back(1);
                BoundingBox bbox;
                bbox.start_x = faceRec.x;
                bbox.start_y = faceRec.y;
                bbox.width = faceRec.width;
                bbox.height = faceRec.height;
                bbox.center_x = bbox.start_x + bbox.width / 2.0;
                bbox.center_y = bbox.start_y + bbox.height / 2.0;
                bboxes.push_back(bbox);
                pos_num++;
                
//                //加负例
//                BoundingBox nbbox;
//                nbbox.start_x = image.cols * 3 / 4;
//                nbbox.start_y = image.rows * 3 / 4;
//                nbbox.width = image.cols / 4 - 10;
//                nbbox.height = image.rows / 4 - 10;
//                nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
//                nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
//                images.push_back(image);
//                ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
//                ground_truth_faces.push_back(-1);
//                bboxes.push_back(nbbox);

                //翻转图片, add by xujj
                cv::Mat_<uchar> flippedImage;
                flip(image, flippedImage, 1);
                images.push_back(flippedImage);
                
                cv::Mat_<float> flipped_ground_truth_shape(ground_truth_shape.rows, 2);
                for ( int p = 0; p < ground_truth_shape.rows; p++ ){
                    if ( p <= 16){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(16-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(16-p,1);
                    }
                    if ( p >=17 && p <= 26 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(17+26-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(17+26-p,1);
                    }
                    if ( p >= 27 && p <= 30 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(p,1);
                    }
                    if ( p >= 31 && p <= 35 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(31+35-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(31+35-p,1);
                    }
                    if ( p >= 36 && p <= 39 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(36+45-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(36+45-p,1);
                    }
                    if ( p >= 40 && p <= 41 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(40+47-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(40+47-p,1);
                    }
                    if ( p >= 42 && p <= 45 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(36+45-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(36+45-p,1);
                    }
                    if ( p >= 46 && p <= 47 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(40+47-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(40+47-p,1);
                    }
                    if ( p >= 48 && p <= 54 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(48+54-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(48+54-p,1);
                    }
                    if ( p >= 55 && p<= 59 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(55+59-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(55+59-p,1);
                    }
                    if ( p >= 60 && p <= 64 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(60+64-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(60+64-p,1);
                    }
                    if ( p >= 65 && p <= 67 ){
                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(65+67-p, 0);
                        flipped_ground_truth_shape(p,1) = ground_truth_shape(65+67-p,1);
                    }
//                    if ( p >= 68 && p <= 69 ){
//                        flipped_ground_truth_shape(p,0) = image.cols - ground_truth_shape(68+69-p, 0);
//                        flipped_ground_truth_shape(p,1) = ground_truth_shape(68+69-p,1);
//                    }
                }
                ground_truth_shapes.push_back(flipped_ground_truth_shape);
                ground_truth_faces.push_back(1);
                BoundingBox flipped_bbox;
                flipped_bbox.start_x = image.cols - (faceRec.x + faceRec.width);
                flipped_bbox.start_y = faceRec.y;
                flipped_bbox.width = faceRec.width;
                flipped_bbox.height = faceRec.height;
                flipped_bbox.center_x = flipped_bbox.start_x + flipped_bbox.width / 2.0;
                flipped_bbox.center_y = flipped_bbox.start_y + flipped_bbox.height / 2.0;
                bboxes.push_back(flipped_bbox);
                pos_num++;

//                //加负例
//                BoundingBox fbbox;
//                fbbox.start_x = image.cols * 3 / 4;
//                fbbox.start_y = image.rows * 3 / 4;
//                fbbox.width = image.cols / 4 - 10;
//                fbbox.height = image.rows / 4 - 10;
//                fbbox.center_x = fbbox.start_x + fbbox.width / 2.0;
//                fbbox.center_y = fbbox.start_y + fbbox.height / 2.0;
//                images.push_back(flippedImage);
//                ground_truth_shapes.push_back(ReProjection(ProjectShape(flipped_ground_truth_shape, bbox), fbbox));
//                ground_truth_faces.push_back(-1);
//                bboxes.push_back(fbbox);

                count++;
                if (count%100 == 0){
                    std::cout << count << " images loaded\n";
                }
                break;
            }
         }

	}
	std::cout << "get " << bboxes.size() << " faces\n";
	fin.close();

    for ( int i=0; i<pos_num; i++){
        cv::Mat_<uchar> image = images[i];
        cv::Mat_<float> ground_truth_shape = ground_truth_shapes[i];
        BoundingBox bbox = bboxes[i];
        //加负例
        BoundingBox nbbox;
        nbbox.width = std::max(image.cols / 4 - 10, image.rows / 4 - 10);
        nbbox.height = nbbox.width;
        
        nbbox.start_x = image.cols - nbbox.width - 50;
        nbbox.start_y = image.rows - nbbox.height - 50;

        nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
        nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
        images.push_back(image);
        ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
        ground_truth_faces.push_back(-1);
        bboxes.push_back(nbbox);
        
        { //再添一个比较接近人脸的
            time_t current_time;
            current_time = time(0);
            cv::RNG rd(current_time);
            BoundingBox nbbox;
            nbbox.width = bbox.width + rd.uniform(-50, 50);
            nbbox.height = nbbox.width;
            
            nbbox.start_x = 20;
            nbbox.start_y = 20;
            
            nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
            nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
            images.push_back(image);
            ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
            ground_truth_faces.push_back(-1);
            bboxes.push_back(nbbox);
        }
        
        //TODO：据说添加4倍左右的负例比较好，还需要继续生成
    }

    return pos_num;
}


// float CalculateError(cv::Mat_<float>& ground_truth_shape, cv::Mat_<float>& predicted_shape){
// 	cv::Mat_<float> temp;
// 	float sum = 0;
// 	for (int i = 0; i<ground_truth_shape.rows; i++){
// 		sum += norm(ground_truth_shape.row(i) - predicted_shape.row(i));
// 	}
//     return sum / (ground_truth_shape.rows);
// }

float CalculateError(cv::Mat_<float>& ground_truth_shape, cv::Mat_<float>& predicted_shape){
    cv::Mat_<float> temp;
    if ( ground_truth_shape.rows >= 68 ){
        temp = ground_truth_shape.rowRange(36, 41)-ground_truth_shape.rowRange(42, 47);
    }
    else{
        temp = ground_truth_shape.rowRange(0, 1)-ground_truth_shape.rowRange(1, 2);
    }
//    temp = ground_truth_shape.rowRange(0, 7)-ground_truth_shape.rowRange(9, 16); //add by xujj
    float x =mean(temp.col(0))[0];
    float y = mean(temp.col(1))[1];
    float interocular_distance = sqrt(x*x+y*y);
    float sum = 0;
    for (int i=0;i<ground_truth_shape.rows;i++){
        sum += norm(ground_truth_shape.row(i)-predicted_shape.row(i));
    }
    return sum/(ground_truth_shape.rows*interocular_distance);
}



void DrawPredictImage(cv::Mat_<uchar> image, cv::Mat_<float>& shape){
	for (int i = 0; i < shape.rows; i++){
		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	cv::imshow("show image", image);
	cv::waitKey(0);
}

BoundingBox GetBoundingBox(cv::Mat_<float>& shape, int width, int height){
	float min_x = 100000.0, min_y = 100000.0;
	float max_x = -1.0, max_y = -1.0;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);
	}
	BoundingBox bbox;
	float scale = 0.6;
	bbox.start_x = min_x - (max_x - min_x) * (scale - 0.5);
	if (bbox.start_x < 0.0)
	{
		bbox.start_x = 0.0;
	}
	bbox.start_y = min_y - (max_y - min_y) * (scale - 0.5);
	if (bbox.start_y < 0.0)
	{
		bbox.start_y = 0.0;
	}
	bbox.width = (max_x - min_x) * scale * 2.0;
	if (bbox.width >= width){
		bbox.width = width - 1.0;
	}
	bbox.height = (max_y - min_y) * scale * 2.0;
	if (bbox.height >= height){
		bbox.height = height - 1.0;
	}
	bbox.center_x = bbox.start_x + bbox.width / 2.0;
	bbox.center_y = bbox.start_y + bbox.height / 2.0;
	return bbox;
}
