#include "utils.h"
#include <dlib/config.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#ifndef DLIB_NO_GUI_SUPPORT
#define DLIB_NO_GUI_SUPPORT
#endif
#include <dlib/all/source.cpp>

//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")

// project the global shape coordinates to [-1, 1]x[-1, 1]

dlib::frontal_face_detector fdetector = dlib::get_frontal_face_detector();;

int NUM_LANDMARKS = 29;
int debug_on_ = 0;


void DrawImage(cv::Mat_<uchar> image, cv::Mat_<float>& ishape){
    cv::Mat_<float> shape = reConvertShape(ishape);
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, cv::Scalar(255,255,255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
            cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), cv::Scalar(0,255,0));
    }
    cv::line(image, cv::Point2f(shape(36, 0), shape(36, 1)), cv::Point2f(shape(41, 0), shape(41, 1)), cv::Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(42, 0), shape(42, 1)), cv::Point2f(shape(47, 0), shape(47, 1)), cv::Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(30, 0), shape(30, 1)), cv::Point2f(shape(35, 0), shape(35, 1)), cv::Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(48, 0), shape(48, 1)), cv::Point2f(shape(59, 0), shape(59, 1)), cv::Scalar(0,255,0));
    cv::line(image, cv::Point2f(shape(60, 0), shape(60, 1)), cv::Point2f(shape(67, 0), shape(67, 1)), cv::Scalar(0,255,0));
    cv::imshow("show image", image);
    cv::waitKey(0);
}

void DrawImageNoShow(cv::Mat image, cv::Mat_<float>& ishape){
    cv::Mat_<float> shape = reConvertShape(ishape);
    for (int i = 0; i < shape.rows; i++){
        cv::circle(image, cv::Point2f(shape(i, 1), shape(i, 0)), 2, cv::Scalar(255,255,255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
            cv::line(image, cv::Point2f(shape(i-1, 1), shape(i-1, 0)), cv::Point2f(shape(i, 1), shape(i, 0)), cv::Scalar(255,255,0));
    }
    cv::line(image, cv::Point2f(shape(36, 1), shape(36, 0)), cv::Point2f(shape(41, 1), shape(41, 0)), cv::Scalar(255,255,255));
    cv::line(image, cv::Point2f(shape(42, 1), shape(42, 0)), cv::Point2f(shape(47, 1), shape(47, 0)), cv::Scalar(255,255,255));
    cv::line(image, cv::Point2f(shape(30, 1), shape(30, 0)), cv::Point2f(shape(35, 1), shape(35, 0)), cv::Scalar(255,255,255));
    cv::line(image, cv::Point2f(shape(48, 1), shape(48, 0)), cv::Point2f(shape(59, 1), shape(59, 0)), cv::Scalar(255,255,255));
    cv::line(image, cv::Point2f(shape(60, 1), shape(60, 0)), cv::Point2f(shape(67, 1), shape(67, 0)), cv::Scalar(255,255,255));
}

void DrawImageNoShowOrientation(cv::Mat image, cv::Mat_<float>& ishape, int orient){
    cv::Mat_<float> shape = reConvertShape(ishape);
    if ( orient == CASCADE_ORIENT_TOP_LEFT){
        for (int i = 0; i < shape.rows; i++){
            cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, cv::Scalar(255,255,255));
            if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
                cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), cv::Scalar(255,255,0));
        }
        cv::line(image, cv::Point2f(shape(36, 0), shape(36, 1)), cv::Point2f(shape(41, 0), shape(41, 1)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(42, 0), shape(42, 1)), cv::Point2f(shape(47, 0), shape(47, 1)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(30, 0), shape(30, 1)), cv::Point2f(shape(35, 0), shape(35, 1)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(48, 0), shape(48, 1)), cv::Point2f(shape(59, 0), shape(59, 1)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(60, 0), shape(60, 1)), cv::Point2f(shape(67, 0), shape(67, 1)), cv::Scalar(255,255,255));
    }
    else if ( orient == CASCADE_ORIENT_TOP_RIGHT ){
        for (int i = 0; i < shape.rows; i++){
            cv::circle(image, cv::Point2f(shape(i, 1), shape(i, 0)), 2, cv::Scalar(255,255,255));
            if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
                cv::line(image, cv::Point2f(shape(i-1, 1), shape(i-1, 0)), cv::Point2f(shape(i, 1), shape(i, 0)), cv::Scalar(255,255,0));
        }
        cv::line(image, cv::Point2f(shape(36, 1), shape(36, 0)), cv::Point2f(shape(41, 1), shape(41, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(42, 1), shape(42, 0)), cv::Point2f(shape(47, 1), shape(47, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(30, 1), shape(30, 0)), cv::Point2f(shape(35, 1), shape(35, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(48, 1), shape(48, 0)), cv::Point2f(shape(59, 1), shape(59, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(shape(60, 1), shape(60, 0)), cv::Point2f(shape(67, 1), shape(67, 0)), cv::Scalar(255,255,255));
    }
    else if ( orient == CASCADE_ORIENT_BOTTOM_LEFT ){
        int w = image.cols;
        for (int i = 0; i < shape.rows; i++){
            cv::circle(image, cv::Point2f(w - shape(i, 1), shape(i, 0)), 2, cv::Scalar(255,255,255));
            if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
                cv::line(image, cv::Point2f(w - shape(i-1, 1), shape(i-1, 0)), cv::Point2f(w - shape(i, 1), shape(i, 0)), cv::Scalar(255,255,0));
        }
        cv::line(image, cv::Point2f(w - shape(36, 1), shape(36, 0)), cv::Point2f(w - shape(41, 1), shape(41, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(w - shape(42, 1), shape(42, 0)), cv::Point2f(w - shape(47, 1), shape(47, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(w - shape(30, 1), shape(30, 0)), cv::Point2f(w - shape(35, 1), shape(35, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(w - shape(48, 1), shape(48, 0)), cv::Point2f(w - shape(59, 1), shape(59, 0)), cv::Scalar(255,255,255));
        cv::line(image, cv::Point2f(w - shape(60, 1), shape(60, 0)), cv::Point2f(w - shape(67, 1), shape(67, 0)), cv::Scalar(255,255,255));
    }
}


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

void getEulerAngles(cv::Mat &rotCameraMatrix,cv::Vec3d &eulerAngles){
    
    cv::Mat cameraMatrix,rotMatrix,transVect,rotMatrixX,rotMatrixY,rotMatrixZ;
    double* _r = rotCameraMatrix.ptr<double>();
    double projMatrix[12] = {_r[0],_r[1],_r[2],0,
        _r[3],_r[4],_r[5],0,
        _r[6],_r[7],_r[8],0};
    
    decomposeProjectionMatrix( cv::Mat(3,4,CV_64FC1,projMatrix),
                              cameraMatrix,
                              rotMatrix,
                              transVect,
                              rotMatrixX,
                              rotMatrixY,
                              rotMatrixZ,
                              eulerAngles);
}

cv::Vec3d getShapeEulerAngles(cv::Mat_<float>& shape, BoundingBox& box)
{
    cv::Vec3d eulerAngles;
    // 2D image points. If you change the image, you need to change vector
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d(shape(30, 1), shape(30, 0)) );    // Nose tip
    image_points.push_back( cv::Point2d(shape(16, 1), shape(16, 0)) );    // Chin
    image_points.push_back( cv::Point2d(shape(36, 1), shape(36, 0)) );     // Left eye left corner
    image_points.push_back( cv::Point2d(shape(45, 1), shape(45, 0)) );    // Right eye right corner
    image_points.push_back( cv::Point2d(shape(48, 1), shape(48, 0)) );    // Left Mouth corner
    image_points.push_back( cv::Point2d(shape(54, 1), shape(54, 0)) );    // Right mouth corner
    
    // 3D model points.
    static std::vector<cv::Point3d> model_points;
    if ( model_points.size() == 0 ){
        model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
        model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
        model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
        model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
        model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
        model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner
    }
    
    // Camera internals
    double focal_length = 2*box.width; // Approximate focal length.
    cv::Point2d center = cv::Point2d(box.center_x,box.center_y);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    
    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    
    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
    
    
    // Project a 3D point (0, 0, 1000.0) onto the image plane.
    // We use this to draw a line sticking out of the nose
    
//    std::vector<cv::Point3d> nose_end_point3D;
//    std::vector<cv::Point2d> nose_end_point2D;
//    nose_end_point3D.push_back(cv::Point3d(0,0,0.7*w));
//    
//    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
//    
//    
//    for(int i=0; i < image_points.size(); i++)
//    {
//        circle(image, image_points[i], 3, cv::Scalar(0,0,255), -1);
//    }
//    
//    cv::line(image,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
    
    cv::Mat rotCameraMatrix;
    Rodrigues(rotation_vector, rotCameraMatrix);
    getEulerAngles(rotCameraMatrix, eulerAngles);
    std::cout << "yaw:" << eulerAngles[1] << "   pitch:" << eulerAngles[0] << "    roll:" << eulerAngles[2] <<  std::endl;
    return eulerAngles;
}

std::vector<cv::Mat_<float>> GetCategoryMeanShapes(std::vector<cv::Mat_<float> >& all_shapes, std::vector<int>& ground_truth_faces, std::vector<int> & ground_truth_categorys,
                             std::vector<BoundingBox>& all_bboxes)
{
    std::vector<cv::Mat_<float>> category_mean_shapes;
    cv::Vec3d eangle;
    int countFront = 0, countLeft = 0, countRight = 0, countMouth = 0;
    cv::Mat_<float> mean_shape_front = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
    cv::Mat_<float> mean_shape_left = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
    cv::Mat_<float> mean_shape_right = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
    cv::Mat_<float> mean_shape_mouth = cv::Mat::zeros(all_shapes[0].rows, 2, CV_32FC1);
    
    for ( int i=0; i< all_shapes.size(); i++){
        ground_truth_categorys[i] = 0;
        if ( ground_truth_faces[i] == 1 ){
            eangle = getShapeEulerAngles(all_shapes[i], all_bboxes[i]);
            double yaw = eangle[1];
            if ( yaw > 40 ){
                ground_truth_categorys[i] = ground_truth_categorys[i] | CASCADE_CATEGORY_RIGHT;
                countRight++;
                mean_shape_right += ProjectShape(all_shapes[i], all_bboxes[i]);
            }
            else if ( yaw < -40 ){
                ground_truth_categorys[i] = ground_truth_categorys[i] | CASCADE_CATEGORY_LEFT;
                countLeft++;
                mean_shape_left += ProjectShape(all_shapes[i], all_bboxes[i]);
            }
            else{
                ground_truth_categorys[i] = ground_truth_categorys[i] | CASCADE_CATEGORY_FRONT;
                countFront++;
                mean_shape_front += ProjectShape(all_shapes[i], all_bboxes[i]);
            }
            float d1 = sqrtf((all_shapes[i](36,0) - all_shapes[i](48,0)) * (all_shapes[i](36,0) - all_shapes[i](48,0)) + (all_shapes[i](36,1) - all_shapes[i](48,1)) * (all_shapes[i](36,1) - all_shapes[i](48,1)));
            float d2 = sqrtf((all_shapes[i](45,0) - all_shapes[i](54,0)) * (all_shapes[i](45,0) - all_shapes[i](54,0)) + (all_shapes[i](45,1) - all_shapes[i](54,1)) * (all_shapes[i](45,1) - all_shapes[i](54,1)));
            float d3 = sqrtf((all_shapes[i](62,0) - all_shapes[i](66,0)) * (all_shapes[i](62,0) - all_shapes[i](66,0)) + (all_shapes[i](62,1) - all_shapes[i](66,1)) * (all_shapes[i](62,1) - all_shapes[i](66,1)));
            if ( d3 > 0.1 * ( d1 + d2) ){
                ground_truth_categorys[i] = ground_truth_categorys[i] | CASCADE_CATEGORY_OPEN_MOUTH;
                countMouth++;
                mean_shape_mouth += ProjectShape(all_shapes[i], all_bboxes[i]);
            }
        }
    }
    mean_shape_front = 1.0 / countFront * mean_shape_front;
    mean_shape_left = 1.0 / countLeft * mean_shape_left;
    mean_shape_right = 1.0 / countRight * mean_shape_right;
    mean_shape_mouth = 1.0 / countMouth * mean_shape_mouth;
    category_mean_shapes.push_back(mean_shape_front);
    category_mean_shapes.push_back(mean_shape_left);
    category_mean_shapes.push_back(mean_shape_right);
    category_mean_shapes.push_back(mean_shape_mouth);
    
    std::cout << "front:" << countFront << "  left:" << countLeft << "   right:" << countRight << "   open mouth:" << countMouth << std::endl;
    
    return category_mean_shapes;
}




void getSimilarityTransformAcc(const cv::Mat_<float>& shape_to,
                            const cv::Mat_<float>& shape_from,
                            cv::Mat_<float>& rotation, float& scale){
    
//    getSimilarityTransform(shape_to, shape_from, rotation, scale);
//    return;
    
    //int table[] = {0,1,6,7,16,17,21,22,26,27,31,33,35,36,39,42,45,48,51,54,57};
    //int table[] = {0,1,6,7,16,17,26,27,33,36,45,48,54};
    int table[] = {0,1,4,5,8,9,12,13,16,17,19,21,22,24,26,27,33,36,45,48,54};
    cv::Mat_<float> from(21, 2);
    cv::Mat_<float> to(21,2);
    for ( int i=0; i<21; i++ ){
        from(i,0) = shape_from(table[i], 0);
        from(i,1) = shape_from(table[i], 1);
        to(i,0) = shape_to(table[i], 0);
        to(i,1) = shape_to(table[i], 1);
    }
    
    rotation = cv::Mat(2, 2, 0.0);
    scale = 0;
    
    // center the data
    float center_x_1 = 0.0;
    float center_y_1 = 0.0;
    float center_x_2 = 0.0;
    float center_y_2 = 0.0;
    for (int i = 0; i < to.rows; i++){
        center_x_1 += to(i, 0);
        center_y_1 += to(i, 1);
        center_x_2 += from(i, 0);
        center_y_2 += from(i, 1);
    }
    center_x_1 /= to.rows;
    center_y_1 /= to.rows;
    center_x_2 /= from.rows;
    center_y_2 /= from.rows;
    
    cv::Mat_<float> temp1 = to;
    cv::Mat_<float> temp2 = from;
    for (int i = 0; i < to.rows; i++){
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
    for (int i = 0; i < to.rows; i++){
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

cv::Mat_<float> LoadGroundTruthShape(const char* name, int& gender){
	int landmarks = 0;
	std::ifstream fin;
	std::string temp;
	fin.open(name, std::fstream::in);
	getline(fin, temp);// read first line
	fin >> temp >> landmarks;
	cv::Mat_<float> shape(landmarks, 2);
	getline(fin, temp); // read '\n' of the second line
	getline(fin, temp); // read third line
	for (int i = 0; i<landmarks; i++){
		fin >> shape(i, 0) >> shape(i, 1);
	}
    getline(fin, temp); //读回车
    getline(fin, temp); //读}
    getline(fin, temp); //如果存在,读性别
    if ( temp == "f" ) gender = -1;
    else if ( temp == "m" ) gender = 1;
    else gender = 0;
    
	fin.close();
//    //add by xujj
//    if (  shape.rows != 68 || shape.cols != 2) return shape; //错误，有调用者去处理
//    
//    cv::Mat_<float> stemp(17, 2);
//    for ( int i=0; i<17; i++){
//        int si;
//        if( i==0){
//            si = 8;
//        }
//        else if ( i % 2 == 0 ){
//            si = 17 - i/2;
//        }
//        else{
//            si = (i-1)/2;
//        }
//        stemp(i,0) = shape(si,0);
//        stemp(i,1) = shape(si,1);
//    }
//    for ( int i=0; i<17; i++){
//        shape(i,0) = stemp(i,0);
//        shape(i,1) = stemp(i,1);
//    }
    
	return shape;
}

int symmetricPoint(int p){
    int sp;
    if ( p <= 16){
        sp = 16 - p;
    }
    if ( p >=17 && p <= 26 ){
        sp = 17 + 26 - p;
    }
    if ( p >= 27 && p <= 30 ){
        sp = p;

    }
    if ( p >= 31 && p <= 35 ){
        sp = 31 + 35 - p;
    }
    if ( p >= 36 && p <= 39 ){
        sp = 36 + 45 - p;
    }
    if ( p >= 40 && p <= 41 ){
        sp = 40 + 47 - p;
    }
    if ( p >= 42 && p <= 45 ){
        sp = 36 + 45 - p;
    }
    if ( p >= 46 && p <= 47 ){
        sp = 40+47-p;
    }
    if ( p >= 48 && p <= 54 ){
        sp = 48+54-p;
    }
    if ( p >= 55 && p<= 59 ){
        sp = 55+59-p;
    }
    if ( p >= 60 && p <= 64 ){
        sp = 60+64-p;
    }
    if ( p >= 65 && p <= 67 ){
        sp = 65+67-p;
    }
    return sp;
}

int adjointPoint(int p){
    int sp;
    if ( p < 15 ) sp = p + 2;
    else if ( p == 15 ) sp = 16;
    else if ( p == 16 ) sp = 15;
    else if ( p == 26 ) sp = 17;
    else if ( p == 35 ) sp = 30;
    else if ( p == 41 ) sp = 46;
    else if ( p == 47 ) sp = 40;
    else if ( p == 59 ) sp = 48;
//    else if ( p == 67 ) sp = 60;
    else if ( p == 61 ) sp = 67;
    else if ( p == 62 ) sp = 66;
    else if ( p == 63 ) sp = 65;
    else if ( p == 65 ) sp = 56;
    else if ( p == 66 ) sp = 57;
    else if ( p == 67 ) sp = 58;
    else sp = p+1;
    return sp;
}

cv::Mat_<float> convertShape(cv::Mat_<float> shape){
    cv::Mat_<float> result(NUM_LANDMARKS,2);
//    int table[] = {17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8};
    int table[] = {0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67};
    int table29[] = {36,37,38,39,40,41,42,43,44,45,46,47,27,30,31,33,35,48,50,52,54,56,58,18,19,20,23,24,25};
    if ( NUM_LANDMARKS == 68 ){
        for ( int i=0; i<68; i++){
            result(i,0) = shape(table[i],0);
            result(i,1) = shape(table[i],1);
        }
    }
    else if ( NUM_LANDMARKS == 29){
        for ( int i=0; i<29; i++){
            result(i,0) = shape(table29[i],0);
            result(i,1) = shape(table29[i],1);
        }
    }
    else if ( NUM_LANDMARKS == 5 ){
        result(0,0) = (shape(37,0) + shape(38,0) + shape(40,0) + shape(41,0))/4;
        result(0,1) = (shape(37,1) + shape(38,1) + shape(40,1) + shape(41,1))/4;
        result(1,0) = (shape(43,0) + shape(44,0) + shape(46,0) + shape(47,0))/4;
        result(1,1) = (shape(43,1) + shape(44,1) + shape(46,1) + shape(47,1))/4;
        result(2,0) = shape(33,0);
        result(2,1) = shape(33,1);
        result(3,0) = shape(48,0);
        result(3,1) = shape(48,1);
        result(4,0) = shape(54,0);
        result(4,1) = shape(54,1);
    }
    return result;
}

cv::Mat_<float> reConvertShape(cv::Mat_<float> shape){
    cv::Mat_<float> result(68,2);
//    int table[] = {17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8};
    int table[] = {0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67};
    int table29[] = {36,37,38,39,40,41,42,43,44,45,46,47,27,30,31,33,35,48,50,52,54,56,58,18,19,20,23,24,25};
    int table5[] = {37,44, 33, 48, 54};
    if ( NUM_LANDMARKS == 68 ){
        for ( int i=0; i<68; i++){
            result(table[i],0) = shape(i,0);
            result(table[i],1) = shape(i,1);
        }
    }
    else if ( NUM_LANDMARKS == 29 ){
        for ( int i=0; i<29; i++){
            result(table29[i],0) = shape(i,0);
            result(table29[i],1) = shape(i,1);
        }
        for ( int i=0; i<18; i++ ){
            result(i,0) = shape(23,0);
            result(i,1) = shape(23,1);
        }
        result(21,0) = shape(25,0);
        result(21,1) = shape(25,1);
        result(22,0) = shape(26,0);
        result(22,1) = shape(26,1);
        result(26,0) = shape(28,0);
        result(26,1) = shape(28,1);
        result(28,0) = shape(12,0);
        result(28,1) = shape(12,1);
        result(29,0) = shape(12,0);
        result(29,1) = shape(12,1);
        result(32,0) = shape(15,0);
        result(32,1) = shape(15,1);
        result(34,0) = shape(15,0);
        result(34,1) = shape(15,1);
        result(49,0) = shape(17,0);
        result(49,1) = shape(17,1);
        result(51,0) = shape(18,0);
        result(51,1) = shape(18,1);
        result(53,0) = shape(19,0);
        result(53,1) = shape(19,1);
        result(55,0) = shape(20,0);
        result(55,1) = shape(20,1);
        result(57,0) = shape(21,0);
        result(57,1) = shape(21,1);
        result(59,0) = shape(22,0);
        result(59,1) = shape(22,1);
        for ( int i=60; i<68; i++ ){
            result(i,0) = 0;
            result(i,1) = 0;
        }
    }
    else if ( NUM_LANDMARKS == 5 ){
        for ( int i=0; i<68; i++ ){
            result(i,0) = 0;
            result(i,1) = 0;
        }
        for ( int i=0; i<5; i++ ){
            result(table5[i],0) = shape(i,0);
            result(table5[i],1) = shape(i,1);
        }
    }
    return result;
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


BoundingBox CalculateBoundingBox(cv::Mat_<float>& shape){
    BoundingBox bbx;
    float left_x = 10000;
    float right_x = 0;
    float top_y = 10000;
    float bottom_y = 0;
    for (int i=0; i < shape.rows;i++){
        if (shape(i,0) < left_x)
            left_x = shape(i,0);
        if (shape(i,0) > right_x)
            right_x = shape(i,0);
        if (shape(i,1) < top_y)
            top_y = shape(i,1);
        if (shape(i,1) > bottom_y)
            bottom_y = shape(i,1);
    }
    bbx.start_x = left_x;
    bbx.start_y = top_y;
    bbx.height  = bottom_y - top_y;
    bbx.width   = right_x - left_x;
    if ( bbx.height > bbx.width){
        float delta = bbx.height - bbx.width;
        bbx.width += delta/2;
        bbx.height -= delta/2;
        bbx.start_x -= delta/4;
        bbx.start_y += delta/4;
    }
    
    if ( bbx.width > bbx.height){
        float delta = bbx.width - bbx.height;
        bbx.width -= delta/2;
        bbx.height += delta/2;
        bbx.start_x += delta/4;
        bbx.start_y -= delta/4;
    }
    //上面有可能越界，需要处理么？
    
    bbx.center_x = bbx.start_x + bbx.width/2.0;
    bbx.center_y = bbx.start_y + bbx.height/2.0;
    return bbx;
}


BoundingBox CalculateBoundingBoxRotation(cv::Mat_<float>& shape, cv::Mat_<float>& rotation){
    BoundingBox bbx = CalculateBoundingBox(shape);
    float factor = 1.0 - std::abs(std::abs(rotation(0,0)) - std::abs(rotation(1,0)));
    factor = 0.05 * factor;
    factor = factor * bbx.width;
    bbx.start_x += factor;
    bbx.start_y += factor;
    bbx.width -= 2*factor;
    bbx.height -= 2*factor;
    return bbx;
}


int LoadImages(std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<float> >& ground_truth_shapes,
    std::vector<int>& ground_truth_faces,
    std::vector<int>& ground_truth_genders,
	//const std::vector<cv::Mat_<float> >& current_shapes,
	std::vector<BoundingBox>& bboxes,
	std::string file_names,
    std::string neg_file_names){
	
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

    time_t current_time;
    current_time = time(0);
    cv::RNG rd(current_time);
    
	//std::cout << name << std::endl;
	while (fin >> name){
		//std::cout << "reading file: " << name << std::endl;
//		std::cout << name << std::endl;
		std::string pts = name.substr(0, name.length() - 3) + "pts";
        
        cv::Mat_<uchar> image = cv::imread(("/Users/xujiajun/developer/dataset/helen/" + name).c_str(), 0);
//        cv::imshow("show image", image);
//        cv::waitKey(0);
        int gender;
        cv::Mat_<float> ground_truth_shape = LoadGroundTruthShape(("/Users/xujiajun/developer/dataset/helen/" + pts).c_str(), gender);
        if ( ground_truth_shape.rows != 68 || ground_truth_shape.cols != 2){
            std::cout<<"error:" << pts << std::endl;
            continue;
        }

//		if (image.cols > 2000){
//			cv::resize(image, image, cv::Size(image.cols / 4, image.rows / 4), 0, 0, cv::INTER_LINEAR);
//			ground_truth_shape /= 4.0;
//		}
//		else if (image.cols > 1500 && image.cols <= 2000){
//			cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
//			ground_truth_shape /= 3.0;
//		}
//        else if (image.cols > 1000 && image.cols <= 1500){
//            cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
//            ground_truth_shape /= 2.0;
//        }
        
        //截取一些做负例
        /*
        BoundingBox obox = CalculateBoundingBox(ground_truth_shape);
        current_time = time(0);
        if ( obox.width > image.cols / 2 ){
            cv::Rect rect;
            rect.width = obox.width * rd.uniform(0.25, 0.55);
            rect.height = rect.width;
            rect.x = obox.start_x + rd.uniform(-rect.width/2.0, obox.width - rect.width/2.0);
            rect.y = obox.start_y + rd.uniform(-rect.height/2.0, obox.height - rect.width/2.0);
            if ( rect.width > 160 && rect.x > 0 && rect.y > 0 && rect.x + rect.width < image.cols && rect.y + rect.height < image.rows ){
                cv::Mat_<uchar> subImage = image(rect);
                std::stringstream ss;
                ss << current_time << rect.width << ".jpg";
                std::string fname;
                ss >> fname;
                std::string file_name = "/Users/xujiajun/developer/dataset/helen/negsub/sub_" + fname;
                imwrite(file_name, subImage);
            }
        }
        */
        
        
        float scale = image.cols / 500.0;
        if ( image.cols > 500 ){
            cv::resize(image, image, cv::Size(image.cols / scale, image.rows / scale), 0, 0, cv::INTER_LINEAR);
            ground_truth_shape /= scale;
        }

//        std::vector<cv::Rect> faces;
//        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0
//                                      |cv::CASCADE_FIND_BIGGEST_OBJECT
//                                      //                                      |cv::CASCADE_DO_ROUGH_SEARCH
//                                      , cv::Size(60, 60));
//        
//        dlib::cv_image<uchar>cimg(image);
//        std::vector<dlib::rectangle> faces;
//        faces = fdetector(cimg);
//        
//        if ( debug_on_ ){
//            if ( faces.size() == 0 ){
//                cv::imshow("no detect", image);
//                cv::waitKey(0);
//            }
//        }
//
//        for (int i = 0; i < faces.size(); i++){
//            cv::Rect faceRec;
//            faceRec.x = faces[i].left();
//            faceRec.y = faces[i].top();
//            faceRec.width = faces[i].right() - faces[i].left();
//            faceRec.height = faces[i].bottom() - faces[i].top();
//            
//            
//            if (ShapeInRect(ground_truth_shape, faceRec)){
            	// check if the detected face rectangle is in the ground_truth_shape
                images.push_back(image);
                ground_truth_shapes.push_back(convertShape(ground_truth_shape));
                ground_truth_faces.push_back(1);
                ground_truth_genders.push_back(gender);
                BoundingBox bbox = CalculateBoundingBox(ground_truth_shape);
//                bbox.start_x = faceRec.x;
//                bbox.start_y = faceRec.y;
//                bbox.width = faceRec.width;
//                bbox.height = faceRec.height;
//                bbox.center_x = bbox.start_x + bbox.width / 2.0;
//                bbox.center_y = bbox.start_y + bbox.height / 2.0;
                bboxes.push_back(bbox);
                pos_num++;
        
//        cv::Mat_<float> temps = convertShape(ground_truth_shape);
//        cv::Vec3d eangle = getShapeEulerAngles(temps, bbox);
//        DrawImage(image, temps);
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
                ground_truth_shapes.push_back(convertShape(flipped_ground_truth_shape));
                ground_truth_faces.push_back(1);
                ground_truth_genders.push_back(gender);
                BoundingBox flipped_bbox;
                flipped_bbox.start_x = image.cols - (bbox.start_x + bbox.width);
                flipped_bbox.start_y = bbox.start_y;
                flipped_bbox.width = bbox.width;
                flipped_bbox.height = bbox.height;
                flipped_bbox.center_x = flipped_bbox.start_x + flipped_bbox.width / 2.0;
                flipped_bbox.center_y = flipped_bbox.start_y + flipped_bbox.height / 2.0;
                bboxes.push_back(flipped_bbox);
                pos_num++;
       
//        cv::Mat_<float> ftemps = convertShape(flipped_ground_truth_shape);
//        cv::Vec3d feangle = getShapeEulerAngles(ftemps, flipped_bbox);
//        DrawImage(image, ftemps);
                count++;
                if (count%100 == 0){
                    std::cout << count << " images loaded\n";
                }
             //   break;
        //    }
        //}
	}
	std::cout << "get " << bboxes.size() << " faces\n";
	fin.close();

    //开始加负例
    int neg_num = 0;
    fin.open(neg_file_names.c_str(), std::ifstream::in);
    while (fin >> name){
//        std::cout << name << std::endl;
        cv::Mat_<uchar> image = cv::imread(("/Users/xujiajun/developer/dataset/helen/" + name).c_str(), 0);
        cv::Mat_<float> ground_truth_shape = ground_truth_shapes[neg_num % pos_num];
        BoundingBox bbox = bboxes[neg_num % pos_num];
        BoundingBox nbbox;
        nbbox.width = std::min(image.cols / 5, image.rows / 5);
        nbbox.height = nbbox.width;
        nbbox.start_x = image.cols - nbbox.width - 20;
        nbbox.start_y = image.rows - nbbox.height - 20;
        nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
        nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
        images.push_back(image);
        ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
        ground_truth_faces.push_back(-1);
        ground_truth_genders.push_back(0);
        bboxes.push_back(nbbox);
        neg_num++;
        
        cv::transpose(image, image); //转置一次图片
        ground_truth_shape = ground_truth_shapes[neg_num % pos_num];
        bbox = bboxes[neg_num % pos_num];
        nbbox.width = std::min(image.cols / 5, image.rows / 5);
        nbbox.height = nbbox.width;
        nbbox.start_x = image.cols - nbbox.width - 20;
        nbbox.start_y = image.rows - nbbox.height - 20;
        nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
        nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
        images.push_back(image);
        ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
        ground_truth_faces.push_back(-1);
        ground_truth_genders.push_back(0);
        bboxes.push_back(nbbox);
        neg_num++;
        
        if ( neg_num >= 4 * pos_num ) break;
    }
    fin.close();
    
    std::cout << "add negative samples:" << neg_num << std::endl;
//    for ( int i=0; i<pos_num; i++){
//        cv::Mat_<uchar> image = images[i];
//        cv::Mat_<float> ground_truth_shape = ground_truth_shapes[i];
//        BoundingBox bbox = bboxes[i];
//        //加负例
//        BoundingBox nbbox;
//        nbbox.width = std::max(image.cols / 4 - 10, image.rows / 4 - 10);
//        nbbox.height = nbbox.width;
//        
//        nbbox.start_x = image.cols - nbbox.width - 50;
//        nbbox.start_y = image.rows - nbbox.height - 50;
//
//        nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
//        nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
//        images.push_back(image);
//        ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
//        ground_truth_faces.push_back(-1);
//        bboxes.push_back(nbbox);
//        
//        { //再添一个比较接近人脸的
//            time_t current_time;
//            current_time = time(0);
//            cv::RNG rd(current_time);
//            BoundingBox nbbox;
//            nbbox.width = bbox.width + rd.uniform(-50, 50);
//            nbbox.height = nbbox.width;
//            
//            nbbox.start_x = 20;
//            nbbox.start_y = 20;
//            
//            nbbox.center_x = nbbox.start_x + nbbox.width / 2.0;
//            nbbox.center_y = nbbox.start_y + nbbox.height / 2.0;
//            images.push_back(image);
//            ground_truth_shapes.push_back(ReProjection(ProjectShape(ground_truth_shape, bbox), nbbox));
//            ground_truth_faces.push_back(-1);
//            bboxes.push_back(nbbox);
//        }
//        
//        //TODO：据说添加4倍左右的负例比较好，还需要继续生成
//    }

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
    else if ( ground_truth_shape.rows == 29 ){
        temp = ground_truth_shape.rowRange(0, 5)-ground_truth_shape.rowRange(6, 11);
    }
    else if ( ground_truth_shape.rows == 5 ) {
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

float CalculateError2(cv::Mat_<float>& ground_truth_shape, cv::Mat_<float>& predicted_shape, int stage, int landmark){
    cv::Mat_<float> temp;
    if ( ground_truth_shape.rows >= 68 ){
        temp = ground_truth_shape.rowRange(36, 41)-ground_truth_shape.rowRange(42, 47);
    }
    else if ( ground_truth_shape.rows == 29 ){
        temp = ground_truth_shape.rowRange(0, 5)-ground_truth_shape.rowRange(6, 11);
    }
    else if ( ground_truth_shape.rows == 5 ) {
        temp = ground_truth_shape.rowRange(0, 1)-ground_truth_shape.rowRange(1, 2);
    }
    //    temp = ground_truth_shape.rowRange(0, 7)-ground_truth_shape.rowRange(9, 16); //add by xujj
    float x =mean(temp.col(0))[0];
    float y = mean(temp.col(1))[1];
    float interocular_distance = sqrt(x*x+y*y);
    float sum = 0;
    float result;
    
    for (int i=0;i<ground_truth_shape.rows;i++){
        sum += norm(ground_truth_shape.row(i)-predicted_shape.row(i));
    }
    result = sum/(ground_truth_shape.rows*interocular_distance);

    sum = norm(ground_truth_shape.row(landmark)-predicted_shape.row(landmark));
    float result2 = sum/interocular_distance;
    if ( result2 < 0.1 || result2 > 0.3 ) result = result2;

//    if ( stage > 1 ){
//        sum = norm(ground_truth_shape.row(landmark)-predicted_shape.row(landmark));
//        float result2 = sum/interocular_distance;
//        if ( result2 > 0.2 && result > 0.1 ){
//            result = result2;
//        }
//    }
    //std::cout << "error2:" << result << std::endl;
    return result;
}


//void DrawPredictImage(cv::Mat_<uchar> image, cv::Mat_<float>& shape){
//	for (int i = 0; i < shape.rows; i++){
//		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
//	}
//	cv::imshow("show image", image);
//	cv::waitKey(0);
//}

void DrawPredictImage(cv::Mat_<uchar> &image, cv::Mat_<float>& ishape){
    cv::Mat_<uchar> temp_image = image.clone();
    cv::Mat_<float> shape = reConvertShape(ishape);
    
    for (int i = 0; i < shape.rows; i++){
        cv::circle(temp_image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, cv::Scalar(255,255,255));
        if ( i > 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!=68 && i!=69)
            cv::line(temp_image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), cv::Scalar(0,255,0));
    }
    cv::line(temp_image, cv::Point2f(shape(36, 0), shape(36, 1)), cv::Point2f(shape(41, 0), shape(41, 1)), cv::Scalar(0,255,0));
    cv::line(temp_image, cv::Point2f(shape(42, 0), shape(42, 1)), cv::Point2f(shape(47, 0), shape(47, 1)), cv::Scalar(0,255,0));
    cv::line(temp_image, cv::Point2f(shape(30, 0), shape(30, 1)), cv::Point2f(shape(35, 0), shape(35, 1)), cv::Scalar(0,255,0));
    cv::line(temp_image, cv::Point2f(shape(48, 0), shape(48, 1)), cv::Point2f(shape(59, 0), shape(59, 1)), cv::Scalar(0,255,0));
    cv::line(temp_image, cv::Point2f(shape(60, 0), shape(60, 1)), cv::Point2f(shape(67, 0), shape(67, 1)), cv::Scalar(0,255,0));
    cv::imshow("show image", temp_image);
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

//int colorDistance(uchar p1, uchar p2){
//    int p = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]);
//    return (int)sqrt(p)/3;
//}
