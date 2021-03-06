/*****************************************************************************
 *   Non-Rigid Face Tracking
 ******************************************************************************
 *   by Jason Saragih, 5th Dec 2012
 *   http://jsaragih.org/
 ******************************************************************************
 *   Ch6 of the book "Mastering OpenCV with Practical Computer Vision Projects"
 *   Copyright Packt Publishing 2012.
 *   http://www.packtpub.com/cool-projects-with-opencv/book
 *****************************************************************************/
/*
 annotate: annotation tool
 Jason Saragih (2012)
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "headers.h"


#define MODEL_NAME "mv0904"

//==============================================================================
using namespace cv;
using namespace std;
class annotate{
public:
    int idx;                       //index of image to annotate
    int pidx;                      //index of point to manipulate
    int gender;
    cv::Mat_<float> shape;
    cv::Mat image;
    cv::Mat image_clean;
    //  Mat image;                     //current image to display
    //  Mat image_clean;               //clean image to display
    const char* wname;             //display window name
    vector<string> instructions;   //annotation instructions
    string message;
    std::vector<std::string> lists;
    std::string file_name;
    std::string shape_file_name;
    CascadeRegressor face_detector;
    CascadeRegressor gender_detector;
    bool imageScaled;
    
//    Ptr<FaceRecognizer> model;


    annotate(){
        wname = "Annotate"; idx = 0; pidx = -1; gender = 0;
        message = "";
//        model = createFisherFaceRecognizer();
//        model->load("/Users/xujiajun/developer/face3000/face3000/gender_at.yml");
    }

    int set_current_image(const int cidx = 0){ //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
        pidx = -1;
        idx = cidx;
        gender = 0;
        file_name = lists[idx];
        image = cv::imread(file_name, 1);
        float scale = max(image.cols,image.rows) / 1024.0;
        imageScaled = false;
//        if ( scale > 1.0 || scale < 0.5 ){
            cv::resize(image, image, cv::Size(image.cols / scale, image.rows / scale), 0, 0, cv::INTER_LINEAR);
            imageScaled = true;
//s        }
        shape_file_name  = file_name.substr(0, file_name.length() - 3) + "pts";
        int landmarks = 0;
        std::ifstream fin;
        std::string temp;
        fin.open(shape_file_name, std::fstream::in);
        if ( !fin ){
//            cout << "pts文件不存在" << std::endl;
            std::vector<cv::Mat_<float>> shapes;
            cv::Mat_<uchar> grayImage;
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            cv::Rect rect;
            struct timeval t1, t2;
            gettimeofday(&t1, NULL);
            bool ret = face_detector.detectOne(grayImage, rect, shape, 0);
            gettimeofday(&t2, NULL);
            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << " isface:" << ret << endl;
            if ( ret > 0 ){
//                int gender = gender_detector.detectGender(grayImage, shape);
//                cout << "gender: " << gender << std::endl;
                shape = reConvertShape(shape);
            }
            else{
                //用平均脸在中间
                std::cout << "no faces" << std::endl;
                BoundingBox bbox;
                bbox.width = min(image.cols, image.rows) / 2;
                bbox.height = bbox.width;
                bbox.start_x = image.cols/2 - bbox.width/2;
                bbox.start_y = image.rows/2 - bbox.height/2;
                bbox.center_x = bbox.start_x + bbox.width/2;
                bbox.center_y = bbox.start_y + bbox.height/2;
                shape = reConvertShape(ReProjection(face_detector.params_.mean_shape_, bbox));
            }
        }
        else{
            getline(fin, temp);// read first line
            fin >> temp >> landmarks;
            cv::Mat_<float> s(landmarks, 2);
            getline(fin, temp); // read '\n' of the second line
            getline(fin, temp); // read third line
            for (int i = 0; i<landmarks; i++){
                fin >> s(i, 0) >> s(i, 1);
            }
            shape = s;
            getline(fin, temp);
            getline(fin, temp);
            getline(fin, temp);
            if ( temp == "f" ) gender = -1;
            else if ( temp == "m" ) gender = 1;
            else gender = 0;
            fin.close();
            if ( imageScaled ){
                s /= scale;
            }
        }
        set_clean_image();
        return 1;
    }

    void draw_image(){  //画图片，画各点，选中点高亮
        cv::Mat_<uchar> grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::Mat_<float> cshape = convertShape(shape);
        int gen = gender_detector.detectGender(grayImage, cshape);
        cout << "gender: " << gen << std::endl;
        if ( gen == -1 ) draw_alert("Female");
        if ( gen == 1 ) draw_alert("Male");
        
        for (int i = 0; i < shape.rows; i++){
            if ( i == pidx ){
                cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 4, Scalar(0,0,255));
            }
            else{
                cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 4, Scalar(255,255,255));
            }
            if ( i != 0 && i != 17 && i != 22 && i != 27 && i!= 36 && i != 42 && i!= 48 && i!= 60 && i!=68 && i!=69)
                cv::line(image, cv::Point2f(shape(i-1, 0), shape(i-1, 1)), cv::Point2f(shape(i, 0), shape(i, 1)), Scalar(0,255,0));
        }
        cv::line(image, cv::Point2f(shape(36, 0), shape(36, 1)), cv::Point2f(shape(41, 0), shape(41, 1)), Scalar(0,255,0));
        cv::line(image, cv::Point2f(shape(42, 0), shape(42, 1)), cv::Point2f(shape(47, 0), shape(47, 1)), Scalar(0,255,0));
        cv::line(image, cv::Point2f(shape(30, 0), shape(30, 1)), cv::Point2f(shape(35, 0), shape(35, 1)), Scalar(0,255,0));
        cv::line(image, cv::Point2f(shape(48, 0), shape(48, 1)), cv::Point2f(shape(59, 0), shape(59, 1)), Scalar(0,255,0));
        cv::line(image, cv::Point2f(shape(60, 0), shape(60, 1)), cv::Point2f(shape(67, 0), shape(67, 1)), Scalar(0,255,0));
        set_capture_instructions();
        if ( shape(37,1) > shape(41,1) || shape(38,1)>shape(40,1)){
            draw_alert("Error: left eye!!!!!");
        }
        if ( shape(43,1) > shape(47,1) || shape(44,1) > shape(46,1)){
            draw_alert("Error: right eye!!!!!");
        }
        if ( shape(51,1) > shape(62,1) || shape(62,1)>shape(66,1) || shape(66,1)>shape(57,1) || shape(50,1) > shape(61,1) || shape(61,1)>shape(67,1) || shape(67,1)>shape(58,1) || shape(52,1)>shape(63,1) || shape(63,1)>shape(65,1) || shape(65,1) > shape(56,1)){
//            draw_alert("Error: mouth!!!!!");
        }
        if ( shape(21,0) > shape(22,0)){
//            draw_alert("Error: eyebow!!!!!");
        }
        draw_instructions();
        draw_message();
        
/* 这一段是测试检测性别的, 效果不好
        cv::Mat_<uchar> grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::Rect rect;
        BoundingBox box = CalculateBoundingBox(shape);
        rect.x = box.start_x;
        rect.y = box.start_y;
        rect.width = box.width;
        rect.height = box.height;
        
        float adj = 0.08 * box.width;
        rect.x -= adj;
        rect.y -= 2*adj;
        rect.width += 2*adj;
        rect.height += 2*adj;
        if ( rect.x < 0) rect.x = 0;
        if ( rect.y < 0) rect.y = 0;
        if ( rect.width > image.cols) rect.width = image.cols;
        if ( rect.height > image.rows ) rect.height = image.rows;
        if (( rect.x + rect.width ) > image.cols ) rect.x = image.cols - rect.width - 1;
        if (( rect.y + rect.height) > image.rows ) rect.y = image.rows - rect.height - 1;
        if ( rect.x < 0) rect.x = 0;
        if ( rect.y < 0) rect.y = 0;
        cv::rectangle(image, rect, Scalar(255,255,255));
        cv::Mat face = grayImage(rect);
        cv::Mat face_resized;
        cv::resize(face, face_resized, cv::Size(200,200));
        int prediction = model->predict(face_resized);
        if ( prediction == 0 ){
            draw_alert("Male");
        }
        else if ( prediction == 1){
            draw_alert("Female");
        }
*/
        
        if ( gender == -1 ) draw_alert2("<Female>");
        if ( gender == 1 ) draw_alert2("<Male>");
        cv::imshow(wname, image);
    }

    void save_pts(){ //保存pts到文件
        std::ofstream fout;
        fout.open(shape_file_name, std::fstream::out);
        if ( !fout ){
            std::cout<<"file open fail" << std::endl;
        }
        else{
            //        version: 1
            //        n_points:  68
            //        {
            //        }
            fout << "version: 1" << endl;
            fout << "n_points:  68" << endl;
            fout << "{" << std::endl;
            for (int i=0; i<68; i++ ){
                fout << shape(i,0) << " " << shape(i,1) << std::endl;
            }
            fout << "}" << std::endl;
            if ( gender == -1 ) fout << "f" << std::endl;
            else if ( gender == 1 ) fout << "m" << std::endl;
        }
        fout.close();
        if ( imageScaled ){
            imwrite(file_name, image_clean);
        }
        message = "Save Success!";
    }

    int delete_image(){
        //lists[idx]文件删除，然后挪到下一个或前一个
        std::cout << "delete " << file_name << std::endl;
        remove(file_name.c_str());
        remove(shape_file_name.c_str());
        std::vector<std::string>::iterator it = lists.begin()+idx;
        lists.erase(it);
        if ( lists.size() == 0 ) return -1;
        if ( idx >= lists.size() ) idx = lists.size() - 1;
        set_current_image(idx);
        return idx;
    }

    void set_clean_image(){
        image_clean = image.clone();
    }
    void copy_clean_image(){
        image_clean.copyTo(image);
    }

    void draw_instructions(){
        if(image.empty())return;
        this->draw_strings(image,instructions);
    }

    void draw_alert(std::string alert){
        cv::Size size = getTextSize(alert,FONT_HERSHEY_COMPLEX,1.0f,1,NULL);
        putText(image,alert,cv::Point(image.cols/2 - size.width/2,image.rows - 50),FONT_HERSHEY_COMPLEX,1.0f,
                Scalar(20,20,255),1,CV_AA);
        putText(image,alert,cv::Point(image.cols/2 - size.width/2 + 1,image.rows - 49),FONT_HERSHEY_COMPLEX,1.0f,
                Scalar(20,20,150),1,CV_AA);
    }
    
    void draw_alert2(std::string alert){
        cv::Size size = getTextSize(alert,FONT_HERSHEY_COMPLEX,1.0f,1,NULL);
        putText(image,alert,cv::Point(image.cols/2 - size.width/2,image.rows - 20),FONT_HERSHEY_COMPLEX,1.0f,
                Scalar(20,255,20),1,CV_AA);
        putText(image,alert,cv::Point(image.cols/2 - size.width/2 + 1,image.rows - 19),FONT_HERSHEY_COMPLEX,1.0f,
                Scalar(20,200,20),1,CV_AA);
    }

    void draw_message(){
        cv::Size size = getTextSize(message,FONT_HERSHEY_COMPLEX,1.0f,1,NULL);
        putText(image,message,cv::Point(image.cols/2 - size.width/2, 100),FONT_HERSHEY_COMPLEX,1.0f,
                Scalar(255,0,0),1,CV_AA);
        putText(image,message,cv::Point(image.cols/2 - size.width/2 + 1,101),FONT_HERSHEY_COMPLEX,1.0f,
                Scalar(255,0,0),1,CV_AA);
        message = "";
    }

    void draw_points(){

    }
    void draw_chosen_point(){
        //    if(pidx >= 0)circle(image,data.points[idx][pidx],1,CV_RGB(0,255,0),2,CV_AA);
    }
    void draw_connections(){
        //    int m = data.connections.size();
        //    if(m == 0)this->draw_points();
        //    else{
        //      if(data.connections[m-1][1] < 0){
        //    int i = data.connections[m-1][0];
        //    data.connections[m-1][1] = i;
        //    data.draw_connect(image,idx); this->draw_points();
        //    circle(image,data.points[idx][i],1,CV_RGB(0,255,0),2,CV_AA);
        //    data.connections[m-1][1] = -1;
        //      }else{data.draw_connect(image,idx); this->draw_points();}
        //    }
    }
    void draw_symmetry(){
        //    this->draw_points(); this->draw_connections();
        //    for(int i = 0; i < int(data.symmetry.size()); i++){
        //      int j = data.symmetry[i];
        //      if(j != i){
        //    circle(image,data.points[idx][i],1,CV_RGB(255,255,0),2,CV_AA);
        //    circle(image,data.points[idx][j],1,CV_RGB(255,255,0),2,CV_AA);
        //      }
        //    }
        //    if(pidx >= 0)circle(image,data.points[idx][pidx],1,CV_RGB(0,255,0),2,CV_AA);
    }
    void set_capture_instructions(){
        instructions.clear();
        instructions.push_back(string("Select Key: (") + file_name + string(")"));
        instructions.push_back(string(" p - Previous image"));
        instructions.push_back(string(" n - Next image"));
        instructions.push_back(string(" s - Save annotations"));
        instructions.push_back(string(" d - Delete image and annotations"));
        instructions.push_back(string(" m,f - Set male/female"));
        instructions.push_back(string(" q - Quit"));
    }
    void set_pick_points_instructions(){
        instructions.clear();
        instructions.push_back(string("Pick Points"));
        instructions.push_back(string("q - done"));
    }
    void set_connectivity_instructions(){
        instructions.clear();
        instructions.push_back(string("Pick Connections"));
        instructions.push_back(string("q - done"));
    }
    void set_symmetry_instructions(){
        instructions.clear();
        instructions.push_back(string("Pick Symmetric Points"));
        instructions.push_back(string("q - done"));
    }
    void set_move_points_instructions(){
        instructions.clear();
        instructions.push_back(string("Move Points"));
        instructions.push_back(string("p - next image"));
        instructions.push_back(string("o - previous image"));
        instructions.push_back(string("q - done"));
    }
    void initialise_symmetry(const int index){
        //    int n = data.points[index].size(); data.symmetry.resize(n);
        //    for(int i = 0; i < n; i++)data.symmetry[i] = i;
    }
    void replicate_annotations(const int index){
        //    if((index < 0) || (index >= int(data.points.size())))return;
        //    for(int i = 0; i < int(data.points.size()); i++){
        //      if(i == index)continue;
        //      data.points[i] = data.points[index];
        //    }
    }
    int find_closest_point(const Point2f p,
                           const double thresh = 10.0){
        int n = shape.rows;
        int imin = -1;
        double dmin = 10000000.0;
        for(int i = 0; i < n; i++){
            Point2f ps;
            ps.x = shape(i,0);
            ps.y = shape(i,1);
            double d = norm(p - ps);
            if (d < dmin){
                imin = i;
                dmin = d;
            }
        }
        if (dmin < thresh)
            return imin;
        else
            return -1;
        return 0;
    }

protected:
    void draw_strings(Mat img,
                      const vector<string> &text){
        for(int i = 0; i < int(text.size()); i++)this->draw_string(img,text[i],i+1);
    }
    void draw_string(Mat img,
                     const string text,
                     const int level)
    {
        cv::Size size = getTextSize(text,FONT_HERSHEY_COMPLEX,0.6f,1,NULL);
        size.height += 6;
        putText(img,text,cv::Point(0,level*size.height),FONT_HERSHEY_COMPLEX,0.6f,
                Scalar::all(0),1,CV_AA);
        putText(img,text,cv::Point(1,level*size.height+1),FONT_HERSHEY_COMPLEX,0.6f,
                Scalar::all(255),1,CV_AA);
    }
} annotation;


//==============================================================================
void p_MouseCallback(int event, int x, int y, int flags, void* param)
{
    if(event == CV_EVENT_LBUTTONDOWN){
        if ( annotation.pidx >= 0 ){
            annotation.shape(annotation.pidx, 0) = x;
            annotation.shape(annotation.pidx, 1) = y;
            annotation.pidx = -1;
            annotation.copy_clean_image();
            annotation.draw_image();
        }
        else{
            int imin = annotation.find_closest_point(Point2f(x,y));
            annotation.pidx = imin;
            annotation.copy_clean_image();
            annotation.draw_image();
            //    if(imin >= 0){ //add connection
            //      int m = annotation.data.connections.size();
            //      if(m == 0)annotation.data.connections.push_back(Vec2i(imin,-1));
            //      else{
            //    if(annotation.data.connections[m-1][1] < 0)//1st connecting point chosen
            //      annotation.data.connections[m-1][1] = imin;
            //    else annotation.data.connections.push_back(Vec2i(imin,-1));
            //      }
            //      annotation.draw_connections();
            //      imshow(annotation.wname,annotation.image);
            //    }
            std::cout<<"mouse clicked:" << x << " " << y << " point:" << imin << std::endl;
        }
    }
}

void keyPressed(int key){
    if ( annotation.pidx >=0 ){
        int dx = 0, dy = 0;
        if ( key == 32 ){
            annotation.pidx = -1;
            annotation.copy_clean_image();
            annotation.draw_image();
            return;
        }
        else if ( key == 63232 ){
            dx = 0; dy = -1;
        }
        else if ( key == 63233 ){
            dx = 0; dy = 1;
        }
        else if ( key == 63234 ){
            dx = -1; dy = 0;
        }
        else if ( key == 63235 ){
            dx = 1; dy = 0;
        }
        if ( dx !=0 || dy !=0 ){
            annotation.shape(annotation.pidx, 0) += dx;
            annotation.shape(annotation.pidx, 1) += dy;
            annotation.copy_clean_image();
            annotation.draw_image();
        }
    }
    else{

    }
}

//
////==============================================================================
//void pp_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
//{
//  if(event == CV_EVENT_LBUTTONDOWN){
////    annotation.data.points[0].push_back(Point2f(x,y));
//    annotation.draw_points(); imshow(annotation.wname,annotation.image);
//  }
//}
//
//
////==============================================================================
//void pc_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
//{
//  if(event == CV_EVENT_LBUTTONDOWN){
//    int imin = annotation.find_closest_point(Point2f(x,y));
////    if(imin >= 0){ //add connection
////      int m = annotation.data.connections.size();
////      if(m == 0)annotation.data.connections.push_back(Vec2i(imin,-1));
////      else{
////    if(annotation.data.connections[m-1][1] < 0)//1st connecting point chosen
////      annotation.data.connections[m-1][1] = imin;
////    else annotation.data.connections.push_back(Vec2i(imin,-1));
////      }
////      annotation.draw_connections();
////      imshow(annotation.wname,annotation.image);
////    }
//  }
//}
////==============================================================================
//void ps_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
//{
//  if(event == CV_EVENT_LBUTTONDOWN){
//    int imin = annotation.find_closest_point(Point2f(x,y));
//    if(imin >= 0){
//      if(annotation.pidx < 0)annotation.pidx = imin;
//      else{
////    annotation.data.symmetry[annotation.pidx] = imin;
////    annotation.data.symmetry[imin] = annotation.pidx;
//    annotation.pidx = -1;
//      }
//      annotation.draw_symmetry();
//      imshow(annotation.wname,annotation.image);
//    }
//  }
//}
////==============================================================================
//void mv_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
//{
//  if(event == CV_EVENT_LBUTTONDOWN){
//    if(annotation.pidx < 0){
//      annotation.pidx = annotation.find_closest_point(Point2f(x,y));
//    }else annotation.pidx = -1;
//    annotation.copy_clean_image();
//    annotation.draw_connections();
//    annotation.draw_chosen_point();
//    imshow(annotation.wname,annotation.image);
//  }else if(event == CV_EVENT_MOUSEMOVE){
//    if(annotation.pidx >= 0){
////      annotation.data.points[annotation.idx][annotation.pidx] = Point2f(x,y);
//      annotation.copy_clean_image();
//      annotation.draw_connections();
//      annotation.draw_chosen_point();
//      imshow(annotation.wname,annotation.image);
//    }
//  }
//}

////==============================================================================
//bool
//parse_help(int argc,char** argv)
//{
//  for(int i = 1; i < argc; i++){
//    string str = argv[i];
//    if(str.length() == 2){if(strcmp(str.c_str(),"-h") == 0)return true;}
//    if(str.length() == 6){if(strcmp(str.c_str(),"--help") == 0)return true;}
//  }return false;
//}
////==============================================================================
//string
//parse_odir(int argc,char** argv)
//{
//  string odir = "data/";
//  for(int i = 1; i < argc; i++){
//    string str = argv[i];
//    if(str.length() != 2)continue;
//    if(strcmp(str.c_str(),"-d") == 0){
//      if(argc > i+1){odir = argv[i+1]; break;}
//    }
//  }
//  if(odir[odir.length()-1] != '/')odir += "/";
//  return odir;
//}
////==============================================================================
//int
//parse_ifile(int argc,
//        char** argv,
//        string& ifile)
//{
//  for(int i = 1; i < argc; i++){
//    string str = argv[i];
//    if(str.length() != 2)continue;
//    if(strcmp(str.c_str(),"-m") == 0){ //MUCT data
//      if(argc > i+1){ifile = argv[i+1]; return 2;}
//    }
//    if(strcmp(str.c_str(),"-v") == 0){ //video file
//      if(argc > i+1){ifile = argv[i+1]; return 1;}
//    }
//  }
//  ifile = ""; return 0;
//}
//


int parse_path(string path){
    if ( path.length() == 0 ){
        return 0;
    }
    string::size_type pos = path.rfind('.');
    std::string ext = path.substr(pos == std::string::npos?path.length():pos);
    if ( ext == ".mp4" || ext == ".MP4" || ext == ".mov"){
        return 1;
    }
    else if ( ext == ".jpg" || ext == ".png" || ext == ".JPG" || ext == ".PNG"){
        return 2;
    }
    else if ( ext.length() == 0 ){
        return 3;
    }

    return -1;
}

std::vector<std::string> get_file_lists(string path){
    std::vector<std::string> lists;
    //lists.push_back("data/test.jpg");
    struct dirent* ent = NULL;
    DIR *pDir;
    pDir = opendir(path.c_str());
    if (pDir == NULL) {
        //被当作目录，但是执行opendir后发现又不是目录，比如软链接就会发生这样的情况。
        return lists;
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
            string::size_type pos = _fileName.rfind('.');
            std::string ext = _fileName.substr(pos);
            if (ext == ".jpg" || ext == ".png" || ext == ".JPG" || ext == ".PNG" || ext == ".jpeg"){
                lists.push_back(fullFilePath);
            }
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
            //不处理嵌套目录
            //            vector<std::string> l = List(fullDirPath.c_str());
            //            for ( int i=0; i<l.size(); i++){
            //                lists.push_back(l[i]);
            //            }
        }
    }
    return lists;
}

std::string video_process(std::string path){
    std::string vpath;
    std::string fname;
    vpath = path.substr(0, path.length()-4);

    string::size_type pos = vpath.rfind('/');
    fname = vpath.substr(pos == std::string::npos?0:pos);

    mkdir(vpath.c_str(), 0755);
    VideoCapture cam;
    cam.open(path);
    if(!cam.isOpened()){
        cout << "Failed opening video file." << endl;
        return 0;
    }
    int serial = 0;
    int interval = 0;
    Mat im;
    while(cam.read(im)){
        interval++;
        if ( interval % 5 != 0 ) continue;
        char str[1024];
        sprintf(str, "%s/%s-%s%04d.jpg",vpath.c_str(),fname.c_str(), "img-", serial++);
        imwrite(str, im);
    }
    return vpath;
}

//==============================================================================
int annotate_main(const char *path, const char *model)
{
    //如果path为空，则读取当前data目录下的所有jpg或png文件，如果还没有pts文件，则自动生成一个
    //如果path为video文件，则读取video，按间隔5帧保存jpg到video名的临时目录。
    //如果path为目录名，则参照data进行
    //如果path为jpg或png文件，则单个进行
    //根据path后缀来区分
    std::cout << "标注工具(MAC版)使用说明:" << std::endl;
    std::cout << "1、在本地创建一个工作目录作为当前目录，把face.zip在当前目录解压" << std::endl;
    std::cout << "2、启动：打开一个终端窗口，进入工作目录，在命令行模式下，键入:" << std::endl;
    std::cout << "   $face3000 annotate [图片文件名|视频文件名|目录名]" << std::endl;
    std::cout << "       图片文件名支持 .jpg, .png" << std::endl;
    std::cout << "       视频文件名支持 .mp4, .mov" << std::endl;
    std::cout << "       目录名可为任意合法目录，如果是相对路径，则会在当前工作目录下寻找" << std::endl;
    std::cout << "       如果没有带文件名或者目录名，则会打开当前工作目录下的data目录，里面有部分标注示例，可参考" << std::endl;
    std::cout << "3、工作流程：1）如果已收集了一批jpg或者png图片，可在当前工作目录下创建work目录，把图片拷贝进来，然后 face3000 annotate work即可" << std::endl;
    std::cout << "           2）如果用电脑或者手机拍摄了一段视频（.mp4或.mov）可拷贝视频文件到工作目录，face3000 annotate video文件名.mp4(.mov) 即可。" << std::endl;
    std::cout << "              程序会自动在视频文件所在目录以该文件名为名创建一个目录，把视频文件中的帧每隔五帧生成.jpg图片，并自动打开开始标注." << std::endl;
    std::cout << "              如果标注部分内容退出后，下次再次标注，可直接face3000 annotate vidio文件名(不带.mp4 .mov后缀，这是一个自动创建的目录)" << std::endl;
    std::cout << "4、标注工具使用指令：" << std::endl;
    std::cout << "   P: Previous，前一张图片" << std::endl;
    std::cout << "   N: Next，后一张图片" << std::endl;
    std::cout << "   S: Save，保存标注结果。（如果你改动了一下标注，没有保存时，按P或N翻动图片，则改动信息将丢失）" << std::endl;
    std::cout << "   D: Delete，删除图片及标注信息" << std::endl;
    std::cout << "   Q: Quit，退出" << std::endl;
    std::cout << "5、标注方法：" << std::endl;
    std::cout << "   打开一张图片时，如果图片还没有经过标注（没有标注数据文件.pts），则程序会自动检测并生成初步的标注数据。如果已有标注，则打开已有的数据" << std::endl;
    std::cout << "   用鼠标点击任意一个标注点，标注点会选中变成红色，这时候再点击鼠标，改标注点将移动到新点击的地方。工具不支持鼠标拖拽，所以需注意标注点为红色时，表示可移动" << std::endl;
    std::cout << "   在标注点为红色时，也可以用上下左右剪头来微调移动标注点位置，此时需注意按空格键来确定已移动到目标位置" << std::endl;
    std::cout << "6、注意事项：" << std::endl;
    std::cout << "   1）眼眶、嘴的标注点，容易有标注点上下颠倒顺序错误的情况，这种情况下，工具会有红色错误提示，务必改正！" << std::endl;
    std::cout << "   2）标注时，需注意保存标注结果，如果没保存的情况下翻页将丢失标注改动信息" << std::endl;
    std::cout << "   3）可运行 face3000 annotate 不带参数来打开缺省data目录下的示例，来了解每个标注点在不同姿态下的合理位置" << std::endl;
    std::cout << std::endl;
    std::cout << "标注内容要求：" << std::endl;
    std::cout << "1、可给自己或家人拍一小段录像作为原始素材，每秒的视频将生成5-6张图片。" << std::endl;
    std::cout << "2、录像最好分几小段，分别在不同的环境及光线条件下拍摄，比如强烈阳光下、背光、晚上灯光下、较暗的情况等" << std::endl;
    std::cout << "3、拍摄是脸部姿态，最好有左右倾侧，左右旋转，仰头，低头，闭眼，扬眉，张嘴等动作，注意动作幅度适中，不必过大！" << std::endl;
    std::cout << "4、除了拍摄录像，如果能收集一些图片亦可，推荐收集老人、小孩、外国不同肤色人等" << std::endl;
    std::cout << "5、图片中人员可以带眼镜，有少许脸部的遮挡图片也可以接受一小部分" << std::endl;
    std::cout << "6、视频中的图片，稍微有点运动模糊可以接受" << std::endl;
    std::cout << std::endl;


//    const char *ModelName = model;
    std::string ModelName = model;
    annotation.face_detector.LoadCascadeRegressor(ModelName);
    annotation.gender_detector.LoadCascadeRegressor("mv0913g");

    std::string current_dir = "";
    std::vector<std::string> lists;

    int current_index=0;

    std::string spath = std::string(path);
    int type = parse_path(spath);
    if ( type == 0 ){
        //没有参数，默认打开data目录
        lists = get_file_lists("data");
    }
    else if ( type == 1 ){
        //视频文件，同文件名创建一个目录，每个5帧把视频保存为jpg
        std::string vpath = video_process(spath);
        lists = get_file_lists(vpath);
    }
    else if ( type == 2 ){
        //图片文件
        lists.push_back(spath);
    }
    else{
        //目录
        lists = get_file_lists(spath);
    }

    if ( lists.size() == 0 ) return 0;

    annotation.lists = lists;
    //    annotation.set_capture_instructions();
    namedWindow(annotation.wname);

    setMouseCallback(annotation.wname, p_MouseCallback, 0);
    annotation.set_current_image(current_index); //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
    std::cout<<"current:" << current_index << " file:" << annotation.file_name << std::endl;
    annotation.draw_image(); //画图片，画各点，选中点高亮

    while (true) {


        //imshow(annotation.wname, annotation.image);

        int c = waitKey(0);
        if (c == 'q') break;
        else if (c == 'p'){ //前一张
            current_index--;
            if ( current_index < 0 ) current_index = 0;
            annotation.set_current_image(current_index); //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
            std::cout<<"current:" << current_index << " file:" << annotation.file_name << std::endl;
            annotation.draw_image(); //画图片，画各点，选中点高亮
        }
        else if (c == 'n'){ //后一张
            if ( current_index < annotation.lists.size() - 1) current_index++;
            annotation.set_current_image(current_index); //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
            std::cout<<"current:" << current_index << " file:" << annotation.file_name << std::endl;
            annotation.draw_image(); //画图片，画各点，选中点高亮
        }
        else if (c == 's'){ //保存
            annotation.save_pts();
            annotation.set_current_image(current_index); //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
            std::cout<<"current:" << current_index << " file:" << annotation.file_name << std::endl;
            annotation.draw_image(); //画图片，画各点，选中点高亮
        }
        else if (c == 'd'){ //删除
            current_index = annotation.delete_image();
            if ( current_index == -1 ) break;
            annotation.set_current_image(current_index); //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
            std::cout<<"current:" << current_index << " file:" << annotation.file_name << std::endl;
            annotation.draw_image(); //画图片，画各点，选中点高亮
        }
        else if ( c == 'm'){
            annotation.gender = 1;
            annotation.draw_image();
        }
        else if ( c == 'f'){
            annotation.gender = -1;
            annotation.draw_image();
        }
        else if (c==63232){//up
            keyPressed(c);
        }
        else if (c == 63233 ){//down
            keyPressed(c);
        }
        else if (c == 63234){//left
            keyPressed(c);
        }
        else if (c == 63235){//right
            keyPressed(c);
        }
        else if (c == 32){//space
            keyPressed(c);
        }
        else{
            std::cout<<"key:"<< c << std::endl;
        }
    }

    //
    //  //get data
    //  namedWindow(annotation.wname);
    //  if(type == 2){ //MUCT data
    //
    //  }else{
    //    //open video stream
    //    VideoCapture cam;
    //    if(type == 1)cam.open(path); else cam.open(0);
    //    if(!cam.isOpened()){
    //      cout << "Failed opening video file." << endl
    //       << "usage: ./annotate [-v video] [-m muct_dir] [-d output_dir]"
    //       << endl; return 0;
    //    }
    //    //get images to annotate
    //    annotation.set_capture_instructions();
    //    while(cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999){
    //      Mat im,img; cam >> im; annotation.image = im.clone();
    //      annotation.draw_instructions();
    //      imshow(annotation.wname,annotation.image); int c = waitKey(10);
    //      if(c == 'q')break;
    //      else if(c == 's'){
    ////    int idx = annotation.data.imnames.size(); char str[1024];
    ////    if     (idx < 10)sprintf(str,"%s00%d.png",odir.c_str(),idx);
    ////    else if(idx < 100)sprintf(str,"%s0%d.png",odir.c_str(),idx);
    ////    else               sprintf(str,"%s%d.png",odir.c_str(),idx);
    ////    imwrite(str,im); annotation.data.imnames.push_back(str);
    //    im = Scalar::all(255); imshow(annotation.wname,im); waitKey(10);
    //      }
    //    }
    ////    if(annotation.data.imnames.size() == 0)return 0;
    ////    annotation.data.points.resize(annotation.data.imnames.size());
    //
    //    //annotate first image
    //    setMouseCallback(annotation.wname,pp_MouseCallback,0);
    //    annotation.set_pick_points_instructions();
    //    annotation.set_current_image(0);
    //    annotation.draw_instructions();
    //    annotation.idx = 0;
    //    while(1){ annotation.draw_points();
    //      imshow(annotation.wname,annotation.image); if(waitKey(0) == 'q')break;
    //    }
    ////    if(annotation.data.points[0].size() == 0)return 0;
    //    annotation.replicate_annotations(0);
    //  }
    ////  save_ft(fname.c_str(),annotation.data);
    //
    //  //annotate connectivity
    //  setMouseCallback(annotation.wname,pc_MouseCallback,0);
    //  annotation.set_connectivity_instructions();
    //  annotation.set_current_image(0);
    //  annotation.draw_instructions();
    //  annotation.idx = 0;
    //  while(1){ annotation.draw_connections();
    //    imshow(annotation.wname,annotation.image); if(waitKey(0) == 'q')break;
    //  }
    ////  save_ft(fname.c_str(),annotation.data);
    //
    //  //annotate symmetry
    //  setMouseCallback(annotation.wname,ps_MouseCallback,0);
    //  annotation.initialise_symmetry(0);
    //  annotation.set_symmetry_instructions();
    //  annotation.set_current_image(0);
    //  annotation.draw_instructions();
    //  annotation.idx = 0; annotation.pidx = -1;
    //  while(1){ annotation.draw_symmetry();
    //    imshow(annotation.wname,annotation.image); if(waitKey(0) == 'q')break;
    //  }
    ////  save_ft(fname.c_str(),annotation.data);
    //
    //  //annotate the rest
    //  if(type != 2){
    //    setMouseCallback(annotation.wname,mv_MouseCallback,0);
    //    annotation.set_move_points_instructions();
    //    annotation.idx = 1; annotation.pidx = -1;
    //    while(1){
    //      annotation.set_current_image(annotation.idx);
    //      annotation.draw_instructions();
    //      annotation.set_clean_image();
    //      annotation.draw_connections();
    //      imshow(annotation.wname,annotation.image); 
    //      int c = waitKey(0);
    //      if     (c == 'q')break;
    //      else if(c == 'p'){annotation.idx++; annotation.pidx = -1;}
    //      else if(c == 'o'){annotation.idx--; annotation.pidx = -1;}
    //      if(annotation.idx < 0)annotation.idx = 0;
    ////      if(annotation.idx >= int(annotation.data.imnames.size()))
    ////    annotation.idx = annotation.data.imnames.size()-1;
    //    }
    //  }
    ////  save_ft(fname.c_str(),annotation.data); destroyWindow("Annotate");
    return 0;
}
//==============================================================================

int annotate_filter(const char *path)
{
    const char *ModelName = MODEL_NAME;
    annotation.face_detector.LoadCascadeRegressor(ModelName);

    std::vector<std::string> lists;
    std::string spath = std::string(path);
    lists = get_file_lists(spath);
    for ( int i=0; i<lists.size(); i++ ){
        std::string file_name = lists[i];
        cv::Mat_<uchar> image = cv::imread(file_name, 0);
        std::vector<cv::Mat_<float>> shapes;
        std::cout << "process:" << file_name;
        std::vector<cv::Rect> rects = annotation.face_detector.detectMultiScale(image, shapes, 1.1, 1, 0, min(image.rows,image.cols) / 3);
        if ( rects.size() == 0 ){
//            std::string dfile =  file_name + ".bak";
//            int ret = rename(file_name.c_str(), dfile.c_str());
//            std::cout << dfile << " " << file_name << " " << ret << std::endl;
            std::cout << ":removed" << std::endl;
            remove(file_name.c_str());
        }
        else{
            std::cout << std::endl;
        }

    }
    return 0;
}
