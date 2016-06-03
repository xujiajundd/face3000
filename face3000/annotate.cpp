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
#include <iostream>
#include <fstream>
//==============================================================================
using namespace cv;
using namespace std;
class annotate{
public:
  int idx;                       //index of image to annotate
  int pidx;                      //index of point to manipulate
    cv::Mat_<float> shape;
    cv::Mat_<cv::Vec3b> image;
    cv::Mat_<cv::Vec3b> image_clean;
//  Mat image;                     //current image to display
//  Mat image_clean;               //clean image to display
  const char* wname;             //display window name
  vector<string> instructions;   //annotation instructions
  std::vector<std::string> lists;
    std::string file_name;
    
  annotate(){wname = "Annotate"; idx = 0; pidx = -1;}

  int set_current_image(const int cidx = 0){ //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
      pidx = -1;
      idx = cidx;
      file_name = lists[idx];
      image = cv::imread(file_name, 1);
      return 1;
  }
    
    void draw_image(){  //画图片，画各点，选中点高亮
        cv::imshow(wname, image);
    }
    
    void save_pts(){ //保存pts到文件
        
    }
    
    int delete_image(){
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
    instructions.push_back(string("Select expressive frames."));
    instructions.push_back(string("s - use this frame"));
    instructions.push_back(string("q - done"));
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
//        int n = data.points[idx].size(),imin = -1; double dmin = -1;
//        for(int i = 0; i < n; i++){
//          double d = norm(p-data.points[idx][i]);
//          if((imin < 0) || (d < dmin)){imin = i; dmin = d;}
//        }
//        if((dmin >= 0) && (dmin < thresh))return imin; else return -1;
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
    Size size = getTextSize(text,FONT_HERSHEY_COMPLEX,0.6f,1,NULL);
    putText(img,text,Point(0,level*size.height),FONT_HERSHEY_COMPLEX,0.6f,
        Scalar::all(0),1,CV_AA);
    putText(img,text,Point(1,level*size.height+1),FONT_HERSHEY_COMPLEX,0.6f,
        Scalar::all(255),1,CV_AA);
  }
} annotation;


//==============================================================================
void p_MouseCallback(int event, int x, int y, int flags, void* param)
{
    if(event == CV_EVENT_LBUTTONDOWN){
        int imin = annotation.find_closest_point(Point2f(x,y));
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
        std::cout<<"mouse clicked:" << x << " " << y << std::endl;
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
    return -1;
}

std::vector<std::string> get_file_lists(string path){
    std::vector<std::string> lists;
    lists.push_back("data/test.jpg");
    return lists;
}

std::string video_process(std::string path){
    std::string vpath;
    return vpath;
}

//==============================================================================
int annotate_main(const char *path)
{
//如果path为空，则读取当前data目录下的所有jpg或png文件，如果还没有pts文件，则自动生成一个
//如果path为video文件，则读取video，按间隔5帧保存jpg到video名的临时目录。
//如果path为目录名，则参照data进行
//如果path为jpg或png文件，则单个进行
//根据path后缀来区分
    std::string current_dir = "";
    std::vector<std::string> lists;
//    cv::Mat_<float> shape;
//    cv::Mat_<cv::Vec3b> image;
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
    
    annotation.lists = lists;
    namedWindow(annotation.wname);
    
    setMouseCallback(annotation.wname, p_MouseCallback, 0);
    
    while (true) {
        annotation.set_current_image(current_index); //读取图片，读取pts，如果没有pts，调用shape程序自动计算一个
        annotation.draw_image(); //画图片，画各点，选中点高亮
        //imshow(annotation.wname, annotation.image);
        
        int c = waitKey(0);
        if (c == 'q') break;
        else if (c == 'p'){ //前一张
            current_index--;
            if ( current_index < 0 ) current_index = 0;
        }
        else if (c == 'n'){ //后一张
            if ( current_index < annotation.lists.size() - 1) current_index++;
        }
        else if (c == 's'){ //保存
            annotation.save_pts();
        }
        else if (c == 'd'){ //删除
            current_index = annotation.delete_image();
            if ( current_index == -1 ) break;
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
