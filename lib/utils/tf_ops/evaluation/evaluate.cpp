#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <cstring>
#include <cassert>

#include <dirent.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>

BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)

typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > Polygon;


using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// easy, moderate and hard evaluation level
enum DIFFICULTY{EASY=0, MODERATE=1, HARD=2};

// evaluation metrics: image, ground or 3D
enum METRIC{IMAGE=0, GROUND=1, BOX3D=2};

// evaluation parameter
const int32_t MIN_HEIGHT[3]     = {40, 25, 25};     // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[3]  = {0, 1, 2};        // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[3] = {0.15, 0.3, 0.5}; // maximum truncation level of the groundtruth used for evaluation

// evaluated object classes
enum CLASSES{CAR=0, PEDESTRIAN=1, CYCLIST=2};
const int NUM_CLASS = 3;
const int NUM_GROUNDTRUTH_ELEM = 15; 
const int NUM_DETECTION_ELEM = 14;

// parameters varying per class
vector<string> CLASS_NAMES;
vector<string> CLASS_NAMES_CAP;
// the minimum overlap required for 2D evaluation on the image/ground plane and 3D evaluation
const double MIN_OVERLAP[3][3] = {{0.7, 0.5, 0.5}, {0.7, 0.5, 0.5}, {0.7, 0.5, 0.5}};

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;

// initialize class names
void initGlobals () {
  CLASS_NAMES.push_back("car");
  CLASS_NAMES.push_back("pedestrian");
  CLASS_NAMES.push_back("cyclist");
  CLASS_NAMES_CAP.push_back("Car");
  CLASS_NAMES_CAP.push_back("Pedestrian");
  CLASS_NAMES_CAP.push_back("Cyclist");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  double         similarity_ground; // ground orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0), similarity_ground(0) {}
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  double ry;
  double  t1, t2, t3;
  double h, w, l;
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  double  ry;
  double  t1, t2, t3;
  double  h, w, l;
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tDetection (tBox box,double thresh) :
    box(box),thresh(thresh) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tBox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};


/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/
vector<int32_t> indices;

vector<tDetection> loadDetections(string file_name, bool &compute_aos,
        vector<bool> &eval_image, vector<bool> &eval_ground,
        vector<bool> &eval_3d, bool &success) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return detections;
  }
  while (!feof(fp)) {
    tDetection d;
    double trash;
    char str[255];
    if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1,
                   &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3,
                   &d.ry, &d.thresh)==16) {

        // d.thresh = 1;
      d.box.type = str;
      detections.push_back(d);

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if(d.box.alpha == -10)
        compute_aos = false;

      // a class is only evaluated if it is detected at least once
      for (int c = 0; c < NUM_CLASS; c++) {
        if (!strcasecmp(d.box.type.c_str(), CLASS_NAMES[c].c_str())) {
          if (!eval_image[c] && d.box.x1 >= 0)
            eval_image[c] = true;
          if (!eval_ground[c] && d.t1 != -1000 && d.t3 != -1000 && d.w > 0 && d.l > 0)
            eval_ground[c] = true;
          if (!eval_3d[c] && d.t1 != -1000 && d.t2 != -1000 && d.t3 != -1000 && d.h > 0 && d.w > 0 && d.l > 0)
            eval_3d[c] = true;
          break;
        }
      }
    }
  }
  fclose(fp);
  success = true;
  return detections;
}

vector<vector<tGroundtruth>> loadGroundtruth(const float * groundtruth, const int gt_num, const int num_images){
    vector<vector<tGroundtruth>> all_groundtruth;
    
    int counter = 0;
    for (int i=0; i<num_images; i++){
        vector<tGroundtruth> gt_in_one_image;
        for (int j=0; j<gt_num; j++){
            double t1, t2, t3, l, h, w, ry;

            // for box
            t1 = groundtruth[counter++]; // 6
            t2 = groundtruth[counter++]; // 7
            t3 = groundtruth[counter++]; // 8

            l = groundtruth[counter++]; // 9
            h = groundtruth[counter++]; // 10
            w = groundtruth[counter++]; // 11

            ry = groundtruth[counter++]; // 12
            tGroundtruth gt;
            gt.box.type = "DontCare";
            gt.truncation = 0.;
            gt.occlusion = 0;
            gt.box.alpha = 0;
            gt.box.x1 = 0; 
            gt.box.y1 = 0;
            gt.box.x2 = 100;
            gt.box.y2 = 100;
            gt.t1 = t1;
            gt.t2 = t2;
            gt.t3 = t3;
            gt.l = l;
            gt.h = h;
            gt.w = w;
            gt.ry = ry;

            gt_in_one_image.push_back(gt);
        }
        all_groundtruth.push_back(gt_in_one_image);
    }
    return all_groundtruth;
}

vector<vector<tDetection>> loadDetections_iou(const float* detections, const int det_num, const int num_images){
    vector<vector<tDetection> > all_detections;

    int counter = 0;
    for (int i=0; i<num_images; i++){
        vector<tDetection> detections_in_one_image;
        for (int j=0; j < det_num; j++){
            double t1, t2, t3, l, h, w, ry;

            // for box
            t1 = detections[counter++]; // 6
            t2 = detections[counter++]; // 7
            t3 = detections[counter++]; // 8

            l = detections[counter++]; // 9
            h = detections[counter++]; // 10
            w = detections[counter++]; // 11

            ry = detections[counter++]; // 12

            tDetection det;
            det.box.type = "DontCare";
            det.box.x1 = 0;
            det.box.y1 = 0;
            det.box.x2 = 100;
            det.box.y2 = 100;
            det.box.alpha = 0.;
            det.thresh = 0.;
            det.t1 = t1;
            det.t2 = t2;
            det.t3 = t3;
            det.h = h;
            det.w = w;
            det.l = l;
            det.ry = ry;
            detections_in_one_image.push_back(det);            
        }
        all_detections.push_back(detections_in_one_image);
    }
    return all_detections;
}

vector< vector<tDetection> > loadDetections(const float* detections, const int* num_list, const int num_images, 
        bool &compute_aos, vector<bool> &eval_image, vector<bool> &eval_ground, vector<bool> &eval_3d) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<vector<tDetection> > all_detections;

  int counter = 0;
  for (int i=0; i<num_images; i++) {
      const int num_detections = num_list[i];
      vector<tDetection> detections_in_one_image;
      for (int j=0; j<num_detections; j++) {
          double x1, y1, x2, y2, t1, t2, t3, h, w, l, thresh, alpha, ry; 
          int cls;

          // for box
          cls = int(detections[counter++]); // 0
          x1 = detections[counter++]; // 1
          y1 = detections[counter++]; // 2
          x2 = detections[counter++]; // 3
          y2 = detections[counter++]; // 4
          alpha = detections[counter++]; // 5

          t1 = detections[counter++]; // 6
          t2 = detections[counter++]; // 7
          t3 = detections[counter++]; // 8
 
          h = detections[counter++]; // 9 
          w = detections[counter++]; // 10
          l = detections[counter++]; // 11

          ry = detections[counter++]; // 12
          thresh = detections[counter++]; // 13

          tDetection det;
          switch ((CLASSES)cls) {
              case 0: det.box.type="Car"; break;
              case 1: det.box.type="Pedestrian"; break;
              case 2: det.box.type="Cyclist"; break;
              default: det.box.type="DontCare";
          }
          det.box.x1 = x1;
          det.box.y1 = y1;
          det.box.x2 = x2;
          det.box.y2 = y2;
          det.box.alpha = alpha;
          det.thresh = thresh;
          det.t1 = t1;
          det.t2 = t2;
          det.t3 = t3;
          det.h = h;
          det.w = w;
          det.l = l;
          det.ry = ry;
          detections_in_one_image.push_back(det);    

          if (det.box.alpha == -10)
              compute_aos = false;
          for (int c = 0; c < NUM_CLASS; c++) {
              if (!strcasecmp(det.box.type.c_str(), CLASS_NAMES[c].c_str()) || !strcasecmp(det.box.type.c_str(), CLASS_NAMES_CAP[c].c_str())) {
                if (!eval_image[c] && det.box.x1 >= 0)
                  eval_image[c] = true;
                if (!eval_ground[c] && det.t1 != -1000 && det.t3 != -1000 && det.w > 0 && det.l > 0)
                  eval_ground[c] = true;
                if (!eval_3d[c] && det.t1 != -1000 && det.t2 != -1000 && det.t3 != -1000 && det.h > 0 && det.w > 0 && det.l > 0) 
                  eval_3d[c] = true;
                break;
              }
          }
      }
      all_detections.push_back(detections_in_one_image);
  }
  
  return all_detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name,bool &success) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {
    tGroundtruth g;
    char str[255];
    if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &g.h,      &g.w,        &g.l,       &g.t1,
                   &g.t2,      &g.t3,        &g.ry )==15) {
      g.box.type = str;
      groundtruth.push_back(g);
    }
  }
  fclose(fp);
  success = true;
  return groundtruth;
}

vector<vector<tGroundtruth> > loadGroundtruth(const string* names, const int num_images, bool& success) {
  vector<vector<tGroundtruth> > all_groundtruth;
  for (int i=0; i<num_images; i++) {
    vector<tGroundtruth> groundtruth;
    string file_name = names[i];
    FILE *fp = fopen(file_name.c_str(),"r");
    if (!fp) {
      success = false;
      return all_groundtruth;
    }

    while (!feof(fp)) {
      tGroundtruth g;
      char str[255];
      if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                     str, &g.truncation, &g.occlusion, &g.box.alpha,
                     &g.box.x1, &g.box.y1, &g.box.x2, &g.box.y2,
                     &g.h, &g.w, &g.l, &g.t1, &g.t2, &g.t3, &g.ry)==15) {
        g.box.type = str;
        groundtruth.push_back(g);
      }
    }
    fclose(fp);

    all_groundtruth.push_back(groundtruth);
  }

  return all_groundtruth;
}

void saveStats (const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori) {

  if (fp_det==0 || fp_ori==0) return;

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(aos.empty())
    return;
  for (int32_t i=0; i<aos.size(); i++)
    fprintf(fp_ori,"%f ",aos[i]);
  fprintf(fp_ori,"\n");
}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double imageBoxOverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

inline double imageBoxOverlap(tDetection a, tGroundtruth b, int32_t criterion=-1){
  return imageBoxOverlap(a.box, b.box, criterion);
}

// compute polygon of an oriented bounding box
template <typename T>
Polygon toPolygon(const T& g) {
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(g.ry); mref(0, 1) = sin(g.ry);
    mref(1, 0) = -sin(g.ry); mref(1, 1) = cos(g.ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {g.l / 2, g.l / 2, -g.l / 2, -g.l / 2,
                     g.w / 2, -g.w / 2, -g.w / 2, g.w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += g.t1;
        gc(1, i) += g.t3;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    Polygon poly;
    append(poly, points);
    return poly;
}

// measure overlap between bird's eye view bounding boxes, parametrized by (ry, l, w, tx, tz)
inline double groundBoxOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double inter_area = in.empty() ? 0 : area(in.front());
    double union_area = area(un.front());
    double o;
    if(criterion==-1)     // union
        o = inter_area / union_area;
    else if(criterion==0) // bbox_a
        o = inter_area / area(dp);
    else if(criterion==1) // bbox_b
        o = inter_area / area(gp);

    return o;
}

// measure overlap between 3D bounding boxes, parametrized by (ry, h, w, l, tx, ty, tz)
inline double box3DOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double ymax = min(d.t2, g.t2);
    double ymin = max(d.t2 - d.h, g.t2 - g.h);

    double inter_area = in.empty() ? 0 : area(in.front());
    double inter_vol = inter_area * max(0.0, ymax - ymin);

    double det_vol = d.h * d.l * d.w;
    double gt_vol = g.h * g.l * g.w;

    double o;
    if(criterion==-1)     // union
        o = inter_vol / (det_vol + gt_vol - inter_vol);
    else if(criterion==0) // bbox_a
        o = inter_vol / det_vol;
    else if(criterion==1) // bbox_b
        o = inter_vol / gt_vol;

    return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, DIFFICULTY difficulty){

  // extract ground truth bounding boxes for current evaluation class
  for(int32_t i=0;i<gt.size(); i++){

    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if(!strcasecmp(gt[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;

    // classes with a neighboring class
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Pedestrian") && !strcasecmp("Person_sitting", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Car") && !strcasecmp("Van", gt[i].box.type.c_str()))
      valid_class = 0;

    // classes not used for evaluation
    else
      valid_class = -1;

    // ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    if(gt[i].occlusion>MAX_OCCLUSION[difficulty] || gt[i].truncation>MAX_TRUNCATION[difficulty] || height<=MIN_HEIGHT[difficulty])
      ignore = true;

    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if(valid_class==1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }

    // neighboring class, or current class but ignored
    else if(valid_class==0 || (ignore && valid_class==1))
      ignored_gt.push_back(1);

    // all other classes which are FN in the evaluation
    else
      ignored_gt.push_back(-1);
  }

  // extract dontcare areas
  for(int32_t i=0;i<gt.size(); i++)
    if(!strcasecmp("DontCare", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for(int32_t i=0;i<det.size(); i++){

    // neighboring classes are not evaluated
    int32_t valid_class;
    if(!strcasecmp(det[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;
    else
      valid_class = -1;

    int32_t height = fabs(det[i].box.y1 - det[i].box.y2);

    // set ignored vector for detections
    if(height<MIN_HEIGHT[difficulty])
      ignored_det.push_back(1);
    else if(valid_class==1)
      ignored_det.push_back(0);
    else
      ignored_det.push_back(-1);
  }
}

tPrData computeStatistics(CLASSES current_class, const vector<tGroundtruth> &gt,
        const vector<tDetection> &det, const vector<tGroundtruth> &dc,
        const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det,
        bool compute_fp, double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
        METRIC metric, bool compute_aos=false, bool compute_aos_3d=false, double thresh=0, bool debug=false){

  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<double> delta_ground;     // holds angular difference for TPs in BEV or 3D
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // detections with a low score are ignored for computing precision (needs FP)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
        ignored_threshold[i] = true;

  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j], gt[i], -1);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>MIN_OVERLAP[metric][current_class] && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if(compute_fp && overlap>MIN_OVERLAP[metric][current_class] && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if(compute_fp && overlap>MIN_OVERLAP[metric][current_class] && valid_detection==NO_DETECTION && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==NO_DETECTION && ignored_gt[i]==0) {
      stat.fn++;
    }

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if(valid_detection!=NO_DETECTION && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=NO_DETECTION){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos)
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

      // compute heading difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos_3d)
        delta_ground.push_back(abs(gt[i].ry - det[det_idx].ry));

      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // if FP are requested, consider stuff area
  if(compute_fp) {

    // count fp
    for (int32_t i = 0; i < det.size(); i++) {

      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if (!(assigned_detection[i] || ignored_det[i] == -1 || ignored_det[i] == 1 || ignored_threshold[i]))
        stat.fp++;
    }

    // do not consider detections overlapping with stuff area
    int32_t nstuff = 0;
    for (int32_t i = 0; i < dc.size(); i++) {
      for (int32_t j = 0; j < det.size(); j++) {

        // detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
        if (assigned_detection[j])
          continue;
        if (ignored_det[j] == -1 || ignored_det[j] == 1)
          continue;
        if (ignored_threshold[j])
          continue;

        // compute overlap and assign to stuff area, if overlap exceeds class specific value
        double overlap = boxoverlap(det[j], dc[i], 0);
        if (overlap > MIN_OVERLAP[metric][current_class]) {
          assigned_detection[j] = true;
          nstuff++;
        }
      }
    }

    // FP = no. of all not to ground truth assigned detections - detections assigned to stuff areas
    stat.fp -= nstuff;

    // if all orientation values are valid, the AOS is computed

    if(compute_aos){
      vector<double> tmp;
      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for(int32_t i=0; i<delta.size(); i++)
        tmp.push_back((1.0+cos(delta[i]))/2.0);
      // be sure, that all orientation deltas are computed
      assert(tmp.size()==stat.fp+stat.tp);
      assert(delta.size()==stat.tp);


      // get the mean orientation similarity for this image
      if(stat.tp>0 || stat.fp>0)
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);

      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      else
        stat.similarity = -1;
    }

    if (compute_aos_3d) {
        vector<double> tmp_ground;
        tmp_ground.assign(stat.fp, 0);
        for (int32_t i = 0; i < delta_ground.size(); i++)
            tmp_ground.push_back((1.0 + cos(delta_ground[i])) / 2.0);

        assert(tmp_ground.size()==stat.fp+stat.tp);
        assert(delta_ground.size()==stat.tp);

        if(stat.tp>0 || stat.fp>0)
            stat.similarity_ground = accumulate(tmp_ground.begin(), tmp_ground.end(), 0.0);

        // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
        else
            stat.similarity_ground = -1;
    }
  }
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

bool eval_class (FILE *fp_det, FILE *fp_ori,CLASSES current_class,
                 const vector< vector<tGroundtruth> > &groundtruth,
                 const vector< vector<tDetection> > &detections, bool compute_aos, bool compute_aos_ground,
                 double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
                 vector<double> &precision, vector<double> &aos, vector<double> &aos_ground,
                 DIFFICULTY difficulty, METRIC metric) {

    assert(groundtruth.size() == detections.size());

    // init
    int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
    vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
    vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty
    vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

    // for all test images do
    for (int32_t i=0; i<groundtruth.size(); i++){

      // holds ignored ground truth, ignored detections and dontcare areas for current frame
      vector<int32_t> i_gt, i_det;
      vector<tGroundtruth> dc;

      // only evaluate objects of current class and ignore occluded, truncated objects
      cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty);
      ignored_gt.push_back(i_gt);
      ignored_det.push_back(i_det);
      dontcare.push_back(dc);

      // compute statistics to get recall values
      tPrData pr_tmp = tPrData();
      pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false, boxoverlap, metric, false, false);

      // add detection scores to vector over all images
      for(int32_t j=0; j<pr_tmp.v.size(); j++)
        v.push_back(pr_tmp.v[j]);
    }

    // get scores that must be evaluated for recall discretization
    thresholds = getThresholds(v, n_gt);

    // compute TP,FP,FN for relevant scores
    vector<tPrData> pr;
    pr.assign(thresholds.size(),tPrData());
    for (int32_t i=0; i<groundtruth.size(); i++){
      // for all scores/recall thresholds do:
      for(int32_t t=0; t<thresholds.size(); t++){
        tPrData tmp = tPrData();
        tmp = computeStatistics(current_class, groundtruth[i], detections[i], dontcare[i],
                                ignored_gt[i], ignored_det[i], true, boxoverlap, metric,
                                compute_aos, compute_aos_ground, thresholds[t], t==38);

        // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
        pr[t].tp += tmp.tp;
        pr[t].fp += tmp.fp;
        pr[t].fn += tmp.fn;
        if(tmp.similarity!=-1)
          pr[t].similarity += tmp.similarity;
        if(tmp.similarity_ground!=-1)
          pr[t].similarity_ground += tmp.similarity_ground;
      }
    }

    // compute recall, precision and AOS
    vector<double> recall;
    precision.assign(N_SAMPLE_PTS, 0);
    if(compute_aos) aos.assign(N_SAMPLE_PTS, 0);
    if(compute_aos_ground) aos_ground.assign(N_SAMPLE_PTS, 0);

    double r=0;
    for (int32_t i=0; i<thresholds.size(); i++){
      r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
      recall.push_back(r);
      precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);
      if(compute_aos)
        aos[i] = pr[i].similarity/(double)(pr[i].tp + pr[i].fp);
      if(compute_aos_ground)
        aos_ground[i] = pr[i].similarity_ground/(double)(pr[i].tp + pr[i].fp);
    }

    // filter precision and AOS using max_{i..end}(precision)
    for (int32_t i=0; i<thresholds.size(); i++){
      precision[i] = *max_element(precision.begin()+i, precision.end());
      if(compute_aos) {
        aos[i] = *max_element(aos.begin()+i, aos.end());
      }
      if(compute_aos_ground) {
        aos_ground[i] = *max_element(aos_ground.begin()+i, aos_ground.end());
      }
    }

    // save statisics and finish with success
    saveStats(precision, aos, fp_det, fp_ori);
    return true;
}

void printAp(string file_name, vector<double> vals[]){

  float sum[3] = {0, 0, 0};
  for (int v = 0; v < 3; ++v)
    for (int i = 0; i < vals[v].size(); i = i + 4)
      sum[v] += vals[v][i];
  printf("%s AP: %f %f %f\n", file_name.c_str(), sum[0] / 11 * 100, sum[1] / 11 * 100, sum[2] / 11 * 100);

}
void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos) {

    char command[1024];

    // save plot data to file
    FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(), "w");
    // printf("save %s\n", (dir_name + "/" + file_name + ".txt").c_str());

    for (int32_t i = 0; i < (int) N_SAMPLE_PTS; i++)
        fprintf(fp, "%f %f %f %f\n", (double) i / (N_SAMPLE_PTS - 1.0),
                vals[0][i], vals[1][i], vals[2][i]);
    fclose(fp);

    float sum[3] = {0, 0, 0};
    for (int v = 0; v < 3; ++v)
        for (int i = 0; i < vals[v].size(); i = i + 4)
            sum[v] += vals[v][i];

    printf("%s : %f %f %f\n", file_name.c_str(), sum[0] / 11 * 100, sum[1] /
            11 * 100, sum[2] / 11 * 100);



  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

vector<int32_t> getEvalIndices(const string& result_dir) {

    DIR* dir;
    dirent* entity;
    dir = opendir(result_dir.c_str());
    if (dir) {
        while (entity = readdir(dir)) {
            string path(entity->d_name);
            int32_t len = path.size();
            if (len < 10) continue;
            int32_t index = atoi(path.substr(len - 10, 10).c_str());
            indices.push_back(index);
        }
    }
    return indices;
}

bool eval(string gt_dir, string result_dir){

  // set some global parameters
  initGlobals();

  // ground truth and result directories
  // string gt_dir         = "data/object/label_2";
  // string result_dir     = "results/" + result_sha;
  string plot_dir       = result_dir + "/plot";

  // create output directories
  system(("rm -rf " + plot_dir).c_str());
  system(("mkdir " + plot_dir).c_str());

  // hold detections and ground truth in memory
  vector< vector<tGroundtruth> > groundtruth;
  vector< vector<tDetection> >   detections;

  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=true;
  bool compute_aos_ground=false;
  vector<bool> eval_image(NUM_CLASS, false);
  vector<bool> eval_ground(NUM_CLASS, false);
  vector<bool> eval_3d(NUM_CLASS, false);

  // for all images read groundtruth and detections
  //mail->msg("Loading detections...");
  std::vector<int32_t> indices = getEvalIndices(result_dir + "/data/");
  //printf("number of files for evaluation: %d\n", (int)indices.size());

  for (int32_t i=0; i<indices.size(); i++) {

    // file name
    char file_name[256];
    sprintf(file_name,"%06d.txt",indices.at(i));

    // read ground truth and result poses
    bool gt_success,det_success;
    vector<tGroundtruth> gt = loadGroundtruth(gt_dir + "/" + file_name,gt_success);
    vector<tDetection> det = loadDetections(result_dir + "/data/" + file_name, compute_aos, eval_image, eval_ground, eval_3d, det_success);
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success) {
      // mail->msg("ERROR: Couldn't read: %s of ground truth. Please write me an email!", file_name);
      return false;
    }
    if (!det_success) {
      // mail->msg("ERROR: Couldn't read: %s", file_name);
      return false;
    }
  }
  // mail->msg("  done.");

  // holds pointers for result files
  FILE *fp_det=0, *fp_ori=0;

  // eval image 2D bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (eval_image[c]) {
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection.txt").c_str(), "w");
      if(compute_aos) {
        fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_orientation.txt").c_str(),"w");
      }
      vector<double> precision[3], aos[3], aos_ground[3];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos,compute_aos_ground, imageBoxOverlap, precision[0], aos[0],aos_ground[0], EASY, IMAGE)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos,compute_aos_ground, imageBoxOverlap, precision[1], aos[1],aos_ground[0], MODERATE, IMAGE)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos,compute_aos_ground, imageBoxOverlap, precision[2], aos[2],aos_ground[0], HARD, IMAGE)) {
        // mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_AP", CLASS_NAMES[c], precision, 0);
      if(compute_aos){
        saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation_AOS", CLASS_NAMES[c], aos, 1);
        fclose(fp_ori);
      }
    }
  }

  // don't evaluate AOS for birdview boxes and 3D boxes
  compute_aos = false;
  compute_aos_ground = true;
  // eval bird's eye view bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (eval_ground[c]) {
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_ground.txt").c_str(), "w");
      vector<double> precision[3], aos[3], aos_ground[3];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos,compute_aos_ground, groundBoxOverlap, precision[0], aos[0],aos_ground[0], EASY, GROUND)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos,compute_aos_ground, groundBoxOverlap, precision[1], aos[1],aos_ground[1], MODERATE, GROUND)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos,compute_aos_ground, groundBoxOverlap, precision[2], aos[2],aos_ground[2], HARD, GROUND)) {
        // mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_BEV_AP", CLASS_NAMES[c], precision, 0);
      if(compute_aos_ground)
          saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation_BEV_AHS", CLASS_NAMES[c], aos_ground, 1);
    }
  }

  // eval 3D bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (eval_3d[c]) {
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_3D_ground.txt").c_str(), "w");
      vector<double> precision[3], aos[3], aos_ground[3];
      if(   !eval_class(fp_det, fp_ori,cls, groundtruth, detections, compute_aos,compute_aos_ground, box3DOverlap, precision[0], aos[0],aos_ground[0], EASY, BOX3D)
         || !eval_class(fp_det, fp_ori,cls, groundtruth, detections, compute_aos,compute_aos_ground, box3DOverlap, precision[1], aos[1],aos_ground[1], MODERATE, BOX3D)
         || !eval_class(fp_det, fp_ori,cls, groundtruth, detections, compute_aos,compute_aos_ground, box3DOverlap, precision[2], aos[2],aos_ground[2], HARD, BOX3D)) {
        // mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_3D_AP", CLASS_NAMES[c], precision, 0);
      if(compute_aos_ground)
          saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation_3D_AHS", CLASS_NAMES[c], aos_ground, 1);    
    }
  }


  // success
  return true;
}


bool calc_intersections(const float *dets, const float *gts, const int det_num, const int gt_num, const int num_images, float* IoU3DMatrics, float* IoUBeVMatrics){
    // dets: [bs, n, 7], gts: [bs, m, 7]
    // det_num = n, gt_num = m 
    // return : [bs, n, m]
    vector<vector<tDetection>> detections;
    vector<vector<tGroundtruth>> groundtruth; 
    vector<tDetection> cur_detections;
    vector<tGroundtruth> cur_groundtruth;

    float iou_bev, iou_3d;
    int img_base_idx, det_base_idx;

    // load Detection result first
    detections = loadDetections_iou(dets, det_num, num_images);
    groundtruth = loadGroundtruth(gts, gt_num, num_images);
    // now calc the 3d iou

    for (int i = 0; i < num_images; i++){
        cur_detections = detections[i];
        cur_groundtruth = groundtruth[i];
        
        img_base_idx = i * det_num * gt_num;
        for (int n = 0; n < det_num; n++){
            det_base_idx = img_base_idx + n * gt_num;
            for (int m=0; m < gt_num; m++){
                iou_bev = groundBoxOverlap(cur_detections[n], cur_groundtruth[m]); 
                iou_3d = box3DOverlap(cur_detections[n], cur_groundtruth[m]);
                IoUBeVMatrics[det_base_idx + m] = iou_bev;
                IoU3DMatrics[det_base_idx + m] = iou_3d; 
            }
        }
    }
}


bool calc_intersections_matching(const float *dets, const float *gts, const int bs, float* IoU3DMatrics, float* IoUBeVMatrics){
    // dets: [-1, 7], gts: [-1, 7]
    // return : [-1]
    vector<vector<tDetection>> detections;
    vector<vector<tGroundtruth>> groundtruth; 
    vector<tDetection> cur_detections;
    vector<tGroundtruth> cur_groundtruth;

    float iou_bev, iou_3d;
    int img_base_idx;

    int det_num = bs, num_images=1;

    // load Detection result first
    detections = loadDetections_iou(dets, det_num, num_images);
    groundtruth = loadGroundtruth(gts, det_num, num_images);
    // now calc the 3d iou

    for (int i = 0; i < num_images; i++){
        cur_detections = detections[i];
        cur_groundtruth = groundtruth[i];
        
        img_base_idx = i * det_num;
        for (int n = 0; n < det_num; n++){
            iou_bev = groundBoxOverlap(cur_detections[n], cur_groundtruth[n]); 
            iou_3d = box3DOverlap(cur_detections[n], cur_groundtruth[n]);
            IoUBeVMatrics[img_base_idx + n] = iou_bev;
            IoU3DMatrics[img_base_idx + n] = iou_3d; 
        }
    }
}


bool eval(const float* dets, const string* names, const int* num_list, const int num_images, float* precision_image, float* aos_image, float* precision_ground, float* aos_ground, float* precision_3d, float* aos_3d){
  // set some global parameters
  initGlobals();

  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=false;
  bool compute_aos_ground=false;
  vector<bool> eval_image(NUM_CLASS, false);
  vector<bool> eval_ground(NUM_CLASS, false);
  vector<bool> eval_3d(NUM_CLASS, false);

  vector<vector<tGroundtruth> > groundtruth;
  vector<vector<tDetection> > detections;

  bool success;
  groundtruth = loadGroundtruth(names, num_images, success);
  std::cout << "loadGroundtruth: no error!" << std::endl;

  detections = loadDetections(dets, num_list, num_images, compute_aos, eval_image, eval_ground, eval_3d);
  std::cout << "loadDetections: no error!" << std::endl;

  // holds pointers for result files
  FILE *fp_det=0, *fp_ori=0;

  int elem_num = NUM_CLASS * 3 * (int)N_SAMPLE_PTS;
  // precision_image = new float[elem_num];
  // aos_image = new float[elem_num];
  // precision_ground = new float[elem_num];
  // aos_ground = new float[elem_num];
  // precision_3d = new float[elem_num];
  // aos_3d = new float[elem_num];

  // eval image 2D bounding boxes
  for (int i=0; i<elem_num; i++) {
      precision_image[i] = -1.;
      aos_image[i] = -1.;
      precision_ground[i] = -1.;
      aos_ground[i] = -1.;
      precision_3d[i] = -1.;
      aos_3d[i] = -1.;
  }

  int cnt1 = 0, cnt2 = 0;
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    vector<double> precision[3], aos[3], aos_grd[3];
    if (eval_image[c]) {
      if( eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, imageBoxOverlap, precision[0], aos[0], aos_grd[0], EASY, IMAGE) &&
          eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, imageBoxOverlap, precision[1], aos[1], aos_grd[1], MODERATE, IMAGE) &&
          eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, imageBoxOverlap, precision[2], aos[2], aos_grd[2], HARD, IMAGE)){
        for (int i=0; i<3; i++) {
          for (int j=0; j<precision[i].size(); j++) {
              int index = c * 3 * N_SAMPLE_PTS + i * N_SAMPLE_PTS + j;
              precision_image[index] = float(precision[i][j]);
          }
          for (int j=0; j<aos[i].size(); j++) {
              int index = c * 3 * N_SAMPLE_PTS + i * N_SAMPLE_PTS + j;
              aos_image[index] = float(aos[i][j]);
          }
        }
      }
    }
  }

  // don't evaluate AOS for birdview boxes and 3D boxes
  // eval bird's eye view bounding boxes
  compute_aos = false;
  cnt1 = 0; cnt2 = 0;
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    vector<double> precision[3], aos[3], aos_grd[3];
    if (eval_ground[c]) {
      if( eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, groundBoxOverlap, precision[0], aos[0], aos_grd[0], EASY, GROUND) && 
          eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, groundBoxOverlap, precision[1], aos[1], aos_grd[1], MODERATE, GROUND) && 
          eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, groundBoxOverlap, precision[2], aos[2], aos_grd[2], HARD, GROUND)){
        for (int i=0; i<3; i++) {
          for (int j=0; j<precision[i].size(); j++) {
              int index = c * 3 * N_SAMPLE_PTS + i * N_SAMPLE_PTS + j;
              precision_ground[index] = float(precision[i][j]);
          }
          for (int j=0; j<aos[i].size(); j++) {
              int index = c * 3 * N_SAMPLE_PTS + i * N_SAMPLE_PTS + j;
              aos_ground[index] = float(aos[i][j]);
          }
        }
      }
    }
  }

  // eval 3D bounding boxes
  cnt1 = 0, cnt2 = 0;
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    vector<double> precision[3], aos[3], aos_grd[3];
    if (eval_3d[c]) {
      if( eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, box3DOverlap, precision[0], aos[0], aos_grd[0],  EASY, BOX3D) && 
          eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, box3DOverlap, precision[1], aos[1], aos_grd[1], MODERATE, BOX3D) &&
          eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, compute_aos_ground, box3DOverlap, precision[2], aos[2], aos_grd[2], HARD, BOX3D)){
        for (int i=0; i<3; i++) {
          for (int j=0; j<precision[i].size(); j++) {
              int index = c * 3 * N_SAMPLE_PTS + i * N_SAMPLE_PTS + j;
              precision_3d[index] = float(precision[i][j]);
          }
          for (int j=0; j<aos[i].size(); j++) {
              int index = c * 3 * N_SAMPLE_PTS + i * N_SAMPLE_PTS + j;
              aos_3d[index] = float(aos[i][j]);
          }
        }
      }
    }
  }

  return true;
}


int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=3) {
    cout << "Usage: ./eval_detection_3d_offline gt_dir result_dir" << endl;
    return 1;
  }

  // read arguments
  string gt_dir = argv[1];
  string result_dir = argv[2];

  // init notification mail
  // Mail *mail;
  // mail = new Mail();
  // mail->msg("Thank you for participating in our evaluation!");

  // run evaluation
  if (eval(gt_dir, result_dir)) {
    //mail->msg("Your evaluation results are available at:");
    //mail->msg(result_dir.c_str());
  } else {
    system(("rm -r " + result_dir + "/plot").c_str());
    // mail->msg("An error occured while processing your results.");
  }

  // send mail and exit
  // delete mail;

  return 0;
}
