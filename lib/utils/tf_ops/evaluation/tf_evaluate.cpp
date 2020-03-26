#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "evaluate.cpp"

using namespace tensorflow;

REGISTER_OP("Evaluate")
    .Input("detections: float32")
    .Input("names: string")
    .Input("numlist: int32")
    .Output("precision_image: float32")
    .Output("aos_image: float32")
    .Output("precision_ground: float32")
    .Output("aos_ground: float32")
    .Output("precision_3d: float32")
    .Output("aos_3d: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({NUM_CLASS, 3, (int)N_SAMPLE_PTS});
        c->set_output(0, output);
        c->set_output(1, output);
        c->set_output(2, output);
        c->set_output(3, output);
        c->set_output(4, output);
        c->set_output(5, output);
        return Status::OK();
    });

REGISTER_OP("CalcIou")
    .Input("detections: float32") // [bs, num_dets, 7]
    .Input("groundtruths: float32") // [bs, num_gts, 7]
    .Output("iou_bev: float32") // [bs, num_dets, num_gts]
    .Output("iou_3d: float32") // [bs, num_dets, num_gts] 
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // [bs, num_dets, 7]
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // [bs, num_gts, 7]
        c->WithRank(c->input(1), 3, &dims2);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims2, 1)}); // [bs, num_dets, num_gts]
        c->set_output(0, output);
        return Status::OK();
    });

REGISTER_OP("CalcMatchingIou")
    .Input("detections: float32") // [-1, 7]
    .Input("groundtruths: float32") // [-1, 7]
    .Output("iou_bev: float32") // [-1]
    .Output("iou_3d: float32") // [-1] 
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // [-1, 7]
        c->WithRank(c->input(0), 2, &dims1);
        // [-1]
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0)});
        c->set_output(0, output);
        c->set_output(1, output);
        return Status::OK();
    });


float randomf(){
    return (rand()+0.5)/(RAND_MAX+1.0);
}

static double get_time(){
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC,&tp);
    return tp.tv_sec+tp.tv_nsec*1e-9;
}



void evaluate_cpu(const float* detections, const string* names, const int* num_list, const int num_images, 
    float* precision_image, float* aos_image, float* precision_ground, float* aos_ground, float* precision_3d, float* aos_3d) {
    eval(detections, names, num_list, num_images, precision_image, aos_image, precision_ground, aos_ground, precision_3d, aos_3d);    
}


class EvaluateOp: public OpKernel{
    public:
        explicit EvaluateOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& detections_tensor=context->input(0);
            OP_REQUIRES(context, detections_tensor.dims()==2 && detections_tensor.shape().dim_size(1)==NUM_DETECTION_ELEM, 
                        errors::InvalidArgument("Evaluate expects (n,m) detections shape"));
            int n_boxes = detections_tensor.shape().dim_size(0);
            int c = detections_tensor.shape().dim_size(1);

            const Tensor& names_tensor=context->input(1);
            OP_REQUIRES(context, names_tensor.dims()==1, errors::InvalidArgument("Evaluate expects (n_images,) names shape"));
            int n_images = names_tensor.shape().dim_size(0);
            const Tensor& numlist_tensor=context->input(2);
            OP_REQUIRES(context, numlist_tensor.dims()==1 && names_tensor.shape().dim_size(0) == numlist_tensor.shape().dim_size(0), 
                        errors::InvalidArgument("Evaluate expects (n_images,) numlist shape"));

            Tensor* precision_image_tensor = nullptr;
            Tensor* aos_image_tensor = nullptr;
            Tensor* precision_ground_tensor = nullptr;
            Tensor* aos_ground_tensor = nullptr;
            Tensor* precision_3d_tensor = nullptr;
            Tensor* aos_3d_tensor = nullptr;

            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NUM_CLASS, 3, (int)N_SAMPLE_PTS}, &precision_image_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{NUM_CLASS, 3, (int)N_SAMPLE_PTS}, &aos_image_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{NUM_CLASS, 3, (int)N_SAMPLE_PTS}, &precision_ground_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{NUM_CLASS, 3, (int)N_SAMPLE_PTS}, &aos_ground_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{NUM_CLASS, 3, (int)N_SAMPLE_PTS}, &precision_3d_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape{NUM_CLASS, 3, (int)N_SAMPLE_PTS}, &aos_3d_tensor));

            auto detections_flat = detections_tensor.flat<float>();
            const float *detections = &(detections_flat(0));
            auto names_flat = names_tensor.flat<string>();
            const string *names = &(names_flat(0));
            auto numlist_flat = numlist_tensor.flat<int>();
            const int *numlist = &(numlist_flat(0));

            auto precision_image_flat = precision_image_tensor->flat<float>();
            float *precision_image = &(precision_image_flat(0));
            auto aos_image_flat = aos_image_tensor->flat<float>();
            float *aos_image = &(aos_image_flat(0));
            auto precision_ground_flat = precision_ground_tensor->flat<float>();
            float *precision_ground = &(precision_ground_flat(0));
            auto aos_ground_flat = aos_ground_tensor->flat<float>();
            float *aos_ground = &(aos_ground_flat(0));
            auto precision_3d_flat = precision_3d_tensor->flat<float>();
            float *precision_3d = &(precision_3d_flat(0));
            auto aos_3d_flat = aos_3d_tensor->flat<float>();
            float *aos_3d = &(aos_3d_flat(0));

            evaluate_cpu(detections, names, numlist, n_images, precision_image, aos_image, precision_ground, aos_ground, precision_3d, aos_3d);
        }
};
REGISTER_KERNEL_BUILDER(Name("Evaluate").Device(DEVICE_CPU),EvaluateOp);


void calc_intersections_cpu(const float *dets, const float *gts, const int det_num, const int gt_num, const int num_images, float* IoU3DMatrics, float* IoUBeVMatrics){
    calc_intersections(dets, gts, det_num, gt_num, num_images, IoU3DMatrics, IoUBeVMatrics);
}

class CalcIouOp: public OpKernel{
    public:
        explicit CalcIouOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& detections_tensor=context->input(0);
            OP_REQUIRES(context, detections_tensor.dims()==3 && detections_tensor.shape().dim_size(2)==7, errors::InvalidArgument("Calculate IoU expects (bs, -1, 7) detections shape"));
            int bs = detections_tensor.shape().dim_size(0);
            int det_num = detections_tensor.shape().dim_size(1);
            
            const Tensor& groundtruths_tensor = context->input(1);
            OP_REQUIRES(context, groundtruths_tensor.dims()==3 && groundtruths_tensor.shape().dim_size(0)==bs && groundtruths_tensor.shape().dim_size(2)==7, errors::InvalidArgument("Calculate IoU expects (bs, -1, 7) gt shape"));
            int gt_num = groundtruths_tensor.shape().dim_size(1);

            Tensor* iou_bev_tensor = nullptr;
            Tensor* iou_3d_tensor = nullptr; 

            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{bs, det_num, gt_num}, &iou_bev_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{bs, det_num, gt_num}, &iou_3d_tensor));

            auto detections_flat = detections_tensor.flat<float>();
            const float *detections = &(detections_flat(0));
            auto groundtruths_flat = groundtruths_tensor.flat<float>();
            const float* groundtruths = &(groundtruths_flat(0));
        
            auto iou_bev_flat = iou_bev_tensor->flat<float>();
            float* iou_bev = &(iou_bev_flat(0));
            auto iou_3d_flat = iou_3d_tensor->flat<float>();
            float* iou_3d = &(iou_3d_flat(0));

            calc_intersections_cpu(detections, groundtruths, det_num, gt_num, bs, iou_3d, iou_bev);
        }
};
REGISTER_KERNEL_BUILDER(Name("CalcIou").Device(DEVICE_CPU),CalcIouOp);


void calc_intersections_matching_cpu(const float *dets, const float *gts, const int bs, float* IoU3DMatrics, float* IoUBeVMatrics){
    calc_intersections_matching(dets, gts, bs, IoU3DMatrics, IoUBeVMatrics);
}

class CalcMatchingIouOp: public OpKernel{
    public:
        explicit CalcMatchingIouOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& detections_tensor=context->input(0);
            OP_REQUIRES(context, detections_tensor.dims()==2 && detections_tensor.shape().dim_size(1)==7, errors::InvalidArgument("Calculate IoU expects (-1, 7) detections shape"));
            int bs = detections_tensor.shape().dim_size(0);
            
            const Tensor& groundtruths_tensor = context->input(1);
            OP_REQUIRES(context, groundtruths_tensor.dims()==2 && groundtruths_tensor.shape().dim_size(0)==bs && groundtruths_tensor.shape().dim_size(1) == 7, errors::InvalidArgument("Calculate IoU expects (-1, 7) gt shape"));

            Tensor* iou_bev_tensor = nullptr;
            Tensor* iou_3d_tensor = nullptr; 

            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{bs}, &iou_bev_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{bs}, &iou_3d_tensor));

            auto detections_flat = detections_tensor.flat<float>();
            const float *detections = &(detections_flat(0));
            auto groundtruths_flat = groundtruths_tensor.flat<float>();
            const float* groundtruths = &(groundtruths_flat(0));
        
            auto iou_bev_flat = iou_bev_tensor->flat<float>();
            float* iou_bev = &(iou_bev_flat(0));
            auto iou_3d_flat = iou_3d_tensor->flat<float>();
            float* iou_3d = &(iou_3d_flat(0));

            calc_intersections_matching_cpu(detections, groundtruths, bs, iou_3d, iou_bev);
        }
};
REGISTER_KERNEL_BUILDER(Name("CalcMatchingIou").Device(DEVICE_CPU),CalcMatchingIouOp);
