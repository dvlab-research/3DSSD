#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("PointsFuse")
    .Attr("down_sample_rate: float") // img_feature down_sampling rate
    .Input("points: float32") // [batch_size, points_num, 3] 
    .Input("img_feature: float32") // [batch_size, height, width, channels]
    .Input("calib: float32") // [batch_size, 3, 4]
    .Output("out_2d_locations: float32") // [batch_size, points_num, 2]
    .Output("out_2d_features: float32") // [batch_size, points_num, channels]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        float down_sample_rate;
        TF_RETURN_IF_ERROR(c->GetAttr("down_sample_rate", &down_sample_rate));
        ::tensorflow::shape_inference::ShapeHandle dims1; // [batch_size, points_num, 3] 
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // [batch_size, h, w, c] 
        c->WithRank(c->input(1), 4, &dims2);
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), 2});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims2, 3)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("PointsFuseGrad")
    .Attr("down_sample_rate: float")
    .Input("img_feature: float32") // [batch_size, h, w, c]
    .Input("img_pc: float32") // [batch_size, points_num, 2]
    .Input("feature_grad: float32") // [batch_size, points_num, c]
    .Output("img_grad: float32") // [batch_size, h, w, c]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        float down_sample_rate;
        TF_RETURN_IF_ERROR(c->GetAttr("down_sample_rate", &down_sample_rate));
        c->set_output(0, c->input(0));
        return Status::OK();
    });

// then register point fuse op
void pointsFuseLauncher(const int b, const int n, const int h, const int w, const int c, const float down_sample_rate, const float* points, const float* img_feature, const float *calib, float* out_2d_locations, float* out_2d_features);
class PointsFuseGpuOp: public OpKernel{
    public:
        explicit PointsFuseGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("down_sample_rate", &down_sample_rate_));
            OP_REQUIRES(context, down_sample_rate_ > 0, errors::InvalidArgument("PointsFuse method expects positive down_sample_rate"));
        }

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor = context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3 && points_tensor.shape().dim_size(2)==3, errors::InvalidArgument("PointsFuse expects (batch_size, num_points, 3) points shape"));
            int b = points_tensor.shape().dim_size(0); // batch_size 
            int n = points_tensor.shape().dim_size(1); // points_num

            const Tensor& img_feature_tensor = context->input(1);
            OP_REQUIRES(context, img_feature_tensor.dims()==4 && img_feature_tensor.shape().dim_size(0)==b, errors::InvalidArgument("PointsFuse expects (batch_size, h, w, c) image feature shape"));
            int h = img_feature_tensor.shape().dim_size(1);
            int w = img_feature_tensor.shape().dim_size(2);
            int c = img_feature_tensor.shape().dim_size(3);

            const Tensor& calib_tensor = context->input(2);
            OP_REQUIRES(context, calib_tensor.dims()==3 && calib_tensor.shape().dim_size(0)==b && calib_tensor.shape().dim_size(1)==3 && calib_tensor.shape().dim_size(2)==4, errors::InvalidArgument("PointsFuse expects (batch_size, 3, 4) calib_P2 matrix shape"));
            
            Tensor *out_2d_locations_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,2}, &out_2d_locations_tensor));

            Tensor *out_2d_features_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1,TensorShape{b,n,c}, &out_2d_features_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float* points = &(points_flat(0));
            auto img_feature_flat = img_feature_tensor.flat<float>();
            const float* img_feature = &(img_feature_flat(0));
            auto calib_flat = calib_tensor.flat<float>();
            const float* calib = &(calib_flat(0));

            auto out_2d_locations_flat = out_2d_locations_tensor->flat<float>();
            float* out_2d_locations = &(out_2d_locations_flat(0));
            auto out_2d_features_flat = out_2d_features_tensor->flat<float>();
            float* out_2d_feature = &(out_2d_features_flat(0));
           
            cudaMemset(out_2d_locations, 0, sizeof(float)*b*n*2);
            cudaMemset(out_2d_feature, 0, sizeof(float) * b*n*c);

            // CUDA code
            pointsFuseLauncher(b, n, h, w, c, down_sample_rate_, points, img_feature, calib, out_2d_locations, out_2d_feature);
        }
    private:
        float down_sample_rate_;
};
REGISTER_KERNEL_BUILDER(Name("PointsFuse").Device(DEVICE_GPU),PointsFuseGpuOp);


// then register point fuse grad op
void pointsFuseGradLauncher(const int b, const int n, const int h, const int w, const int c, const float down_sample_rate, const float* img_feature, const float* img_pc, const float* feature_grad, float* img_grad);
class PointsFuseGradGpuOp: public OpKernel{
    public:
        explicit PointsFuseGradGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("down_sample_rate", &down_sample_rate_));
            OP_REQUIRES(context, down_sample_rate_ > 0, errors::InvalidArgument("PointsFuseGrad method expects positive down_sample_rate"));
        }

        void Compute(OpKernelContext * context) override {
            const Tensor& img_feature_tensor = context->input(0);
            OP_REQUIRES(context, img_feature_tensor.dims()==4, errors::InvalidArgument("PointsFuseGrad expects (batch_size, h, w, c) image feature shape"));
            int b = img_feature_tensor.shape().dim_size(0);
            int h = img_feature_tensor.shape().dim_size(1);
            int w = img_feature_tensor.shape().dim_size(2);
            int c = img_feature_tensor.shape().dim_size(3);

            const Tensor& img_pc_tensor = context->input(1);
            OP_REQUIRES(context, img_pc_tensor.dims()==3 && img_pc_tensor.shape().dim_size(0) == b && img_pc_tensor.shape().dim_size(2)==2, errors::InvalidArgument("PointsFuseGrad expects (batch_size, points_num, 2) image feature shape"));
            int n = img_pc_tensor.shape().dim_size(1);

            const Tensor& feature_grad_tensor = context->input(2); // [batch_size, points_num, channels]
            OP_REQUIRES(context, feature_grad_tensor.dims()==3 && feature_grad_tensor.shape().dim_size(0)==b && feature_grad_tensor.shape().dim_size(1)==n && feature_grad_tensor.shape().dim_size(2)==c, errors::InvalidArgument("PointsFuseGrad expects (batch_size, points_num, c) image feature shape"));
            
            Tensor *img_grad_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,h,w,c}, &img_grad_tensor));

            auto img_feature_flat = img_feature_tensor.flat<float>();
            const float* img_feature = &(img_feature_flat(0));
            auto img_pc_flat = img_pc_tensor.flat<float>();
            const float* img_pc = &(img_pc_flat(0));
            auto feature_grad_flat = feature_grad_tensor.flat<float>();
            const float * feature_grad = &(feature_grad_flat(0));

            auto img_grad_flat = img_grad_tensor->flat<float>();
            float* img_grad = &(img_grad_flat(0));
            cudaMemset(img_grad, 0, sizeof(float) * b*h*w*c);

            // CUDA code
            pointsFuseGradLauncher(b, n, h, w, c, down_sample_rate_, img_feature, img_pc, feature_grad, img_grad);
        }
    private:
        float down_sample_rate_;
};
REGISTER_KERNEL_BUILDER(Name("PointsFuseGrad").Device(DEVICE_GPU),PointsFuseGradGpuOp);

