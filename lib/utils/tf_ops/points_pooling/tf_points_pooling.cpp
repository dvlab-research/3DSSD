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

REGISTER_OP("PointsPooling")
    .Attr("l: int")
    .Attr("h: int")
    .Attr("w: int")
    .Attr("sample_num: int")
    .Input("pc: float32") // [bs, proposal_num, point_num, c]
    .Input("box_3d: float32") // [bs, proposal_num, 6]
    .Input("pc_loc: float32") // [bs, proposal_num, point_num, 3]
    .Output("out_features: float32") // [bs, proposal_num, l, h, w, sample_num, c]
    .Output("out_idx: int32") // [bs, proposal_num, l, h, w, sample_num]
    .Output("sampled_num_lists: int32") // [bs, proposal_num, l, h, w]
    .Output("pillars: float32") // [bs, proposal_num, l, h, w, 3], ctrs for each pillars
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // [bs, proposal_num, point_num, c]
        c->WithRank(c->input(0), 4, &dims1);
        int l, h, w, sample_num;
        TF_RETURN_IF_ERROR(c->GetAttr("l", &l));
        TF_RETURN_IF_ERROR(c->GetAttr("h", &h));
        TF_RETURN_IF_ERROR(c->GetAttr("w", &w));
        TF_RETURN_IF_ERROR(c->GetAttr("sample_num", &sample_num));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), l, h, w, sample_num, c->Dim(dims1, 3)});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), l, h, w, sample_num});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), l, h, w});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), l, h, w, 3});
        c->set_output(3, output4);
        return Status::OK();
    });

REGISTER_OP("PointsPoolingGrad")
    .Input("pc: float32") // [bs, proposal_num, point_num, c]
    .Input("out_idx: int32") // [bs, proposal_num, l, h, w, sample_num]
    .Input("sampled_num_lists: int32") // [bs, proposal_num, l, h, w]
    .Input("features_grad: float32") // [bs, proposal_num, l, h, w, sample_num, c]
    .Output("pc_grad: float32") // [bs, proposal_num, point_num, c]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void pointsPoolingLauncher(const int bs, const int proposal_num, const int point_num, const int channel_num, 
    const int l, const int h, const int w, const int sample_num, 
    const float* pc, const float* box_3d, const float* pc_loc, 
    float* out_features, int* out_idx, int *sampled_num_lists, float* pillars);

class PointsPoolingGpuOp: public OpKernel{
    public:
        explicit PointsPoolingGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("l", &l_));
            OP_REQUIRES(context, l_ > 0, errors::InvalidArgument("PointsPooling method expects positive length"));

            OP_REQUIRES_OK(context, context->GetAttr("h", &h_));
            OP_REQUIRES(context, h_ > 0, errors::InvalidArgument("PointsPooling method expects positive height"));

            OP_REQUIRES_OK(context, context->GetAttr("w", &w_));
            OP_REQUIRES(context, w_ > 0, errors::InvalidArgument("PointsPooling method expects positive width"));

            OP_REQUIRES_OK(context, context->GetAttr("sample_num", &sample_num_));
            OP_REQUIRES(context, sample_num_ > 0, errors::InvalidArgument("PointsPooling method expects positive sample number"));
        }

        void Compute(OpKernelContext * context) override {
            const Tensor& pc_tensor = context->input(0);
            OP_REQUIRES(context, pc_tensor.dims()==4, 
                errors::InvalidArgument("PointsPooling expects (bs, proposal_num, num_points, channel) points shape"));
            int bs = pc_tensor.shape().dim_size(0);
            int proposal_num = pc_tensor.shape().dim_size(1);
            int point_num = pc_tensor.shape().dim_size(2); 
            int channel_num = pc_tensor.shape().dim_size(3); 

            const Tensor& box_3d_tensor = context->input(1);
            OP_REQUIRES(context, box_3d_tensor.dims()==3 && 
                box_3d_tensor.shape().dim_size(0)==bs && 
                box_3d_tensor.shape().dim_size(1)==proposal_num && 
                box_3d_tensor.shape().dim_size(2)==6, 
                errors::InvalidArgument("PointsPooling expects (bs, proposal_num, 6) proposal tensor shape"));

            const Tensor& pc_loc_tensor = context->input(2);
            OP_REQUIRES(context, pc_loc_tensor.dims()==4 && 
                pc_loc_tensor.shape().dim_size(0)==bs && 
                pc_loc_tensor.shape().dim_size(1)==proposal_num && 
                pc_loc_tensor.shape().dim_size(2)==point_num && 
                pc_loc_tensor.shape().dim_size(3)==3, 
                errors::InvalidArgument("PointsPooling expects (proposal_num, points_num, 3) points location tensor shape"));
            

            Tensor * out_features_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,
                TensorShape{bs, proposal_num, l_, h_, w_, sample_num_, channel_num}, &out_features_tensor));

            Tensor * out_idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1,
                TensorShape{bs, proposal_num, l_, h_, w_, sample_num_}, &out_idx_tensor));

            Tensor *sampled_num_lists_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2,
                TensorShape{bs, proposal_num, l_, h_, w_}, &sampled_num_lists_tensor));

            Tensor *pillars_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3,
                TensorShape{bs, proposal_num, l_, h_, w_, 3}, &pillars_tensor));

            auto pc_flat = pc_tensor.flat<float>();
            const float *pc = &(pc_flat(0));
            auto box_3d_flat = box_3d_tensor.flat<float>();
            const float *box_3d = &(box_3d_flat(0));
            auto pc_loc_flat = pc_loc_tensor.flat<float>();
            const float *pc_loc = &(pc_loc_flat(0));
          
            auto out_features_flat = out_features_tensor->flat<float>();
            float * out_features = &(out_features_flat(0));
            auto out_idx_flat = out_idx_tensor->flat<int>();
            int * out_idx = &(out_idx_flat(0));
            auto sampled_num_lists_flat = sampled_num_lists_tensor->flat<int>();
            int * sampled_num_lists = &(sampled_num_lists_flat(0));
            auto pillars_flat = pillars_tensor->flat<float>();
            float* pillars = &(pillars_flat(0));

            cudaMemset(out_features, 0, 
                sizeof(float) * bs * proposal_num * l_ *h_ *w_ * sample_num_ * channel_num);
            cudaMemset(out_idx, 0, 
                sizeof(int) * bs * proposal_num * l_ * h_ * w_ * sample_num_);
            cudaMemset(sampled_num_lists, 0, 
                sizeof(int) * bs * proposal_num * l_ * h_ * w_);
            cudaMemset(pillars, 0, 
                sizeof(float)* bs * proposal_num * l_ * h_ * w_ * 3);
 
            pointsPoolingLauncher(bs, proposal_num, point_num, channel_num, 
                l_, h_, w_, sample_num_, 
                pc, box_3d, pc_loc,
                out_features, out_idx, sampled_num_lists, pillars);
        }
    private:
        int l_, h_, w_, sample_num_;
};
REGISTER_KERNEL_BUILDER(Name("PointsPooling").Device(DEVICE_GPU),PointsPoolingGpuOp);



void pointsPoolingGradLauncher(const int bs, const int proposal_num, const int point_num, const int channel_num, 
    const int l, const int h, const int w, const int sample_num, 
    const float* pc, const int* out_idx, const int *sampled_num_lists, const float* features_grad, 
    float *pc_grad);

class PointsPoolingGradGpuOp: public OpKernel{
    public:
        explicit PointsPoolingGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& pc_tensor = context->input(0);
            OP_REQUIRES(context, pc_tensor.dims()==4, 
                errors::InvalidArgument("PointsPooling expects (bs, proposal_num, point_num, channel_num) points shape"));
            int bs = pc_tensor.shape().dim_size(0);
            int proposal_num = pc_tensor.shape().dim_size(1);
            int point_num = pc_tensor.shape().dim_size(2);
            int channel_num = pc_tensor.shape().dim_size(3);

            const Tensor& out_idx_tensor=context->input(1);
            OP_REQUIRES(context,out_idx_tensor.dims()==6 &&
                out_idx_tensor.shape().dim_size(0)==bs && 
                out_idx_tensor.shape().dim_size(1)==proposal_num, 
                errors::InvalidArgument("Wrong arguments for out_idx_tensor in grad ops"));
            int l = out_idx_tensor.shape().dim_size(2);
            int h = out_idx_tensor.shape().dim_size(3);
            int w = out_idx_tensor.shape().dim_size(4);
            int sample_num = out_idx_tensor.shape().dim_size(5);
            

            const Tensor& sampled_num_lists_tensor = context->input(2);
            OP_REQUIRES(context,sampled_num_lists_tensor.dims()==5 && 
                sampled_num_lists_tensor.shape().dim_size(0)==bs && 
                sampled_num_lists_tensor.shape().dim_size(1)==proposal_num && 
                sampled_num_lists_tensor.shape().dim_size(2) == l && 
                sampled_num_lists_tensor.shape().dim_size(3) == h && 
                sampled_num_lists_tensor.shape().dim_size(4) == w, 
                errors::InvalidArgument("Wrong shape for grad ops of sampled_num_lists tensor"));

            const Tensor& features_grad_tensor = context->input(3);
            OP_REQUIRES(context,features_grad_tensor.dims()==7 && 
                features_grad_tensor.shape().dim_size(0)==bs && 
                features_grad_tensor.shape().dim_size(1)==proposal_num && 
                features_grad_tensor.shape().dim_size(2) == l && 
                features_grad_tensor.shape().dim_size(3) == h && 
                features_grad_tensor.shape().dim_size(4) == w && 
                features_grad_tensor.shape().dim_size(5)==sample_num && 
                features_grad_tensor.shape().dim_size(6)==channel_num, 
                errors::InvalidArgument("Wrong shape for grad ops of features_grad out put tensor"));

            Tensor *pc_grad_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,
                TensorShape{bs, proposal_num, point_num, channel_num}, &pc_grad_tensor));

            auto pc_flat = pc_tensor.flat<float>();
            const float *pc = &(pc_flat(0));
            auto out_idx_flat = out_idx_tensor.flat<int>();
            const int *out_idx = &(out_idx_flat(0));
            auto sampled_num_lists_flat = sampled_num_lists_tensor.flat<int>();
            const int* sampled_num_lists = &(sampled_num_lists_flat(0));
            auto features_grad_flat = features_grad_tensor.flat<float>();
            const float* features_grad = &(features_grad_flat(0));
          
            auto pc_grad_flat = pc_grad_tensor->flat<float>();
            float* pc_grad = &(pc_grad_flat(0));

            cudaMemset(pc_grad, 0, 
                sizeof(float) * bs * proposal_num * point_num * channel_num);
 
            pointsPoolingGradLauncher(bs, proposal_num, point_num, channel_num, 
                l, h, w, sample_num, 
                pc, out_idx, sampled_num_lists, features_grad, 
                pc_grad);
        }
};
REGISTER_KERNEL_BUILDER(Name("PointsPoolingGrad").Device(DEVICE_GPU),PointsPoolingGradGpuOp);
