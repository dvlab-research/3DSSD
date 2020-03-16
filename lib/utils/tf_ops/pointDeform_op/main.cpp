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
#include <iostream>

using namespace tensorflow;

REGISTER_OP("SelectionKRadius")
    .Attr("radius: float")
    .Input("idx: int32")
    .Input("val: float32")
    .Output("outi: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1;
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2;
        c->WithRank(c->input(1), 3, &dims2);
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("NearestSelect")
    .Attr("radius: float")
    .Input("xyz: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dim1;
        c->WithRank(c->input(0), 3, &dim1); // batch_size * npoint * 3
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1), 8});
        c->set_output(0, output1); // batch_size * npoint * 8
        return Status::OK();
    });
REGISTER_OP("AddOffset")
    .Input("offset: float32")
    .Input("group_xyz: float32")
    .Input("idx: int32")
    .Output("group_xyz_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });
REGISTER_OP("AddOffsetGrad")
    .Input("offset: float32")
    .Input("group_xyz: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("offset_grad: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void add_offset_cpu(int b, int n, const float *offset, const float* group_xyz, const int* idx, float *group_xyz_out) {
    // add offset cpu
    // offset[b, n, 24]
    // group_xyz[b, n, 8, 3]
    // idx [b, n, 8], offset for each
    // group_xyz_out, [b, n, 8, 3]
    for (int i=0; i < b; i++){
        for (int j=0; j<n; j ++){
            for (int k=0; k < 8; k++){
                int cur_index = idx[j * 8 + k];
                float offset1 = offset[cur_index * 24 + k * 3];
                float offset2 = offset[cur_index * 24 + k * 3 + 1];
                float offset3 = offset[cur_index * 24 + k * 3 + 2];
                group_xyz_out[j * 8 * 3 + k * 3] = group_xyz[j * 8 * 3 + k * 3] + offset1;
                group_xyz_out[j * 8 * 3 + k * 3 + 1] = group_xyz[j * 8 * 3 + k * 3 + 1] + offset2;
                group_xyz_out[j * 8 * 3 + k * 3 + 2] = group_xyz[j * 8 * 3 + k * 3 + 2] + offset3; 
            }
        }
        offset += n * 24;
        group_xyz += n * 24;
        idx += n * 8;
        group_xyz_out += n * 24;
    }
}

void add_offset_grad_cpu(int b, int n, const float *offset, const float *group_xyz, const int *idx, const float *grad_out, float *offset_grad){
    // offset: [b, n, 24]
    // group_xyz: [b, n, 8, 3]
    // idx: [b, n, 8]
    // grad_out: [b, n, 8, 3]
    // offset_grad: [b, n, 24]
    for (int i=0; i<b;i++){
        for (int j=0; j<n;j++){
            for (int k=0; k<8;k++){
                int cur_index = idx[j * 8 + k];
                offset_grad[cur_index * 24 + k * 3] += grad_out[j * 8 * 3 + k * 3];
                offset_grad[cur_index * 24 + k * 3 + 1] += grad_out[j * 8 * 3 + k * 3 + 1];
                offset_grad[cur_index * 24 + k * 3 + 2] += grad_out[j * 8 * 3 + k * 3 + 2];
            }
        }
        offset_grad += n * 24;
        grad_out += n * 24;
        idx += n * 8;
        group_xyz += n * 24;
        offset += n*24;
    }
}


void selectionKRadiusLauncher(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out);
class SelectionKRadiusOp: public OpKernel {
public:
    explicit SelectionKRadiusOp(OpKernelConstruction * context):OpKernel(context){
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& idx_tensor = context->input(0);
        const Tensor& val_tensor = context->input(1);
        int b = idx_tensor.shape().dim_size(0);
        int m = idx_tensor.shape().dim_size(1);
        int k = idx_tensor.shape().dim_size(2);

        Tensor* idx_out_tensor = nullptr;
        Tensor* val_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,k}, &idx_out_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,k}, &val_out_tensor));

        auto idx_flat = idx_tensor.flat<int>();
        auto val_flat = val_tensor.flat<float>();
        const int* idx = &(idx_flat(0));
        const float* val = &(val_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        auto val_out_flat = val_out_tensor->flat<float>();
        int* idx_out = &(idx_out_flat(0));
        float* val_out = &(val_out_flat(0));
        selectionKRadiusLauncher(b, m, k, radius_, idx, val, idx_out, val_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionKRadius").Device(DEVICE_GPU),SelectionKRadiusOp);

void nearestSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out);
class NearestSelectOp : public OpKernel {
public:
    explicit NearestSelectOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 8}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));

        nearestSelectLauncher(b, n, radius_, xyz, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("NearestSelect").Device(DEVICE_GPU), NearestSelectOp);

class AddOffsetOp : public OpKernel {
    public:
        explicit AddOffsetOp(OpKernelConstruction * context): OpKernel(context) {
        }
    
        void Compute(OpKernelContext* context) override {
            const Tensor& offset_tensor=context->input(0);
            OP_REQUIRES(context, offset_tensor.dims()==3, errors::InvalidArgument("AddOffsetOp expects (b,num_points,3*8) points shape"));
            int b = offset_tensor.shape().dim_size(0);
            int n = offset_tensor.shape().dim_size(1);
    
            const Tensor& group_xyz_tensor = context->input(1);
            OP_REQUIRES(context, group_xyz_tensor.dims()==4 && group_xyz_tensor.shape().dim_size(0)==b && group_xyz_tensor.shape().dim_size(1)==n && group_xyz_tensor.shape().dim_size(2)==8 && group_xyz_tensor.shape().dim_size(3)==3, errors::InvalidArgument("AddOffsetOp expects (b,num_points,8, 3) group_xyz shape"));
    
            const Tensor& idx_tensor = context->input(2);
            OP_REQUIRES(context, idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0) == b && idx_tensor.shape().dim_size(1)==n && idx_tensor.shape().dim_size(2)==8, errors::InvalidArgument("AddOffsetOp expects (b,num_points,8) idx_tensor shape"));
    
            Tensor * group_xyz_out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,8,3}, &group_xyz_out_tensor));        
            auto offset_flat = offset_tensor.flat<float>();
            const float *offset = &(offset_flat(0));
            auto group_xyz_flat = group_xyz_tensor.flat<float>();
            const float *group_xyz = &(group_xyz_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx= &(idx_flat(0));
            auto group_xyz_out_flat = group_xyz_out_tensor->flat<float>();
            float *group_xyz_out= &(group_xyz_out_flat(0));

            add_offset_cpu(b, n, offset, group_xyz, idx, group_xyz_out);            
        }
}; 
REGISTER_KERNEL_BUILDER(Name("AddOffset").Device(DEVICE_CPU),AddOffsetOp);

class AddOffsetGradOp : public OpKernel {
    public:
        explicit AddOffsetGradOp(OpKernelConstruction * context): OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& offset_tensor=context->input(0);
            OP_REQUIRES(context, offset_tensor.dims()==3, errors::InvalidArgument("AddOffsetOp expects (b,num_points,3*8) points shape"));
            int b = offset_tensor.shape().dim_size(0);
            int n = offset_tensor.shape().dim_size(1);

            const Tensor& group_xyz_tensor = context->input(1);
            OP_REQUIRES(context, group_xyz_tensor.dims()==4 && group_xyz_tensor.shape().dim_size(0)==b && group_xyz_tensor.shape().dim_size(1)==n && group_xyz_tensor.shape().dim_size(2)==8 && group_xyz_tensor.shape().dim_size(3)==3, errors::InvalidArgument("AddOffsetOp expects (b,num_points,8, 3) group_xyz shape"));

            const Tensor& idx_tensor = context->input(2);
            OP_REQUIRES(context, idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0) == b && idx_tensor.shape().dim_size(1)==n && idx_tensor.shape().dim_size(2)==8, errors::InvalidArgument("AddOffsetOp expects (b,num_points,8) idx_tensor shape"));            

            const Tensor& grad_out_tensor = context->input(3);
            OP_REQUIRES(context, grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0) == b && grad_out_tensor.shape().dim_size(1)==n && grad_out_tensor.shape().dim_size(2)==8, errors::InvalidArgument("AddOffsetGradOp expects (b,num_points,8, 3) grad_out_tensor shape")); 

            Tensor * offset_grad_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,24}, &offset_grad_tensor));

            auto offset_flat = offset_tensor.flat<float>();
            const float *offset = &(offset_flat(0));
            auto group_xyz_flat = group_xyz_tensor.flat<float>();
            const float *group_xyz = &(group_xyz_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx= &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto offset_grad_flat = offset_grad_tensor->flat<float>();
            float *offset_grad= &(offset_grad_flat(0));
            memset(offset_grad, 0, sizeof(float)*b*n*24);

            add_offset_grad_cpu(b, n, offset, group_xyz, idx, grad_out, offset_grad);
        }
};
REGISTER_KERNEL_BUILDER(Name("AddOffsetGrad").Device(DEVICE_CPU),AddOffsetGradOp);
