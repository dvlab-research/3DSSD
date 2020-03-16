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

REGISTER_OP("PointsNms")
  .Input("iou_matrix: float32")
  .Input("points_sample: int32")
  .Attr("merge_function: int")
  .Attr("iou_thresh: float")
  .Output("keep_inds: int32") // [n]
  .Output("nmsed_points_sample: int32") // [n, npoint]
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // n * n
    c->WithRank(c->input(0), 2, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // n * npoints
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints
    ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0)});
    ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
    c->set_output(0, output1);
    c->set_output(1, output2);
    return Status::OK();
  });


REGISTER_OP("PointsInsideBoxes")
  .Input("points: float32") // [npoint, 3]
  .Input("anchors: float32") // [anchors_num, 6]
  .Output("points_sample_mask: int32") // [n, npoint]
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // npoint * 3
    c->WithRank(c->input(0), 2, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // box_num * 6 
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims1, 0)});
    c->set_output(0, output);
    return Status::OK();
  });


REGISTER_OP("PointsNmsBlock")
  .Input("points_sample: int32")
  .Attr("merge_function: int")
  .Attr("iou_thresh: float")
  .Attr("num_to_keep: int")
  .Output("keep_inds: int32") // [n]
  .Output("nmsed_points_sample: int32") // [n, npoint]
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims; // n * npoints
    c->WithRank(c->input(0), 2, &dims);
    // batch_size * npoints
    int num_to_keep;
    TF_RETURN_IF_ERROR(c->GetAttr("num_to_keep", &num_to_keep));
    ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({num_to_keep});
    ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims, 0), c->Dim(dims, 1)});
    c->set_output(0, output1);
    c->set_output(1, output2);
    return Status::OK();
  });


REGISTER_OP("PointsIou")
  .Input("points_sample_mask: int32") // [n, npoint]
  .Output("iou_matrix: float32") // [n, n]
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims; // n * npoint
    c->WithRank(c->input(0), 2, &dims);
    // batch_size * npoints
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims, 0), c->Dim(dims, 0)});
    c->set_output(0, output);
    return Status::OK();
  });


void points_nms_gpu(const int n, const int npoint, const int merge_function, float iou_thresh, const float *iou_matrix, const int *points_sample, int *keep_inds, int *nmsed_points_sample);
class PointsNmsGpuOp: public OpKernel{
    public:
        explicit PointsNmsGpuOp(OpKernelConstruction * context): OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("merge_function", &merge_function_));
            OP_REQUIRES(context, merge_function_ == 1 || merge_function_ == 0 || merge_function_ == 2, errors::InvalidArgument("PointsNMS only support 0 or 1 or 2 value"));
            OP_REQUIRES_OK(context, context->GetAttr("iou_thresh", &iou_thresh_));
            OP_REQUIRES(context, iou_thresh_ >= 0, errors::InvalidArgument("Points nms iou thresh only support [0, 1]"));
        }

        void Compute(OpKernelContext * context) override{
            const Tensor& iou_matrix_tensor = context->input(0);
            OP_REQUIRES(context, iou_matrix_tensor.dims()==2 && iou_matrix_tensor.shape().dim_size(0)==iou_matrix_tensor.shape().dim_size(1), errors::InvalidArgument("PointsNMS operation expects (n, n) shape"));
            int n = iou_matrix_tensor.shape().dim_size(0);

            const Tensor& points_sample_tensor = context->input(1);
            OP_REQUIRES(context, points_sample_tensor.dims()==2 && points_sample_tensor.shape().dim_size(0)==n, errors::InvalidArgument("expect (n, npoints) shape for points NMS op"));
            int npoints = points_sample_tensor.shape().dim_size(1);

            // then allocate the output
            Tensor *keep_inds_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{n}, &keep_inds_tensor));

            Tensor *nmsed_points_sample_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{n, npoints}, &nmsed_points_sample_tensor));

            auto iou_matrix_flat = iou_matrix_tensor.flat<float>();
            const float *iou_matrix = &(iou_matrix_flat(0));
            auto points_sample_flat = points_sample_tensor.flat<int>();
            const int *points_sample = &(points_sample_flat(0));
            auto keep_inds_flat = keep_inds_tensor->flat<int>();
            int *keep_inds = &(keep_inds_flat(0));
            auto nmsed_points_sample_flat = nmsed_points_sample_tensor->flat<int>();
            int *nmsed_points_sample = &(nmsed_points_sample_flat(0));

            points_nms_gpu(n, npoints, merge_function_, iou_thresh_, iou_matrix, points_sample, keep_inds, nmsed_points_sample);
        }
    private:
        float iou_thresh_;
        int merge_function_;
};
REGISTER_KERNEL_BUILDER(Name("PointsNms").Device(DEVICE_GPU), PointsNmsGpuOp) 



void points_nms_block_gpu(const int n, const int npoint, const int merge_function, const float iou_thresh, const int num_to_keep, const int *points_sample, int *keep_inds, int *nmsed_points_sample);
class PointsNmsBlockGpuOp: public OpKernel{
    public:
        explicit PointsNmsBlockGpuOp(OpKernelConstruction * context): OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("merge_function", &merge_function_));
            OP_REQUIRES(context, merge_function_ == 1 || merge_function_ == 0 || merge_function_ == 2, errors::InvalidArgument("PointsNMS only support 0 or 1 or 2 value"));

            OP_REQUIRES_OK(context, context->GetAttr("iou_thresh", &iou_thresh_));
            OP_REQUIRES(context, iou_thresh_ >= 0, errors::InvalidArgument("Points nms iou thresh only support [0, 1]"));

            OP_REQUIRES_OK(context, context->GetAttr("num_to_keep", &num_to_keep_));
            OP_REQUIRES(context, num_to_keep_ >= 0, errors::InvalidArgument("Points nms num_to_keep must greater than 0"));
        }

        void Compute(OpKernelContext * context) override{

            const Tensor& points_sample_tensor = context->input(0);
            OP_REQUIRES(context, points_sample_tensor.dims()==2, errors::InvalidArgument("expect (n, npoints) shape for points NMS op"));
            int n = points_sample_tensor.shape().dim_size(0);
            int npoints = points_sample_tensor.shape().dim_size(1);

            // then allocate the output
            Tensor *keep_inds_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{num_to_keep_}, &keep_inds_tensor));

            Tensor *nmsed_points_sample_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{n, npoints}, &nmsed_points_sample_tensor));

            auto points_sample_flat = points_sample_tensor.flat<int>();
            const int *points_sample = &(points_sample_flat(0));
            auto keep_inds_flat = keep_inds_tensor->flat<int>();
            int *keep_inds = &(keep_inds_flat(0));
            auto nmsed_points_sample_flat = nmsed_points_sample_tensor->flat<int>();
            int *nmsed_points_sample = &(nmsed_points_sample_flat(0));

            points_nms_block_gpu(n, npoints, merge_function_, iou_thresh_, num_to_keep_, points_sample, keep_inds, nmsed_points_sample);
        }
    private:
        float iou_thresh_;
        int merge_function_;
        int num_to_keep_;
};
REGISTER_KERNEL_BUILDER(Name("PointsNmsBlock").Device(DEVICE_GPU), PointsNmsBlockGpuOp) 



void points_iou_gpu(const int n, const int npoint, const int* points_sample_mask, float* iou_matrix);
class PointsIouGpuOp: public OpKernel{
    public:
        explicit PointsIouGpuOp(OpKernelConstruction * context): OpKernel(context){}

        void Compute(OpKernelContext * context) override{
            const Tensor& points_sample_mask_tensor = context->input(0);
            OP_REQUIRES(context, points_sample_mask_tensor.dims()==2, errors::InvalidArgument("PointsNMS operation expects (n, npoint) shape"));
            int n = points_sample_mask_tensor.shape().dim_size(0);
            int npoint = points_sample_mask_tensor.shape().dim_size(1);

            // then allocate the output
            Tensor *iou_matrix_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{n, n}, &iou_matrix_tensor));


            auto points_sample_mask_flat = points_sample_mask_tensor.flat<int>();
            const int *points_sample_mask = &(points_sample_mask_flat(0));

            auto iou_matrix_flat = iou_matrix_tensor->flat<float>();
            float*iou_matrix = &(iou_matrix_flat(0));

            points_iou_gpu(n, npoint, points_sample_mask, iou_matrix);
        }
};
REGISTER_KERNEL_BUILDER(Name("PointsIou").Device(DEVICE_GPU), PointsIouGpuOp) 


void points_inside_boxes_gpu(const int n, const int npoint, const float *points, const float* anchors, int* points_sample_mask);
class PointsInsideBoxesGpuOp: public OpKernel{
    public:
        explicit PointsInsideBoxesGpuOp(OpKernelConstruction * context): OpKernel(context){}

        void Compute(OpKernelContext * context) override{
            const Tensor& points_tensor = context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==2 && points_tensor.shape().dim_size(1)==3, errors::InvalidArgument("PointsInsideBoxesOp operation expects (npoint, 3) shape"));
            int npoint = points_tensor.shape().dim_size(0);

            const Tensor& anchors_tensor = context->input(1);
            OP_REQUIRES(context, anchors_tensor.dims()==2 && anchors_tensor.shape().dim_size(1)==6, errors::InvalidArgument("PointsInsideBoxesOp operation expects (anchors_num, 6) shape"));
            int n = anchors_tensor.shape().dim_size(0);

            // then allocate the output
            Tensor *points_sample_mask_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{n, npoint}, &points_sample_mask_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float* points = &(points_flat(0));
            auto anchors_flat = anchors_tensor.flat<float>();
            const float* anchors = &(anchors_flat(0));

            auto points_sample_mask_flat = points_sample_mask_tensor->flat<int>();
            int *points_sample_mask = &(points_sample_mask_flat(0));

            points_inside_boxes_gpu(n, npoint, points, anchors, points_sample_mask);
        }
};
REGISTER_KERNEL_BUILDER(Name("PointsInsideBoxes").Device(DEVICE_GPU), PointsInsideBoxesGpuOp) 
