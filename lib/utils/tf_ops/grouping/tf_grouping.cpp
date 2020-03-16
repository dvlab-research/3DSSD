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

REGISTER_OP("CalculatePointsIou")
    .Input("batch_points: float32") // [b, n, 3]
    .Input("batch_anchors_corners: float32") // [b, num_anchors, 8, 3]
    .Input("batch_label_corners: float32") // [bs, gt_num, 8, 3]
    .Output("out: float32") // [b, num_anchors, gt_num]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * num_anchors * 8 * 3
        c->WithRank(c->input(1), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * gt_num * 8 * 3
        c->WithRank(c->input(2), 4, &dims2);
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims2, 1)}); // batch_size, num_anchors, gt_num
        c->set_output(0, output1);
        return Status::OK();
    });
REGISTER_OP("QueryCornersPoint")
    .Input("batch_points: float32") // [b, n, 3]
    .Input("batch_anchors_corners: float32") // [b, num_proposals, 8, 3]
    .Output("out: int32") // [b, num_proposals, n]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * num_proposals * 8 * 3
        c->WithRank(c->input(1), 4, &dims2);
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 1)}); // batch_size, num_proposals, npoint
        c->set_output(0, output1);
        return Status::OK();
    });
REGISTER_OP("QueryCubePoint")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Attr("subcube_num: int")
    .Input("xyz1: float32") // batch_size, ndataset, 3
    .Input("xyz2: float32") // batch_size, npoint, 3
    .Output("idx: int32") // batch_size, npoint, nsample
    .Output("pts_cnt: int32") // batch_size, npoint, subcube_num^3
    .Output("subcube_location: float32") // batch_size, npoint, subcube_num ^ 3, 3
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        int subcube_num;
        TF_RETURN_IF_ERROR(c->GetAttr("subcube_num", &subcube_num));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), subcube_num * subcube_num * subcube_num});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), subcube_num * subcube_num * subcube_num, 3});
        c->set_output(2, output3);
        return Status::OK();
    });
REGISTER_OP("QueryBallPointDynamicShape")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("radius: float32") // [bs, npoint, split_bin_num]
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .Output("radius_idx: int32") // [bs, npoint, nsample, 2]
    .Output("radius_rate: float32") // [bs, npoint, nsample, 2]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample, 2});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample, 2});
        c->set_output(3, output4);
        return Status::OK();
    });
REGISTER_OP("QueryDynamicRadiusForPoints")
    .Input("xyz1: float32") // [bs, npoint, nsample, 3]
    .Input("radius: float32") // [bs, npoint, split_bin_num]
    .Output("radius_idx: int32") // [bs, npoint, nsample, 2]
    .Output("radius_rate: float32") // [bs, npoint, nsample, 2]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size, npoint, nsample, 3
        c->WithRank(c->input(0), 4, &dims1);
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims1, 2), 2}); // bs, npoint, nsample, 2
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims1, 2), 2});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("QueryTargetDistanceForPoints")
    .Attr("split_bin_num: int")
    .Input("xyz1: float32") // [bs, npoint, 3]
    .Input("gt_boxes_3d: float32") // [bs, npoint, 7]
    .Output("target_dist: float32") // [bs, npoint, split_bin_num]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // bs, npoint, 3
        c->WithRank(c->input(0), 3, &dims1);
        int split_bin_num;
        TF_RETURN_IF_ERROR(c->GetAttr("split_bin_num", &split_bin_num));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), split_bin_num}); // bs, npoint, nsample, 2
        c->set_output(0, output1);
        return Status::OK();
    });
REGISTER_OP("QueryBallPoint")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("QueryBallPointDilated")
    .Attr("min_radius: float")
    .Attr("max_radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32") // bs * n * 3
    .Input("xyz2: float32") // bs * m * 3
    .Output("idx: int32") // bs * m * nsample
    .Output("pts_cnt: int32") // bs * m
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("QueryBallPointWithidx")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("sort_idx: int32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("QueryBallPointDynamicRadius")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("radius: float32") // [bs, npoint]
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("SelectionSort")
    .Attr("k: int")
    .Input("dist: float32")
    .Output("outi: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("GroupPoint")
    .Input("points: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims2, 2), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("GroupPointGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });



// CalculatePointsIou
// void calculatePointsIouLauncher(int b, int n, int anchors_num, int gt_num, const float* batch_points, const float* batch_anchors_corners, const float* batch_label_corners, float* out);
class CalculatePointsIouGpuOp : public OpKernel {
    public:
        explicit CalculatePointsIouGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& batch_points_tensor = context->input(0); // [b, n, 3]
            OP_REQUIRES(context, batch_points_tensor.dims()==3 && batch_points_tensor.shape().dim_size(2)==3, errors::InvalidArgument("CalculatePointsIou expects (batch_size, ndataset, 3) batch points shape."));
            int b = batch_points_tensor.shape().dim_size(0);
            int n = batch_points_tensor.shape().dim_size(1);

            const Tensor& batch_anchors_corners_tensor = context->input(1);
            OP_REQUIRES(context, batch_anchors_corners_tensor.dims()==4 && batch_anchors_corners_tensor.shape().dim_size(0)==b && batch_anchors_corners_tensor.shape().dim_size(2)==8 && batch_anchors_corners_tensor.shape().dim_size(3)==3, errors::InvalidArgument("CalculatePointsIou expects (batch_size, num_proposals, 8, 3) xyz2 shape."));
            int num_anchors = batch_anchors_corners_tensor.shape().dim_size(1);

            const Tensor& batch_label_corners_tensor = context->input(2);
            OP_REQUIRES(context, batch_label_corners_tensor.dims()==4 && batch_label_corners_tensor.shape().dim_size(0)==b && batch_label_corners_tensor.shape().dim_size(2)==8 && batch_label_corners_tensor.shape().dim_size(3)==3, errors::InvalidArgument("CalculatePointsIou expects (batch_size, gt_num, 8, 3) xyz2 shape."));
            int gt_num = batch_label_corners_tensor.shape().dim_size(1);

            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,num_anchors,gt_num}, &out_tensor));

            auto batch_points_flat = batch_points_tensor.flat<float>();
            const float *batch_points = &(batch_points_flat(0));
            auto batch_anchors_corners_flat = batch_anchors_corners_tensor.flat<float>();
            const float *batch_anchors_corners= &(batch_anchors_corners_flat(0));
            auto batch_label_corners_flat = batch_label_corners_tensor.flat<float>();
            const float *batch_label_corners = &(batch_label_corners_flat(0));

            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            // calculatePointsIouLauncher(b, n, num_anchors, gt_num, batch_points, batch_anchors_corners, batch_label_corners, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("CalculatePointsIou").Device(DEVICE_GPU), CalculatePointsIouGpuOp);


// QueryCornersPoint
void queryCornersPointLauncher(int b, int n, int proposals_num, const float* batch_points, const float* batch_anchors_corners, int *out);
class QueryCornersPointGpuOp : public OpKernel {
    public:
        explicit QueryCornersPointGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& batch_points_tensor = context->input(0); // [b, n, 3]
            OP_REQUIRES(context, batch_points_tensor.dims()==3 && batch_points_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryCornersPoints expects (batch_size, ndataset, 3) batch points shape."));
            int b = batch_points_tensor.shape().dim_size(0);
            int n = batch_points_tensor.shape().dim_size(1);

            const Tensor& batch_anchors_corners_tensor = context->input(1);
            OP_REQUIRES(context, batch_anchors_corners_tensor.dims()==4 && batch_anchors_corners_tensor.shape().dim_size(0)==b && batch_anchors_corners_tensor.shape().dim_size(2)==8 && batch_anchors_corners_tensor.shape().dim_size(3)==3, errors::InvalidArgument("QueryCornersPoints expects (batch_size, num_proposals, 8, 3) xyz2 shape."));
            int proposals_num = batch_anchors_corners_tensor.shape().dim_size(1);

            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,proposals_num,n}, &out_tensor));

            auto batch_points_flat = batch_points_tensor.flat<float>();
            const float *batch_points = &(batch_points_flat(0));
            auto batch_anchors_corners_flat = batch_anchors_corners_tensor.flat<float>();
            const float *batch_anchors_corners= &(batch_anchors_corners_flat(0));
            auto out_flat = out_tensor->flat<int>();
            int *out = &(out_flat(0));
            queryCornersPointLauncher(b, n, proposals_num, batch_points, batch_anchors_corners, out);
        }
};
REGISTER_KERNEL_BUILDER(Name("QueryCornersPoint").Device(DEVICE_GPU), QueryCornersPointGpuOp);


// QueryCubePoint 
void queryCubePointLauncher(int b, int n, int m, float radius, int nsample, int subcube_num, int total_subcube_num, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt, float* subcube_location);
class QueryCubePointGpuOp : public OpKernel {
    public:
        explicit QueryCubePointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryCubePoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryCubePoint expects positive nsample"));

            OP_REQUIRES_OK(context, context->GetAttr("subcube_num", &subcube_num_));
            OP_REQUIRES(context, subcube_num_ > 0, errors::InvalidArgument("QueryCubePoint expects positive subcube_num"));
            total_subcube_num_ = subcube_num_ * subcube_num_ * subcube_num_;
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryCubePoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryCubePoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,total_subcube_num_}, &pts_cnt_tensor));
            Tensor *subcube_location_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,m,total_subcube_num_,3}, &subcube_location_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            auto subcube_location_flat = subcube_location_tensor->flat<float>();
            float *subcube_location = &(subcube_location_flat(0));
            queryCubePointLauncher(b,n,m,radius_,nsample_,subcube_num_,total_subcube_num_,xyz1,xyz2,idx,pts_cnt,subcube_location);
        }
    private:
        float radius_;
        int nsample_;
        int subcube_num_;
        int total_subcube_num_;
};
REGISTER_KERNEL_BUILDER(Name("QueryCubePoint").Device(DEVICE_GPU), QueryCubePointGpuOp);

// QueryBallPointDynamicShape 
void queryBallPointDynamicShapeLauncher(int b, int n, int m, int split_bin_num, int nsample, const float *xyz1, const float *xyz2, const float* radius, int *idx, int *pts_cnt, int* radius_idx, float* radius_rate);
class QueryBallPointDynamicShapeGpuOp : public OpKernel {
    public:
        explicit QueryBallPointDynamicShapeGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPointDynamicShapeGpuOp expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            const Tensor& radius_tensor = context->input(2);
            OP_REQUIRES(context, radius_tensor.dims()==3 && radius_tensor.shape().dim_size(1)==m, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, split_bin_num) radius shape."));
            int split_bin_num = radius_tensor.shape().dim_size(2);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));
            Tensor *radius_idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,m,nsample_,2}, &radius_idx_tensor));
            Tensor *radius_rate_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,m,nsample_,2}, &radius_rate_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto radius_flat = radius_tensor.flat<float>();
            const float *radius = &(radius_flat(0));

            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));

            auto radius_idx_flat = radius_idx_tensor->flat<int>();
            int *radius_idx = &(radius_idx_flat(0));
            auto radius_rate_flat = radius_rate_tensor->flat<float>();
            float *radius_rate = &(radius_rate_flat(0));
            queryBallPointDynamicShapeLauncher(b,n,m,split_bin_num,nsample_,xyz1,xyz2,radius,idx,pts_cnt,radius_idx,radius_rate);
        }
    private:
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPointDynamicShape").Device(DEVICE_GPU), QueryBallPointDynamicShapeGpuOp);

// QueryDynamicRadiusForPoints 
void queryDynamicRadiusForPointsLauncher(int b, int n, int nsample, int split_bin_num, const float *xyz1, const float* radius, int* radius_idx, float* radius_rate);
class QueryDynamicRadiusForPointsGpuOp : public OpKernel {
    public:
        explicit QueryDynamicRadiusForPointsGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==4 && xyz1_tensor.shape().dim_size(3)==3, errors::InvalidArgument("QueryDynamicRadiusForPoints expects (batch_size, npoint, nsample, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);
            int nsample = xyz1_tensor.shape().dim_size(2);

            const Tensor& radius_tensor = context->input(1);
            OP_REQUIRES(context, radius_tensor.dims()==3 && radius_tensor.shape().dim_size(1)==n, errors::InvalidArgument("QueryDynamicRadiusForPoints expects (batch_size, npoint, split_bin_num) radius shape."));
            int split_bin_num = radius_tensor.shape().dim_size(2);

            Tensor *radius_idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,nsample,2}, &radius_idx_tensor));
            Tensor *radius_rate_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,nsample,2}, &radius_rate_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto radius_flat = radius_tensor.flat<float>();
            const float *radius = &(radius_flat(0));

            auto radius_idx_flat = radius_idx_tensor->flat<int>();
            int *radius_idx = &(radius_idx_flat(0));
            auto radius_rate_flat = radius_rate_tensor->flat<float>();
            float *radius_rate = &(radius_rate_flat(0));
            queryDynamicRadiusForPointsLauncher(b,n,nsample,split_bin_num,xyz1,radius,radius_idx,radius_rate);
        }
};
REGISTER_KERNEL_BUILDER(Name("QueryDynamicRadiusForPoints").Device(DEVICE_GPU), QueryDynamicRadiusForPointsGpuOp);


// QueryTargetDistanceForPoints
void queryTargetDistanceForPointsLauncher(int b, int n, int split_bin_num, const float* xyz1, const float* gt_boxes_3d, float* target_dist);
class QueryTargetDistanceForPointsGpuOp : public OpKernel {
    public:
        explicit QueryTargetDistanceForPointsGpuOp(OpKernelConstruction* context) : OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("split_bin_num", &split_bin_num_));
            OP_REQUIRES(context, split_bin_num_ > 0, errors::InvalidArgument("QueryTargetDistanceForPointsGpuOp expects positive split_bin_num"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryTargetDistanceForPointsGpuOp expects (batch_size, npoint, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& gt_boxes_3d_tensor = context->input(1);
            OP_REQUIRES(context, gt_boxes_3d_tensor.dims()==3 && gt_boxes_3d_tensor.shape().dim_size(1)==n && gt_boxes_3d_tensor.shape().dim_size(2)==7, errors::InvalidArgument("QueryTargetDistanceForPointsGpuOp expects (batch_size, npoint, 7) radius shape."));

            Tensor *target_dist_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,split_bin_num_}, &target_dist_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto gt_boxes_3d_flat = gt_boxes_3d_tensor.flat<float>();
            const float *gt_boxes_3d = &(gt_boxes_3d_flat(0));

            auto target_dist_flat = target_dist_tensor->flat<float>();
            float *target_dist = &(target_dist_flat(0));
            queryTargetDistanceForPointsLauncher(b,n,split_bin_num_,xyz1,gt_boxes_3d,target_dist);
        }
    private:
        int split_bin_num_;
};
REGISTER_KERNEL_BUILDER(Name("QueryTargetDistanceForPoints").Device(DEVICE_GPU), QueryTargetDistanceForPointsGpuOp);

// QueryBallPoint
void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
class QueryBallPointGpuOp : public OpKernel {
    public:
        explicit QueryBallPointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU), QueryBallPointGpuOp);


// QueryBallPointWithidx
void queryBallPointWithidxLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, const int* sort_idx, int *idx, int *pts_cnt);
class QueryBallPointWithidxGpuOp : public OpKernel {
    public:
        explicit QueryBallPointWithidxGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPointWithidx expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPointWithidx expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPointWithidx expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPointWithidx expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            const Tensor& sort_idx_tensor = context->input(2);
            OP_REQUIRES(context, sort_idx_tensor.dims()==3 && sort_idx_tensor.shape().dim_size(1)==m && sort_idx_tensor.shape().dim_size(2)==n, errors::InvalidArgument("QueryBallPointWithidx expects (batch_size, npoint, ndataset) xyz2 shape."));

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto sort_idx_flat = sort_idx_tensor.flat<int>();
            const int *sort_idx = &(sort_idx_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointWithidxLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,sort_idx,idx,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPointWithidx").Device(DEVICE_GPU), QueryBallPointWithidxGpuOp);


// QueryBallPointDilated
void queryBallPointDilatedLauncher(int b, int n, int m, float min_radius, float max_radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
class QueryBallPointDilatedGpuOp : public OpKernel {
    public:
        explicit QueryBallPointDilatedGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("min_radius", &min_radius_));
            OP_REQUIRES(context, min_radius_ >= 0, errors::InvalidArgument("QueryBallPointDilated expects positive min_radius"));

            OP_REQUIRES_OK(context, context->GetAttr("max_radius", &max_radius_));
            OP_REQUIRES(context, max_radius_ > 0, errors::InvalidArgument("QueryBallPointDilated expects positive max_radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPointDilated expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPointDilated expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPointDilated expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointDilatedLauncher(b,n,m,min_radius_,max_radius_,nsample_,xyz1,xyz2,idx,pts_cnt);
        }
    private:
        float min_radius_;
        float max_radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPointDilated").Device(DEVICE_GPU), QueryBallPointDilatedGpuOp);


// QueryBallPointDynamicRadius 
void queryBallPointDynamicRadiusLauncher(int b, int n, int m, int nsample, const float *xyz1, const float *xyz2, const float* radius, int *idx, int *pts_cnt);
class QueryBallPointDynamicRadiusGpuOp : public OpKernel {
    public:
        explicit QueryBallPointDynamicRadiusGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("queryBallPointDynamicRadius expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            const Tensor& radius_tensor = context->input(2);
            OP_REQUIRES(context, radius_tensor.dims()==2 && radius_tensor.shape().dim_size(1)==m, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto radius_flat = radius_tensor.flat<float>();
            const float *radius = &(radius_flat(0));

            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointDynamicRadiusLauncher(b,n,m,nsample_,xyz1,xyz2,radius,idx,pts_cnt);
        }
    private:
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPointDynamicRadius").Device(DEVICE_GPU), QueryBallPointDynamicRadiusGpuOp);

// SelectionSort
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out);
class SelectionSortGpuOp : public OpKernel {
    public:
        explicit SelectionSortGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("SelectionSort expects positive k"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& dist_tensor = context->input(0);
            OP_REQUIRES(context, dist_tensor.dims()==3, errors::InvalidArgument("SelectionSort expects (b,m,n) dist shape."));
            int b = dist_tensor.shape().dim_size(0);
            int m = dist_tensor.shape().dim_size(1);
            int n = dist_tensor.shape().dim_size(2);

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,n}, &outi_tensor));
            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,n}, &out_tensor));

            auto dist_flat = dist_tensor.flat<float>();
            const float *dist = &(dist_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            selectionSortLauncher(b,n,m,k_,dist,outi,out);
        }
    private:
        int k_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionSort").Device(DEVICE_GPU), SelectionSortGpuOp);


// GroupPoint
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            groupPointLauncher(b,n,c,m,nsample,points,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU),GroupPointGpuOp);


// GroupPointGrad
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points);
class GroupPointGradGpuOp: public OpKernel{
    public:
        explicit GroupPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPointGrad expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            const Tensor& grad_out_tensor=context->input(2);
            OP_REQUIRES(context,grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==m && grad_out_tensor.shape().dim_size(2)==nsample && grad_out_tensor.shape().dim_size(3)==c, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample, channel) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,c}, &grad_points_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            cudaMemset(grad_points, 0, sizeof(float)*b*n*c);
            groupPointGradLauncher(b,n,c,m,nsample,grad_out,idx,grad_points);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_GPU),GroupPointGradGpuOp);


