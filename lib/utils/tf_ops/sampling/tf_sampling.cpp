#include <cstdio>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("ProbSample")
  .Input("inp: float32")
  .Input("inpr: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ncategory
    c->WithRank(c->input(0), 2, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("FarthestPointSample")
  .Attr("npoint: int")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * c
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("FarthestPointSampleWithDistance")
  .Attr("npoint: int")
  .Input("dist: float32") // batch_size, ndataset, ndataset, distance_matrix
  .Output("out: int32") // batch_size, npoint
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * c
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("FarthestPointSampleWithPreidx")
  .Attr("npoint: int")
  .Input("inp: float32") // batch_size, ndataset, c
  .Input("preidx: int32") // batch_size, m1
  .Output("out: int32") // batch_size, m
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * c
    c->WithRank(c->input(0), 3, &dims1);
    int npoint;
    TF_RETURN_IF_ERROR(c->GetAttr("npoint", &npoint));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("GatherPoint")
  .Input("inp: float32")
  .Input("idx: int32")
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * 3
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims2);
    // batch_size * npoints * 3
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("GatherPointGrad")
  .Input("inp: float32") // [b, n, c]
  .Input("idx: int32") // [b, m]
  .Input("out_g: float32") // [b, m, c]
  .Output("inp_g: float32") // [b, n, c]
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });
REGISTER_OP("GatherByMask")
  .Attr("proposal_num: int")
  .Input("inp: float32")
  .Input("mask: float32")
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * points_num * c
    c->WithRank(c->input(0), 3, &dims1);
    int proposal_num;
    TF_RETURN_IF_ERROR(c->GetAttr("proposal_num", &proposal_num));
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), proposal_num, c->Dim(dims1, 2)});
    c->set_output(0, output);
    return Status::OK();
  });

void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out);
class ProbSampleGpuOp: public OpKernel{
  public:
    explicit ProbSampleGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      const Tensor& inpr_tensor=context->input(1);
      auto inp_flat=inp_tensor.flat<float>();
      auto inpr_flat=inpr_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      const float * inpr=&(inpr_flat(0));
      OP_REQUIRES(context,inp_tensor.dims()==2,errors::InvalidArgument("ProbSample expects (batch_size,num_choices) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      OP_REQUIRES(context,inpr_tensor.dims()==2 && inpr_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ProbSample expects (batch_size,num_points) inpr shape"));
      int m=inpr_tensor.shape().dim_size(1);
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      probsampleLauncher(b,n,m,inp,inpr,temp,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("ProbSample").Device(DEVICE_GPU), ProbSampleGpuOp);

void farthestpointsamplingLauncher(int b,int n,int c,int m,const float * inp,float * temp,int * out);
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSample expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,c) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int c=inp_tensor.shape().dim_size(2);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingLauncher(b,n,c,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);


void farthestpointsamplingwithdistLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
class FarthestPointSampleWithDistanceGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleWithDistanceGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
                    OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSampleWithDistance expects positive npoint"));
                }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0); // batch_size, ndataset, ndataset, distance matrix
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(1) == inp_tensor.shape().dim_size(2),errors::InvalidArgument("FarthestPointSampleWithDistance expects (batch_size,num_points,num_points) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingwithdistLauncher(b,n,m,inp,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSampleWithDistance").Device(DEVICE_GPU),FarthestPointSampleWithDistanceGpuOp);

void farthestpointsamplingwithpreidxLauncher(int b,int n,int c,int m,int m1,const float * inp,const int* preidx,float * temp,int * out);
class FarthestPointSampleWithPreidxGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleWithPreidxGpuOp(OpKernelConstruction* context):OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("npoint", &npoint_));
      OP_REQUIRES(context, npoint_ > 0, errors::InvalidArgument("FarthestPointSampleWithPreidx expects positive npoint"));
    }
    void Compute(OpKernelContext * context)override{
      int m = npoint_;

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("FarthestPointSampleWithPreidx expects (batch_size,num_points,c) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int c=inp_tensor.shape().dim_size(2);

      const Tensor& preidx_tensor = context->input(1); 
      OP_REQUIRES(context,preidx_tensor.dims()==2 && preidx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("FarthestPointSampleWithPreidx expects (batch_size,m1) preidx shape"));
      int m1=preidx_tensor.shape().dim_size(1);

      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto preidx_flat = preidx_tensor.flat<int>();
      const int* preidx=&(preidx_flat(0));

      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));
      farthestpointsamplingwithpreidxLauncher(b,n,c,m,m1,inp,preidx,temp,out);
    }
    private:
        int npoint_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSampleWithPreidx").Device(DEVICE_GPU),FarthestPointSampleWithPreidxGpuOp);

void gatherpointLauncher(int b,int n,int m,int c,const float * inp,const int * idx,float * out);
class GatherPointGpuOp: public OpKernel{
  public:
    explicit GatherPointGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("GatherPoint expects (batch_size,num_points,c) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int c=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPoint expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,c},&out_tensor));
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      gatherpointLauncher(b,n,m,c,inp,idx,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPoint").Device(DEVICE_GPU),GatherPointGpuOp);

void scatteraddpointLauncher(int b,int n,int m,int c,const float * out_g,const int * idx,float * inp_g);
class GatherPointGradGpuOp: public OpKernel{
  public:
    explicit GatherPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0); // [b, n, c]
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_points,c) inp"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int c=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1); //[b,m]
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      const Tensor& out_g_tensor=context->input(2); // [b, m, c]
      OP_REQUIRES(context,out_g_tensor.dims()==3 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==m && out_g_tensor.shape().dim_size(2)==c,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_result,c) out_g shape"));
      auto out_g_flat=out_g_tensor.flat<float>();
      const float * out_g=&(out_g_flat(0));
      Tensor * inp_g_tensor=NULL; // [b, n, c]
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,c},&inp_g_tensor));
      auto inp_g_flat=inp_g_tensor->flat<float>();
      float * inp_g=&(inp_g_flat(0));
      cudaMemset(inp_g,0,b*n*c*sizeof(float));
      scatteraddpointLauncher(b,n,m,c,out_g,idx,inp_g);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPointGrad").Device(DEVICE_GPU),GatherPointGradGpuOp);


void GatherByMaskLauncher(int b,int n,int c,int proposal_num,const float *inp,const float *mask,float *out);
class GatherByMaskGpuOp: public OpKernel{
  public:
    explicit GatherByMaskGpuOp(OpKernelConstruction * context):OpKernel(context){
        OP_REQUIRES_OK(context, context->GetAttr("proposal_num", &proposal_num_));
        OP_REQUIRES(context, proposal_num_ > 0, errors::InvalidArgument("GatherByMask expects positive proposal number"));
    }
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("GatherByMask expects (bs,num_points,c) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int c=inp_tensor.shape().dim_size(2);

      const Tensor& mask_tensor =context->input(1);
      OP_REQUIRES(context,mask_tensor.dims()==2 && mask_tensor.shape().dim_size(0)==b && mask_tensor.shape().dim_size(1)==n,errors::InvalidArgument("GatherByMask expects (bs,num_points) mask shape"));

      auto inp_flat=inp_tensor.flat<float>();
      const float *inp = &(inp_flat(0));
      auto mask_flat = mask_tensor.flat<float>();
      const float* mask = &(mask_flat(0));

      Tensor *out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,proposal_num_,c},&out_tensor));
      auto out_flat=out_tensor->flat<float>();
      float *out=&(out_flat(0));
      GatherByMaskLauncher(b,n,c,proposal_num_,inp,mask,out);
    }
    private:
        int proposal_num_;
};
REGISTER_KERNEL_BUILDER(Name("GatherByMask").Device(DEVICE_GPU),GatherByMaskGpuOp);
