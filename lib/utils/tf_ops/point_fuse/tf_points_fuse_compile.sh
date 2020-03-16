#/bin/bash
/usr/local/cuda-9.0/bin/nvcc tf_points_fuse_g.cu -o tf_points_fuse_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 tf_points_fuse.cpp tf_points_fuse_g.cu.o -o tf_points_fuse_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
