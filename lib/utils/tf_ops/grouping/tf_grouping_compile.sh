#/bin/bash
/usr/local/cuda-9.0/bin/nvcc -ccbin /usr/bin/gcc-5 tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC


# TF1.4
/usr/bin/gcc-5 -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I ~/anaconda3/envs/pc_detector/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I ~/anaconda3/envs/pc_detector/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L ~/anaconda3/envs/pc_detector/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
