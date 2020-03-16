# evaluation 
cd evaluation
g++ -std=c++11 tf_evaluate.cpp -o tf_evaluate_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# evaluation plot
cd evaluation_plot
g++ -std=c++11 tf_evaluate.cpp -o tf_evaluate_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# grouping
cd grouping
/usr/local/cuda-9.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# interpolation
cd interpolation
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# nms
cd nms
/usr/local/cuda-9.0/bin/nvcc nms_kernel.cu -o nms_kernel.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_nms.cpp nms_kernel.cu.o -o tf_nms_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# point Deformable
cd pointDeform_op
/usr/local/cuda-9.0/bin/nvcc pointDeform.cu -o pointDeform_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 main.cpp pointDeform_g.cu.o -o tf_pointDeform_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# pointSift
cd pointSIFT_op
/usr/local/cuda-9.0/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# point fuse
cd point_fuse
/usr/local/cuda-9.0/bin/nvcc tf_points_fuse_g.cu -o tf_points_fuse_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_points_fuse.cpp tf_points_fuse_g.cu.o -o tf_points_fuse_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# points pooling
cd points_pooling
/usr/local/cuda-9.0/bin/nvcc tf_points_pooling_g.cu -o tf_points_pooling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_points_pooling.cpp tf_points_pooling_g.cu.o -o tf_points_pooling_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# sampling
cd sampling
/usr/local/cuda-9.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..
