# evaluation 
TFPATH=$1 # e.g: /usr/local/lib/python3.5/dist-packages/tensorflow
CUDAPATH=$2 # e.g: /usr/local/cuda-9.0
OPSPATH="lib/utils/tf_ops"

# voxel operation
cd lib/builder/voxel_generator
./build.sh
cd dist
pip install points2voxel-0.0.1-cp36-cp36m-linux_x86_64.whl
cd ../../../..

# evaluation
cd ${OPSPATH}/evaluation
g++ -std=c++11 tf_evaluate.cpp -o tf_evaluate_so.so -shared -fPIC -I ${TFPATH}/include -I ${CUDAPATH}/include -I ${TFPATH}/include/external/nsync/public -lcudart -L ${CUDAPATH}/lib64/ -L${TFPATH} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# grouping
cd grouping
$CUDAPATH/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TFPATH/include -I $CUDAPATH/include -I $TFPATH/include/external/nsync/public -lcudart -L $CUDAPATH/lib64/ -L$TFPATH -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# interpolation
cd interpolation
$CUDAPATH/bin/nvcc tf_interpolate_g.cu -o tf_interpolate_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_interpolate.cpp tf_interpolate_g.cu.o -o tf_interpolate_so.so -shared -fPIC -I $TFPATH/include -I $CUDAPATH/include -I $TFPATH/include/external/nsync/public -lcudart -L $CUDAPATH/lib64/ -L$TFPATH -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# points pooling
cd points_pooling
$CUDAPATH/bin/nvcc tf_points_pooling_g.cu -o tf_points_pooling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_points_pooling.cpp tf_points_pooling_g.cu.o -o tf_points_pooling_so.so -shared -fPIC -I $TFPATH/include -I $CUDAPATH/include -I $TFPATH/include/external/nsync/public -lcudart -L $CUDAPATH/lib64/ -L$TFPATH -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# sampling
cd sampling
$CUDAPATH/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TFPATH/include -I $CUDAPATH/include -I $TFPATH/include/external/nsync/public -lcudart -L $CUDAPATH/lib64/ -L$TFPATH -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

# nms
cd nms
./build.sh
cd ..

