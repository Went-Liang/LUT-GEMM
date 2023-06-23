#include <omp.h>
#include <iostream>
#include <Eigen/Core>
#include <ctime>
#include <cstdlib>
#include <unsupported/Eigen/CXX11/Tensor>


template<typename T, typename STYPE, typename BTYPE>
void data_generator(const int q, const int g,
                    const unsigned m, const unsigned n, const unsigned k,
                    Eigen::Tensor<T, 2, Eigen::RowMajor>& W, Eigen::Tensor<T, 2, Eigen::RowMajor>& X, Eigen::Tensor<T, 2, Eigen::RowMajor>& Y,
                    Eigen::Tensor<STYPE, 3, Eigen::RowMajor>& A, Eigen::Tensor<BTYPE, 3, Eigen::RowMajor>& B){
    srand((unsigned)time(NULL));
    int subgroup_size = sizeof(BTYPE) * 8;
    Eigen::Tensor<STYPE, 3, Eigen::RowMajor> bitB(m, n, q);
    Eigen::Tensor<STYPE, 3, Eigen::RowMajor> bitB_(m, n, q);
    Eigen::Tensor<STYPE, 3, Eigen::RowMajor> completeA(m, n, q);

    X.setRandom();
    A.setRandom();
//    X.setConstant(1);
//    A.setConstant(1);

    for(unsigned i = 0; i < m; ++i){
        for(unsigned j = 0; j < n; ++j){
            Eigen::array<Eigen::Index, 3> offsets = {i, j, 0};
            Eigen::array<Eigen::Index, 3> offsets_ = {i, (int)ceil(j / g), 0};
            Eigen::array<Eigen::Index, 3> extents = {1, 1, q};
            completeA.slice(offsets, extents) = A.slice(offsets_, extents);
        }
    }
    for(unsigned i = 0; i < m; ++i){
        for(unsigned j = 0; j < n; ++j){
            for(unsigned z = 0; z < q; ++z) {
                bitB(i, j, z) = (STYPE)((rand()%2) > 0.5? 1.0: -1.0);
                bitB_(i, j, z) = (STYPE)(bitB(i, j, z) == 1? 1.0: 0.0);
            }
        }
    }

    Eigen::Tensor<STYPE, 1> powlist(subgroup_size);
    for(int i = 0; i < subgroup_size; ++i)
        powlist(i) = (STYPE)pow(2, i);

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(0, 0) };
    for(unsigned i = 0; i < m; ++i){
        for(unsigned j = 0; j < (int)ceil((float)n / (float)subgroup_size); ++j){
            for(unsigned z = 0; z < q; ++z) {
                BTYPE sum = 0;
                for(unsigned c = 0; c < subgroup_size; ++c) {
                    if(j * subgroup_size + c >= n)continue;
                    sum += (BTYPE) (powlist(c) * bitB_(i, j * subgroup_size + c, z));
                }
                B(i, j, z) = sum;
            }
        }
    }

    auto Z = completeA * bitB;
    Eigen::array<int, 1> dims({2});
    W = Z.sum(dims);


    product_dims = Eigen::IndexPair<int>(1, 0);
    Y = W.contract(X, product_dims);

//    std::cout << bitB << "\n";
}


template<typename T, typename STYPE, typename BTYPE>
void checkAB(const int q, const int g,
             const unsigned m, const unsigned n, const unsigned k,
             const Eigen::Tensor<T, 2, Eigen::RowMajor> X, const Eigen::Tensor<T, 2, Eigen::RowMajor> Y,
             const Eigen::Tensor<STYPE, 3, Eigen::RowMajor> A, const Eigen::Tensor<BTYPE, 3, Eigen::RowMajor> B){
    int subgroup_size = sizeof(BTYPE) * 8;
    Eigen::Tensor<T, 2, Eigen::RowMajor> res(m, k);
    res.setZero();

    for(int iq = 0; iq < q; ++iq){
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                for(int ik = 0; ik < k; ++ik) {
                    int group_id = (int) ceil(j / g);
                    STYPE a = A(i, group_id, iq);

                    int subgroup_id = j / subgroup_size;
                    BTYPE decimal_b = B(i, subgroup_id, iq);
                    int bin_id = j % subgroup_size;
                    BTYPE shifted = decimal_b >> bin_id;
                    int bitValue = shifted & 1;
                    STYPE b = (STYPE)(bitValue == 1 ? 1 : -1);

                    T abx = ((T)(a * b)) * X(j, ik);
                    res(i, ik) += abx;
                }
            }
        }
    }
    Eigen::Tensor<T, 2, Eigen::RowMajor> diffArray = (res - Y).abs();
    std::cout << "Max Error: " << diffArray.maximum() << "\n";
}

template<typename T, typename STYPE, typename BTYPE>
void checkAB_(const int q, const int g,
             const unsigned m, const unsigned n, const unsigned k,
              const Eigen::Tensor<T, 2, Eigen::RowMajor> X, const Eigen::Tensor<T, 2, Eigen::RowMajor> Y,
              const Eigen::Tensor<STYPE, 3, Eigen::RowMajor> A, const Eigen::Tensor<BTYPE, 3, Eigen::RowMajor> B){
    int subgroup_size = sizeof(BTYPE) * 8;
    Eigen::Tensor<T, 2, Eigen::RowMajor> res(m, k);
    res.setZero();
    // B -> bitB
    // (A * B) * X

    for(int iq = 0; iq < q; ++iq){
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < k; ++j){
                for(int in = 0; in < n; ++in) {
                    int group_id = (int) ceil(in / g);
                    STYPE a = A(i, group_id, iq);

                    int subgroup_id = in / subgroup_size;
                    BTYPE decimal_b = B(i, subgroup_id, iq);
                    int bin_id = in % subgroup_size;
                    BTYPE shifted = decimal_b >> bin_id;
                    int bitValue = shifted & 1;
                    STYPE b = (STYPE)(bitValue == 1 ? 1 : -1);

                    T abx = ((T)(a * b)) * X(in, j);
                    res(i, j) += abx;
                }
            }
        }
    }
    Eigen::Tensor<T, 2, Eigen::RowMajor> diffArray = (res - Y).abs();
    std::cout << "Max Error: " << diffArray.maximum() << "\n";
}

template<typename T, typename STYPE, typename BTYPE>
void checkAB_1(const int q, const int g,
               const unsigned m, const unsigned n, const unsigned k,
               const Eigen::Tensor<T, 2, Eigen::RowMajor> X, const Eigen::Tensor<T, 2, Eigen::RowMajor> Y,
               const Eigen::Tensor<STYPE, 3, Eigen::RowMajor> A, const Eigen::Tensor<BTYPE, 3, Eigen::RowMajor> B){
    int subgroup_size = sizeof(BTYPE) * 8;
    Eigen::Tensor<T, 2, Eigen::RowMajor> res(m, k);
    res.setZero();
    // B -> bitB
    // (A * B) * X
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < k; ++j){

            for(int in = 0; in < n; ++in) {
                T w{0};
                for(int iq = 0; iq < q; ++iq){
                    int group_id = (int) ceil(in / g);
                    STYPE a = A(i, group_id, iq);

                    int subgroup_id = in / subgroup_size;
                    BTYPE decimal_b = B(i, subgroup_id, iq);
                    int bin_id = in % subgroup_size;
                    BTYPE shifted = decimal_b >> bin_id;
                    int bitValue = shifted & 1;
                    STYPE b = (STYPE)(bitValue == 1 ? 1 : -1);
                    w += a * b;
                }

                T abx = w * X(in, j);
                res(i, j) += abx;
            }
        }
    }
    Eigen::Tensor<T, 2, Eigen::RowMajor> diffArray = (res - Y).abs();
    std::cout << "Max Error: " << diffArray.maximum() << "\n";
}


template<typename T, typename STYPE, typename BTYPE>
__global__ void lut_gemm_kernel(const int q, const int group_size,
                                const unsigned m, const unsigned n, const unsigned k,
                                const unsigned A_col_num, const unsigned B_col_num,
                                const STYPE *A, const BTYPE *B, const T *X,
                                T *Y) {
// grid(m/th, n/g), block(th)
// W(m*n), X(n*k), Y(m*k), AB(m*n*q)
// m = 2, n = 8, g = 8,
    constexpr unsigned int subgroup_size = sizeof(BTYPE) * 8;
    unsigned int lut_num = group_size / subgroup_size;
    unsigned int lut_size = pow(2, subgroup_size);
    unsigned int group_id = blockIdx.y;
    unsigned int row_id = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ T luts[];

    for(int ik = 0; ik < k; ++ik) {
        if (threadIdx.x < lut_size) {
            for (int i = 0; i < lut_num; ++i) {
                T sum = 0;
#pragma unroll
                for (int j = 0; j < subgroup_size; ++j) {
                    sum += X[(group_id * group_size + i * subgroup_size + j) * k + ik] *
                           (((threadIdx.x % lut_size >> j) & 1) ? 1 : -1);
                }
                luts[i * lut_size + threadIdx.x % lut_size] = sum;
            }
        }
        __syncthreads();

        for (int iq = 0; iq < q; ++iq) {
            T sum = 0;
            for (int i = 0; i < lut_num; ++i) {
                BTYPE decimal_b = B[row_id * B_col_num * q + (group_id * lut_num + i) * q + iq];
                sum += luts[i * lut_size + decimal_b];
            }
            sum *= A[row_id * A_col_num * q + group_id * q + iq];
            atomicAdd(&Y[row_id * k + ik], sum);
        }
    }
}

int main() {
    int gpu_rank = 0;
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, gpu_rank);
    cudaSetDevice(gpu_rank);
    printf("GPU %s status: ", deviceProp.name);
    double boostFrequency = deviceProp.clockRate / 1e6;
    int fp32CoresNum = 640;
    double peakPerformance = boostFrequency * fp32CoresNum * 2;
    printf("clock rate %.3f GHz, FP32 cores num %d, FP32 peak throughput %.3f "
            "GFLOPS\n",
            boostFrequency, fp32CoresNum, peakPerformance);
    omp_set_num_threads(omp_get_num_procs());


//    typedef Eigen::bfloat16 T;        // W X Y: before quant
    typedef float T;                    // W X Y: before quant
    typedef float STYPE;                // scaleMat  A
    typedef std::uint8_t BTYPE;         // binaryMat B
    int subgroup_size = sizeof(BTYPE) * 8;
    int q = 3;
    int g = 128;
    unsigned m = 4096;
    unsigned n = 1024;
    unsigned k = 8;
    unsigned A_col_num = (int)ceil((float)n / (float)g);
    unsigned B_col_num = (int)ceil((float)n / (float)subgroup_size);

    Eigen::Tensor<T, 2, Eigen::RowMajor> W(m, n), X(n, k), Y(m, k);
    Eigen::Tensor<STYPE, 3, Eigen::RowMajor> A(m, A_col_num, q);
    Eigen::Tensor<BTYPE, 3, Eigen::RowMajor> B(m, B_col_num, q);

    data_generator<T, STYPE, BTYPE>(q, g, m, n, k,
                                    W, X, Y, A, B);
    checkAB_1<T, STYPE, BTYPE>(q, g, m, n, k,
                               X, Y, A, B);


    STYPE *deviceAPtr;
    BTYPE *deviceBPtr;
    T *deviceXPtr;
    T *deviceYPtr;
    cudaMalloc(&deviceAPtr, (m * A_col_num * q) * sizeof(STYPE));
    cudaMalloc(&deviceBPtr, (m * B_col_num * q) * sizeof(BTYPE));
    cudaMalloc(&deviceXPtr, (n * k) * sizeof(T));
    cudaMalloc(&deviceYPtr, (m * k) * sizeof(T));
    cudaMemcpy(deviceAPtr, A.data(), (m * A_col_num * q) * sizeof(STYPE),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBPtr, B.data(), (m * B_col_num * q) * sizeof(BTYPE),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceXPtr, X.data(), (n * k) * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaMemset(deviceYPtr, 0, (m * k) * sizeof(T));
    Eigen::Tensor<T, 2, Eigen::RowMajor> cuY(m, k);

    int th = 1024;
    dim3 block(th);
    dim3 grid((m - 1) / block.x + 1, (n - 1) / g + 1);

    lut_gemm_kernel<T, STYPE, BTYPE>
            <<<grid, block, pow(2, subgroup_size) * (g / subgroup_size) * sizeof(T)>>>
            (q, g, m, n, k,
             A_col_num, B_col_num,
             deviceAPtr, deviceBPtr, deviceXPtr, deviceYPtr);

    cudaDeviceSynchronize();
    cudaMemcpy(cuY.data(), deviceYPtr, (m * k) * sizeof(T), cudaMemcpyDeviceToHost);
    Eigen::Tensor<T, 2, Eigen::RowMajor> diffArray = (cuY - Y).abs();
    std::cout << "lut-gemm Max Error: " << diffArray.maximum() << "\n";


//    std::cout << W << std::endl;
//    std::cout << "\n";
//    std::cout << X << std::endl;
//    std::cout << "\n";
//    std::cout << Y << std::endl;
//    std::cout << "\n";
//    std::cout << A << std::endl;
//    std::cout << "\n";
//    std::cout << B << std::endl;
//    std::cout << "\n";
//    std::cout << cuY << std::endl;


}
