#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void matmul_kernel(const double* a, const double* b, double* result, ssize_t a_rows, ssize_t a_cols, ssize_t b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        double sum = 0.0;
        for (int k = 0; k < a_cols; k++) {
            sum += a[row * a_cols + k] * b[k * b_cols + col];
        }
        result[row * b_cols + col] = sum;
    }
}

py::array_t<double> matmul_cuda(py::array_t<double> a, py::array_t<double> b) {
    ssize_t a_rows = a.shape(0);
    ssize_t a_cols = a.shape(1);
    ssize_t b_cols = b.shape(1);

    py::array_t<double> result({a_rows, b_cols});

    double* d_a;
    double* d_b;
    double* d_result;

    cudaMalloc(&d_a, a_rows * a_cols * sizeof(double));
    cudaMalloc(&d_b, a_cols * b_cols * sizeof(double));
    cudaMalloc(&d_result, a_rows * b_cols * sizeof(double));

    cudaMemcpy(d_a, a.data(), a_rows * a_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), a_cols * b_cols * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((b_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (a_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, a_rows, a_cols, b_cols);

    cudaMemcpy(result.mutable_data(), d_result, a_rows * b_cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

PYBIND11_MODULE(cuda_function, m) {
    m.def("matmul", matmul_cuda, "CUDA tensor.matmul()");
}
