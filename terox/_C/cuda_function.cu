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

__global__ void conv2d_forward_kernel(const double* a, const double* w, double* output, 
                                       int batch_size, int in_channels, int input_height, int input_width,
                                       int out_channels, int kernel_height, int kernel_width,
                                       int output_height, int output_width, int stride, int padding) {
    int batch = blockIdx.x;
    int oc = blockIdx.y;
    int i = blockIdx.z / output_width;
    int j = blockIdx.z % output_width;

    if (batch < batch_size && oc < out_channels && i < output_height && j < output_width) {
        double sum = 0.0;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int h = i * stride + kh;
                    int w = j * stride + kw;
                    if (h < input_height + 2 * padding && w < input_width + 2 * padding) {
                        sum += a[batch * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                  ic * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                  (h + padding) * (input_width + 2 * padding) +
                                  (w + padding)] * 
                                w[oc * in_channels * kernel_height * kernel_width + 
                                  ic * kernel_height * kernel_width + 
                                  kh * kernel_width + kw];
                    }
                }
            }
        }
        output[batch * out_channels * output_height * output_width + 
               oc * output_height * output_width + 
               i * output_width + j] = sum;
    }
}

py::array_t<double> conv2d_forward(py::array_t<double> a, py::array_t<double> w, int stride, int padding) {
    auto a_buf = a.unchecked<4>();
    auto w_buf = w.unchecked<4>();

    ssize_t batch_size = a_buf.shape(0);
    ssize_t in_channels = a_buf.shape(1);
    ssize_t input_height = a_buf.shape(2);
    ssize_t input_width = a_buf.shape(3);
    ssize_t out_channels = w_buf.shape(0);
    ssize_t kernel_height = w_buf.shape(2);
    ssize_t kernel_width = w_buf.shape(3);

    ssize_t output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    ssize_t output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    py::array_t<double> output({batch_size, out_channels, output_height, output_width});
    double* d_a;
    double* d_w;
    double* d_output;

    cudaMalloc(&d_a, batch_size * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) * sizeof(double));
    cudaMalloc(&d_w, out_channels * in_channels * kernel_height * kernel_width * sizeof(double));
    cudaMalloc(&d_output, batch_size * out_channels * output_height * output_width * sizeof(double));

    cudaMemcpy(d_a, a.data(), batch_size * in_channels * input_height * input_width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), out_channels * in_channels * kernel_height * kernel_width * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blocks(batch_size, out_channels, output_height * output_width);
    dim3 threads(1, 1, 1);

    conv2d_forward_kernel<<<blocks, threads>>>(d_a, d_w, d_output, batch_size, in_channels, input_height, input_width,
                                               out_channels, kernel_height, kernel_width, 
                                               output_height, output_width, stride, padding);
    cudaDeviceSynchronize();

    cudaMemcpy(output.mutable_data(), d_output, batch_size * out_channels * output_height * output_width * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_output);

    return output;
}

__global__ void conv2d_backward_kernel(const double* a, const double* w, const double* grad,
                                        double* grad_a, double* grad_w,
                                        int batch_size, int in_channels, int input_height, int input_width,
                                        int out_channels, int kernel_height, int kernel_width,
                                        int output_height, int output_width, int stride, int padding) {
    int batch = blockIdx.x;
    int oc = blockIdx.y;
    int i = blockIdx.z / output_width;
    int j = blockIdx.z % output_width;

    if (batch < batch_size && oc < out_channels && i < output_height && j < output_width) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int h = i * stride + kh;
                    int w = j * stride + kw;
                    if (h < input_height + 2 * padding && w < input_width + 2 * padding) {
                        // 计算权重梯度
                        atomicAdd(&grad_w[oc * in_channels * kernel_height * kernel_width + 
                                          ic * kernel_height * kernel_width + 
                                          kh * kernel_width + kw],
                                   grad[batch * out_channels * output_height * output_width + 
                                        oc * output_height * output_width + 
                                        i * output_width + j] *
                                   a[batch * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                     ic * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                     (h + padding) * (input_width + 2 * padding) +
                                     (w + padding)]);
                    }
                }
            }
        }

        // 计算输入梯度
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int h = i * stride + kh;
                    int w = j * stride + kw;
                    if (h < input_height + 2 * padding && w < input_width + 2 * padding) {
                        atomicAdd(&grad_a[batch * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                          ic * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                          (h + padding) * (input_width + 2 * padding) +
                                          (w + padding)],
                                       grad[batch * out_channels * output_height * output_width + 
                                            oc * output_height * output_width + 
                                            i * output_width + j] * 
                                       w[oc * in_channels * kernel_height * kernel_width +
                                         ic * kernel_height * kernel_width +
                                         kh * kernel_width + kw]);
                    }
                }
            }
        }
    }
}

std::tuple<py::array_t<double>, py::array_t<double>> conv2d_backward(py::array_t<double> a, py::array_t<double> w, int stride, int padding, py::array_t<double> grad) {
    auto a_buf = a.unchecked<4>();
    auto w_buf = w.unchecked<4>();
    auto grad_buf = grad.unchecked<4>();

    ssize_t batch_size = a_buf.shape(0);
    ssize_t in_channels = a_buf.shape(1);
    ssize_t input_height = a_buf.shape(2);
    ssize_t input_width = a_buf.shape(3);
    ssize_t out_channels = w_buf.shape(0);
    ssize_t kernel_height = w_buf.shape(2);
    ssize_t kernel_width = w_buf.shape(3);
    ssize_t output_height = grad_buf.shape(2);
    ssize_t output_width = grad_buf.shape(3);

    py::array_t<double> grad_a({batch_size, in_channels, input_height, input_width});
    py::array_t<double> grad_w({out_channels, in_channels, kernel_height, kernel_width});
    double* d_a;
    double* d_w;
    double* d_grad;
    double* d_grad_a;
    double* d_grad_w;

    cudaMalloc(&d_a, batch_size * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) * sizeof(double));
    cudaMalloc(&d_w, out_channels * in_channels * kernel_height * kernel_width * sizeof(double));
    cudaMalloc(&d_grad, batch_size * out_channels * output_height * output_width * sizeof(double));
    cudaMalloc(&d_grad_a, batch_size * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) * sizeof(double));
    cudaMalloc(&d_grad_w, out_channels * in_channels * kernel_height * kernel_width * sizeof(double));

    cudaMemcpy(d_a, a.data(), batch_size * in_channels * input_height * input_width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), out_channels * in_channels * kernel_height * kernel_width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, grad.data(), batch_size * out_channels * output_height * output_width * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blocks(batch_size, out_channels, output_height * output_width);
    dim3 threads(1, 1, 1);

    conv2d_backward_kernel<<<blocks, threads>>>(d_a, d_w, d_grad, d_grad_a, d_grad_w, 
                                                 batch_size, in_channels, input_height, input_width,
                                                 out_channels, kernel_height, kernel_width,
                                                 output_height, output_width, stride, padding);
    cudaDeviceSynchronize();

    cudaMemcpy(grad_a.mutable_data(), d_grad_a, batch_size * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_w.mutable_data(), d_grad_w, out_channels * in_channels * kernel_height * kernel_width * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_grad);
    cudaFree(d_grad_a);
    cudaFree(d_grad_w);

    return std::make_tuple(grad_a, grad_w);
}

PYBIND11_MODULE(cuda_function, m) {
    m.def("cuda_matmul", &matmul_cuda, "CUDA tensor.matmul()");
    m.def("cuda_conv2d_forward", &conv2d_forward, "CUDA tensor.conv2d()");
    m.def("cuda_conv2d_backward", &conv2d_backward, "CUDA tensor.conv2d()");
}
