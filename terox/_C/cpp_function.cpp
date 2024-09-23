#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> matmul(py::array_t<double> a, py::array_t<double> b) {
    ssize_t a_rows = a.shape(0);
    ssize_t a_cols = a.shape(1);
    ssize_t b_cols = b.shape(1);

    py::array_t<double> result({a_rows, b_cols});

    auto a_buf = a.unchecked<2>();
    auto b_buf = b.unchecked<2>();
    auto res_buf = result.mutable_unchecked<2>();

    for (ssize_t i = 0; i < a_rows; i++) {
        for (ssize_t j = 0; j < b_cols; j++) {
            res_buf(i, j) = 0;
            for (ssize_t k = 0; k < a_cols; k++) {
                res_buf(i, j) += a_buf(i, k) * b_buf(k, j);
            }
        }
    }

    return result;
}

py::array_t<double> conv2d_forward(py::array_t<double> a, py::array_t<double> w, int stride, int padding) {
    auto a_buf = a.unchecked<4>();  // (batch_size, in_channels, height, width)
    auto w_buf = w.unchecked<4>();  // (out_channels, in_channels, kernel_height, kernel_width)

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
    auto output_buf = output.mutable_unchecked<4>();

    std::vector<double> padded_size(batch_size * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding), 0);
    auto a_padded = padded_size.data();

    for (ssize_t b = 0; b < batch_size; ++b) {
        for (ssize_t c = 0; c < in_channels; ++c) {
            for (ssize_t h = 0; h < input_height; ++h) {
                for (ssize_t w = 0; w < input_width; ++w) {
                    a_padded[b * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                              c * (input_height + 2 * padding) * (input_width + 2 * padding) +
                              (h + padding) * (input_width + 2 * padding) +
                              (w + padding)] = a_buf(b, c, h, w);
                }
            }
        }
    }

    for (ssize_t i = 0; i < output_height; ++i) {
        for (ssize_t j = 0; j < output_width; ++j) {
            for (ssize_t oc = 0; oc < out_channels; ++oc) {
                double sum = 0.0;
                for (ssize_t ic = 0; ic < in_channels; ++ic) {
                    for (ssize_t kh = 0; kh < kernel_height; ++kh) {
                        for (ssize_t kw = 0; kw < kernel_width; ++kw) {
                            sum += a_padded[(b * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                             ic * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                             (i * stride + kh) * (input_width + 2 * padding) +
                                             (j * stride + kw))] * w_buf(oc, ic, kh, kw);
                        }
                    }
                }
                output_buf(b, oc, i, j) = sum;
            }
        }
    }

    return output;
}

std::tuple<py::array_t<double>, py::array_t<double>> conv2d_backward(py::array_t<double> a, py::array_t<double> w, int stride, int padding, py::array_t<double> grad) {
    auto a_buf = a.unchecked<4>();  // (batch_size, in_channels, height, width)
    auto w_buf = w.unchecked<4>();  // (out_channels, in_channels, kernel_height, kernel_width)
    auto grad_buf = grad.unchecked<4>();  // (batch_size, out_channels, grad_height, grad_width)

    ssize_t batch_size = a_buf.shape(0);
    ssize_t in_channels = a_buf.shape(1);
    ssize_t input_height = a_buf.shape(2);
    ssize_t input_width = a_buf.shape(3);
    ssize_t out_channels = w_buf.shape(0);
    ssize_t kernel_height = w_buf.shape(2);
    ssize_t kernel_width = w_buf.shape(3);
    ssize_t output_height = grad_buf.shape(2);
    ssize_t output_width = grad_buf.shape(3);

    std::vector<double> padded_size(batch_size * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding), 0);
    auto a_padded = padded_size.data();

    for (ssize_t b = 0; b < batch_size; ++b) {
        for (ssize_t c = 0; c < in_channels; ++c) {
            for (ssize_t h = 0; h < input_height; ++h) {
                for (ssize_t w = 0; w < input_width; ++w) {
                    a_padded[b * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                              c * (input_height + 2 * padding) * (input_width + 2 * padding) +
                              (h + padding) * (input_width + 2 * padding) +
                              (w + padding)] = a_buf(b, c, h, w);
                }
            }
        }
    }

    py::array_t<double> grad_a({batch_size, in_channels, input_height, input_width});
    py::array_t<double> grad_w({out_channels, in_channels, kernel_height, kernel_width});
    auto grad_a_buf = grad_a.mutable_unchecked<4>();
    auto grad_w_buf = grad_w.mutable_unchecked<4>();

    for (ssize_t i = 0; i < output_height; ++i) {
        for (ssize_t j = 0; j < output_width; ++j) {
            for (ssize_t oc = 0; oc < out_channels; ++oc) {
                for (ssize_t ic = 0; ic < in_channels; ++ic) {
                    for (ssize_t kh = 0; kh < kernel_height; ++kh) {
                        for (ssize_t kw = 0; kw < kernel_width; ++kw) {
                            grad_w_buf(oc, ic, kh, kw) += grad_buf(b, oc, i, j) * a_padded[(b * in_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                                  ic * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                                  (i * stride + kh) * (input_width + 2 * padding) +
                                                  (j * stride + kw))];
                        }
                    }
                    grad_a_buf(b, ic, i * stride + kh, j * stride + kw) += grad_buf(b, oc, i, j) * w_buf(oc, ic, kh, kw);
                }
            }
        }
    }

    return std::make_tuple(grad_a, grad_w);
}

PYBIND11_MODULE(cpp_function, m) {
    m.def("cpp_matmul", &matmul, "C tensor.matmul()");
    m.def("cpp_conv2d_forward", &conv2d_forward, "C tensor.conv2d()");
    m.def("cpp_conv2d_backward", &conv2d_backward, "C tensor.conv2d()");
}
