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

PYBIND11_MODULE(cpp_function, m) {
    m.def("matmul", matmul, "C tensor.matmul()");
}
