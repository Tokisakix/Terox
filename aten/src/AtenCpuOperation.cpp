#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <aten/AtenCpuOperation.h>

void AtenCpuTensorAdd(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
    return;
}

PYBIND11_MODULE(_C, m) {
    auto cpu_module = m.def_submodule("cpu", "cpu");
    cpu_module.def("AtenCpuTensorAdd", &AtenCpuTensorAdd, "AtenCpuTensorAdd");
}