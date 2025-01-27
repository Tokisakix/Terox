#ifndef ATEN_CPU_OPERATION_H
#define ATEN_CPU_OPERATION_H

#include <pybind11/stl.h>

void AtenCpuTensorAdd(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out, int size);

#endif