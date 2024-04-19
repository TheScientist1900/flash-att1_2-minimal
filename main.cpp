#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor myForward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor myForward2(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
    m.def("myForward", torch::wrap_pybind_function(myForward), "myForward");
    m.def("myForward2", torch::wrap_pybind_function(myForward2), "myForward2");
}

