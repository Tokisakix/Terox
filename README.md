# Terox

[Chinese](README_cn.md) | **English**

**Terox is an open source tiny Deep Learning System based on Python, Cython and CUDA.**

![img](asset/terox.png)

Terox is a tiny Python package that provides some features:
- [x] Support automatic differentiation.
- [x] Provides convenient tensor calculation.
- [x] Control the parameters and the model.
- [ ] Provides common computing functions for deep learning.
- [ ] Provides common deep learning components.
- [ ] Provides deep learning model optimizer.
- [ ] Accelerate computing on CPU and GPU.
- [ ] Support distributed computing.

---

## Setup

Terox requires **Python 3.8** or higher. To check your version of Python, run either:

```Shell
python --version # expect python version >= 3.8
```

The next step is to install packages. There are several packages used throughout Terox, and you can install them in your enviroment by running:

```Shell
python -m pip install -r requirements.txt
```

As a final step, you can run the following command to package Terox and install it in your environment:

```Shell
python -m pip install -Ue .
```

Make sure that everything is installed by running python and then checking. If your output is `Terox v0.1 by Tokisakix.`, the installation was successful:

```Python
import terox
print(terox.__version__) # expect output: "Terox v0.1 by Tokisakix."
```

## Test

You can test the correctness of the project by running `pytest` in the root directory of the project:

```Shell
python -m pytest
```

Pytest tests all modules by default, but you can also run the following commands to do some testing:

```Shell
python -m pytest -m <test-name>
```

Where `<test-name>` can select the following test module name:

```Shell
# autodiff test
test_function
test_scalar
test_scalar_opts
test_scalar_overload
test_backward

# module test
test_module
```

## project

You can find the accompanying demonstration project under the `/project` path, which demonstrates some of the uses of **Terox**.

You can run the sample code by going to the project path under '/project' and running the following command:

```Shell
python run.py
```

Examples of projects currently available are:

```
scalar
```