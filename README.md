# Terox

[Chinese](README_cn.md) | **English**

**Terox is an open source tiny Deep Learning System based on Python, Cython and CUDA.**

Terox is a tiny Python package that provides some features:
- [x] Support automatic differentiation.
- [ ] Provides convenient tensor calculation.
- [ ] Control the parameters and the model.
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