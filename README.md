# Terox

[Chinese](README_cn.md) | **English**

**Terox is an open source tiny Deep Learning System based on Python, Cython and CUDA.**

---

## Setup

Terox requires **Python 3.8** or higher. To check your version of Python, run either:

```Shell
python --version
python3 --version
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
print(terox.__version__)
```