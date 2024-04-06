# Terox

**中文** | [英文](README.md)

**Terox 是一个基于 Python、Cython 和 CUDA 的开源微型深度学习系统。**

---

## 设置

Terox 要求 **Python 3.8** 或更高版本。要检查你的 Python 的版本，请运行:

```Shell
python --version
python3 --version
```

下一步是安装第三方软件包。在 Terox 项目中使用了几个包，您可以通过运行以下命令将它们安装到您的环境中:

```Shell
Python -m pip install -r requirements.txt
```

最后一步，您可以运行以下命令将 Terox 打包并安装到您的环境中:

```Shell
python -m pip install -Ue .
```

通过运行 Python 并检查，确保所有内容都已安装。运行下方代码，如果您的输出是 `Terox v0.1 by Tokisakix.`，则安装成功:

```Python
import terox
print(terox.__version__)
```