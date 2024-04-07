# Terox

**中文** | [英文](README.md)

**Terox 是一个基于 Python、Cython 和 CUDA 的开源微型深度学习系统。**

Terox 是一个很精简的 Python 包，它提供了一些特性:
- [x] 支持自动微分。
- [ ] 提供方便的张量计算。
- [ ] 便捷控制参数和模型。
- [ ] 提供深度学习常用的计算函数。
- [ ] 提供常用的深度学习组件。
- [ ] 提供深度学习模型优化器。
- [ ] 提高在 CPU 和 GPU 上的计算速度。
- [ ] 支持分布式计算。

---

## 设置

Terox 要求 **Python 3.8** 或更高版本。要检查你的 Python 的版本，请运行:

```Shell
python --version # 期望 python 版本 >= 3.8
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
print(terox.__version__) # 期望输出: "Terox v0.1 by Tokisakix."
```