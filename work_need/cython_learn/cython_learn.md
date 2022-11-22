# Cython笔记本

# Cython学习笔记

## 参考资料索引:

用户指南 - 在 Cython 中使用 C ++ - 《Cython 3.0 中文文档》 - 书栈网 · BookStack](https://www.bookstack.cn/read/cython-doc-zh/docs-31.md)

[
    Cython/PyPy编程技术 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1284193666899787776)

## Cython的基本使用

### 编译:

#### 1)Setup-tool方法

创建一个文件: setup.py

```
from distutils.core import setup

from Cython.Build import cythonize

# Example_Cython:写着Cython代码的Pyx文件

setup(

        name="Example Cython",

        ext_modules=cythonize(["examples_cy.pyx"])
        )

```

3、让后在同目录根下运行

    python setup.py build_ext --inplace

4、让后等待编译完毕之后，会在built文件目录下生成一个文件:

​	Windows: examples_cy.cp39-win_amd64.pyd，如果是Mac/Linux系统,文件后缀就是So

#### 2)Cythonize方法





## Cython:Jupyter直接交互使用


kernel会帮你自动编译

%load_ext cython

%%cython -a

def func(a,b):

​	return a+b

即可

可以使用后台帮忙编译加载 Cython计算函数
