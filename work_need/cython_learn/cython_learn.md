# Cython笔记本

# Cython学习笔记

## 参考资料索引:

用户指南 - 在 Cython 中使用 C ++ - 《Cython 3.0 中文文档》 - 书栈网 · BookStack](https://www.bookstack.cn/read/cython-doc-zh/docs-31.md)

[
    Cython/PyPy编程技术 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1284193666899787776)

## Cython的基本使用

### 编译:

#### 1)Setup-tool方法 --推荐(简单，方便)

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

在 Cmd中输入Cython --help即可查看所有对应的指令。

**--include-dir**:指定编译时包含的C/C++头文件或其他*.py或*.pyx文件

**--output-file**:指定解析后的C/C++源代码文件的所在路径

**--working**:指定cython解析器的工作目录

**-2或-3**:-2告知cython解析器以python2的方式理解python源代码,-3告知cython解析器以python3的方式去理解python源代码



代码示例:

```python
	cythonize -a -i example_code.pyx -3
```

example_code是示例的pyx代码

-a开关还会生成带注释的源代码html文件,可以查看和Cpython解释器调用多少次



指定编译:

​	

```bash
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing C:\Programs\Python\Python37 -o ./test_cython_code.so  ./test_cython_code.c
```







## Cython:Jupyter直接交互使用


kernel会帮你自动编译

%load_ext cython

%%cython -a

def func(a,b):

​	return a+b

即可

可以使用后台帮忙编译加载 Cython计算函数

针对写好的Cython函数可以直接验算