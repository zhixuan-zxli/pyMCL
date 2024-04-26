from setuptools import setup, Extension
import numpy as np

ext_modules = [
    Extension(
        "binsearchkw",  # Module name
        sources=["binsearchkw.cpp"],  # C source file
        include_dirs=[np.get_include()],  # NumPy headers
        extra_compile_args=["-std=c++11"],  # Optional: C compiler flags
    )
]

setup(
    name='binsearchkw',
    ext_modules=ext_modules,
)
