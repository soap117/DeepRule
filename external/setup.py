import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

if sys.platform == 'win32':
    extra_compile_args = []
else:
    extra_compile_args = ['-Wno-cpp', '-Wno-unused-function']

extensions = [
    Extension(
        "nms", 
        ["nms.pyx"],
        extra_compile_args=extra_compile_args
    )
]

setup(
    name="coco",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
