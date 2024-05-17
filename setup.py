from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "eroder",
        ["eroder.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='Terrain thingy!',
    package_dir={'terrain_erosion_simulator': ''},
    ext_modules = cythonize(extensions, annotate=True)
)
