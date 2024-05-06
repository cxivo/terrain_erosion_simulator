from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Terrain thingy!',
    package_dir={'terrain_erosion_simulator': ''},
    ext_modules = cythonize("eroder.pyx", annotate=True)
)
