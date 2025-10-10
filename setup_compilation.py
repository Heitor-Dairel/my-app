from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("modules.math.fibonacci", ["src/backend/modules/math/fibonacci.pyx"])
]

setup(
    name="my_app",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)

# python setup_compilation.py build_ext --inplace
