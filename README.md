# FindBlas

Python module for finding installed BLAS library in a system, along with its headers. Intended to be used for easily linking Cython-wrapped C/C++ code that calls BLAS functions to the corresponding system's library.

BLAS (basic linear algebra subroutines) is a standard module for  fast linear algebra computations, widely used in packages for mathematical, statistical, and scientific computing. Unlike other tools such as R, Python does not come with a default BLAS installation, and any C/C++ or FORTRAN code intended to be wrapped (called from) Python that uses BLAS, requires either a system-wide install that would register `-lblas`, or manually supplying the path to the library before compiling the code. Some packages circumvent the requirement by supplying their own non-optimized replacements of BLAS functions that follow the same API and letting the user manually alter the linkage before compiling, but this approach leads to extra efforts and most users end up running slow code.

This package eases the usage of BLAS functions in wrapped code. It can find either Python installs (e.g. `conda install openblas`, `pip install mkl`) or system installs (e.g. `apt-get install libopenblas-base libopenblas-dev`), and will also work in Windows with libraries from `m2w64` (e.g. `conda install -c msys2 m2w64-openblas`, `conda install -c msys2 m2w64-gsl`).

Supports the following BLAS implementations:
* `MKL` (free of charge but not open-source)
* `OpenBLAS` (open-source)
* `ATLAS` (open-source, must be built from source as it makes system-specific optimizations)
* `GSL` (open-source and copyleft, not very optimized)

All of which conform to the CBLAS API (i.e. functions named like `cblas_ddot`, `cblas_sgemm`, etc.).

Also included is a `build_ext_with_blas` class built on top of `Cython.Distutils.build_ext` that can be passed to `distutils` and `setuptools`, and which will automatically add links to BLAS; and a header `findblas.h` that will include the function prototypes from the library that was found.

The `build_ext_with_blas` module works also in builds originating from `readthedocs.org` without explicitly adding a specific BLAS dependency like `mkl`, so you can add `findblas` as a dependency for a Python package and host its documentation on RTD without additional hassle.

## Installation

Package is available in PyPI, can be installed with:

```pip install findblas```

It is recommended to install `numpy` and/or `scipy` as then it will try to take the same BLAS library that those are using. In non-Windows systems, if the file name does not match to any implementation-specific name (e.g. just `libblas.so`), it will additionally try to use use `pyelftools` or system's `readelf` to check if it can identify the version.

## Finding BLAS library

```python
import findblas

blas_path, blas_file, incl_path, incl_file, flags = findblas.find_blas()
```

## Compiling Python extension linked against BLAS

_Example here requires Cython (e.g. `conda install cython`, `pip install cython`)._

Example `cfile.c` using BLAS - **important to incude `findblas.h`!**:
```c
#include "findblas.h"
double inner_prod(double *a, int n)
{
	return cblas_ddot(n, a, 1, a, 1);
}
```

Example `pywrapper.pyx` file wrapping it:
```python
import numpy as np
cimport numpy as np
cdef extern from "cfile.c":
	double inner_prod(double *a, int n)
def call_inner_prod(np.ndarray[double, ndim=1] a):
	return inner_prod(&a[0], a.shape[0])
```

Example `setup.py` for packaging them:
```python
try:
	from setuptools import setup
except:
	from distutils.core import setup
import numpy as np
from distutils.extension import Extension
from findblas.distutils import build_ext_with_blas

setup(
    name  = "inner_prod",
    packages = ["inner_prod"],
    cmdclass = {'build_ext': build_ext_with_blas},
    ext_modules = [Extension("inner_prod", sources=["pywrapper.pyx"], include_dirs=[np.get_include()])]
    )
```

The code can then be compiled with e.g. `python setup.py build_ext --inplace` or `python setup.py install`, and tested like this:
```python
import numpy as np, inner_prod
inner_prod.call_inner_prod( np.arange(10).astype('float64') )
>>> 285.0
```



The `build_ext_with_blas` class can be subclassed in the same way as other `build_ext` modules - e.g. if you want to add compiler-specific arguments:
```python
try:
	from setuptools import setup
except:
	from distutils.core import setup
import numpy as np
from distutils.extension import Extension
from findblas.distutils import build_ext_with_blas

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args += ['/O2']
        else: # everything else that cares about following standards
            for e in self.extensions:
                e.extra_compile_args += ['-Ofast', '-fopenmp', '-march=native', '-std=c99']
                e.extra_link_args += ['-fopenmp']
        build_ext_with_blas.build_extensions(self)

setup(
    name  = "inner_prod",
    packages = ["inner_prod"],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension("inner_prod", sources=["pywrapper.pyx"], include_dirs=[np.get_include()])]
    )
```

## Flags returned

The `find_blas` function can return the following flags (if using `build_ext_with_blas`, these will be available by the preprocessor in C files as if doing e.g. `#define DEFINED_THIS_FLAG`):
* `HAS_MKL` : library found was Intel's MKL.
* `HAS_OPENBLAS` : library found was OpenBLAS.
* `HAS_ATLAS` : library found was ATLAS.
* `HAS_GSL` : library found was the GNU Scientific Library.
* `UNKNWON_BLAS` : specific implementation cannot be determined.
* `HAS_UNDERSCORES` : library has functions named like the FORTRAN versions plus an underscore (e.g. `ddot_`, `sgemm_`) (does NOT exclude having `cblas` functions)
* `NO_CBLAS` : library does not have `cblas` functions (e.g. `cblas_ddot`) - in this case, you might want to either raise an error, or declare function prototypes yourself.

Additionally, the `build_ext_with_blas` module might define the following:
* `NO_CBLAS_HEADER` : no header with cblas functions was found, the prototypes were declared using e.g. `int`, `double` (v.s. e.g. `MKL_INT`, `openblas_int`), and might not be reliable in non-standard systems.
* `MKL_OWN_INCL_CBLAS` : the header `mkl.h` was not found, but there was a `mkl_cblas.h` included, which contains only the cblas functions.
* `OPENBLAS_OWN_INCL` : OpenBLAS header was named `cblas-openblas.h` rather than `cblas.h`.
* `GSL_OWN_INCL_CBLAS` : GSL header is named `gsl_cblas.h` (this is the default name in GSL).
* `INCL_CBLAS` : a standard header named `cblas.h` was included.
* `INCL_BLAS` : a standard header named `blas.h` was included.

If `HAS_MKL` is defined and `MKL_OWN_INCL_CBLAS` is not defined, it means that it included the usual `mkl.h` (which is what should usually happen in MKL).
