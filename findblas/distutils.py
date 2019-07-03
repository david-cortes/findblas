from Cython.Distutils import build_ext
import findblas, re, os, sys, warnings

## https://stackoverflow.com/questions/52905458/link-cython-wrapped-c-functions-against-blas-from-numpy
class build_ext_with_blas( build_ext ):
    """
    'build_ext' module built on top of 'Cython.Distutils.build_ext'.

    Intended to passed to 'setuptools.setup', or to 'distutils.core.setup'.
    """

    def build_extensions(self):
        ## Lookup blas files and headers first
        nocblas_err_msg = "No CBLAS library found - please install one with e.g. "
        nocblas_err_msg += "'pip install mkl mkl-devel' (Win/Mac/Lin)."
        from_rtd = os.environ.get('READTHEDOCS') == 'True'
        if not from_rtd:
            blas_path, blas_file, incl_path, incl_file, flags = findblas.find_blas()
            if (blas_file is None) or (blas_path is None):
                raise ValueError(nocblas_err_msg)
            elif blas_file == "mkl_rt.dll":
                txt = "Found MKL library at:\n" + os.path.join(blas_path, blas_file)
                txt += "\nHowever, it is missing .lib files - please install them with 'pip install mkl-devel'."
                raise ValueError(txt)
            elif bool(re.search(r"\.dll$", blas_file)) and not bool(re.search("libopenblas", blas_file)):
                raise ValueError("Found BLAS library at:\n" + os.path.join(blas_path, blas_file), "\nBut .lib files are missing!")
            else:
                print("Installation: Using BLAS library found in:\n" + os.path.join(blas_path, blas_file) + "\n\n")
        else:
            flags = ['_FOR_RTD']
            blas_path, blas_file, incl_path, incl_file = [None]*4

        ## if no CBLAS and no functions are present, there will be no prototypes for the cblas API
        if "NO_CBLAS" in flags:
            raise ValueError(nocblas_err_msg)

        ## Add findblas' header
        ## if installing with pip or setuptools, will be here (this is the ideal case)
        if os.path.exists(re.sub(r"__init__\.py$", "findblas.h", findblas.__file__)):
            finblas_head_fold = re.sub(r"__init__\.py$", "", findblas.__file__)

        ## if installing with distutils, will be placed here (this should ideally not happen)
        elif os.path.exists(os.path.join(sys.prefix, "include", "findblas.h")):
            finblas_head_fold = os.path.join(sys.prefix, "include")

        ## if the header file doesn't exist, shall raise en error
        else:
            raise ValueError("Could not find header file from 'findblas' - please try reinstalling with 'pip install --force findblas'")

        ## Pass extra flags for the header
        warning_msg = "No CBLAS headers were found - function propotypes might be unreliable."
        mkl_err_msg = "Missing MKL CBLAS headers, please reinstall with e.g. 'conda install mkl-include' or 'pip install mkl-include'."
        gsl_err_msg = "Missing GSL CBLAS headers, please reinstall with e.g. 'conda install gsl'."
        if incl_file == "mkl_cblas.h":
            flags.append("MKL_OWN_INCL_CBLAS")
        elif incl_file == "mkl_blas.h":
            raise ValueError(mkl_err_msg)
        elif incl_file == "cblas-openblas.h":
            flags.append("OPENBLAS_OWN_INCL")
        elif incl_file == "gsl_cblas.h":
            flags.append("GSL_OWN_INCL_CBLAS")
        elif incl_file == "gsl_blas.h":
            raise ValueError(gsl_err_msg)
        elif incl_file == "INCL_CBLAS":
            flags.append("INCL_CBLAS")
        elif incl_file == "blas.h":
            flags.append("INCL_BLAS")
            warnings.warn(warning_msg)
        elif (incl_path is None) or (incl_file is None):
            flags.append("NO_CBLAS_HEADER")
            warnings.warn(warning_msg)
        else:
            pass

        ## Now add them to the extension
        for e in self.extensions:
            if not from_rtd:
                if self.compiler.compiler_type == 'msvc': # visual studio
                    e.extra_link_args += [os.path.join(blas_path, blas_file)]
                else: # everything else which cares about following standards
                    e.extra_link_args += ["-L" + blas_path, "-l:" + blas_file]
            e.define_macros += [(f, None) for f in flags]
            if incl_path is not None:
                e.include_dirs.append(incl_path)
            e.include_dirs.append(finblas_head_fold)

        build_ext.build_extensions(self)
