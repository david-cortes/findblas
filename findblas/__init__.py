import os, sys, re, warnings
from sys import platform
from sysconfig import get_paths
from copy import deepcopy
import platform as platform_module

try:
    import numpy as np
    import numpy.distutils.system_info
except ImportError:
    pass
try:
    import scipy
    import scipy.linalg
except ImportError:
    pass

### TODO: maybe add a 'find_lapack' equivalent


def find_blas(allow_unidentified_blas=True, allow_pep518_paths=True):
    """
    Find installed BLAS library

    Find installed BLAS library either through a system install (e.g. by a package manager, CPACK, or downloading installer from intel's webpage),
    or a python install (e.g. 'conda install mkl mkl-include openblas gsl', 'pip install mkl mkl-include').

    Can find any of: MKL, OpenBLAS, BLIS, ATLAS, GSL - all of which offer the standard CBLAS API (e.g. functions named like 'cblas_dgemm').

    In non-Windows systems, will try to use either 'pyelftools' or system's 'readelf' to inspect the library's functions if the
    library's file name is generic (e.g. 'libblas.so').

    Does not have any external dependencies, but the following are recommended: numpy, scipy, pyelftools, cython.

    Parameters
    ----------
    allow_unidentified_blas : bool
        Whether to allow outputting a BLAS library which cannot be identified as being from one of the supported
        vendors (MKL, OpenBLAS, BLIS, ATLAS, GSL) and on which no standard BLAS function symbols have been found
        through ELF inspectors.

        Typically, SciPy's shared object files will be considered as candidates to output, and they tend to contain
        all of the 'reference' BLAS symbols as part of their exported symbols on linux, but on windows systems, the
        DLLs from SciPy tend not to export BLAS symbols and hence are not linkable for python extensions that require
        them. Note that this library will only inspect ELF formats, which windows DLLs do not conform to.

        If passing 'True', and an unrecognized library has been identified as a candidate, it will ask the user
        through a command line prompt about whether to take the library or not.

        If passing 'False', will not output an unrecognized and uninspected library, and there will be no user prompt.
    allow_pep518_paths : bool
        Whether to also look in temporary paths from a PEP518 build-time environment. Note that these paths will only
        be available during the setup of a given package, but will be removed afterwards, so having them as a hard-coded
        path will not be useful for dynamic linking, but can still be useful for other purposes such as static linking
        or just generically linking against symbols that are guaranteed to be loaded beforehand.

    Returns
    -------
    blas_path : str
        Path where the BLAS library file is located (e.g. '/usr/local/lib')
    blas_file : str
        Name of the file (e.g. 'libblas.so')
    incl_path : str
        Path where the corresponding header(s) can be found (e.g. '/usr/include')
    incl_file : str
        Name of the header file (e.g. 'cblas-openblas.h')
    flags : list
        Potential flags about the library that was found (can e.g. be passed to preprocessor), including:
        - HAS_MKL (MKL library was found)
        - HAS_OPENBLAS (OpenBLAS library was found)
        - HAS_BLIS (BLIS library was found - note that is does not include LAPACK like the others)
        - HAS_ATLAS (ATLAS library was found)
        - HAS_GSL (GSL library was found)
        - UNKNWON_BLAS (Vendor cannot be identified)
        - NO_CBLAS (found library does not possess CBLAS API)
        - HAS_UNDERSCORES (found library contains functions with original names ending in underscores, e.g. 'dgemm_').
    """

    if platform[:3] == "win":
        ext = [".lib", ".dll", ".dll.a", ".a"]
        pref = ""
    elif platform[:3] == "dar":
        ext = [".dylib", ".a"]
        pref = "lib"
    else:
        ext = [".so", ".a"]
        pref = "lib"

    ## Possible file names for each library in different OSes
    ## Tries to look for dynamic-link libraries at first, but in MSVC, linking to the .dll's will fail
    mkl_file_names1 = process_fnames1(["mkl_rt", "mkl_rt.2", "mkl_rt.1"], pref, ext[0])
    openblas_file_names1 = process_fnames1(["openblas", "openblas64", "openblas64_"], pref, ext[0])
    blis_file_names1 = process_fnames1(["blis", "blis-mt"], pref, ext[0])
    atlas_file_names1 = process_fnames1(["atlas", "tatlas", "satlas"], pref, ext[0])
    gsl_file_names1 = process_fnames1(["gslcblas"], pref, ext[0])

    if platform[:3] == "win":
        add_windows_fnames1 = lambda lst: [nm + ext[2] for nm in lst]
        openblas_file_names1 += add_windows_fnames1(["libopenblas", "libopenblas"])
        blis_file_names1 += add_windows_fnames1(["libblis", "libblis-mt"])
        atlas_file_names1 += add_windows_fnames1(["libatlas", "libatlas"])
        gsl_file_names1 += add_windows_fnames1(["libgslcblas", "libgslcblas"])

    mkl_file_names2 = process_fnames1(["mkl_rt", "mkl_rt.2", "mkl_rt.1"], pref, ext[1])
    openblas_file_names2 = process_fnames1(["openblas"], pref, ext[1])
    blis_file_names2 = process_fnames1(["blis", "blis-mt"], pref, ext[1])
    atlas_file_names2 = process_fnames1(["atlas", "tatlas", "satlas"], pref, ext[1])
    gsl_file_names2 = process_fnames1(["gslcblas"], pref, ext[1])

    incl_mkl_name = ["mkl.h", "mkl_cblas.h", "mkl_blas.h"]
    incl_openblas_name = ["cblas-openblas.h"]
    incl_blis_name = ["blis.h"]
    incl_atlas_name = []
    incl_gsl_name = ["gsl_cblas.h", "gsl_blas.h"]
    incl_generic_name = ["cblas.h", "blas.h"]

    ## Will look up each potential file name in folders:
    ## -Suggested by NumPy
    ## -Suggested by SciPy
    ## -Python installation folder
    ## -In PATH or PYTHONPATH
    ## -In system install folders
    candidate_paths = []

    ## Also have to search where are the headers
    mkl_include_paths = []
    openblas_include_paths = []
    blis_include_paths = []
    atlas_include_paths = []
    gsl_include_paths = []
    system_include_paths = []

    # sys.stdout = open(os.devnull, "w")
    import io
    from contextlib import redirect_stdout

    _ = io.StringIO()
    with redirect_stdout(_):
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('mkl')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blas_mkl')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "np.__config__.system_info.blas_mkl_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "scipy.__config__.mkl_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "scipy.__config__.blas_mkl_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('mkl')['include_dirs']",
            mkl_include_paths,
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blas_mkl')['include_dirs']",
            mkl_include_paths,
        )
        _try_add_from_command(
            "np.__config__.system_info.blas_mkl_info['include_dirs']", mkl_include_paths
        )
        _try_add_from_command(
            "scipy.__config__.mkl_info['include_dirs']", mkl_include_paths
        )
        _try_add_from_command(
            "scipy.__config__.blas_mkl_info['include_dirs']", mkl_include_paths
        )

        _try_add_from_command(
            "numpy.distutils.system_info.get_info('openblas')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "scipy.__config__.openblas_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "scipy.__config__.openblas_lapack_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('openblas')['include_dirs']",
            openblas_include_paths,
        )
        _try_add_from_command(
            "scipy.__config__.openblas_info['include_dirs']", openblas_include_paths
        )
        _try_add_from_command(
            "scipy.__config__.openblas_lapack_info['include_dirs']",
            openblas_include_paths,
        )

        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blis')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "scipy.__config__.blis_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blis')['include_dirs']",
            blis_include_paths,
        )
        _try_add_from_command(
            "scipy.__config__.blis_info['include_dirs']", blis_include_paths
        )

        _try_add_from_command(
            "numpy.distutils.system_info.get_info('atlas')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('atlas_blas')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "scipy.__config__.atlas_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('atlas')['include_dirs']",
            atlas_include_paths,
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('atlas_blas')['include_dirs']",
            atlas_include_paths,
        )
        _try_add_from_command(
            "scipy.__config__.atlas_info['include_dirs']", atlas_include_paths
        )

        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blas')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blas_opt')['library_dirs']",
            candidate_paths,
        )
        _try_add_from_command(
            "scipy.__config__.blas_opt_info['library_dirs']", candidate_paths
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blas')['include_dirs']",
            gsl_include_paths,
        )
        _try_add_from_command(
            "numpy.distutils.system_info.get_info('blas_opt')['include_dirs']",
            gsl_include_paths,
        )
        _try_add_from_command(
            "scipy.__config__.blas_opt_info['include_dirs']", gsl_include_paths
        )
    # sys.stdout = sys.__stdout__

    python_fold = sys.prefix
    system_include_paths.append(os.path.join(python_fold, "include"))
    gsl_include_paths.append(os.path.join(python_fold, "include", "gsl"))
    if platform[:3] == "win":
        candidate_paths.append(os.path.join(python_fold, "Library", "bin"))
        candidate_paths.append(os.path.join(python_fold, "Library", "lib"))
        candidate_paths.append(os.path.join(python_fold, "Library", "mingw-w64", "bin"))
        candidate_paths.append(os.path.join(python_fold, "Library", "mingw-w64", "lib"))
        candidate_paths.append(os.path.join(python_fold, "Library", "bin", "gsl"))
        candidate_paths.append(os.path.join(python_fold, "Library", "lib", "gsl"))
        candidate_paths.append(
            os.path.join(python_fold, "Library", "mingw-w64", "bin", "gsl")
        )
        candidate_paths.append(
            os.path.join(python_fold, "Library", "mingw-w64", "lib", "gsl")
        )
        system_include_paths.append(os.path.join(python_fold, "Library", "include"))
        system_include_paths.append(
            os.path.join(python_fold, "Library", "mingw-w64", "include")
        )
        gsl_include_paths.append(os.path.join(python_fold, "Library", "include", "gsl"))
        gsl_include_paths.append(
            os.path.join(python_fold, "Library", "mingw-w64", "include", "gsl")
        )
    else:
        candidate_paths.append(os.path.join(python_fold, "lib"))
    for fld in get_paths().values():
        if bool(re.search("lib", fld)):
            candidate_paths.append(re.sub(r"^(.*lib).*$", r"\1", fld))
        if bool(re.search("include", fld)):
            system_include_paths.append(re.sub(r"^(.*include).*$", r"\1", fld))

    _try_add_from_command("os.environ['PATH'].split(\":\")", candidate_paths)
    _try_add_from_command("os.environ['PATH'].split(\";\")", candidate_paths)
    _try_add_from_command("os.environ['PYTHONPATH'].split(\":\")", candidate_paths)
    _try_add_from_command("os.environ['PYTHONPATH'].split(\";\")", candidate_paths)

    paths_from_sys = [pt for pt in sys.path if bool(re.search("[Ll]ib", pt))]
    if platform[:3] == "win":
        paths_from_sys += [
            os.path.join(re.sub("(.*Library).*$", r"\1", pt), "bin")
            for pt in paths_from_sys
            if bool(re.search("Library", pt))
        ] + [
            os.path.join(re.sub("(.*Library).*$", r"\1", pt), "lib")
            for pt in paths_from_sys
            if bool(re.search("Library", pt))
        ]
        paths_from_sys += [
            os.path.join(
                os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "Library"), "bin"
            )
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        paths_from_sys += [
            os.path.join(
                os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "Library"), "lib"
            )
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        system_include_paths += [
            os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include")
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        gsl_include_paths += [
            os.path.join(
                os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include"), "gsl"
            )
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        candidate_paths.append(os.path.join(python_fold, "lib", "gsl"))
    else:
        paths_from_sys += [
            os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "lib")
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        paths_from_sys += [
            os.path.join(
                os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "lib"), "gsl"
            )
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        system_include_paths += [
            os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include")
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
        gsl_include_paths += [
            os.path.join(
                os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include"), "gsl"
            )
            for pt in paths_from_sys
            if bool(re.search(r"(.*conda\d?).*$", pt))
        ]
    paths_from_sys += [re.sub("(.*lib).*$", r"\1", pt) for pt in paths_from_sys]
    system_include_paths += [
        re.sub("(.*include).*$", r"\1", pt) for pt in paths_from_sys
    ]
    candidate_paths += paths_from_sys

    sys_arch = platform_module.architecture()[0]
    if sys_arch == "64bit":
        sys_arch = "64"
    elif sys_arch == "32bit":
        sys_arch = "32"
    elif sys_arch == "128bit":
        sys_arch = "128"
    else:
        sys_arch = ""

    if platform[:3] != "win":
        candidate_paths += [
            "/opt/intel/lib",
            "/opt/intel/lib/intel64",
            "/opt/intel/mkl/lib",
            "/opt/intel/mkl/lib/intel" + sys_arch,
            "/opt/intel/oneapi/mkl/latest/lib",
            "/opt/intel/oneapi/mkl/latest/lib" + sys_arch,
            "/usr/local/intel/lib",
            "/usr/local/intel/mkl/lib",
            "/usr/local/intel/lib/intel" + sys_arch,
            "/usr/local/intel/mkl/lib/intel" + sys_arch,
            "/usr/local/intel/oneapi/mkl/latest/lib",
            "/usr/local/intel/oneapi/mkl/latest/lib/intel" + sys_arch,
            "/usr/lib64/atlas",
            "/usr/lib/atlas",
            "/usr/local/lib64/atlas",
            "/usr/local/lib/atlas",
            "/usr/lib64/gsl",
            "/usr/lib/gsl",
            "/usr/local/lib64/gsl",
            "/usr/local/lib/gsl",
        ]
        candidate_paths += [
            "/usr/lib/x86" + ("_" if sys_arch == "64" else "") + sys_arch + "-linux-gnu",
            "/usr/lib",
            "/usr/local/lib",
            "/lib64",
            "/lib",
            "/usr/lib64",
            "/usr/local/lib64",
            "/opt/local/lib64",
            "/opt/local/lib",
        ]
        ## AMD blis has the version number hard-coded in the folder name without symlink
        amd_hardcoded_paths = ["/opt/AMD/aocl/"]
        for pt in amd_hardcoded_paths:
            if os.path.exists(pt):
                for d in os.listdir(pt):
                    dd = os.path.join(pt, d)
                    if os.path.isdir(dd):
                        candidate_paths.append(dd)
                        candidate_paths.append(os.path.join(dd, "lib"))
                        blis_include_paths.append(os.path.join(dd, "amd-blis"))

        mkl_include_paths += [
            "/opt/intel/include",
            "/opt/intel/mkl/include",
            "/opt/intel/mkl/include/intel" + sys_arch,
            "/usr/local/intel/include",
            "/usr/local/intel/mkl/include",
            "/usr/local/intel/include/intel" + sys_arch,
            "/usr/local/intel/mkl/include/intel" + sys_arch,
            "/opt/intel/oneapi/mkl/latest/include",
            "/opt/intel/oneapi/mkl/latest/include/oneapi",
            "/usr/local/intel/oneapi/mkl/latest/include",
            "/usr/local/intel/oneapi/mkl/latest/include/intel" + sys_arch,
        ]
        ## openblas is the only good citizen that plays by the rules
        atlas_include_paths += [
            "/usr/lib/atlas",
            "/usr/lib64/atlas",
            "/usr/lib/atlas/include",
            "/usr/lib64/atlas/include",
        ]
        gsl_include_paths += [
            "/usr/include/gsl",
            "/usr/local/include/gsl",
            "/opt/local/include/gsl",
        ]
        system_include_paths += [
            "/usr/include/x86" + "_"
            if sys_arch == "64"
            else "" + sys_arch + "-linux-gnu",
            "/usr/include",
            "/usr/local/include",
            "/opt/local/include",
        ]

        if platform[:3] == "dar":
            candidate_paths.append("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework")

            for blas_lib in ["openblas", "blis", "atlas", "blas"]:
                path_homebrew_blas = f"/opt/homebrew/opt/{blas_lib}"
                if os.path.exists(path_homebrew_blas):
                    if os.path.exists(os.path.join(path_homebrew_blas, "lib")):
                        candidate_paths.append(os.path.join(path_homebrew_blas, "lib"))
                    include_path_this = os.path.join(path_homebrew_blas, "include")
                    if os.path.exists(include_path_this):
                        if blas_lib == "openblas":
                            openblas_include_paths.append(include_path_this)
                        elif blas_lib == "blis":
                            blis_include_paths.append(include_path_this)
                        elif blas_lib == "atlas":
                            atlas_include_paths.append(include_path_this)
                        else:
                            system_include_paths.append(include_path_this)

                path_homebrew_blas_versioned = f"/opt/homebrew/Cellar/{blas_lib}"
                if os.path.exists(path_homebrew_blas_versioned):
                    for sub_pt in os.listdir(path_homebrew_blas_versioned):
                        candidate_hb_openblas = os.path.join(path_homebrew_blas_versioned, sub_pt)
                        if os.path.isdir(candidate_hb_openblas):
                            if os.path.exists(os.path.join(candidate_hb_openblas, "lib")):
                                candidate_paths.append(os.path.join(candidate_hb_openblas, "lib"))
                            include_path_this = os.path.join(path_homebrew_blas, "include")
                            if os.path.exists(include_path_this):
                                if blas_lib == "openblas":
                                    openblas_include_paths.append(include_path_this)
                                elif blas_lib == "blis":
                                    blis_include_paths.append(include_path_this)
                                elif blas_lib == "atlas":
                                    atlas_include_paths.append(include_path_this)
                                else:
                                    system_include_paths.append(include_path_this)

    else:
        ## Try to lookup default MKL installation
        intel_folder = os.path.join(
            os.environ["ProgramFiles" + "(x86)" if sys_arch != "32" else ""],
            "IntelSWTools",
        )
        if os.path.exists(intel_folder):
            curr_path = intel_folder
            fold_c_n_l = [
                fld
                for fld in os.listdir(curr_path)
                if bool(re.search("compilers_and_libraries", fld))
            ]
            if len(fold_c_n_l) > 0:
                for f_c_n_l in fold_c_n_l:
                    curr_path = os.path.join(intel_folder, f_c_n_l)
                    if os.path.exists(os.path.join(curr_path, "windows")):
                        curr_path = os.path.join(curr_path, "windows")

                        ### lookup redist
                        if os.path.exists(
                            os.path.join(curr_path, "redist", "intel" + sys_arch, "mkl")
                        ):
                            candidate_paths.append(
                                os.path.join(
                                    curr_path, "redist", "intel" + sys_arch, "mkl"
                                )
                            )

                        ## lookup mkl
                        if os.path.exists(
                            os.path.join(curr_path, "mkl", "lib", "intel" + sys_arch)
                        ):
                            candidate_paths.append(
                                os.path.join(
                                    curr_path, "mkl", "lib", "intel" + sys_arch
                                )
                            )

                        ## lookup include
                        if os.path.exists(os.path.join(curr_path, "mkl", "include")):
                            mkl_include_paths.append(
                                os.path.join(curr_path, "mkl", "include")
                            )

        ## Try to add C:\Windows\System32
        try:
            ## https://stackoverflow.com/questions/41630224/python-does-not-find-system32
            is_wow64 = (
                platform.architecture()[0] == "32bit"
                and "ProgramFiles(x86)" in os.environ
            )
            candidate_paths += os.path.join(
                os.environ["SystemRoot"], "SysNative" if is_wow64 else "System32"
            )
        except Exception:
            pass

        ## Try to add visual studio headers
        try:
            ## https://stackoverflow.com/questions/335408/where-does-visual-studio-look-for-c-header-files/335426#335426
            vs_path = os.path.join(
                os.environ["ProgramFiles" + "(x86)" if sys_arch != "32" else ""],
                "Microsoft Visual Studio",
            )
            if os.path.exists(os.path.join(vs_path)):
                for yr in os.listdir(vs_path):
                    vr_path = os.path.join(
                        vs_path, yr, "Community", "VC", "Tools", "MSVC"
                    )
                    if os.path.exists(vr_path):
                        system_include_paths += [
                            os.path.join(vr_path, v, "include")
                            for v in os.listdir(vr_path)
                            if os.path.exists(os.path.join(vr_path, v, "include"))
                        ]
            winkits_folder = os.path.join(
                os.environ["ProgramFiles" + "(x86)" if sys_arch != "32" else ""],
                "Windows Kits",
            )
            if os.path.exists(winkits_folder):
                for v in winkits_folder:
                    i_fold = os.path.join(winkits_folder, v, "Include")
                    if os.path.exists(i_fold):
                        for vr in os.listdir(i_fold):
                            h_fold = os.path.join(i_fold, vr, "ucrt")
                            if os.path.exists(h_fold):
                                system_include_paths += h_fold
        except Exception:
            pass

    ## Potential cases of PEP518 environments
    paths_pep518 = []
    if allow_pep518_paths:
        for path in candidate_paths:
            if bool(re.search(r"[Oo]verlay", path)):
                clean_path = re.sub(r"^(.*[Oo]verlay).*$", r"\1", path)

                paths_pep518.append(clean_path)

                paths_pep518.append(os.path.join(clean_path, "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "gsl"))
                paths_pep518.append(os.path.join(clean_path, "lib", "gsl", "lib"))
                gsl_include_paths.append(os.path.join(clean_path, "lib", "gsl", "include"))
                paths_pep518.append(os.path.join(clean_path, "lib", "openblas"))
                paths_pep518.append(os.path.join(clean_path, "lib", "openblas", "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "blis"))
                paths_pep518.append(os.path.join(clean_path, "lib", "blis", "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "openblas-openmp"))
                paths_pep518.append(
                    os.path.join(clean_path, "lib", "openblas-openmp", "lib")
                )
                paths_pep518.append(os.path.join(clean_path, "lib", "blis-openmp"))
                paths_pep518.append(os.path.join(clean_path, "lib", "blis-openmp", "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "openblas-pthread"))
                paths_pep518.append(
                    os.path.join(clean_path, "lib", "openblas-pthread", "lib")
                )
                paths_pep518.append(os.path.join(clean_path, "lib", "blis-pthread"))
                paths_pep518.append(os.path.join(clean_path, "lib", "blis-pthread", "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "openblas-serial"))
                paths_pep518.append(
                    os.path.join(clean_path, "lib", "openblas-serial", "lib")
                )
                paths_pep518.append(os.path.join(clean_path, "lib", "blis-serial"))
                paths_pep518.append(os.path.join(clean_path, "lib", "blis-serial", "lib"))
                openblas_include_paths.append(
                    os.path.join(clean_path, "lib", "openblas", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "lib", "openblas-openmp", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "lib", "openblas-pthread", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "lib", "openblas-serial", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "lib", "blis", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "lib", "blis-openmp", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "lib", "blis-pthread", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "lib", "blis-serial", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "lib", "amd-blis", "include")
                )
                paths_pep518.append(os.path.join(clean_path, "lib", "gslcblas"))
                paths_pep518.append(os.path.join(clean_path, "lib", "cblas"))
                paths_pep518.append(os.path.join(clean_path, "lib", "blas"))
                paths_pep518.append(os.path.join(clean_path, "lib", "mkl"))
                paths_pep518.append(os.path.join(clean_path, "lib", "mkl", "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "mkl", "lib", "intel"))
                mkl_include_paths.append(os.path.join(clean_path, "lib", "mkl", "include"))
                paths_pep518.append(os.path.join(clean_path, "lib", "atlas"))
                atlas_include_paths.append(
                    os.path.join(clean_path, "lib", "atlas", "include")
                )
                paths_pep518.append(os.path.join(clean_path, "lib", "atlas", "lib"))
                paths_pep518.append(os.path.join(clean_path, "lib", "libatlas"))
                atlas_include_paths.append(
                    os.path.join(clean_path, "lib", "libatlas", "include")
                )
                paths_pep518.append(os.path.join(clean_path, "lib", "libatlas", "lib"))

                paths_pep518.append(os.path.join(clean_path, "Lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "gsl"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "gsl", "lib"))
                gsl_include_paths.append(os.path.join(clean_path, "Lib", "gsl", "include"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "openblas"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "openblas", "lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis", "lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "openblas-openmp"))
                paths_pep518.append(
                    os.path.join(clean_path, "Lib", "openblas-openmp", "lib")
                )
                paths_pep518.append(os.path.join(clean_path, "Lib", "openblas-pthread"))
                paths_pep518.append(
                    os.path.join(clean_path, "Lib", "openblas-pthread", "lib")
                )
                paths_pep518.append(os.path.join(clean_path, "Lib", "openblas-serial"))
                paths_pep518.append(
                    os.path.join(clean_path, "Lib", "openblas-serial", "lib")
                )
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis-openmp"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis-openmp", "lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis-pthread"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis-pthread", "lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis-serial"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blis-serial", "lib"))
                openblas_include_paths.append(
                    os.path.join(clean_path, "Lib", "openblas", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "Lib", "openblas-openmp", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "Lib", "openblas-pthread", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "Lib", "openblas-serial", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "Lib", "blis", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "Lib", "blis-openmp", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "Lib", "blis-pthread", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "Lib", "blis-serial", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "Lib", "amd-blis", "include")
                )
                paths_pep518.append(os.path.join(clean_path, "Lib", "gslcblas"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "cblas"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "blas"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "mkl"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "mkl", "lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "mkl", "lib", "intel"))
                mkl_include_paths.append(os.path.join(clean_path, "Lib", "mkl", "include"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "atlas"))
                atlas_include_paths.append(
                    os.path.join(clean_path, "Lib", "atlas", "include")
                )
                paths_pep518.append(os.path.join(clean_path, "Lib", "atlas", "lib"))
                paths_pep518.append(os.path.join(clean_path, "Lib", "libatlas"))
                atlas_include_paths.append(
                    os.path.join(clean_path, "Lib", "libatlas", "include")
                )
                paths_pep518.append(os.path.join(clean_path, "Lib", "libatlas", "lib"))

                system_include_paths.append(os.path.join(clean_path, "include"))
                gsl_include_paths.append(os.path.join(clean_path, "include", "gsl"))
                gsl_include_paths.append(
                    os.path.join(clean_path, "include", "gsl", "include")
                )
                gsl_include_paths.append(os.path.join(clean_path, "include", "gslcblas"))
                system_include_paths.append(os.path.join(clean_path, "include", "cblas"))
                system_include_paths.append(os.path.join(clean_path, "include", "blas"))
                mkl_include_paths.append(os.path.join(clean_path, "include", "mkl"))
                mkl_include_paths.append(os.path.join(clean_path, "include", "mkl", "lib"))
                mkl_include_paths.append(
                    os.path.join(clean_path, "include", "mkl", "lib", "intel")
                )
                mkl_include_paths.append(
                    os.path.join(clean_path, "include", "mkl", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas", "include")
                )
                blis_include_paths.append(os.path.join(clean_path, "include", "blis"))
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas-openmp")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas-openmp", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas-pthread")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas-pthread", "include")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas-serial")
                )
                openblas_include_paths.append(
                    os.path.join(clean_path, "include", "openblas-serial", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis-openmp")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis-openmp", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis-pthread")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis-pthread", "include")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis-serial")
                )
                blis_include_paths.append(
                    os.path.join(clean_path, "include", "blis-serial", "include")
                )
                blis_include_paths.append(os.path.join(clean_path, "include", "amd-blis"))
                atlas_include_paths.append(os.path.join(clean_path, "include", "atlas"))
                atlas_include_paths.append(
                    os.path.join(clean_path, "include", "atlas", "include")
                )
                atlas_include_paths.append(os.path.join(clean_path, "include", "libatlas"))
                atlas_include_paths.append(
                    os.path.join(clean_path, "lib", "libatlas", "include")
                )

    candidate_paths += paths_pep518

    ## Try getting MKL from pip too
    try:
        import pip
        import io
        from contextlib import redirect_stdout

        pip_outp = io.StringIO()
        try:
            try:
                with redirect_stdout(pip_outp):
                    pip.main(["show", "-f", "mkl"])
            except Exception:
                from pip._internal import main as pip_main

                with redirect_stdout(pip_outp):
                    pip_main(["show", "-f", "mkl"])
        except Exception:
            with redirect_stdout(pip_outp):
                os.system("pip show -f mkl")

        pip_outp = pip_outp.getvalue()
        pip_outp = pip_outp.split("\n")
        for ln in pip_outp:
            if bool(re.search(r"^Location", ln)):
                files_root = re.sub(r"^Location:\s+", "", ln)
                files_root = files_root.rstrip()
                break

        for ln in pip_outp:
            ## exts: .o, .so, .a, .lib, .dll, .dynlib
            if bool(re.search(r"mkl_rt(\.\d)?\.[solibdyaSOLIBDYA]+", ln)):
                candidate_paths.append(
                    os.path.join(
                        files_root,
                        re.sub(
                            r"^\s*(.*)[\/\\]+[lib]*mkl_rt(\.\d)?\.[solibdyaSOLIBDYA]+(\.\d)?",
                            r"\1",
                            ln,
                        ),
                    )
                )

    except Exception:
        pass

    ## Discard duplicated paths, but keep the order
    search_paths = _deduplicate_paths(candidate_paths)

    flags_found = list()
    blas_file = None
    blas_path = None

    ### Start looking for each library in selected paths
    def search_blas_lib(search_paths, blas_names, suff=None):
        if suff is not None:
            search_paths_new = deepcopy(search_paths)
            for s in suff:
                search_paths_new += [pt + s for pt in search_paths]
            search_paths = search_paths_new

        blas_path = None
        blas_file = None
        for blas_name in blas_names:
            if blas_file is None:
                for pt in search_paths:
                    if os.path.exists(os.path.join(pt, blas_name)):
                        blas_file = blas_name
                        blas_path = pt
                        break
            else:
                break
        return blas_path, blas_file

    ### First try dynamic-link libraries
    ## MKL
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(search_paths, mkl_file_names1)
        if blas_file is not None:
            flags_found.append("HAS_MKL")

    ## OpenBLAS
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(
            search_paths,
            openblas_file_names1,
            ["openblas", "openblas-openmp", "openblas-pthread", "openblas-serial"],
        )
        if blas_file is not None:
            flags_found.append("HAS_OPENBLAS")
            flags_found.append("HAS_UNDERSCORES")

    ## BLIS
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(
            search_paths,
            blis_file_names1,
            ["blis", "blis-openmp", "blis-pthread", "blis-serial"],
        )
        if blas_file is not None:
            flags_found.append("HAS_BLIS")
            flags_found.append("HAS_UNDERSCORES")

    ## ATLAS
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(search_paths, atlas_file_names1)
        if blas_file is not None:
            flags_found.append("HAS_ATLAS")
            flags_found.append("HAS_UNDERSCORES")

    ## GSL
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(search_paths, gsl_file_names1)
        if blas_file is not None:
            flags_found.append("HAS_GSL")

    ### Then try static libraries
    ## MKL
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(search_paths, mkl_file_names2)
        if blas_file is not None:
            flags_found.append("HAS_MKL")

    ## OpenBLAS
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(
            search_paths,
            openblas_file_names2,
            ["openblas", "openblas-openmp", "openblas-pthread", "openblas-serial"],
        )
        if blas_file is not None:
            flags_found.append("HAS_OPENBLAS")
            flags_found.append("HAS_UNDERSCORES")

    ## BLIS
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(
            search_paths,
            blis_file_names2,
            ["blis", "blis-openmp", "blis-pthread", "blis-serial"],
        )
        if blas_file is not None:
            flags_found.append("HAS_BLIS")
            flags_found.append("HAS_UNDERSCORES")

    ## ATLAS
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(search_paths, atlas_file_names2)
        if blas_file is not None:
            flags_found.append("HAS_ATLAS")
            flags_found.append("HAS_UNDERSCORES")

    ## GSL
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(search_paths, gsl_file_names2)
        if blas_file is not None:
            flags_found.append("HAS_GSL")

    ### Generic
    if blas_file is None:
        blas_path, blas_file = search_blas_lib(
            search_paths, [pref + "cblas" + e for e in ext]
        )
        if blas_file is not None:
            if platform[:3] != "win":
                found_syms = _find_symbols(blas_path, blas_file)
                if found_syms[1] is not None:
                    flags_found += found_syms[1]
        else:
            blas_path, blas_file = search_blas_lib(
                search_paths, [pref + "blas" + e for e in ext] + [pref + "BLAS" + e for e in ext]
            )
            if blas_file is not None:
                if platform[:3] != "win":
                    found_syms = _find_symbols(blas_path, blas_file)
                    if found_syms[1] is not None:
                        flags_found += found_syms[1]

    ### Try regex matching
    def check_is_blas(pt, fname, allow_unidentified_blas):
        ask_user = True
        is_blas = False
        flags_found = []
        if platform[:3] != "win":
            found_syms = _find_symbols(pt, fname)
            if found_syms[0] is True:
                if found_syms[1] is not None:
                    is_blas = True
                    flags_found += found_syms[1]
                ask_user = False

        if ask_user and allow_unidentified_blas:
            ## if not, ask the user explicitly
            txt = "Found file with name matching 'blas'\n"
            txt += pt + fname + "\n"
            txt += "Is this a BLAS library? [y/n]: "
            while True:
                use_this = input(txt)
                if use_this.lower()[0] == "y":
                    is_blas = True
                    flags_found.append("UNKNWON_BLAS")
                    break
                elif use_this.lower()[0] == "n":
                    is_blas = False
                    break
        return is_blas, flags_found

    if blas_file is None:
        candidate_files_dyna = []
        candidate_files_stat = []
        candidate_paths_dyna = []
        candidate_paths_stat = []
        for pt in search_paths:
            if os.path.exists(pt):
                nfound_dyna = len(candidate_files_dyna)
                nfound_stat = len(candidate_files_stat)
                candidate_files_dyna += [
                    f
                    for f in os.listdir(pt)
                    if bool(re.search("blas", f))
                    and bool(re.search(r"\." + ext[0][1:] + r"$", f))
                ]
                candidate_files_stat += [
                    f
                    for f in os.listdir(pt)
                    if bool(re.search("blas", f))
                    and bool(re.search(r"\." + ext[1][1:] + r"$", f))
                ]
                nfound_dyna = len(candidate_files_dyna) - nfound_dyna
                nfound_stat = len(candidate_files_stat) - nfound_stat
                candidate_paths_dyna += [pt] * nfound_dyna
                candidate_paths_stat += [pt] * nfound_stat

        candidate_files = candidate_files_dyna + candidate_files_stat
        candidate_paths = candidate_paths_dyna + candidate_paths_stat
        for f in range(len(candidate_files)):
            is_blas, temp = check_is_blas(
                candidate_paths[f], candidate_files[f], allow_unidentified_blas
            )
            if is_blas:
                blas_file = candidate_files[f]
                blas_path = candidate_paths[f]
                flags_found += temp
                break

    err_msg = "Could not locate MKL, OpenBLAS, BLIS, ATLAS or GSL libraries - you'll need to manually modify setup.py to add BLAS path."
    if blas_file is None:
        try:
            import numpy as np

            path_np = re.sub(r"\\", "/", np.__file__)
            path_np = re.sub(r"(/+)?__init__\.py$", "", path_np)
            path_np = os.path.join(path_np, ".libs")
            files_np = [f for f in os.listdir(path_np) if bool(re.search("blas", f))]
            blas_file = files_np[0]
            blas_path = path_np
            if platform[:3] != "win":
                found_syms = _find_symbols(blas_path, blas_file)
                if found_syms[0] is True:
                    if found_syms[1] is None:
                        raise ValueError(err_msg)
                    else:
                        flags_found += found_syms[1]

            warnings.warn(
                "No BLAS library found - taking NumPy's linalg's file as library."
            )
        except Exception:
            try:
                import scipy.linalg

                blas_from_scipy = re.sub(r"\\", "/", scipy.linalg.cython_blas.__file__)
                blas_file = re.sub(r".*/(.*)$", r"\1", blas_from_scipy)
                blas_path = re.sub(r"(.*)/.*$", r"\1", blas_from_scipy)
                if platform[:3] != "win":
                    found_syms = _find_symbols(blas_path, blas_file)
                    if found_syms[0] is True:
                        if found_syms[1] is None:
                            raise ValueError(err_msg)
                        else:
                            flags_found += found_syms[1]

                warnings.warn(
                    "No BLAS library found - taking SciPy's linalg.cython_blas file as library."
                )

            except Exception:
                raise ValueError(err_msg)

    ### Now lookup the include path
    def get_inc_paths(blas_path, include_paths, system_include_paths):
        include_paths += [
            blas_path,
            re.sub("/lib.?$", "/include", blas_path),
            re.sub(r"^(.*)/lib/(.*)$", r"\1/include/\2", blas_path),
            re.sub(r"^(.*)Library.*$", r"\1include", blas_path),
        ]
        include_paths += system_include_paths
        return _deduplicate_paths(include_paths)

    def search_incl_kwds(search_paths, blas_names, keywords):
        incl_file = None
        incl_path = None
        for incl_name in blas_names:
            for pt in search_paths:
                if incl_file is not None:
                    break
                if os.path.exists(os.path.join(pt, incl_name)):
                    with open(os.path.join(pt, incl_name)) as h:
                        for line in h:
                            if incl_file is not None:
                                break
                            for kw in keywords:
                                if bool(re.search(kw, line)):
                                    incl_file = incl_name
                                    incl_path = pt
                                    break

        return incl_path, incl_file

    if "HAS_MKL" in flags_found:
        search_paths = get_inc_paths(blas_path, mkl_include_paths, system_include_paths)
        incl_path, incl_file = search_incl_kwds(
            search_paths, incl_mkl_name + incl_generic_name, ["MKL"]
        )

    elif "HAS_OPENBLAS" in flags_found:
        search_paths = get_inc_paths(
            blas_path, openblas_include_paths, system_include_paths
        )
        incl_path, incl_file = search_incl_kwds(
            search_paths, incl_openblas_name + incl_generic_name, ["openblas"]
        )

    elif "HAS_BLIS" in flags_found:
        search_paths = get_inc_paths(
            blas_path, openblas_include_paths, system_include_paths
        )
        incl_path, incl_file = search_incl_kwds(
            search_paths, incl_openblas_name + incl_generic_name, ["blis", "amd-blis"]
        )

    elif "HAS_ATLAS" in flags_found:
        search_paths = get_inc_paths(
            blas_path, atlas_include_paths, system_include_paths
        )
        incl_path, incl_file = search_incl_kwds(
            search_paths, incl_atlas_name + incl_generic_name, ["atlas", "ATLAS"]
        )

    elif "HAS_GSL" in flags_found:
        search_paths = get_inc_paths(blas_path, gsl_include_paths, system_include_paths)
        incl_path, incl_file = search_incl_kwds(
            search_paths, incl_gsl_name + incl_generic_name, ["GSL_CBLAS"]
        )

    else:
        flags_found.append("UNKNWON_BLAS")
        search_paths = get_inc_paths(blas_path, [], system_include_paths)
        all_kwds = ["MKL", "openblas", "blis", "atlas", "ATLAS"]
        if "NO_CBLAS" not in flags_found:
            all_kwds += ["GSL_CBLAS", "cblas", "CBLAS"]
        all_kwds += ["ddot", "DDOT"]
        incl_path, incl_file = search_incl_kwds(
            search_paths, incl_generic_name, all_kwds
        )

    return blas_path, blas_file, incl_path, incl_file, flags_found


def _deduplicate_paths(candidate_paths):
    ## Discards duplicated paths, but keep the order
    seen_paths = set()
    search_paths = list()
    for pt in candidate_paths:
        pt = re.sub(r"\\", "/", pt)
        pt = re.sub(r"/+", "/", pt)
        if pt not in seen_paths:
            search_paths.append(pt)
            seen_paths.add(pt)
    return search_paths


def process_fnames1(lst, pref, ext):
    out = []
    for nm in lst:
        tmp = nm.split(".")
        # Note: on linux, MKL can be named as 'libmkl_rt.so.2',
        # while on windows and mac will be named as e.g. 'mkl_rt.1.dll'
        if (len(tmp) == 2) and (platform[:3] not in ("win", "dar")):
            out.append(pref + tmp[0] + ext + "." + tmp[1])
        else:
            out.append(pref + nm + ext)
    return out


def _try_add_from_command(str_expr, candidate_paths):
    try:
        exec("candidate_paths += " + str_expr)
    except Exception:
        pass


def _find_symbols(pt, fname):
    try:
        from elftools.elf.elffile import ELFFile

        with open(os.path.join(pt, fname), "rb") as f:
            elffile = ELFFile(f)
            symtab = elffile.get_section_by_name(".symtab")
            if symtab.get_symbol_by_name("openblas_get_config") is not None:
                return True, ["HAS_OPENBLAS", "HAS_UNDERSCORES"]
            if symtab.get_symbol_by_name("bli_axpyd") is not None:
                return True, ["HAS_BLIS", "HAS_UNDERSCORES"]
            if symtab.get_symbol_by_name("mkl_dcsrgemv") is not None:
                return True, ["HAS_MKL"]
            if symtab.get_symbol_by_name("ddot_") is not None:
                found_syms = ["HAS_UNDERSCORES"]
                if symtab.get_symbol_by_name("cblas_ddot") is None:
                    found_syms += ["NO_CBLAS"]
                return True, found_syms
            if symtab.get_symbol_by_name("cblas_ddot") is not None:
                return True, []
            if (symtab.get_symbol_by_name("ddot") is not None) or (
                symtab.get_symbol_by_name("DDOT") is not None
            ):
                return True, ["NO_CBLAS"]
        return True, None

    except Exception:
        try:
            import subprocess

            symbols = subprocess.check_output(
                ["readelf", "-s", os.path.join(pt, fname)]
            )
            symbols = str(symbols).split()
            has_cblas = False
            has_underscores = False
            has_ddot = False
            for s in symbols:
                if bool(re.search("openblas", s)):
                    return True, ["HAS_OPENBLAS", "HAS_UNDERSCORES"]
                if bool(re.search("bli_axpyd", s)):
                    return True, ["HAS_BLIS", "HAS_UNDERSCORES"]
                if bool(re.search("mkl_dcsrgemv", s)):
                    return True, ["HAS_MKL"]
                if bool(re.search(r"ddot_[^a-z]", s)):
                    has_underscores = True
                if bool(re.search("cblas_ddot", s)):
                    has_cblas = True
                if bool(re.search("ddot", s)):
                    has_ddot = True

            flags_found = []
            if not has_cblas:
                flags_found.append("NO_CBLAS")
            if has_underscores:
                flags_found.append("HAS_UNDERSCORES")

            if has_ddot:
                return True, flags_found
            else:
                return True, None
        except Exception:
            return False, None
