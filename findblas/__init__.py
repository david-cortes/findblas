import os, sys, re, warnings
from sys import platform
from sysconfig import get_paths
import platform as platform_module
try:
	import numpy as np
	import numpy.distutils.system_info
except:
	pass
try:
	import scipy
	import scipy.linalg
except:
	pass

def find_blas():
	"""
	Find installed BLAS library

	Find installed BLAS library either through a system install (e.g. by a package manager, CPACK, or downloading installer from intel's webpage),
	or a python install (e.g. 'conda install mkl mkl-include openblas gsl', 'pip install mkl mkl-include').

	Can find any of: MKL, OpenBLAS, ATLAS - GSL, all of which offer the standard CBLAS API (e.g. functions named like 'cblas_dgemm').

	In non-Windows systems, will try to use either 'pyelftools' or system's 'readelf' to inspect the library's functions if the
	library's file name is generic (e.g. 'libblas.so').

	Does not have any external dependencies, but the following are recommended: numpy, scipy, pyelftools, cython.

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
	mkl_file_names1 = [pref + "mkl_rt" + ext[0]]
	openblas_file_names1 = [pref + "openblas" + ext[0]]
	atlas_file_names1 = [pref + "atlas" + ext[0], pref + "tatlas" + ext[0], pref + "satlas" + ext[0]]
	gsl_file_names1 = [pref + "gslcblas" + ext[0]]

	if platform[:3] == "win":
		openblas_file_names1 += ["libopenblas" + ext[2], "libopenblas" + ext[3]]
		atlas_file_names1 += ["libatlas" + ext[2], "libatlas" + ext[3]]
		gsl_file_names1 += ["libgslcblas" + ext[2], "libgslcblas" + ext[3]]

	mkl_file_names2 = [pref + "mkl_rt" + ext[1]]
	openblas_file_names2 = [pref + "openblas" + ext[1]]
	atlas_file_names2 = [pref + "atlas" + ext[1], pref + "tatlas" + ext[1], pref + "satlas" + ext[1]]
	gsl_file_names2 = [pref + "gslcblas" + ext[1]]

	incl_mkl_name = ["mkl.h", "mkl_cblas.h", "mkl_blas.h"]
	incl_openblas_name = ["cblas-openblas.h"]
	incl_atlas_name = []
	incl_gsl_name = ["gsl_cblas.h", "gsl_blas.h"]
	incl_generic_name = ['cblas.h', 'blas.h']

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
	atlas_include_paths = []
	gsl_include_paths = []
	system_include_paths = []

	sys.stdout = open(os.devnull, "w")
	_try_add_from_command("numpy.distutils.system_info.get_info('mkl')['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('blas_mkl')['library_dirs']", candidate_paths)
	_try_add_from_command("np.__config__.system_info.blas_mkl_info['library_dirs']", candidate_paths)
	_try_add_from_command("scipy.__config__.mkl_info['library_dirs']", candidate_paths)
	_try_add_from_command("scipy.__config__.blas_mkl_info['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('mkl')['include_dirs']", mkl_include_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('blas_mkl')['include_dirs']", mkl_include_paths)
	_try_add_from_command("np.__config__.system_info.blas_mkl_info['include_dirs']", mkl_include_paths)
	_try_add_from_command("scipy.__config__.mkl_info['include_dirs']", mkl_include_paths)
	_try_add_from_command("scipy.__config__.blas_mkl_info['include_dirs']", mkl_include_paths)

	_try_add_from_command("numpy.distutils.system_info.get_info('openblas')['library_dirs']", candidate_paths)
	_try_add_from_command("scipy.__config__.openblas_info['library_dirs']", candidate_paths)
	_try_add_from_command("scipy.__config__.openblas_lapack_info['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('openblas')['include_dirs']", openblas_include_paths)
	_try_add_from_command("scipy.__config__.openblas_info['include_dirs']", openblas_include_paths)
	_try_add_from_command("scipy.__config__.openblas_lapack_info['include_dirs']", openblas_include_paths)

	_try_add_from_command("numpy.distutils.system_info.get_info('atlas')['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('atlas_blas')['library_dirs']", candidate_paths)
	_try_add_from_command("scipy.__config__.atlas_info['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('atlas')['include_dirs']", atlas_include_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('atlas_blas')['include_dirs']", atlas_include_paths)
	_try_add_from_command("scipy.__config__.atlas_info['include_dirs']", atlas_include_paths)

	_try_add_from_command("numpy.distutils.system_info.get_info('blas')['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('blas_opt')['library_dirs']", candidate_paths)
	_try_add_from_command("scipy.__config__.blas_opt_info['library_dirs']", candidate_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('blas')['include_dirs']", gsl_include_paths)
	_try_add_from_command("numpy.distutils.system_info.get_info('blas_opt')['include_dirs']", gsl_include_paths)
	_try_add_from_command("scipy.__config__.blas_opt_info['include_dirs']", gsl_include_paths)
	sys.stdout = sys.__stdout__

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
		candidate_paths.append(os.path.join(python_fold, "Library", "mingw-w64", "bin", "gsl"))
		candidate_paths.append(os.path.join(python_fold, "Library", "mingw-w64", "lib", "gsl"))
		system_include_paths.append(os.path.join(python_fold, "Library", "include"))
		system_include_paths.append(os.path.join(python_fold, "Library", "mingw-w64", "include"))
		gsl_include_paths.append(os.path.join(python_fold, "Library", "include", "gsl"))
		gsl_include_paths.append(os.path.join(python_fold, "Library", "mingw-w64", "include", "gsl"))
	else:
		candidate_paths.append(os.path.join(python_fold, "lib"))
	for fld in get_paths().values():
		if bool(re.search("lib", fld)):
			candidate_paths.append(re.sub(r"^(.*lib).*$", r"\1", fld))
		if bool(re.search("include", fld)):
			system_include_paths.append(re.sub(r"^(.*include).*$", r"\1", fld))

	_try_add_from_command("os.environ['PATH'].split(\":\")", candidate_paths)
	_try_add_from_command("os.environ['PYTHONPATH'].split(\":\")", candidate_paths)

	paths_from_sys = [pt for pt in sys.path if bool(re.search("[Ll]ib", pt))]
	if platform[:3] == "win":
		paths_from_sys += [os.path.join(re.sub("(.*Library).*$", r"\1", pt), "bin") for pt in paths_from_sys if bool(re.search("Library", pt))] \
		 + [os.path.join(re.sub("(.*Library).*$", r"\1", pt), "lib") for pt in paths_from_sys if bool(re.search("Library", pt))]
		paths_from_sys += [os.path.join(os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "Library"), "bin") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		paths_from_sys += [os.path.join(os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "Library"), "lib") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		system_include_paths += [os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		gsl_include_paths += [os.path.join(os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include"), "gsl") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		candidate_paths.append(os.path.join(python_fold, "lib", "gsl"))
	else:
		paths_from_sys += [os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "lib") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		paths_from_sys += [os.path.join(os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "lib"), "gsl") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		system_include_paths += [os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
		gsl_include_paths += [os.path.join(os.path.join(re.sub(r"(.*conda\d?).*$", r"\1", pt), "include"), "gsl") for pt in paths_from_sys if bool(re.search(r"(.*conda\d?).*$", pt))]
	paths_from_sys += [re.sub("(.*lib).*$", r"\1", pt) for pt in paths_from_sys]
	system_include_paths += [re.sub("(.*include).*$", r"\1", pt) for pt in paths_from_sys]
	candidate_paths += paths_from_sys

	sys_arch = platform_module.architecture()[0]
	if sys_arch == '64bit':
		sys_arch = '64'
	elif sys_arch == '32bit':
		sys_arch = '32'
	elif sys_arch == '128bit':
		sys_arch = '128'
	else:
		sys_arch = ''

	if platform[:3] != "win":
		candidate_paths += ['/opt/intel/lib', '/opt/intel/lib/intel64', '/opt/intel/mkl/lib', '/opt/intel/mkl/lib/intel'+sys_arch,
		'/usr/local/intel/lib', '/usr/local/intel/mkl/lib', '/usr/local/intel/lib/intel'+sys_arch, '/usr/local/intel/mkl/lib/intel'+sys_arch,
		'/usr/lib64/atlas', '/usr/lib/atlas', '/usr/local/lib64/atlas', '/usr/local/lib/atlas',
		'/usr/lib64/gsl', '/usr/lib/gsl', '/usr/local/lib64/gsl', '/usr/local/lib/gsl']
		candidate_paths += ['/usr/lib/x86' + "_" if sys_arch=='64' else '' +sys_arch+'-linux-gnu',
		'/usr/lib', '/usr/local/lib', '/lib64', '/lib', '/usr/lib64', '/usr/local/lib64', '/opt/local/lib64', '/opt/local/lib']

		mkl_include_paths += ['/opt/intel/include', '/opt/intel/mkl/include', '/opt/intel/mkl/include/intel'+sys_arch,
			'/usr/local/intel/include', '/usr/local/intel/mkl/include', '/usr/local/intel/include/intel'+sys_arch, '/usr/local/intel/mkl/include/intel'+sys_arch]
		## openblas is the only good citizen that plays by the rules
		atlas_include_paths += ['/usr/lib/atlas', '/usr/lib64/atlas', '/usr/lib/atlas/include', '/usr/lib64/atlas/include']
		gsl_include_paths += ['/usr/include/gsl', '/usr/local/include/gsl', '/opt/local/include/gsl']
		system_include_paths += ['/usr/include/x86' + "_" if sys_arch=='64' else '' +sys_arch+'-linux-gnu', '/usr/include', '/usr/local/include', '/opt/local/include']
	else:
		## Try to lookup default MKL installation
		intel_folder = os.path.join(os.environ["ProgramFiles" + "(x86)" if sys_arch != '32' else ''], "IntelSWTools")
		if os.path.exists(intel_folder):
			curr_path = intel_folder
			fold_c_n_l = [fld for fld in os.listdir(curr_path) if bool(re.search("compilers_and_libraries", fld))]
			if len(fold_c_n_l) > 0:
				for f_c_n_l in fold_c_n_l:

					curr_path = os.path.join(intel_folder, f_c_n_l)
					if os.path.exists(os.path.join(curr_path, "windows")):
						curr_path = os.path.join(curr_path, "windows")
						
						### lookup redist
						if os.path.exists(os.path.join(curr_path, "redist", "intel"+sys_arch, "mkl")):
							candidate_paths.append(os.path.join(curr_path, "redist", "intel"+sys_arch, "mkl"))

						## lookup mkl
						if os.path.exists(os.path.join(curr_path, "mkl", "lib", "intel"+sys_arch)):
							candidate_paths.append(os.path.join(curr_path, "mkl", "lib", "intel"+sys_arch))

						## lookup include
						if os.path.exists(os.path.join(curr_path, "mkl", "include")):
							mkl_include_paths.append(os.path.join(curr_path, "mkl", "include"))

		## Try to add C:\Windows\System32
		try:
			## https://stackoverflow.com/questions/41630224/python-does-not-find-system32
			is_wow64 = (platform.architecture()[0] == '32bit' and 'ProgramFiles(x86)' in os.environ)
			candidate_paths += os.path.join(os.environ['SystemRoot'], 'SysNative' if is_wow64 else 'System32')
		except:
			pass

		## Try to add visual studio headers
		try:
			## https://stackoverflow.com/questions/335408/where-does-visual-studio-look-for-c-header-files/335426#335426
			vs_path = os.path.join(os.environ["ProgramFiles" + "(x86)" if sys_arch != '32' else ''], "Microsoft Visual Studio")
			if os.path.exists(os.path.join(vs_path)):
				for yr in os.listdir(vs_path):
					vr_path = os.path.join(vs_path, yr, "Community", "VC", "Tools", "MSVC")
					if os.path.exists(vr_path):
						system_include_paths += [os.path.join(vr_path, v, "include") for v in os.listdir(vr_path) if os.path.exists(os.path.join(vr_path, v, "include"))]
			winkits_folder = os.path.join(os.environ["ProgramFiles" + "(x86)" if sys_arch != '32' else ''], "Windows Kits")
			if os.path.exists(winkits_folder):
				for v in winkits_folder:
					i_fold = os.path.join(winkits_folder, v, "Include")
					if os.path.exists(i_fold):
						for vr in os.listdir(i_fold):
							h_fold = os.path.join(i_fold, vr, "ucrt")
							if os.path.exists(h_fold):
								system_include_paths += h_fold
		except:
			pass

	## Discard duplicated paths, but keep the order
	search_paths = _deduplicate_paths(candidate_paths)

	flags_found = list()
	blas_file = None
	blas_path = None

	### Start looking for each library in selected paths
	def search_blas_lib(search_paths, blas_names):
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
		blas_path, blas_file = search_blas_lib(search_paths, openblas_file_names1)
		if blas_file is not None:
			flags_found.append("HAS_OPENBLAS")
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
		blas_path, blas_file = search_blas_lib(search_paths, openblas_file_names2)
		if blas_file is not None:
			flags_found.append("HAS_OPENBLAS")
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
		blas_path, blas_file = search_blas_lib(search_paths, [pref + "blas" + e for e in ext])
		if blas_file is not None:
			if platform[:3] != "win":
				found_syms = _find_symbols(blas_path, blas_file)
				if found_syms is not None:
					flags_found += found_syms[2]

	### Try regex matching
	def check_is_blas(pt, fname):
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


		if ask_user:
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
				candidate_files_dyna += [f for f in os.listdir(pt) if bool(re.search("blas", f)) and bool(re.search(r"\." + ext[0][1:] + r"$", f))]
				candidate_files_stat += [f for f in os.listdir(pt) if bool(re.search("blas", f)) and bool(re.search(r"\." + ext[1][1:] + r"$", f))]
				nfound_dyna = len(candidate_files_dyna) - nfound_dyna
				nfound_stat = len(candidate_files_stat) - nfound_stat
				candidate_paths_dyna += [pt] * nfound_dyna
				candidate_paths_stat += [pt] * nfound_stat

		candidate_files = candidate_files_dyna + candidate_files_stat
		candidate_paths = candidate_paths_dyna + candidate_paths_stat
		for f in range(len(candidate_files)):
			is_blas, temp = check_is_blas(candidate_paths[f], candidate_files[f])
			if is_blas:
				blas_file = candidate_files[f]
				blas_path = candidate_paths[f]
				flags_found += temp
				break

	err_msg = "Could not locate MKL, OpenBLAS, ATLAS or GSL libraries - you'll need to manually modify setup.py to add BLAS path."
	if blas_file is None:
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

			warnings.warn("No BLAS library found - taking SciPy's linalg.cython_blas file as library.")

		except:
			raise ValueError(err_msg)

	### Now lookup the include path
	def get_inc_paths(blas_path, include_paths, system_include_paths):
		include_paths += [blas_path, re.sub("/lib.?$", "/include", blas_path),
			re.sub(r"^(.*)/lib/(.*)$", r"\1/include/\2", blas_path),
			re.sub(r"^(.*)Library.*$", r"\1include", blas_path)]
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

	if 'HAS_MKL' in flags_found:
		search_paths = get_inc_paths(blas_path, mkl_include_paths, system_include_paths)
		incl_path, incl_file = search_incl_kwds(search_paths, incl_mkl_name + incl_generic_name, ["MKL"])

	elif 'HAS_OPENBLAS' in flags_found:
		search_paths = get_inc_paths(blas_path, openblas_include_paths, system_include_paths)
		incl_path, incl_file = search_incl_kwds(search_paths, incl_openblas_name + incl_generic_name, ["openblas"])

	elif 'HAS_ATLAS' in flags_found:
		search_paths = get_inc_paths(blas_path, atlas_include_paths, system_include_paths)
		incl_path, incl_file = search_incl_kwds(search_paths, incl_atlas_name + incl_generic_name, ["atlas", "ATLAS"])
		
	elif 'HAS_GSL' in flags_found:
		search_paths = get_inc_paths(blas_path, gsl_include_paths, system_include_paths)
		incl_path, incl_file = search_incl_kwds(search_paths, incl_gsl_name + incl_generic_name, ["GSL_CBLAS"])
		
	else:
		flags_found.append('UNKNWON_BLAS')
		search_paths = get_inc_paths(blas_path, [], system_include_paths)
		all_kwds = ["MKL", "openblas", "atlas", "ATLAS"]
		if 'NO_CBLAS' not in flags_found:
			all_kwds += ["GSL_CBLAS", "cblas", "CBLAS"]
		all_kwds += ["ddot", "DDOT"]
		incl_path, incl_file = search_incl_kwds(search_paths, incl_generic_name, all_kwds)

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

def _try_add_from_command(str_expr, candidate_paths):
	try:
		exec("candidate_paths += " + str_expr)
	except:
		pass

def _find_symbols(pt, fname):
	try:
		from elftools.elf.elffile import ELFFile
		with open(os.path.join(pt, fname), 'rb') as f:
			elffile = ELFFile(f)
			symtab = elffile.get_section_by_name('.symtab')
			if symtab.get_symbol_by_name('openblas_get_config') is not None:
				return True, ["HAS_OPENBLAS", "HAS_UNDERSCORES"]
			if symtab.get_symbol_by_name('mkl_dcsrgemv') is not None:
				return True, ["HAS_MKL"]
			if symtab.get_symbol_by_name('ddot_') is not None:
				found_syms = ["HAS_UNDERSCORES"]
				if symtab.get_symbol_by_name('cblas_ddot') is None:
					found_syms += ['NO_CBLAS']
				return True, found_syms
			if symtab.get_symbol_by_name('cblas_ddot') is not None:
				return True, []
			if (symtab.get_symbol_by_name('ddot') is not None) or (symtab.get_symbol_by_name('DDOT') is not None):
				return True, ["NO_CBLAS"]
		return True, None

	except:
		try:
			import subprocess
			symbols = subprocess.check_output(['readelf', '-s', os.path.join(pt, fname)])
			symbols = str(symbols).split()
			has_cblas = False
			has_underscores = False
			has_ddot = False
			for s in symbols:
				if bool(re.search("openblas", s)):
					return True, ["HAS_OPENBLAS", "HAS_UNDERSCORES"]
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
		except:
			return False, None
