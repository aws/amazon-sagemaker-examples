# tkConfig.sh --
#
# This shell script (for sh) is generated automatically by Tk's
# configure script.  It will create shell variables for most of
# the configuration options discovered by the configure script.
# This script is intended to be included by the configure scripts
# for Tk extensions so that they don't have to figure this all
# out for themselves.  This file does not duplicate information
# already provided by tclConfig.sh, so you may need to use that
# file in addition to this one.
#
# The information in this file is specific to a single platform.

# Tk's version number.
TK_VERSION='8.6'
TK_MAJOR_VERSION='8'
TK_MINOR_VERSION='6'
TK_PATCH_LEVEL='.13'

# -D flags for use with the C compiler.
TK_DEFS='-DPACKAGE_NAME=\"tk\" -DPACKAGE_TARNAME=\"tk\" -DPACKAGE_VERSION=\"8.6\" -DPACKAGE_STRING=\"tk\ 8.6\" -DPACKAGE_BUGREPORT=\"\" -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DUSE_THREAD_ALLOC=1 -D_REENTRANT=1 -D_THREAD_SAFE=1 -DHAVE_PTHREAD_ATTR_SETSTACKSIZE=1 -DHAVE_PTHREAD_ATFORK=1 -DTCL_THREADS=1 -DMODULE_SCOPE=extern\ __attribute__\(\(__visibility__\(\"hidden\"\)\)\) -DHAVE_HIDDEN=1 -DHAVE_CAST_TO_UNION=1 -DTCL_SHLIB_EXT=\".so\" -DNDEBUG=1 -DTCL_CFG_OPTIMIZED=1 -D_LARGEFILE64_SOURCE=1 -DTCL_WIDE_INT_IS_LONG=1 -DHAVE_SYS_TIME_H=1 -DTIME_WITH_SYS_TIME=1 -DHAVE_INTPTR_T=1 -DHAVE_UINTPTR_T=1 -DHAVE_PW_GECOS=1 -DHAVE_XKBKEYCODETOKEYSYM=1 -DXKEYCODETOKEYSYM_IS_DEPRECATED=1 '

# Flag, 1: we built a shared lib, 0 we didn't
TK_SHARED_BUILD=1


# TK_DBGX used to be used to distinguish debug vs. non-debug builds.
# This was a righteous pain so the core doesn't do that any more.
TK_DBGX=

# The name of the Tk library (may be either a .a file or a shared library):
TK_LIB_FILE='libtk8.6.so'

# Additional libraries to use when linking Tk.
TK_LIBS='-lX11   -lpthread -ldl -lz  -lpthread -lm'

# Top-level directory in which Tk's platform-independent files are
# installed.
TK_PREFIX='/workspaces/amazon-sagemaker-examples/venv'

# Top-level directory in which Tk's platform-specific files (e.g.
# executables) are installed.
TK_EXEC_PREFIX='/workspaces/amazon-sagemaker-examples/venv'

# -I switch(es) to use to make all of the X11 include files accessible:
TK_XINCLUDES=''

# Linker switch(es) to use to link with the X11 library archive.
TK_XLIBSW='-lX11'

# -l flag to pass to the linker to pick up the Tk library
TK_LIB_FLAG='-ltk8.6'

# String to pass to linker to pick up the Tk library from its
# build directory.
TK_BUILD_LIB_SPEC='-L/workspaces/amazon-sagemaker-examples/venv/lib -ltk8.6'

# String to pass to linker to pick up the Tk library from its
# installed directory.
TK_LIB_SPEC='-L/workspaces/amazon-sagemaker-examples/venv/lib -ltk8.6'

# String to pass to the compiler so that an extension can
# find installed Tk headers.
TK_INCLUDE_SPEC='-I/workspaces/amazon-sagemaker-examples/venv/include'

# Location of the top-level source directory from which Tk was built.
# This is the directory that contains a README file as well as
# subdirectories such as generic, unix, etc.  If Tk was compiled in a
# different place than the directory containing the source files, this
# points to the location of the sources, not the location where Tk was
# compiled.
TK_SRC_DIR='/workspaces/amazon-sagemaker-examples/venv/include'

# Needed if you want to make a 'fat' shared library library
# containing tk objects or link a different wish.
TK_CC_SEARCH_FLAGS='"-Wl,-rpath,${LIB_RUNTIME_DIR}"'
TK_LD_SEARCH_FLAGS='"-Wl,-rpath,${LIB_RUNTIME_DIR}"'

# The name of the Tk stub library (.a):
TK_STUB_LIB_FILE='libtkstub8.6.a'

# -l flag to pass to the linker to pick up the Tk stub library
TK_STUB_LIB_FLAG='-ltkstub8.6'

# String to pass to linker to pick up the Tk stub library from its
# build directory.
TK_BUILD_STUB_LIB_SPEC='-L/workspaces/amazon-sagemaker-examples/venv/lib -ltkstub8.6'

# String to pass to linker to pick up the Tk stub library from its
# installed directory.
TK_STUB_LIB_SPEC='-L/workspaces/amazon-sagemaker-examples/venv/lib -ltkstub8.6'

# Path to the Tk stub library in the build directory.
TK_BUILD_STUB_LIB_PATH='/workspaces/amazon-sagemaker-examples/venv/lib/libtkstub8.6.a'

# Path to the Tk stub library in the install directory.
TK_STUB_LIB_PATH='/workspaces/amazon-sagemaker-examples/venv/lib/libtkstub8.6.a'
