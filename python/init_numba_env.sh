# For colour output; also supports jupyter_nb.
export NUMBA_COLOR_SCHEME=dark_bg

# Level 2 is a bit faster for compilation itself.
# Level 3 is faster??? for running CUDA kernels.
export NUMBA_OPT=2

# Disable jit entirely for debugging purposes.
#export NUMBA_DISABLE_JIT=1

# Options for caching
#export NUMBA_DEBUG_CACHE=1
