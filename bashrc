#*---------------------------------------------------------------------------*#
#                                                                             #
# cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          #
# Developed at UDESC - State University of Santa Catarina                     #
# Website: https://www.udesc.br                                               #
# Github: https://github.com/geoenergiaUDESC/cudaLBM                          #
#                                                                             #
#*---------------------------------------------------------------------------*#


#*---------------------------------------------------------------------------*#

# USER-DEFINED ENVIRONMENT VARIABLES

# Define the architecture detection mode
# Options: "Automatic", "Manual"
export CUDALBM_ARCHITECTURE_DETECTION="Automatic"
#export CUDALBM_ARCHITECTURE_DETECTION="Manual"

# Define the architecture type
# If CUDALBM_ARCHITECTURE_DETECTION is set to "Manual", specify the
# architecture type here. Otherwise, no need to modify this variable.
export CUDALBM_ARCHITECTURE_VERSION="89"

#*---------------------------------------------------------------------------*#

#*---------------------------------------------------------------------------*#

# DO NOT MODIFY THIS SECTION UNLESS YOU ABSOLUTELY KNOW WHAT YOU ARE DOING

# Define the third party install directory
export CUDALBM_PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Export the build directory
export CUDALBM_BUILD_DIR="$CUDALBM_PROJECT_DIR/build"
export CUDALBM_BIN_DIR="$CUDALBM_BUILD_DIR/bin"
export CUDALBM_INCLUDE_DIR="$CUDALBM_BUILD_DIR/include"

# Determine the active CUDA installation path
if command -v nvcc > /dev/null 2>&1; then
    # Resolve the path to nvcc and extract the CUDA directory
    NVCC_PATH=$(command -v nvcc)
    # Resolve symlinks to get the actual path
    RESOLVED_NVCC_PATH=$(readlink -f "$NVCC_PATH" 2>/dev/null || echo "$NVCC_PATH")
    # Extract the base directory (remove trailing '/bin/nvcc')
    export CUDALBM_CUDA_DIR=$(dirname "$(dirname "$RESOLVED_NVCC_PATH")")
else
    echo "Error: nvcc not found. Ensure CUDA is installed and in your PATH." >&2
    return 1
fi

export PATH=$CUDALBM_CUDA_DIR/bin:$PATH
export LIBRARY_PATH=$CUDALBM_CUDA_DIR/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDALBM_CUDA_DIR/lib64:$LD_LIBRARY_PATH

# Export the path to the installed UCX
export CUDALBM_UCX_DIR=$CUDALBM_BUILD_DIR/ucx
export PATH=$CUDALBM_UCX_DIR/bin:$PATH
export LIBRARY_PATH=$CUDALBM_UCX_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDALBM_UCX_DIR/lib:$LD_LIBRARY_PATH

# Export the path to the installed OpenMPI
export CUDALBM_MPI_DIR=$CUDALBM_BUILD_DIR/OpenMPI
export PATH=$CUDALBM_MPI_DIR/bin:$PATH
export LIBRARY_PATH=$CUDALBM_MPI_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDALBM_MPI_DIR/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CUDALBM_MPI_DIR/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDALBM_MPI_DIR/include:$CPLUS_INCLUDE_PATH

# Export the path to the compiled executable
export PATH=$CUDALBM_PROJECT_DIR/build/bin:$PATH

#*---------------------------------------------------------------------------*#