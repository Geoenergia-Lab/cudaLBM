/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Functions used to handle errors

Namespace
    LBM

SourceFiles
    errorHandler.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_ERRORHANDLER_CUH
#define __MBLBM_ERRORHANDLER_CUH

namespace LBM
{
    /**
     * @brief Checks for CUDA runtime errors and handles them by reporting and terminating
     *
     * @param err CUDA error code to check (cudaError_t enum value)
     * @param loc Source location information (automatically captured at call site if not specified)
     *
     * @note This function is noexcept and will terminate the program if an error is detected
     * @warning This function immediately terminates the program when a CUDA error is encountered
     *
     * If the provided error code is not equal to cudaSuccess, this function will:
     * 1. Print a detailed error message to stderr including:
     *    - Source file name where the error was detected
     *    - Line number in the source file
     *    - Function name where the error was detected
     *    - Numeric error code
     *    - Human-readable error description from cudaGetErrorString()
     * 2. Terminate the program with the error code as exit status
     *
     * Example output:
     * @code
     * CUDA error at example.cu(123):kernelLauncher(): [77] cudaErrorIllegalAddress.
     * @endcode
     *
     * @par Typical usage pattern:
     * @code
     * cudaError_t result = cudaMalloc(&devicePtr, size);
     * checkCudaErrors(result);  // Will terminate if allocation failed
     * @endcode
     *
     * @see cudaError_t
     * @see cudaGetErrorString()
     * @see std::source_location
     **/
    void checkCudaErrors(
        const cudaError_t err,
        const std::source_location &loc = std::source_location::current()) noexcept
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error at " << loc.file_name() << "(" << loc.line() << "):" << loc.function_name() << ": [" << static_cast<int>(err) << "] " << cudaGetErrorString(err) << "." << std::endl;
            std::exit(static_cast<int>(err));
        }
    }

    /**
     * @overload Inline version of checkCudaErrors
     **/
    inline void checkCudaErrorsInline(
        const cudaError_t err,
        const std::source_location &loc = std::source_location::current()) noexcept
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error at " << loc.file_name() << "(" << loc.line() << "):" << loc.function_name() << ": [" << static_cast<int>(err) << "] " << cudaGetErrorString(err) << "." << std::endl;
            std::exit(static_cast<int>(err));
        }
    }

    /**
     * @brief Handles runtime errors by printing detailed information and terminating the program
     *
     * @param err Error code to report and use as exit status
     * @param errorString Descriptive message explaining the error
     * @param loc Source location information (automatically captured at call site if not specified)
     *
     * @note This function is noexcept and will always terminate the program
     * @warning This function immediately terminates the program after printing the error message
     *
     * The error message format includes:
     * - File name where the error occurred
     * - Line number in the source file
     * - Function name where the error occurred
     * - Numeric error code
     * - Descriptive error string
     *
     * Example output:
     * @code
     * Run time error at example.cpp(42):myFunction(): [404] Resource not found
     * @endcode
     *
     * @par Example usage:
     * @code
     * if (fileNotFound) {
     *     errorHandler(404, "Resource not found");
     * }
     * @endcode
     **/
    void errorHandler(
        const int err,
        const std::string &errorString,
        const std::source_location &loc = std::source_location::current()) noexcept
    {
        std::cerr << "runTimeError:" << std::endl;
        std::cerr << "{" << std::endl;
        std::cerr << "    fileName: " << loc.file_name() << std::endl;
        std::cerr << "    line: " << loc.line() << std::endl;
        std::cerr << "    functionName: " << loc.function_name() << std::endl;
        std::cerr << "    errorCode: " << err << std::endl;
        std::cerr << "    errorMessage: " << errorString << std::endl;
        std::cerr << "};" << std::endl;
        // std::cerr
        //     << "Run time error error at "
        //     << loc.file_name()
        //     << "("
        //     << loc.line()
        //     << "):"
        //     << loc.function_name()
        //     << ": ["
        //     << err
        //     << "] "
        //     << errorString
        //     << std::endl;
        std::exit(err);
    }

    /**
     * @brief Type used for MPI errors
     * @note Has to be enumerated because there are only so many MPI error codes
     **/
    typedef enum Enum : int
    {
        SUCCESS = 0,                    // Successful return code.
        ERR_BUFFER = 1,                 // Invalid buffer pointer.
        ERR_COUNT = 2,                  // Invalid count argument.
        ERR_TYPE = 3,                   // Invalid datatype argument.
        ERR_TAG = 4,                    // Invalid tag argument.
        ERR_COMM = 5,                   // Invalid communicator.
        ERR_RANK = 6,                   // Invalid rank.
        ERR_REQUEST = 7,                // Invalid MPI_Request handle.
        ERR_ROOT = 8,                   // Invalid root.
        ERR_GROUP = 9,                  // Null group passed to function.
        ERR_OP = 10,                    // Invalid operation.
        ERR_TOPOLOGY = 11,              // Invalid topology.
        ERR_DIMS = 12,                  // Illegal dimension argument.
        ERR_ARG = 13,                   // Invalid argument.
        ERR_UNKNOWN = 14,               // Unknown error.
        ERR_TRUNCATE = 15,              // Message truncated on receive.
        ERR_OTHER = 16,                 // Other error; use Error_string.
        ERR_INTERN = 17,                // Internal error code.
        ERR_IN_STATUS = 18,             // Look in status for error value.
        ERR_PENDING = 19,               // Pending request.
        ERR_ACCESS = 20,                // Permission denied.
        ERR_AMODE = 21,                 // Unsupported amode passed to open.
        ERR_ASSERT = 22,                // Invalid assert.
        ERR_BAD_FILE = 23,              // Invalid file name (for example, path name too long).
        ERR_BASE = 24,                  // Invalid base.
        ERR_CONVERSION = 25,            // An error occurred in a user-supplied data-conversion function.
        ERR_DISP = 26,                  // Invalid displacement.
        ERR_DUP_DATAREP = 27,           // Conversion functions could not be registered because a data representation identifier that was already defined was passed to Register_datarep.
        ERR_FILE_EXISTS = 28,           // File exists.
        ERR_FILE_IN_USE = 29,           // File operation could not be completed, as the file is currently open by some process.
        ERR_FILE = 30,                  // Invalid file handle.
        ERR_INFO_KEY = 31,              // Illegal info key.
        ERR_INFO_NOKEY = 32,            // No such key.
        ERR_INFO_VALUE = 33,            // Illegal info value.
        ERR_INFO = 34,                  // Invalid info object.
        ERR_IO = 35,                    // I/O error.
        ERR_KEYVAL = 36,                // Illegal key value.
        ERR_LOCKTYPE = 37,              // Invalid locktype.
        ERR_NAME = 38,                  // Name not found.
        ERR_NO_MEM = 39,                // Memory exhausted.
        ERR_NOT_SAME = 40,              // Collective argument not identical on all processes, or collective routines called in a different order by different processes.
        ERR_NO_SPACE = 41,              // Not enough space.
        ERR_NO_SUCH_FILE = 42,          // File (or directory) does not exist.
        ERR_PORT = 43,                  // Invalid port.
        ERR_PROC_ABORTED = 74,          // Operation failed because a remote peer has aborted.
        ERR_QUOTA = 44,                 // Quota exceeded.
        ERR_READ_ONLY = 45,             // Read-only file system.
        ERR_RMA_CONFLICT = 46,          // Conflicting accesses to window.
        ERR_RMA_SYNC = 47,              // Erroneous RMA synchronization.
        ERR_SERVICE = 48,               // Invalid publish/unpublish.
        ERR_SIZE = 49,                  // Invalid size.
        ERR_SPAWN = 50,                 // Error spawning.
        ERR_UNSUPPORTED_DATAREP = 51,   // Unsupported datarep passed to MPI_File_set_view.
        ERR_UNSUPPORTED_OPERATION = 52, // Unsupported operation, such as seeking on a file that supports only sequential access.
        ERR_WIN = 53,                   // Invalid window.
        T_ERR_MEMORY = 54,              // Out of memory.
        T_ERR_NOT_INITIALIZED = 55,     // Interface not initialized.
        T_ERR_CANNOT_INIT = 56,         // Interface not in the state to be initialized.
        T_ERR_INVALID_INDEX = 57,       // The enumeration index is invalid.
        T_ERR_INVALID_ITEM = 58,        // The item index queried is out of range.
        T_ERR_INVALID_HANDLE = 59,      // The handle is invalid.
        T_ERR_OUT_OF_HANDLES = 60,      // No more handles available.
        T_ERR_OUT_OF_SESSIONS = 61,     // No more sessions available.
        T_ERR_INVALID_SESSION = 62,     // Session argument is not a valid session.
        T_ERR_CVAR_SET_NOT_NOW = 63,    // Variable cannot be set at this moment.
        T_ERR_CVAR_SET_NEVER = 64,      // Variable cannot be set until end of execution.
        T_ERR_PVAR_NO_STARTSTOP = 65,   // Variable cannot be started or stopped.
        T_ERR_PVAR_NO_WRITE = 66,       // Variable cannot be written or reset.
        T_ERR_PVAR_NO_ATOMIC = 67,      // Variable cannot be read and written atomically.
        ERR_RMA_RANGE = 68,             // Target memory is not part of the window (in the case of a window created with MPI_Win_create_dynamic, target memory is not attached.
        ERR_RMA_ATTACH = 69,            // Memory cannot be attached (e.g., because of resource exhaustion).
        ERR_RMA_FLAVOR = 70,            // Passed window has the wrong flavor for the called function.
        ERR_RMA_SHARED = 71,            // Memory cannot be shared (e.g., some process in the group of the specified communicator cannot expose shared memory).
        T_ERR_INVALID = 72,             // Invalid use of the interface or bad parameter values(s).
        T_ERR_INVALID_NAME = 73,        // The variable or category name is invalid.
        ERR_SESSION = 78,               // Invalid session
        ERR_LASTCODE = 93               // Last error code.
    } mpiError_t;
}

#endif