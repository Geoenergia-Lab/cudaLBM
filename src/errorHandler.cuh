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
     * @brief Utility class for handling CUDA and general runtime errors.
     *
     * Provides static methods to check error codes and terminate with a
     * formatted error message. Constructors can be used for immediate checking.
     **/
    class errorHandler
    {
    public:
        /**
         * @brief Construct and check a CUDA error.
         * @param[in] err CUDA error code to check.
         *
         * If the error is not cudaSuccess, the program terminates with a
         * detailed error report.
         **/
        [[nodiscard]] errorHandler(const cudaError_t err) noexcept
        {
            check(err);
        }

        /**
         * @brief Construct and check a general error with a custom message.
         * @param[in] err Integer error code.
         * @param[in] errorString Human-readable error description.
         *
         * If the error is non-zero, the program terminates with the provided
         * error code and message.
         **/
        [[nodiscard]] errorHandler(const int err, const name_t &errorString) noexcept
        {
            check(err, errorString);
        }

        ~errorHandler() noexcept = default;

        /**
         * @brief Check a CUDA error and terminate if not successful.
         * @param[in] err CUDA error code.
         *
         * If err != cudaSuccess, prints an error report and calls std::exit().
         * This version is not marked inline, suitable for calls outside
         * performance-critical loops.
         **/
        static void check(const cudaError_t err, const std::source_location &loc = std::source_location::current()) noexcept
        {
            checkImpl(err, loc);
        }
        static inline void checkLast(const std::source_location &loc = std::source_location::current()) noexcept
        {
            check(cudaGetLastError(), loc);
        }

        /**
         * @brief Inline version of check(cudaError_t).
         * @param[in] err CUDA error code.
         *
         * Identical to check() but gives the compiler an inline hint.
         * Use this in tight loops where function call overhead matters.
         **/
        static inline void checkInline(const cudaError_t err, const std::source_location &loc = std::source_location::current()) noexcept
        {
            checkImpl(err, loc);
        }
        static inline void checkLastInline(const std::source_location &loc = std::source_location::current()) noexcept
        {
            checkInline(cudaGetLastError(), loc);
        }

        /**
         * @brief Check a general error with custom message and terminate if non-zero.
         * @param[in] err Integer error code.
         * @param[in] errorString Descriptive error message.
         *
         * If err != 0, prints a report including the given string and calls std::exit().
         **/
        static void check(const int err, const name_t &errorString, const std::source_location &loc = std::source_location::current()) noexcept
        {
            checkImpl(err, errorString, loc);
        }

    private:
        /**
         * @brief Implementation of check(cudaError_t)
         **/
        static inline void checkImpl(const cudaError_t err, const std::source_location &loc) noexcept
        {
            if (err != cudaSuccess)
            {
                exit(err, loc);
            }
        }

        /**
         * @brief Implementation of check(int, errorString)
         **/
        static inline void checkImpl(const int err, const name_t &errorString, const std::source_location &loc) noexcept
        {
            if (err != 0)
            {
                exit(err, errorString, loc);
            }
        }

        /**
         * @brief Terminate program with a formatted error report (integer code).
         * @param[in] err Integer error code.
         * @param[in] errorString Human-readable message.
         *
         * Outputs error details to stderr and calls std::exit(err).
         **/
        static inline void exit(const int err, const name_t &errorString, const std::source_location &loc) noexcept
        {
            std::cerr << std::endl;
            std::cerr << "runTimeError:" << std::endl;
            std::cerr << "{" << std::endl;
            std::cerr << "    fileName: " << loc.file_name() << std::endl;
            std::cerr << "    line: " << loc.line() << std::endl;
            std::cerr << "    functionName: " << loc.function_name() << std::endl;
            std::cerr << "    errorCode: " << err << std::endl;
            std::cerr << "    errorMessage: " << errorString << std::endl;
            std::cerr << "};" << std::endl;
            std::cerr << std::endl;
            std::exit(err);
        }

        /**
         * @brief Terminate program with a formatted CUDA error report.
         * @param[in] err CUDA error code.
         *
         * Converts the CUDA error to an integer and a string, then calls exit(int,string).
         **/
        static inline void exit(const cudaError_t err, const std::source_location &loc) noexcept
        {
            exit(static_cast<int>(err), cudaGetErrorString(err), loc);
        }
    };

    /**
     * @brief Type used for MPI errors
     * @note Has to be enumerated because there are only so many MPI error codes
     **/
    namespace errorCode
    {
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
}

#endif