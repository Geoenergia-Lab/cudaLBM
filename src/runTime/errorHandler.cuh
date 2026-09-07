/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
         * @brief Check a CUDA error and terminate if not successful.
         * @tparam T The type of the error code to check
         * @param[in] err CUDA error code.
         * @param[in] loc Location of the error in the source code.
         * If err != cudaSuccess, prints an error report and calls std::exit().
         * This version is not marked inline, suitable for calls outside
         * performance-critical loops.
         **/
        template <typename T>
        __host__ static void handle(const T err, const std::source_location &loc = std::source_location::current()) noexcept
        {
            handle_impl(err, loc);
        }

        /**
         * @brief Inline version of handle
         * @tparam T The type of the error code to check
         * @param[in] err CUDA error code.
         * @param[in] loc Location of the error in the source code.
         * Identical to handle() but gives the compiler an inline hint.
         * Use this in tight loops where function call overhead matters.
         **/
        template <typename T>
        __host__ static inline void handleInline(const T err, const std::source_location &loc = std::source_location::current()) noexcept
        {
            handle_impl(err, loc);
        }

    private:
        /**
         * @brief Implementation of handle
         * @tparam T The type of the error code to check
         * @param[in] err Integer error code.
         * @param[in] loc Location of the error in the source code.
         **/
        template <typename T>
        __host__ static inline void handle_impl(const T err, const std::source_location &loc) noexcept
        {
            if constexpr (std::is_same_v<T, cudaError_t>)
            {
                if (err != cudaSuccess)
                {
                    update_codes_and_print(err, loc);
                }
            }

            if constexpr (std::is_same_v<T, runTime::error::code>)
            {
                if (err != runTime::error::NO_ERROR)
                {
                    update_codes_and_print(err, loc);
                }
            }
        }

        /**
         * @brief Terminate program with a formatted error report (integer code).
         * @tparam T The type of the error code to check
         * @param[in] err Integer error code.
         * @param[in] loc Location of the error in the source code.
         **/
        template <typename T>
        __host__ static inline void update_codes_and_print(const T err, const std::source_location &loc) noexcept
        {
            runTime::update_codes(err);
            IO::printError(
                "runTimeError",
                "fileName", base_name(loc),
                "line", loc.line(),
                "functionName", loc.function_name(),
                "errorCode", err,
                "errorMessage", get_error_string(err));
        }

        template <typename T>
        __host__ [[nodiscard]] static inline constexpr const char *get_error_string(const T code) noexcept
        {
            if constexpr (std::is_same_v<T, cudaError_t>)
            {
                return cudaGetErrorString(code);
            }

            if constexpr (std::is_same_v<T, runTime::error::code>)
            {
                return runTime::error::messages()[code];
            }
        }

        __host__ [[nodiscard]] static inline constexpr const char *base_name(const std::source_location &loc) noexcept
        {
            const char *path = loc.file_name();
            const char *base = path;
            for (const char *p = path; *p != '\0'; ++p)
            {
                if (*p == '/' || *p == '\\')
                {
                    base = p + 1;
                }
            }
            return base;
        }
    };
}

#endif