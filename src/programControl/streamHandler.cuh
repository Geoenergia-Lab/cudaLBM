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
    A class handling CUDA streams

Namespace
    LBM

SourceFiles
    streamHandler.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_STREAMHANDLER_CUH
#define __MBLBM_STREAMHANDLER_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @brief Helper function to get the correct stream index for a given device and direction
         * @param[in] idxDev The device index
         * @param[in] idxStr The stream index for the direction
         **/
        __host__ [[nodiscard]] static inline constexpr host::label_t idxStream(const host::label_t idxDev, const host::label_t idxStr) noexcept
        {
            return (static_cast<host::label_t>(3) * idxDev) + idxStr;
        }

        /**
         * @brief Helper function to get the correct stream index for a given device and direction
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @param[in] idxDev The device index
         **/
        template <const int coeff>
        __host__ [[nodiscard]] static inline constexpr host::label_t idxStream(const host::label_t idxDev) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            if constexpr (coeff == -1)
            {
                return idxStream(idxDev, 0); // East stream
            }

            if constexpr (coeff == +1)
            {
                return idxStream(idxDev, 2); // West stream
            }
        }
    }

    /**
     * @class streamHandler
     * @brief Manages a collection of CUDA streams for asynchronous operations
     * @tparam N Number of CUDA streams to manage (must be positive)
     *
     * This class handles the creation, synchronization, and destruction of
     * multiple CUDA streams. It provides thread-safe access to streams and
     * ensures proper cleanup during destruction.
     **/
    class streamHandler
    {
    public:
        /**
         * @brief Default constructor
         * @param[in] deviceIndices Ordinals of the devices for which to create the streams
         **/
        __host__ [[nodiscard]] streamHandler(const std::vector<deviceIndex_t> &deviceIndices) noexcept
            : streams_(createCudaStreams(deviceIndices)) {}

        /**
         * @brief Destructor
         *
         * Automatically synchronizes and destroys all CUDA streams upon
         * object destruction. Ensures proper cleanup of GPU resources.
         **/
        __host__ ~streamHandler() noexcept
        {
            for (const cudaStream_t &stream : streams_)
            {
                errorHandler::handle(cudaStreamSynchronize(stream));
                errorHandler::handle(cudaStreamDestroy(stream));
            }
        }

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] streamHandler(const streamHandler &) = delete;
        __host__ [[nodiscard]] streamHandler &operator=(const streamHandler &) = delete;

        /**
         * @brief Synchronizes a specific CUDA stream
         * @param[in] i Integral constant representing the stream index
         **/
        __host__ inline void synchronize(const host::label_t i) const noexcept
        {
            errorHandler::handleInline(cudaStreamSynchronize(streams_[i]));
        }

        /**
         * @brief Stream access operator
         * @param[in] i Integral constant representing the stream index
         * @return Reference to the requested CUDA stream
         * @warning No bounds checking performed at runtime
         **/
        __host__ const cudaStream_t &operator[](const host::label_t i) const noexcept
        {
            return streams_[i];
        }

        /**
         * @brief Returns all managed CUDA streams
         * @return Const reference to std::array containing all CUDA streams
         **/
        __host__ [[nodiscard]] inline const std::vector<cudaStream_t> &streams() const noexcept
        {
            return streams_;
        }

    private:
        /**
         * @brief Creates and initializes CUDA streams
         * @return std::array of N initialized CUDA streams
         * @param[in] deviceIndices Ordinals of the devices for which to create the streams
         * Private helper function that handles actual stream creation
         * with proper error checking and device synchronization.
         **/
        __host__ [[nodiscard]] static const std::vector<cudaStream_t> createCudaStreams(const std::vector<deviceIndex_t> &deviceIndices)
        {
            std::vector<cudaStream_t> streams(deviceIndices.size() * 3);

            for (host::label_t deviceIdx = 0; deviceIdx < deviceIndices.size(); deviceIdx++)
            {
                errorHandler::handle(cudaSetDevice(deviceIndices[deviceIdx]));
                errorHandler::handle(cudaDeviceSynchronize());
            }

            for (host::label_t deviceIdx = 0; deviceIdx < deviceIndices.size(); deviceIdx++)
            {
                errorHandler::handle(cudaSetDevice(deviceIndices[deviceIdx]));
                for (device::label_t stream = 0; stream < 3; stream++)
                {
                    errorHandler::handle(cudaStreamCreate(&streams[device::idxStream(deviceIdx, stream)]));
                }
            }

            for (host::label_t deviceIdx = 0; deviceIdx < deviceIndices.size(); deviceIdx++)
            {
                errorHandler::handle(cudaSetDevice(deviceIndices[deviceIdx]));
                errorHandler::handle(cudaDeviceSynchronize());
            }

            return streams;
        }

        /**
         * @brief The underlying streams held in a std::array
         **/
        const std::vector<cudaStream_t> streams_;
    };
}

#endif