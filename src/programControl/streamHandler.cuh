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
    /**
     * @class streamHandler
     * @brief Manages a collection of CUDA streams for asynchronous operations
     * @tparam N Number of CUDA streams to manage (must be positive)
     *
     * This class handles the creation, synchronization, and destruction of
     * multiple CUDA streams. It provides thread-safe access to streams and
     * ensures proper cleanup during destruction.
     */
    template <const label_t N>
    class streamHandler
    {
    public:
        /**
         * @brief Default constructor
         *
         * Creates N CUDA streams upon construction. The constructor includes
         * a static assertion to ensure N is positive at compile-time.
         * @pre N must be positive (enforced via static_assert)
         */
        __host__ [[nodiscard]] streamHandler() noexcept
            : streams_(createCudaStreams())
        {
            static_assert(N > 0, "Number of CUDA streams must be positive!");
        }

        /**
         * @brief Destructor
         *
         * Automatically synchronizes and destroys all CUDA streams upon
         * object destruction. Ensures proper cleanup of GPU resources.
         */
        ~streamHandler() noexcept
        {
            if constexpr (N == 1)
            {
                cudaStreamSynchronize(streams_[0]);
                cudaStreamDestroy(streams_[0]);
            }
            else
            {
                for (label_t stream = 0; stream < N; stream++)
                {
                    cudaStreamSynchronize(streams_[stream]);
                }
                for (label_t stream = 0; stream < N; stream++)
                {
                    cudaStreamDestroy(streams_[stream]);
                }
            }
        }

        /**
         * @brief Synchronizes all managed CUDA streams
         *
         * Blocks the host until all operations in all CUDA streams complete.
         * This ensures completion of all asynchronous operations before continuing.
         */
        inline void synchronizeAll() const noexcept
        {
            host::constexpr_for<0, N>(
                [&](const auto stream)
                {
                    cudaStreamSynchronize(streams_[stream]);
                });
        }

        /**
         * @brief Synchronizes a specific CUDA stream
         * @tparam stream_ Index of the stream to synchronize (must be < N)
         * @param[in] stream Integral constant representing the stream index
         * @pre stream_ must be less than N (enforced via static_assert)
         *
         * Blocks the host until all operations in the specified stream complete.
         */
        template <const label_t stream_>
        inline void synchronize(const std::integral_constant<label_t, stream_> stream) const noexcept
        {
            static_assert(stream_ < N, "Attempting to access an invalid stream: stream number must be < N");
            cudaStreamSynchronize(streams_[stream()]);
        }

        /**
         * @brief Stream access operator
         * @tparam stream_ Index of the stream to access (must be < N)
         * @param[in] stream Integral constant representing the stream index
         * @return Reference to the requested CUDA stream
         * @pre stream_ must be less than N (enforced via static_assert)
         * @warning No bounds checking performed at runtime
         */
        template <const label_t stream_>
        __device__ cudaStream_t &operator[](const std::integral_constant<label_t, stream_> stream) const noexcept
        {
            return streams_[stream()];
        }

        /**
         * @brief Returns all managed CUDA streams
         * @return Const reference to std::array containing all CUDA streams
         */
        __host__ [[nodiscard]] inline const std::array<cudaStream_t, N> &streams() const noexcept
        {
            return streams_;
        }

    private:
        /**
         * @brief Creates and initializes CUDA streams
         * @return std::array of N initialized CUDA streams
         *
         * Private helper function that handles actual stream creation
         * with proper error checking and device synchronization.
         */
        __host__ [[nodiscard]] static const std::array<cudaStream_t, N> createCudaStreams() noexcept
        {
            std::array<cudaStream_t, N> streamsLBM;

            for (label_t stream = 0; stream < N; stream++)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                checkCudaErrors(cudaStreamCreate(&streamsLBM[stream]));
                checkCudaErrors(cudaDeviceSynchronize());
            }

            return streamsLBM;
        }

        /**
         * @brief The underlying streams held in a std::array
         */
        const std::array<cudaStream_t, N> streams_;
    };
}

#endif