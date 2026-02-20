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
    Class handling the streaming step

Namespace
    LBM

SourceFiles
    streaming.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_STREAMING_CUH
#define __MBLBM_STREAMING_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../globalFunctions.cuh"
#include "../array/array.cuh"

namespace LBM
{
    /**
     * @class streaming
     * @brief Handles the streaming step in Lattice Boltzmann Method simulations
     *
     * This class manages the streaming (propagation) step of the LBM algorithm,
     * where particle distributions move to neighboring lattice sites. It provides
     * efficient shared memory operations for storing and retrieving population
     * data with optimized periodic boundary handling.
     **/
    class streaming
    {
    public:
        /**
         * @brief Default constructor
         **/
        __device__ __host__ [[nodiscard]] inline consteval streaming() {}

        /**
         * @brief Saves thread population density to shared memory
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam N Size of shared memory array
         * @param[in] pop Population density array for current thread
         * @param[out] s_pop Shared memory array for population storage
         * @param[in] tid Thread ID within block
         *
         * This method stores population data from individual threads into
         * shared memory for efficient access during the streaming step.
         * It uses compile-time loop unrolling for optimal performance.
         **/
        template <class VelocitySet>
        __device__ static inline void save(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            scalar_t *const ptrRestrict s_pop,
            const label_t tid) noexcept
        {
            device::constexpr_for<0, (VelocitySet::Q() - 1)>(
                [&](const auto i)
                {
                    s_pop[q_i<i * block::stride()>() + tid] = pop[q_i<i + 1>()];
                });
        }

        template <class VelocitySet, const std::size_t N>
        __device__ static inline void save(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, N> &s_pop,
            const label_t tid) noexcept
        {
            save<VelocitySet>(pop, s_pop.data(), tid);
        }

        /**
         * @brief Pulls population density from shared memory with periodic boundaries
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam N Size of shared memory array
         * @param[out] pop Population density array to be populated
         * @param[in] s_pop Shared memory array containing population data
         *
         * This method retrieves population data from shared memory, applying
         * periodic boundary conditions to handle data exchange between threads
         * at block boundaries. It implements the D3Q19 streaming pattern.
         **/
        template <class VelocitySet>
        __device__ static inline void pull(
            thread::array<scalar_t, VelocitySet::Q()> &pop,
            const scalar_t *const ptrRestrict s_pop,
            const thread::coordinate &Tx) noexcept
        {
            device::constexpr_for<0, (VelocitySet::Q() - 1)>(
                [&](const auto i)
                {
                    const label_t x = periodic_index<-VelocitySet::template cx<int>(q_i<i + 1>()), block::nx()>(Tx.value<axis::X>());
                    const label_t y = periodic_index<-VelocitySet::template cy<int>(q_i<i + 1>()), block::ny()>(Tx.value<axis::Y>());
                    const label_t z = periodic_index<-VelocitySet::template cz<int>(q_i<i + 1>()), block::nz()>(Tx.value<axis::Z>());
                    pop[q_i<i + 1>()] = s_pop[q_i<i * block::stride()>() + block::idx(x, y, z)];
                });
        }

        template <class VelocitySet, const std::size_t N>
        __device__ static inline void pull(
            thread::array<scalar_t, VelocitySet::Q()> &pop,
            const thread::array<scalar_t, N> &s_pop,
            const thread::coordinate &Tx) noexcept
        {
            pull<VelocitySet>(pop, s_pop.data(), Tx);
        }

    private:
        /**
         * @brief Computes periodic boundary index with optimization for power-of-two dimensions
         * @tparam Shift Direction shift (-1 for backward, +1 for forward)
         * @tparam Dim Dimension size (periodic length)
         * @param[in] idx Current index position
         * @return Shifted index with periodic wrapping
         *
         * This function uses bitwise AND optimization when Dim is power-of-two
         * for improved performance, falling back to modulo arithmetic otherwise.
         **/
        template <const int coeff, const label_t Dim>
        __device__ [[nodiscard]] static inline label_t periodic_index(const label_t idx) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

            if constexpr (Dim > 0 && (Dim & (Dim - 1)) == 0)
            {
                // Power-of-two: use bitwise AND
                if constexpr (coeff == -1)
                {
                    return (idx - 1) & (Dim - 1);
                }

                if constexpr (coeff == 1)
                {
                    return (idx + 1) & (Dim - 1);
                }

                if constexpr (coeff == 0)
                {
                    return idx & (Dim - 1);
                }
            }
            else
            {
                // General case: adjust by adding Dim to ensure nonnegative modulo
                if constexpr (coeff == -1)
                {
                    return (idx - 1 + Dim) % Dim;
                }

                if constexpr (coeff == 1)
                {
                    return (idx + 1) % Dim;
                }

                if constexpr (coeff == 0)
                {
                    return idx % Dim;
                }
            }
        }
    };

}

#endif