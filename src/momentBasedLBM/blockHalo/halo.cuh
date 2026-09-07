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
    A class handling the device halo. This class is used to exchange the
    microscopic velocity components at the edge of a CUDA block

Namespace
    LBM::device

SourceFiles
    halo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @class halo
         * @brief Manages halo regions for inter-block communication in CUDA LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         *
         * This class handles the exchange of distribution functions between adjacent
         * CUDA blocks during LBM simulations. It maintains double-buffered halo regions
         * to support efficient ping-pong swapping between computation steps.
         **/
        template <class VelocitySet, class BoundaryConditions>
        class halo
        {
            using blockStencil = thread::array<const device::label_t, 2>;
            using threadStencil = thread::array<const device::label_t, 2>;

        public:
            /**
             * @brief Loads halo population data from neighboring blocks
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            __device__ static inline constexpr void pull(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                pull_direction<axis::X>(pop, readBuffer, Tx, Bx, point);

                pull_direction<axis::Y>(pop, readBuffer, Tx, Bx, point);

                pull_direction<axis::Z>(pop, readBuffer, Tx, Bx, point);
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @param[out] pop Array containing population values to save
             * @param[in] moments Moment array (rho, U, Pi)
             * @param[in] writeBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             * This device function saves population values to halo regions for
             * neighboring blocks to read.
             **/
            __device__ static inline constexpr void save(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const momentsArray &moments,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                VelocitySet::reconstruct<false>(pop, moments);

                save_direction<axis::X>(pop, writeBuffer, Tx, Bx, point);

                save_direction<axis::Y>(pop, writeBuffer, Tx, Bx, point);

                save_direction<axis::Z>(pop, writeBuffer, Tx, Bx, point);
            }

#include "haloSharedMemoryOperations.cuh"

        private:
            /**
             * @brief Returns the streaming index for a given axis and velocity
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam coeff The velocity component (-1 or 1)
             * @param[in] i The index of the velocity
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] static inline consteval device::label_t streaming_index(const device::label_t i) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                return static_cast<device::label_t>(VelocitySet::template indices_on_face<alpha, coeff>()[i]);
            }

            /**
             * @brief Checks if the current thread is at a block boundary in a given direction, accounting for periodicity
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
             * @tparam isPeriodic Whether the domain is periodic in this direction
             * @param[in] alpha_v The global coordinate in the alpha direction
             * @param[in] Tx Three-dimensional thread coordinates
             * @return True if the thread is at the specified block boundary and not at the domain edge (if non-periodic)
             **/
            template <const axis::type alpha, const int coeff, const bool isPeriodic>
            __device__ [[nodiscard]] static inline constexpr bool boundaryCheck(const device::label_t alpha_v, const thread::coordinate &Tx) noexcept
            {
                if constexpr (coeff == -1)
                {
                    if constexpr (isPeriodic)
                    {
                        return (Tx.value<alpha>() == static_cast<device::label_t>(0));
                    }
                    else
                    {
                        return (Tx.value<alpha>() == static_cast<device::label_t>(0) && alpha_v != static_cast<device::label_t>(0));
                    }
                }

                if constexpr (coeff == 1)
                {
                    if constexpr (isPeriodic)
                    {
                        return (Tx.value<alpha>() == (block::n<alpha>() - static_cast<device::label_t>(1)));
                    }
                    else
                    {
                        return (Tx.value<alpha>() == (block::n<alpha>() - static_cast<device::label_t>(1)) && alpha_v != (device::n<alpha>() - static_cast<device::label_t>(1)));
                    }
                }
            }

            /**
             * @brief Selects between shifted or central thread coordinates based upon a coefficient
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1, 0 or 1)
             * @param[in] dt The left and right hand side shifted thread coordinates
             * @param[in] t The thread coordinate
             **/
            template <const int coeff>
            __device__ [[nodiscard]] static inline constexpr device::label_t thread_stencil(const blockStencil &dt, const device::label_t t) noexcept
            {
                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

                if constexpr (coeff == -1)
                {
                    return dt[0];
                }

                if constexpr (coeff == 0)
                {
                    return t;
                }

                if constexpr (coeff == 1)
                {
                    return dt[1];
                }
            }

            /**
             * @brief Selects between shifted or central block coordinates based upon a coefficient
             * @tparam alpha The axis direction (X, Y, Z or NULL)
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1, 0 or 1)
             * @param[in] t The thread coordinate
             * @param[in] b_shifted The shifted block
             * @param[in] b The current block
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] static inline constexpr device::label_t block_stencil(const device::label_t t, const device::label_t b_shifted, const device::label_t b) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

                if constexpr (coeff == -1)
                {
                    return (t == static_cast<device::label_t>(0)) ? (b_shifted) : (b);
                }

                if constexpr (coeff == 0)
                {
                    return b;
                }

                if constexpr (coeff == 1)
                {
                    return (t == block::n<alpha>() - static_cast<device::label_t>(1)) ? (b_shifted) : (b);
                }
            }

            /**
             * @brief Loads the populations from the halo into the pop array for a particular face
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam PtrIndex The index of the pointer corresponding to the halo face
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            template <const axis::type alpha, const int coeff>
            __device__ static inline constexpr void pull_face(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                const blockStencil resolvedBlock_alpha{Bx.shifted_block<alpha, -1>(), Bx.shifted_block<alpha, +1>()};
                const blockStencil resolvedBlock_beta{Bx.shifted_block<axis::orthogonal<alpha, 0>(), -1>(), Bx.shifted_block<axis::orthogonal<alpha, 0>(), +1>()};
                const blockStencil resolvedBlock_gamma{Bx.shifted_block<axis::orthogonal<alpha, 1>(), -1>(), Bx.shifted_block<axis::orthogonal<alpha, 1>(), +1>()};

                const threadStencil da{Tx.shifted_coordinate<axis::orthogonal<alpha, 0>(), -1>(), Tx.shifted_coordinate<axis::orthogonal<alpha, 0>(), +1>()};
                const threadStencil db{Tx.shifted_coordinate<axis::orthogonal<alpha, 1>(), -1>(), Tx.shifted_coordinate<axis::orthogonal<alpha, 1>(), +1>()};

                const device::label_t b_alpha = block_stencil<alpha, -coeff>(
                    Tx.value<alpha>(),
                    thread_stencil<-coeff>(resolvedBlock_alpha, Bx.value<alpha>()),
                    Bx.value<alpha>());

                device::constexpr_for<0, VelocitySet::template QF<device::label_t>()>(
                    [&](const auto i)
                    {
                        const device::label_t t_a = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(da, Tx.value<axis::orthogonal<alpha, 0>()>());
                        const device::label_t t_b = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(db, Tx.value<axis::orthogonal<alpha, 1>()>());

                        // Then we should select the true block based on the thread
                        const device::label_t b_beta = block_stencil<axis::orthogonal<alpha, 0>(), -VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(
                            Tx.value<axis::orthogonal<alpha, 0>()>(),
                            thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(resolvedBlock_beta, Bx.value<axis::orthogonal<alpha, 0>()>()),
                            Bx.value<axis::orthogonal<alpha, 0>()>());

                        const device::label_t b_gamma = block_stencil<axis::orthogonal<alpha, 1>(), -VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(
                            Tx.value<axis::orthogonal<alpha, 1>()>(),
                            thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(resolvedBlock_gamma, Bx.value<axis::orthogonal<alpha, 1>()>()),
                            Bx.value<axis::orthogonal<alpha, 1>()>());

                        if constexpr (alpha == axis::X)
                        {
                            pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::template QF<device::label_t>()>(t_a, t_b, b_alpha, b_beta, b_gamma)]));
                        }

                        if constexpr (alpha == axis::Y)
                        {
                            pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::template QF<device::label_t>()>(t_a, t_b, b_beta, b_alpha, b_gamma)]));
                        }

                        if constexpr (alpha == axis::Z)
                        {
                            pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::template QF<device::label_t>()>(t_a, t_b, b_beta, b_gamma, b_alpha)]));
                        }
                    });
            }

            /**
             * @brief Loads halo population data from neighboring blocks in a specific direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam isPeriodic Whether the domain is periodic in this direction
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            template <const axis::type alpha>
            __device__ static inline constexpr void pull_direction(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                if (boundaryCheck<alpha, -1, BoundaryConditions::periodic<alpha>()>(point.value<alpha>(), Tx))
                {
                    pull_face<alpha, +1>(pop, readBuffer, Tx, Bx);
                }
                else if (boundaryCheck<alpha, +1, BoundaryConditions::periodic<alpha>()>(point.value<alpha>(), Tx))
                {
                    pull_face<alpha, -1>(pop, readBuffer, Tx, Bx);
                }
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam PtrIndex The index of the pointer corresponding to the halo face
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            template <const axis::type alpha, const int coeff>
            __device__ static inline constexpr void save_face(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                device::constexpr_for<0, VelocitySet::template QF<device::label_t>()>(
                    [&](const auto i)
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::template QF<device::label_t>()>(
                            Tx.value<axis::orthogonal<alpha, 0>()>(),
                            Tx.value<axis::orthogonal<alpha, 1>()>(),
                            Bx.value<axis::X>(),
                            Bx.value<axis::Y>(),
                            Bx.value<axis::Z>())] = pop[q_i<streaming_index<alpha, coeff>(i)>()];
                    });
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks in a specific direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @param[in] pop Array containing population values to save
             * @param[out] writeBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            template <const axis::type alpha>
            __device__ static inline constexpr void save_direction(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                if (boundaryCheck<alpha, -1, BoundaryConditions::periodic<alpha>()>(point.value<alpha>(), Tx))
                {
                    save_face<alpha, -1>(pop, writeBuffer, Tx, Bx);
                }
                else if (boundaryCheck<alpha, +1, BoundaryConditions::periodic<alpha>()>(point.value<alpha>(), Tx))
                {
                    save_face<alpha, +1>(pop, writeBuffer, Tx, Bx);
                }
            }
        };
    }
}

#endif