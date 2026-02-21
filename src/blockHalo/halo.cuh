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
        template <class VelocitySet, const bool x_periodic, const bool y_periodic, const bool z_periodic>
        class halo
        {
        public:
            /**
             * @brief Constructs halo regions from moment data and mesh
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] halo(const host::latticeMesh &mesh, const programControl &programCtrl) noexcept
                : fGhost_(haloFace<VelocitySet>(
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl),
                      mesh,
                      programCtrl)),
                  gGhost_(haloFace<VelocitySet>(
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl),
                      host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl),
                      mesh,
                      programCtrl)) {}

            /**
             * @brief Default destructor
             **/
            __host__ ~halo() {}

            /**
             * @brief Swaps read and write halo buffers
             * @note Synchronizes device before swapping to ensure all operations complete
             **/
            __host__ inline void swap(const label_t i) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::halo::swap, "Must set the device prior to calling."));

                errorHandler::checkInline(cudaDeviceSynchronize());
                std::swap(fGhost_.x0Ref(i), gGhost_.x0Ref(i));
                std::swap(fGhost_.x1Ref(i), gGhost_.x1Ref(i));
                std::swap(fGhost_.y0Ref(i), gGhost_.y0Ref(i));
                std::swap(fGhost_.y1Ref(i), gGhost_.y1Ref(i));
                std::swap(fGhost_.z0Ref(i), gGhost_.z0Ref(i));
                std::swap(fGhost_.z1Ref(i), gGhost_.z1Ref(i));
            }

            /**
             * @brief Provides read-only access to the current read halo
             * @return Collection of const pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> fGhost(const label_t i) const noexcept
            {
                return {fGhost_.x0Const(i), fGhost_.x1Const(i), fGhost_.y0Const(i), fGhost_.y1Const(i), fGhost_.z0Const(i), fGhost_.z1Const(i)};
            }

            /**
             * @brief Provides mutable access to the current write halo
             * @return Collection of mutable pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> gGhost(const label_t i) noexcept
            {
                return {gGhost_.x0(i), gGhost_.x1(i), gGhost_.y0(i), gGhost_.y1(i), gGhost_.z0(i), gGhost_.z1(i)};
            }

            /**
             * @brief Loads halo population data from neighboring blocks
             * @param[out] pop Array to store loaded population values
             * @param[in] fGhost Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            __device__ static inline constexpr void load(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &fGhost,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::halo::load, "Potential issue with condition checking (e.g. West, East, etc)."));

                // No boundary check
                if (West(point.value<axis::X>(), Tx))
                {
                    // West
                    load_face<axis::X, +1, 1>(pop, fGhost, Tx, Bx);
                }
                // No boundary check
                else if (East(point.value<axis::X>(), Tx))
                {
                    // East
                    load_face<axis::X, -1, 0>(pop, fGhost, Tx, Bx);
                }

                // No boundary check
                if (South(point.value<axis::Y>(), Tx))
                {
                    // South
                    load_face<axis::Y, +1, 3>(pop, fGhost, Tx, Bx);
                }
                // No boundary check
                else if (North(point.value<axis::Y>(), Tx))
                {
                    // North
                    load_face<axis::Y, -1, 2>(pop, fGhost, Tx, Bx);
                }

                // No boundary check
                if (Back(point.value<axis::Z>(), Tx))
                {
                    // Back
                    load_face<axis::Z, +1, 5>(pop, fGhost, Tx, Bx);
                }
                // No boundary check
                else if (Front(point.value<axis::Z>(), Tx))
                {
                    // Front
                    load_face<axis::Z, -1, 4>(pop, fGhost, Tx, Bx);
                }
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @param[in] pop Array containing population values to save
             * @param[out] gGhost Collection of pointers to the halo faces
             *
             * This device function saves population values to halo regions for
             * neighboring blocks to read.
             **/
            __device__ static inline constexpr void save(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const thread::array<scalar_t, 10> &moments,
                const device::ptrCollection<6, scalar_t> &gGhost,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::halo::save, "Potential issue with condition checking (e.g. West, East, etc)."));

                VelocitySet::reconstruct<false>(pop, moments);

                if (West(point.value<axis::X>(), Tx))
                {
                    // West
                    save_face<axis::X, 0, -1>(pop, gGhost, Tx, Bx);
                }
                else if (East(point.value<axis::X>(), Tx))
                {
                    // East
                    save_face<axis::X, 1, 1>(pop, gGhost, Tx, Bx);
                }

                if (South(point.value<axis::Y>(), Tx))
                {
                    // South
                    save_face<axis::Y, 2, -1>(pop, gGhost, Tx, Bx);
                }
                else if (North(point.value<axis::Y>(), Tx))
                {
                    // North
                    save_face<axis::Y, 3, 1>(pop, gGhost, Tx, Bx);
                }

                if (Back(point.value<axis::Z>(), Tx))
                {
                    // Back
                    save_face<axis::Z, 4, -1>(pop, gGhost, Tx, Bx);
                }
                else if (Front(point.value<axis::Z>(), Tx))
                {
                    // Front
                    save_face<axis::Z, 5, 1>(pop, gGhost, Tx, Bx);
                }
            }

#include "haloSharedMemoryOperations.cuh"

        private:
            /**
             * @brief The individual halo objects
             **/
            haloFace<VelocitySet> fGhost_;
            haloFace<VelocitySet> gGhost_;

            /**
             * @brief Returns the streaming index for a given axis and velocity
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam v The velocity component (-1 or 1)
             * @param[i] i The index of the velocity
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] static inline consteval label_t streaming_index(const label_t i) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                return velocitySet::template indices_on_face<VelocitySet, alpha, coeff>()[i];
            }

            /**
             * @brief Check if current thread is at western block boundary
             * @param[in] x Global x-coordinate
             * @return True if at western boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline constexpr bool West(const label_t x, const thread::coordinate &Tx) noexcept
            {
                if constexpr (x_periodic)
                {
                    return (Tx.value<axis::X>() == 0);
                }
                else
                {
                    return (Tx.value<axis::X>() == 0 && x != 0);
                }
            }

            /**
             * @brief Check if current thread is at eastern block boundary
             * @param[in] x Global x-coordinate
             * @return True if at eastern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline constexpr bool East(const label_t x, const thread::coordinate &Tx) noexcept
            {
                if constexpr (x_periodic)
                {
                    return (Tx.value<axis::X>() == (block::n<axis::X>() - 1));
                }
                else
                {
                    return (Tx.value<axis::X>() == (block::n<axis::X>() - 1) && x != (device::n<axis::X>() - 1));
                }
            }

            /**
             * @brief Check if current thread is at southern block boundary
             * @param[in] y Global y-coordinate
             * @return True if at southern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline constexpr bool South(const label_t y, const thread::coordinate &Tx) noexcept
            {
                if constexpr (y_periodic)
                {
                    return (Tx.value<axis::Y>() == 0);
                }
                else
                {
                    return (Tx.value<axis::Y>() == 0 && y != 0);
                }
            }

            /**
             * @brief Check if current thread is at northern block boundary
             * @param[in] y Global y-coordinate
             * @return True if at northern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline constexpr bool North(const label_t y, const thread::coordinate &Tx) noexcept
            {
                if constexpr (y_periodic)
                {
                    return (Tx.value<axis::Y>() == (block::n<axis::Y>() - 1));
                }
                else
                {
                    return (Tx.value<axis::Y>() == (block::n<axis::Y>() - 1) && y != (device::n<axis::Y>() - 1));
                }
            }

            /**
             * @brief Check if current thread is at back (z-min) block boundary
             * @param[in] z Global z-coordinate
             * @return True if at back boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline constexpr bool Back(const label_t z, const thread::coordinate &Tx) noexcept
            {
                if constexpr (z_periodic)
                {
                    return (Tx.value<axis::Z>() == 0);
                }
                else
                {
                    return (Tx.value<axis::Z>() == 0 && z != 0);
                }
            }

            /**
             * @brief Check if current thread is at front (z-max) block boundary
             * @param[in] z Global z-coordinate
             * @return True if at front boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline constexpr bool Front(const label_t z, const thread::coordinate &Tx) noexcept
            {
                if constexpr (z_periodic)
                {
                    return (Tx.value<axis::Z>() == (block::n<axis::Z>() - 1));
                }
                else
                {
                    return (Tx.value<axis::Z>() == (block::n<axis::Z>() - 1) && z != (device::n<axis::Z>() - 1));
                }
            }

            /**
             * @brief Computes linear index for a thread within a block
             * @param[in] tx Thread x-coordinate within block
             * @param[in] ty Thread y-coordinate within block
             * @param[in] tz Thread z-coordinate within block
             * @return Linearized index in shared memory
             *
             * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
             **/
            __device__ __host__ [[nodiscard]] static inline constexpr label_t idx_block(const label_t tx, const label_t ty, const label_t tz) noexcept
            {
                return tx + block::nx() * (ty + block::ny() * tz);
            }

            /**
             * @brief Computes the warp number of a particular thread within a block
             * @param[in] tx Thread x-coordinate within block
             * @param[in] ty Thread y-coordinate within block
             * @param[in] tz Thread z-coordinate within block
             * @return The unique ID of the warp corresponding to a particular thread
             *
             * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
             **/
            __device__ __host__ [[nodiscard]] static inline constexpr label_t warpID(const label_t tx, const label_t ty, const label_t tz) noexcept
            {
                return idx_block(tx, ty, tz) / block::warp_size();
            }

            /**
             * @brief Computes the linear index of a thread within a warp
             * @param[in] tx Thread x-coordinate within block
             * @param[in] ty Thread y-coordinate within block
             * @param[in] tz Thread z-coordinate within block
             * @return The unique ID of a thread within a warp, in the range [0, warp_size]
             *
             * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
             **/
            __device__ __host__ [[nodiscard]] static inline constexpr label_t idxWarp(const label_t tx, const label_t ty, const label_t tz) noexcept
            {
                return idx_block(tx, ty, tz) % block::warp_size();
            }

            /**
             * @brief Selects between shifted or central thread coordinates based upon a coefficient
             * @tparam coeff The velocity set coefficient (-1, 0, +1)
             * @param[in] dt The left and right hand side shifted thread coordinates
             * @param[in] t The thread coordinate
             **/
            template <const int coeff>
            __device__ [[nodiscard]] static inline constexpr label_t thread_stencil(const thread::array<label_t, 2> &dt, const label_t t) noexcept
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
             * @tparam alpha The axis (X, Y or Z)
             * @tparam coeff The velocity set coefficient (-1, 0, +1)
             * @param[in] t The thread coordinate
             * @param[in] b_shifted The shifted block
             * @param[in] b The current block
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] static inline constexpr label_t block_stencil(const label_t t, const label_t b_shifted, const label_t b) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

                if constexpr (coeff == -1)
                {
                    return (t == 0) ? (b_shifted) : (b);
                }

                if constexpr (coeff == 0)
                {
                    return b;
                }

                if constexpr (coeff == 1)
                {
                    return (t == block::n<alpha>() - 1) ? (b_shifted) : (b);
                }
            }

            /**
             * @brief Loads the populations from the halo into the pop array for a particular face
             * @tparam alpha The axis direction
             * @tparam PtrIndex The index of the pointer corresponding to the halo face
             * @tparam coeff The normal direction; -1 or +1
             * @param[out] pop Array to store loaded population values
             * @param[in] fGhost Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            template <const axis::type alpha, const int coeff, const label_t PtrIndex>
            __device__ static inline constexpr void load_face(thread::array<scalar_t, VelocitySet::Q()> &pop, const device::ptrCollection<6, const scalar_t> &fGhost, const thread::coordinate &Tx, const block::coordinate &Bx) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                const thread::array<label_t, 2> dBx{Bx.shifted_block<axis::X, -1>(), Bx.shifted_block<axis::X, +1>()};
                const thread::array<label_t, 2> dBy{Bx.shifted_block<axis::Y, -1>(), Bx.shifted_block<axis::Y, +1>()};
                const thread::array<label_t, 2> dBz{Bx.shifted_block<axis::Z, -1>(), Bx.shifted_block<axis::Z, +1>()};

                const thread::array<label_t, 2> da{Tx.shifted_coordinate<axis::orthogonal<alpha, 0>(), -1>(), Tx.shifted_coordinate<axis::orthogonal<alpha, 0>(), +1>()};
                const thread::array<label_t, 2> db{Tx.shifted_coordinate<axis::orthogonal<alpha, 1>(), -1>(), Tx.shifted_coordinate<axis::orthogonal<alpha, 1>(), +1>()};

                device::constexpr_for<0, VelocitySet::QF()>(
                    [&](const auto i)
                    {
                        const label_t t_a = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(da, Tx.value<axis::orthogonal<alpha, 0>()>());
                        const label_t t_b = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(db, Tx.value<axis::orthogonal<alpha, 1>()>());

                        // Then we should select the true block based on the thread
                        const label_t b_x = block_stencil<axis::X, -VelocitySet::template c<int, axis::X>()[streaming_index<alpha, coeff>(i)]>(
                            Tx.value<axis::X>(),
                            thread_stencil<-VelocitySet::template c<int, axis::X>()[streaming_index<alpha, coeff>(i)]>(
                                dBx, Bx.value<axis::X>()),
                            Bx.value<axis::X>());

                        const label_t b_y = block_stencil<axis::Y, -VelocitySet::template c<int, axis::Y>()[streaming_index<alpha, coeff>(i)]>(
                            Tx.value<axis::Y>(),
                            thread_stencil<-VelocitySet::template c<int, axis::Y>()[streaming_index<alpha, coeff>(i)]>(
                                dBy, Bx.value<axis::Y>()),
                            Bx.value<axis::Y>());

                        const label_t b_z = block_stencil<axis::Z, -VelocitySet::template c<int, axis::Z>()[streaming_index<alpha, coeff>(i)]>(
                            Tx.value<axis::Z>(),
                            thread_stencil<-VelocitySet::template c<int, axis::Z>()[streaming_index<alpha, coeff>(i)]>(
                                dBz, Bx.value<axis::Z>()),
                            Bx.value<axis::Z>());

                        pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&fGhost.ptr<PtrIndex>()[idxPop<alpha, i, VelocitySet::QF()>(t_a, t_b, b_x, b_y, b_z)]);
                    });
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @tparam alpha The axis direction
             * @tparam PtrIndex The index of the pointer corresponding to the halo face
             * @tparam coeff The normal direction; -1 or +1
             * @param[out] pop Array to store loaded population values
             * @param[in] fGhost Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            template <const axis::type alpha, const label_t PtrIndex, const int coeff>
            __device__ static inline constexpr void save_face(const thread::array<scalar_t, VelocitySet::Q()> &pop, const device::ptrCollection<6, scalar_t> &gGhost, const thread::coordinate &Tx, const block::coordinate &Bx) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                device::constexpr_for<0, VelocitySet::QF()>(
                    [&](const auto i)
                    {
                        gGhost.ptr<PtrIndex>()[idxPop<alpha, i, VelocitySet::QF()>(Tx.value<axis::orthogonal<alpha, 0>()>(), Tx.value<axis::orthogonal<alpha, 1>()>(), Bx)] = pop[q_i<streaming_index<alpha, coeff>(i)>()];
                    });
            }
        };
    }
}

#endif