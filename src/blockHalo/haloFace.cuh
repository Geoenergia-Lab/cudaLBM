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
    A class handling the individual faces of the device halo.

Namespace
    LBM::device

SourceFiles
    haloFace.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALOFACE_CUH
#define __MBLBM_HALOFACE_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @class haloFace
         * @brief Manages individual halo faces for inter-block communication in CUDA LBM
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         *
         * This class handles the storage and management of distribution functions
         * at block boundaries for all six faces (x0, x1, y0, y1, z0, z1). It provides
         * both read-only and mutable access to halo data for efficient communication
         * between adjacent CUDA blocks during LBM simulations.
         **/
        template <class VelocitySet>
        class haloFace
        {
        public:
            /**
             * @brief Constructs halo faces from moment data and mesh
             * @param[in] fMom Moment representation of distribution functions (10 interlaced moments)
             * @param[in] mesh Lattice mesh defining simulation domain
             * @post All six halo faces are allocated and initialized with population data
             **/
            __host__ [[nodiscard]] haloFace(
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &rho,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &u,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &v,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &w,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_xx,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_xy,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_xz,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_yy,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_yz,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_zz,
                const host::latticeMesh &mesh) noexcept
                : x0_(device::allocateArray(initialise_pop<axis::X, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh))),
                  x1_(device::allocateArray(initialise_pop<axis::X, 1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh))),
                  y0_(device::allocateArray(initialise_pop<axis::Y, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh))),
                  y1_(device::allocateArray(initialise_pop<axis::Y, 1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh))),
                  z0_(device::allocateArray(initialise_pop<axis::Z, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh))),
                  z1_(device::allocateArray(initialise_pop<axis::Z, 1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh))){};

            /**
             * @brief Destructor - releases all allocated device memory
             **/
            ~haloFace() noexcept
            {
                checkCudaErrors(cudaFree(x0_));
                checkCudaErrors(cudaFree(x1_));
                checkCudaErrors(cudaFree(y0_));
                checkCudaErrors(cudaFree(y1_));
                checkCudaErrors(cudaFree(z0_));
                checkCudaErrors(cudaFree(z1_));
            }

            /**
             * @name Read-only Accessors
             * @brief Provide const access to halo face data
             * @return Const pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x0Const() const noexcept
            {
                return x0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x1Const() const noexcept
            {
                return x1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y0Const() const noexcept
            {
                return y0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y1Const() const noexcept
            {
                return y1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z0Const() const noexcept
            {
                return z0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z1Const() const noexcept
            {
                return z1_;
            }

            /**
             * @name Mutable Accessors
             * @brief Provide mutable access to halo face data
             * @return Pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x0() noexcept
            {
                return x0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x1() noexcept
            {
                return x1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y0() noexcept
            {
                return y0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y1() noexcept
            {
                return y1_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z0() noexcept
            {
                return z0_;
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z1() noexcept
            {
                return z1_;
            }

            /**
             * @name Pointer Reference Accessors
             * @brief Provide reference to pointer for swapping operations
             * @return Reference to pointer (used for buffer swapping)
             * @note These methods are specifically for pointer swapping and should not be used elsewhere
             **/
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x0Ref() noexcept
            {
                return x0_;
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x1Ref() noexcept
            {
                return x1_;
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y0Ref() noexcept
            {
                return y0_;
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y1Ref() noexcept
            {
                return y1_;
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z0Ref() noexcept
            {
                return z0_;
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z1Ref() noexcept
            {
                return z1_;
            }

        private:
            /**
             * @brief Halo faces pointers
             **/
            scalar_t *ptrRestrict x0_;
            scalar_t *ptrRestrict x1_;
            scalar_t *ptrRestrict y0_;
            scalar_t *ptrRestrict y1_;
            scalar_t *ptrRestrict z0_;
            scalar_t *ptrRestrict z1_;

            /**
             * @brief Calculate number of elements for a halo face
             * @tparam alpha Direction index (x, y, or z)
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Number of elements in the specified halo face
             **/
            template <const axis::direction alpha>
            __host__ [[nodiscard]] static inline constexpr label_t nFaces(const host::latticeMesh &mesh) noexcept
            {
                assertions::validate_direction<alpha, axis::NOT_NULL>();

                if constexpr (alpha == axis::X)
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nx()) * VelocitySet::QF();
                }

                if constexpr (alpha == axis::Y)
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::ny()) * VelocitySet::QF();
                }

                if constexpr (alpha == axis::Z)
                {
                    return ((mesh.nx() * mesh.ny() * mesh.nz()) / block::nz()) * VelocitySet::QF();
                }
            }

            /**
             * @brief Initialize population data for a specific halo face
             * @tparam alpha Direction index (x, y, or z)
             * @tparam side Face side (0 for min, 1 for max)
             * @param[in] fMom Moment representation of distribution functions
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Initialized population data for the specified halo face
             **/
            template <const axis::direction alpha, const int side>
            __host__ [[nodiscard]] const std::vector<scalar_t> initialise_pop(
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &rho,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &u,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &v,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &w,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_xx,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_xy,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_xz,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_yy,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_yz,
                const host::array<false, scalar_t, VelocitySet, time::instantaneous> &m_zz,
                const host::latticeMesh &mesh) const noexcept
            {
                std::vector<scalar_t> face(nFaces<alpha>(mesh), 0);

                // Loop over all blocks and threads
                for (label_t bz = 0; bz < mesh.nzBlocks(); ++bz)
                {
                    for (label_t by = 0; by < mesh.nyBlocks(); ++by)
                    {
                        for (label_t bx = 0; bx < mesh.nxBlocks(); ++bx)
                        {
                            for (label_t tz = 0; tz < block::nz(); ++tz)
                            {
                                for (label_t ty = 0; ty < block::ny(); ++ty)
                                {
                                    for (label_t tx = 0; tx < block::nx(); ++tx)
                                    {

                                        // Skip out-of-bounds elements (equivalent to GPU version)
                                        if (tx >= mesh.nx() || ty >= mesh.ny() || tz >= mesh.nz())
                                        {
                                            continue;
                                        }

                                        const label_t base = host::idx(tx, ty, tz, bx, by, bz, mesh);

                                        // Contiguous moment access
                                        const thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(
                                            thread::array<scalar_t, 10>{
                                                rho0<scalar_t>() + rho[base],
                                                u[base],
                                                v[base],
                                                w[base],
                                                m_xx[base],
                                                m_xy[base],
                                                m_xz[base],
                                                m_yy[base],
                                                m_yz[base],
                                                m_zz[base]});

                                        // Handle ghost cells (equivalent to threadIdx.x/y/z checks)
                                        handleGhostCells<alpha, side>(face, pop, tx, ty, tz, bx, by, bz, mesh);
                                    }
                                }
                            }
                        }
                    }
                }

                return face;
            }

            /**
             * @brief Populate halo face with population data from boundary cells
             * @tparam alpha Direction index (X, Y, or Z)
             * @tparam side Face side (0 for min, 1 for max)
             * @param[out] face Halo face data to populate
             * @param[in] pop Population density values for current cell
             * @param[in] tx, ty, tz Thread indices within block
             * @param[in] bx, by, bz Block indices
             * @param[in] mesh Lattice mesh for dimensioning
             *
             * This method handles the D3Q19 lattice model, storing appropriate
             * population components based on boundary position and direction.
             **/
            template <const axis::direction alpha, const int side>
            __host__ void static handleGhostCells(
                std::vector<scalar_t> &face,
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const label_t tx, const label_t ty, const label_t tz,
                const label_t bx, const label_t by, const label_t bz,
                const host::latticeMesh &mesh) noexcept
            {
                constexpr const thread::array<label_t, VelocitySet::QF()> indices = velocitySet::template indices_on_face<VelocitySet, alpha, side>();

                host::constexpr_for<0, VelocitySet::QF()>(
                    [&](const auto q)
                    {
                        face[host::idxPop<alpha, q, VelocitySet::QF()>(tx, ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[indices[q_i<q>()]];
                    });

                // if constexpr (alpha == X)
                // {
                //     if constexpr (side == 0)
                //     {
                //         if (tx == 0)
                //         {
                //             // w
                //             face[host::idxPopX<0, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<2>()];
                //             face[host::idxPopX<1, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<8>()];
                //             face[host::idxPopX<2, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<10>()];
                //             face[host::idxPopX<3, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<14>()];
                //             face[host::idxPopX<4, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<16>()];
                //             if constexpr (VelocitySet::Q() == 27)
                //             {
                //                 face[host::idxPopX<5, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<20>()];
                //                 face[host::idxPopX<6, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<22>()];
                //                 face[host::idxPopX<7, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<24>()];
                //                 face[host::idxPopX<8, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<25>()];
                //             }
                //         }
                //     }
                //     if constexpr (side == 1)
                //     {
                //         if (tx == (block::nx() - 1))
                //         {
                //             face[host::idxPopX<0, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<1>()];
                //             face[host::idxPopX<1, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<7>()];
                //             face[host::idxPopX<2, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<9>()];
                //             face[host::idxPopX<3, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<13>()];
                //             face[host::idxPopX<4, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<15>()];
                //             if constexpr (VelocitySet::Q() == 27)
                //             {
                //                 face[host::idxPopX<5, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<19>()];
                //                 face[host::idxPopX<6, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<21>()];
                //                 face[host::idxPopX<7, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<23>()];
                //                 face[host::idxPopX<8, VelocitySet::QF()>(ty, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<26>()];
                //             }
                //         }
                //     }
                // }

                // if constexpr (alpha == Y)
                // {
                //     if constexpr (side == 0)
                //     {
                //         if (ty == 0)
                //         { // s
                //             face[host::idxPopY<0, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<4>()];
                //             face[host::idxPopY<1, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<8>()];
                //             face[host::idxPopY<2, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<12>()];
                //             face[host::idxPopY<3, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<13>()];
                //             face[host::idxPopY<4, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<18>()];
                //             if constexpr (VelocitySet::Q() == 27)
                //             {
                //                 face[host::idxPopY<5, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<20>()];
                //                 face[host::idxPopY<6, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<22>()];
                //                 face[host::idxPopY<7, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<23>()];
                //                 face[host::idxPopY<8, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<26>()];
                //             }
                //         }
                //     }
                //     if constexpr (side == 1)
                //     {
                //         if (ty == (block::ny() - 1))
                //         {
                //             face[host::idxPopY<0, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<3>()];
                //             face[host::idxPopY<1, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<7>()];
                //             face[host::idxPopY<2, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<11>()];
                //             face[host::idxPopY<3, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<14>()];
                //             face[host::idxPopY<4, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<17>()];
                //             if constexpr (VelocitySet::Q() == 27)
                //             {
                //                 face[host::idxPopY<5, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<19>()];
                //                 face[host::idxPopY<6, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<21>()];
                //                 face[host::idxPopY<7, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<24>()];
                //                 face[host::idxPopY<8, VelocitySet::QF()>(tx, tz, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<25>()];
                //             }
                //         }
                //     }
                // }

                // if constexpr (alpha == Z)
                // {
                //     if constexpr (side == 0)
                //     {
                //         if (tz == 0)
                //         { // b
                //             face[host::idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<6>()];
                //             face[host::idxPopZ<1, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<10>()];
                //             face[host::idxPopZ<2, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<12>()];
                //             face[host::idxPopZ<3, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<15>()];
                //             face[host::idxPopZ<4, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<17>()];
                //             if constexpr (VelocitySet::Q() == 27)
                //             {
                //                 face[host::idxPopZ<5, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<20>()];
                //                 face[host::idxPopZ<6, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<21>()];
                //                 face[host::idxPopZ<7, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<24>()];
                //                 face[host::idxPopZ<8, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<26>()];
                //             }
                //         }
                //     }
                //     if constexpr (side == 1)
                //     {
                //         if (tz == (block::nz() - 1))
                //         {
                //             face[host::idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<5>()];
                //             face[host::idxPopZ<1, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<9>()];
                //             face[host::idxPopZ<2, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<11>()];
                //             face[host::idxPopZ<3, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<16>()];
                //             face[host::idxPopZ<4, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<18>()];
                //             if constexpr (VelocitySet::Q() == 27)
                //             {
                //                 face[host::idxPopZ<5, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<19>()];
                //                 face[host::idxPopZ<6, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<22>()];
                //                 face[host::idxPopZ<7, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<23>()];
                //                 face[host::idxPopZ<8, VelocitySet::QF()>(tx, ty, bx, by, bz, mesh.nxBlocks(), mesh.nyBlocks())] = pop[q_i<25>()];
                //             }
                //         }
                //     }
                // }
            }
        };
    }
}

#endif