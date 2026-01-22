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
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &rho,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &u,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &v,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &w,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xx,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xy,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xz,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_yy,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_yz,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_zz,
                const host::latticeMesh &mesh) noexcept
                : x0_(initialise_pop<axis::X, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  x1_(initialise_pop<axis::X, 1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  y0_(initialise_pop<axis::Y, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  y1_(initialise_pop<axis::Y, 1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  z0_(initialise_pop<axis::Z, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  z1_(initialise_pop<axis::Z, 1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)){};

            /**
             * @brief Destructor - releases all allocated device memory
             **/
            ~haloFace() noexcept {}

            /**
             * @name Read-only Accessors
             * @brief Provide const access to halo face data
             * @return Const pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x0Const() const noexcept
            {
                return x0_.constPtr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x1Const() const noexcept
            {
                return x1_.constPtr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y0Const() const noexcept
            {
                return y0_.constPtr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y1Const() const noexcept
            {
                return y1_.constPtr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z0Const() const noexcept
            {
                return z0_.constPtr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z1Const() const noexcept
            {
                return z1_.constPtr();
            }

            /**
             * @name Mutable Accessors
             * @brief Provide mutable access to halo face data
             * @return Pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x0() noexcept
            {
                return x0_.ptr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x1() noexcept
            {
                return x1_.ptr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y0() noexcept
            {
                return y0_.ptr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y1() noexcept
            {
                return y1_.ptr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z0() noexcept
            {
                return z0_.ptr();
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z1() noexcept
            {
                return z1_.ptr();
            }

            /**
             * @name Pointer Reference Accessors
             * @brief Provide reference to pointer for swapping operations
             * @return Reference to pointer (used for buffer swapping)
             * @note These methods are specifically for pointer swapping and should not be used elsewhere
             **/
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x0Ref() noexcept
            {
                return x0_.ptrRef();
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x1Ref() noexcept
            {
                return x1_.ptrRef();
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y0Ref() noexcept
            {
                return y0_.ptrRef();
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y1Ref() noexcept
            {
                return y1_.ptrRef();
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z0Ref() noexcept
            {
                return z0_.ptrRef();
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z1Ref() noexcept
            {
                return z1_.ptrRef();
            }

        private:
            /**
             * @brief Halo face arrays
             **/
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> x0_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> x1_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> y0_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> y1_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> z0_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> z1_;

            /**
             * @brief Calculate number of elements for a halo face
             * @tparam alpha Direction index (x, y, or z)
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Number of elements in the specified halo face
             **/
            template <const axis::type alpha>
            __host__ [[nodiscard]] static inline constexpr label_t nFaces(const host::latticeMesh &mesh) noexcept
            {
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

                return 0;
            }

            /**
             * @brief Initialize population data for a specific halo face
             * @tparam alpha Direction index (x, y, or z)
             * @tparam side Face side (0 for min, 1 for max)
             * @param[in] fMom Moment representation of distribution functions
             * @param[in] mesh Lattice mesh for dimensioning
             * @return Initialized population data for the specified halo face
             **/
            template <const axis::type alpha, const int side>
            __host__ [[nodiscard]] const std::vector<scalar_t> initialise_pop(
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &rho,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &u,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &v,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &w,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xx,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xy,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xz,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_yy,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_yz,
                const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_zz,
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
             * @tparam alpha Direction index (x, y, or z)
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
            template <const axis::type alpha, const int side>
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
            }
        };
    }
}

#endif