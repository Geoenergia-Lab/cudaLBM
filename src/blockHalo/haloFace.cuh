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
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
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
             * @param[in] mesh The lattice mesh
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
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : x0_(initialise_pop<axis::X, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::X>()),
                  x1_(initialise_pop<axis::X, +1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::X>()),
                  y0_(initialise_pop<axis::Y, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Y>()),
                  y1_(initialise_pop<axis::Y, +1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Y>()),
                  z0_(initialise_pop<axis::Z, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Z>()),
                  z1_(initialise_pop<axis::Z, +1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Z>()) {}

            /**
             * @brief Destructor - releases all allocated device memory
             **/
            __host__ ~haloFace() noexcept {}

            /**
             * @name Read-only Accessors
             * @brief Provide const access to halo face data
             * @return Const pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x0Const(const label_t i) const noexcept
            {
                return x0_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x1Const(const label_t i) const noexcept
            {
                return x1_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y0Const(const label_t i) const noexcept
            {
                return y0_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y1Const(const label_t i) const noexcept
            {
                return y1_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z0Const(const label_t i) const noexcept
            {
                return z0_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z1Const(const label_t i) const noexcept
            {
                return z1_.constPtr(i);
            }

            /**
             * @name Mutable Accessors
             * @brief Provide mutable access to halo face data
             * @return Pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x0(const label_t i) noexcept
            {
                return x0_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x1(const label_t i) noexcept
            {
                return x1_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y0(const label_t i) noexcept
            {
                return y0_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y1(const label_t i) noexcept
            {
                return y1_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z0(const label_t i) noexcept
            {
                return z0_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z1(const label_t i) noexcept
            {
                return z1_.ptr(i);
            }

            /**
             * @name Pointer Reference Accessors
             * @brief Provide reference to pointer for swapping operations
             * @return Reference to pointer (used for buffer swapping)
             * @note These methods are specifically for pointer swapping and should not be used elsewhere
             **/
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x0Ref(const label_t i) noexcept
            {
                return x0_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x1Ref(const label_t i) noexcept
            {
                return x1_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y0Ref(const label_t i) noexcept
            {
                return y0_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y1Ref(const label_t i) noexcept
            {
                return y1_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z0Ref(const label_t i) noexcept
            {
                return z0_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z1Ref(const label_t i) noexcept
            {
                return z1_.ptrRef(i);
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
             * @brief Initialize population data for a specific halo face
             * @tparam alpha Direction index (x, y, or z)
             * @tparam coeff Face coeff (-1 for min, 1 for max)
             * @param[in] fMom Moment representation of distribution functions
             * @param[in] mesh The lattice mesh
             * @return Initialized population data for the specified halo face
             **/
            template <const axis::type alpha, const int coeff>
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
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::haloFace::initialise_pop, "Need to sort out indexing and decomposition between devices"));

                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                std::vector<scalar_t> face(mesh.nFacesPerDevice<alpha, VelocitySet::QF()>(), 0);

                // I think it is correct for this loop to be global
                // Because the initial conditions and file that we read from are already GPU-ordered
                host::forAll(
                    mesh.nBlocks(),
                    [&](const label_t bx, const label_t by, const label_t bz,
                        const label_t tx, const label_t ty, const label_t tz)
                    {
                        const label_t base = host::idx(tx, ty, tz, bx, by, bz, mesh);

                        // Handle ghost cells
                        handleGhostCells<alpha, coeff>(
                            face,
                            VelocitySet::reconstruct(
                                {rho[base] + rho0(),
                                 u[base],
                                 v[base],
                                 w[base],
                                 m_xx[base],
                                 m_xy[base],
                                 m_xz[base],
                                 m_yy[base],
                                 m_yz[base],
                                 m_zz[base]}),
                            tx, ty, tz,
                            bx, by, bz,
                            mesh);
                    });

                return face;
            }

            /**
             * @brief Populate halo face with population data from boundary cells
             * @tparam alpha Direction index (x, y, or z)
             * @tparam coeff Face coeff (-1 for min, 1 for max)
             * @param[out] face Halo face data to populate
             * @param[in] pop Population density values for current cell
             * @param[in] tx, ty, tz Thread indices within block
             * @param[in] bx, by, bz Block indices
             * @param[in] mesh The lattice mesh
             *
             * This method handles the D3Q19 lattice model, storing appropriate
             * population components based on boundary position and direction.
             **/
            template <const axis::type alpha, const int coeff>
            __host__ static void handleGhostCells(
                std::vector<scalar_t> &face,
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const label_t tx, const label_t ty, const label_t tz,
                const label_t bx, const label_t by, const label_t bz,
                const host::latticeMesh &mesh) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::haloFace::handleGhostCells, "Potential issue with indexing: idxPop needs to be adapted to multi GPU."));

                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                constexpr const thread::array<label_t, VelocitySet::QF()> indices = velocitySet::template indices_on_face<VelocitySet, alpha, coeff>();

                const blockLabel_t Tx(tx, ty, tz);
                const blockLabel_t Bx(bx, by, bz);

                for (label_t i = 0; i < VelocitySet::QF(); i++)
                {
                    face[host::idxPop<alpha, VelocitySet::QF()>(i, Tx, Bx, mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>())] = pop[indices[i]];
                }
            }
        };
    }
}

#endif