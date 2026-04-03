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
                : West_(initialise_pop<axis::X, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::X>(), integralConstant<int, -1>()),
                  East_(initialise_pop<axis::X, +1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::X>(), integralConstant<int, +1>()),
                  South_(initialise_pop<axis::Y, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Y>(), integralConstant<int, -1>()),
                  North_(initialise_pop<axis::Y, +1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Y>(), integralConstant<int, +1>()),
                  Back_(initialise_pop<axis::Z, -1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Z>(), integralConstant<int, -1>()),
                  Front_(initialise_pop<axis::Z, +1>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Z>(), integralConstant<int, +1>()) {}

            __host__ [[nodiscard]] haloFace(
                const host::scalarField<host::PAGED, VelocitySet, time::instantaneous> &rho,
                const host::vectorField<host::PAGED, VelocitySet, time::instantaneous> &U,
                const host::symmetricTensorField<host::PAGED, VelocitySet, time::instantaneous> &Pi,
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : West_(initialise_pop<axis::X, -1>(rho, U, Pi, mesh), mesh, programCtrl, integralConstant<axis::type, axis::X>(), integralConstant<int, -1>()),
                  East_(initialise_pop<axis::X, +1>(rho, U, Pi, mesh), mesh, programCtrl, integralConstant<axis::type, axis::X>(), integralConstant<int, +1>()),
                  South_(initialise_pop<axis::Y, -1>(rho, U, Pi, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Y>(), integralConstant<int, -1>()),
                  North_(initialise_pop<axis::Y, +1>(rho, U, Pi, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Y>(), integralConstant<int, +1>()),
                  Back_(initialise_pop<axis::Z, -1>(rho, U, Pi, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Z>(), integralConstant<int, -1>()),
                  Front_(initialise_pop<axis::Z, +1>(rho, U, Pi, mesh), mesh, programCtrl, integralConstant<axis::type, axis::Z>(), integralConstant<int, +1>()) {}

            /**
             * @brief Destructor - releases all allocated device memory
             **/
            __host__ ~haloFace() noexcept {}

            /**
             * @name Read-only Accessors
             * @brief Provide const access to halo face data
             * @return Const pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x0Const(const host::label_t i) const noexcept
            {
                return West_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *x1Const(const host::label_t i) const noexcept
            {
                return East_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y0Const(const host::label_t i) const noexcept
            {
                return South_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *y1Const(const host::label_t i) const noexcept
            {
                return North_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z0Const(const host::label_t i) const noexcept
            {
                return Back_.constPtr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr const scalar_t *z1Const(const host::label_t i) const noexcept
            {
                return Front_.constPtr(i);
            }

            /**
             * @name Mutable Accessors
             * @brief Provide mutable access to halo face data
             * @return Pointer to halo face data
             **/
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x0(const host::label_t i) noexcept
            {
                return West_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *x1(const host::label_t i) noexcept
            {
                return East_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y0(const host::label_t i) noexcept
            {
                return South_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *y1(const host::label_t i) noexcept
            {
                return North_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z0(const host::label_t i) noexcept
            {
                return Back_.ptr(i);
            }
            __device__ __host__ [[nodiscard]] inline constexpr scalar_t *z1(const host::label_t i) noexcept
            {
                return Front_.ptr(i);
            }

            /**
             * @name Pointer Reference Accessors
             * @brief Provide reference to pointer for swapping operations
             * @return Reference to pointer (used for buffer swapping)
             * @note These methods are specifically for pointer swapping and should not be used elsewhere
             **/
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x0Ref(const host::label_t i) noexcept
            {
                return West_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & x1Ref(const host::label_t i) noexcept
            {
                return East_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y0Ref(const host::label_t i) noexcept
            {
                return South_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & y1Ref(const host::label_t i) noexcept
            {
                return North_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z0Ref(const host::label_t i) noexcept
            {
                return Back_.ptrRef(i);
            }
            __host__ [[nodiscard]] inline constexpr scalar_t * ptrRestrict & z1Ref(const host::label_t i) noexcept
            {
                return Front_.ptrRef(i);
            }

        private:
            /**
             * @brief Halo face arrays
             **/
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> West_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> East_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> South_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> North_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> Back_;
            device::array<field::SKELETON, scalar_t, VelocitySet, time::instantaneous> Front_;

            /**
             * @brief Initialize population data for a specific halo face
             * @tparam alpha The axis direction (X, Y or Z)
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
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                std::vector<scalar_t> face(mesh.nFaces<alpha, VelocitySet::QF()>(), 0);

                // Loop over all the blocks of the global domain
                for (host::label_t bz = 0; bz < mesh.nBlocks<axis::Z>(); bz++)
                {
                    for (host::label_t by = 0; by < mesh.nBlocks<axis::Y>(); by++)
                    {
                        for (host::label_t bx = 0; bx < mesh.nBlocks<axis::X>(); bx++)
                        {
                            // Handle the ghost cells for this block
                            const host::blockLabel Bx(bx, by, bz);
                            handleGhostCells<alpha, coeff>(
                                face,
                                rho,
                                u, v, w,
                                m_xx, m_xy, m_xz, m_yy, m_yz, m_zz,
                                Bx, mesh);
                        }
                    }
                }

                return face;
            }

            template <const axis::type alpha, const int coeff>
            __host__ [[nodiscard]] const std::vector<scalar_t> initialise_pop(
                const host::scalarField<host::PAGED, VelocitySet, time::instantaneous> &rho,
                const host::vectorField<host::PAGED, VelocitySet, time::instantaneous> &U,
                const host::symmetricTensorField<host::PAGED, VelocitySet, time::instantaneous> &Pi,
                const host::latticeMesh &mesh) const noexcept
            {
                return initialise_pop<alpha, coeff>(
                    rho.self(),
                    U.x(), U.y(), U.z(),
                    Pi.xx(), Pi.xy(), Pi.xz(), Pi.yy(), Pi.yz(), Pi.zz(),
                    mesh);
            }

            /**
             * @brief Populate halo face with population data from boundary cells
             * @tparam alpha The axis direction (X, Y or Z)
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
                const host::blockLabel &Bx,
                const host::latticeMesh &mesh) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                constexpr const thread::array<host::label_t, VelocitySet::QF()> indices = velocitySet::template indices_on_face<VelocitySet, alpha, coeff>();

                // For all velocity coefficients on this face
                for (host::label_t i = 0; i < VelocitySet::QF(); i++)
                {
                    // Second perpendicular axis
                    for (host::label_t tb = 0; tb < block::n<axis::orthogonal<alpha, 1>(), host::label_t>(); tb++)
                    {
                        // First perpendicular axis
                        for (host::label_t ta = 0; ta < block::n<axis::orthogonal<alpha, 0>(), host::label_t>(); ta++)
                        {
                            // Recover the 3D thread coordinates from the orthogonals, the axis and the normal coefficient
                            const host::blockLabel Tx = axis::to_3d<alpha, coeff>(ta, tb);

                            const host::label_t base = host::idx(Tx, Bx, mesh);

                            const thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(
                                {rho[base] + rho0(),
                                 u[base],
                                 v[base],
                                 w[base],
                                 m_xx[base],
                                 m_xy[base],
                                 m_xz[base],
                                 m_yy[base],
                                 m_yz[base],
                                 m_zz[base]});

                            const host::label_t j = host::idxPop<alpha, VelocitySet::QF()>(i, Tx, Bx, mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());

                            if (j >= face.size())
                            {
                                std::cout << j << " is greater than " << face.size() << std::endl;
                            }
                            face[j] = pop[indices[i]];
                        }
                    }
                }
            }
        };
    }
}

#endif