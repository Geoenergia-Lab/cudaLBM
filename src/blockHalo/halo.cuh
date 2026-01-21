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
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         *
         * This class handles the exchange of distribution functions between adjacent
         * CUDA blocks during LBM simulations. It maintains double-buffered halo regions
         * to support efficient ping-pong swapping between computation steps.
         **/
        template <class VelocitySet, const bool xPeriodic, const bool yPeriodic>
        class halo
        {
        public:
            /**
             * @brief Constructs halo regions from moment data and mesh
             * @param[in] mesh Lattice mesh defining simulation domain
             * @param[in] programCtrl Program control parameters
             **/
            __host__ [[nodiscard]] halo(
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : fGhost_(haloFace<VelocitySet>(
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl),
                      mesh)),
                  gGhost_(haloFace<VelocitySet>(
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl),
                      host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl),
                      mesh)){};

            /**
             * @brief Constructs halo regions from moment data and mesh
             * @param[in] rho,u,v,w,m_xx,m_xy,m_xz,m_yy,m_yz,m_zz Moment representation of distribution functions
             * @param[in] mesh Lattice mesh defining simulation domain
             **/
            __host__ [[nodiscard]] halo(
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
                : fGhost_(haloFace<VelocitySet>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)),
                  gGhost_(haloFace<VelocitySet>(rho, u, v, w, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz, mesh)){};

            /**
             * @brief Default destructor
             **/
            ~halo() {};

            /**
             * @brief Swaps read and write halo buffers
             * @note Synchronizes device before swapping to ensure all operations complete
             **/
            __host__ inline void swap() noexcept
            {
                checkCudaErrorsInline(cudaDeviceSynchronize());
                std::swap(fGhost_.x0Ref(), gGhost_.x0Ref());
                std::swap(fGhost_.x1Ref(), gGhost_.x1Ref());
                std::swap(fGhost_.y0Ref(), gGhost_.y0Ref());
                std::swap(fGhost_.y1Ref(), gGhost_.y1Ref());
                std::swap(fGhost_.z0Ref(), gGhost_.z0Ref());
                std::swap(fGhost_.z1Ref(), gGhost_.z1Ref());
            }

            /**
             * @brief Provides read-only access to the current read halo
             * @return Collection of const pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> fGhost() const noexcept
            {
                return {fGhost_.x0Const(), fGhost_.x1Const(), fGhost_.y0Const(), fGhost_.y1Const(), fGhost_.z0Const(), fGhost_.z1Const()};
            }

            /**
             * @brief Provides mutable access to the current write halo
             * @return Collection of mutable pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> gGhost() noexcept
            {
                return {gGhost_.x0(), gGhost_.x1(), gGhost_.y0(), gGhost_.y1(), gGhost_.z0(), gGhost_.z1()};
            }

            /**
             * @brief Loads halo population data from neighboring blocks
             * @param[out] pop Array to store loaded population values
             * @param[in] fGhost Collection of pointers to the halo faces
             *
             * This device function loads population values from neighboring blocks'
             * halo regions based on the current thread's position within its block.
             * It handles all 18 directions of the D3Q19 lattice model.
             **/
            __device__ static inline void load(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &fGhost) noexcept
            {
                const label_t tx = threadIdx.x;
                const label_t ty = threadIdx.y;
                const label_t tz = threadIdx.z;

                const label_t bx = blockIdx.x;
                const label_t by = blockIdx.y;
                const label_t bz = blockIdx.z;

                const label_t txp1 = (tx + 1 + block::nx()) % block::nx();
                const label_t txm1 = (tx - 1 + block::nx()) % block::nx();

                const label_t typ1 = (ty + 1 + block::ny()) % block::ny();
                const label_t tym1 = (ty - 1 + block::ny()) % block::ny();

                const label_t tzp1 = (tz + 1 + block::nz()) % block::nz();
                const label_t tzm1 = (tz - 1 + block::nz()) % block::nz();

                const label_t bxm1 = (bx - 1 + device::NUM_BLOCK_X) % device::NUM_BLOCK_X;
                const label_t bxp1 = (bx + 1 + device::NUM_BLOCK_X) % device::NUM_BLOCK_X;

                const label_t bym1 = (by - 1 + device::NUM_BLOCK_Y) % device::NUM_BLOCK_Y;
                const label_t byp1 = (by + 1 + device::NUM_BLOCK_Y) % device::NUM_BLOCK_Y;

                const label_t bzm1 = (bz - 1 + device::NUM_BLOCK_Z) % device::NUM_BLOCK_Z;
                const label_t bzp1 = (bz + 1 + device::NUM_BLOCK_Z) % device::NUM_BLOCK_Z;

                if (tx == 0)
                { // w
                    pop[q_i<1>()] = __ldg(&fGhost.ptr<1>()[idxPopX<0, VelocitySet::QF()>(ty, tz, bxm1, by, bz)]);
                    pop[q_i<7>()] = __ldg(&fGhost.ptr<1>()[idxPopX<1, VelocitySet::QF()>(tym1, tz, bxm1, ((ty == 0) ? bym1 : by), bz)]);
                    pop[q_i<9>()] = __ldg(&fGhost.ptr<1>()[idxPopX<2, VelocitySet::QF()>(ty, tzm1, bxm1, by, ((tz == 0) ? bzm1 : bz))]);
                    pop[q_i<13>()] = __ldg(&fGhost.ptr<1>()[idxPopX<3, VelocitySet::QF()>(typ1, tz, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)]);
                    pop[q_i<15>()] = __ldg(&fGhost.ptr<1>()[idxPopX<4, VelocitySet::QF()>(ty, tzp1, bxm1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        pop[q_i<19>()] = __ldg(&fGhost.ptr<1>()[idxPopX<5, VelocitySet::QF()>(tym1, tzm1, bxm1, ((ty == 0) ? bym1 : by), ((tz == 0) ? bzm1 : bz))]);
                        pop[q_i<21>()] = __ldg(&fGhost.ptr<1>()[idxPopX<6, VelocitySet::QF()>(tym1, tzp1, bxm1, ((ty == 0) ? bym1 : by), ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                        pop[q_i<23>()] = __ldg(&fGhost.ptr<1>()[idxPopX<7, VelocitySet::QF()>(typ1, tzm1, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), ((tz == 0) ? bzm1 : bz))]);
                        pop[q_i<26>()] = __ldg(&fGhost.ptr<1>()[idxPopX<8, VelocitySet::QF()>(typ1, tzp1, bxm1, ((ty == (block::ny() - 1)) ? byp1 : by), ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    }
                }
                else if (tx == (block::nx() - 1))
                { // e
                    pop[q_i<2>()] = __ldg(&fGhost.ptr<0>()[idxPopX<0, VelocitySet::QF()>(ty, tz, bxp1, by, bz)]);
                    pop[q_i<8>()] = __ldg(&fGhost.ptr<0>()[idxPopX<1, VelocitySet::QF()>(typ1, tz, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), bz)]);
                    pop[q_i<10>()] = __ldg(&fGhost.ptr<0>()[idxPopX<2, VelocitySet::QF()>(ty, tzp1, bxp1, by, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    pop[q_i<14>()] = __ldg(&fGhost.ptr<0>()[idxPopX<3, VelocitySet::QF()>(tym1, tz, bxp1, ((ty == 0) ? bym1 : by), bz)]);
                    pop[q_i<16>()] = __ldg(&fGhost.ptr<0>()[idxPopX<4, VelocitySet::QF()>(ty, tzm1, bxp1, by, ((tz == 0) ? bzm1 : bz))]);
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        pop[q_i<20>()] = __ldg(&fGhost.ptr<0>()[idxPopX<5, VelocitySet::QF()>(typ1, tzp1, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                        pop[q_i<22>()] = __ldg(&fGhost.ptr<0>()[idxPopX<6, VelocitySet::QF()>(typ1, tzm1, bxp1, ((ty == (block::ny() - 1)) ? byp1 : by), ((tz == 0) ? bzm1 : bz))]);
                        pop[q_i<24>()] = __ldg(&fGhost.ptr<0>()[idxPopX<7, VelocitySet::QF()>(tym1, tzp1, bxp1, ((ty == 0) ? bym1 : by), ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                        pop[q_i<25>()] = __ldg(&fGhost.ptr<0>()[idxPopX<8, VelocitySet::QF()>(tym1, tzm1, bxp1, ((ty == 0) ? bym1 : by), ((tz == 0) ? bzm1 : bz))]);
                    }
                }

                if (ty == 0)
                { // s
                    pop[q_i<3>()] = __ldg(&fGhost.ptr<3>()[idxPopY<0, VelocitySet::QF()>(tx, tz, bx, bym1, bz)]);
                    pop[q_i<7>()] = __ldg(&fGhost.ptr<3>()[idxPopY<1, VelocitySet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), bym1, bz)]);
                    pop[q_i<11>()] = __ldg(&fGhost.ptr<3>()[idxPopY<2, VelocitySet::QF()>(tx, tzm1, bx, bym1, ((tz == 0) ? bzm1 : bz))]);
                    pop[q_i<14>()] = __ldg(&fGhost.ptr<3>()[idxPopY<3, VelocitySet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, bz)]);
                    pop[q_i<17>()] = __ldg(&fGhost.ptr<3>()[idxPopY<4, VelocitySet::QF()>(tx, tzp1, bx, bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        pop[q_i<19>()] = __ldg(&fGhost.ptr<3>()[idxPopY<5, VelocitySet::QF()>(txm1, tzm1, ((tx == 0) ? bxm1 : bx), bym1, ((tz == 0) ? bzm1 : bz))]);
                        pop[q_i<21>()] = __ldg(&fGhost.ptr<3>()[idxPopY<6, VelocitySet::QF()>(txm1, tzp1, ((tx == 0) ? bxm1 : bx), bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                        pop[q_i<24>()] = __ldg(&fGhost.ptr<3>()[idxPopY<7, VelocitySet::QF()>(txp1, tzp1, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                        pop[q_i<25>()] = __ldg(&fGhost.ptr<3>()[idxPopY<8, VelocitySet::QF()>(txp1, tzm1, ((tx == (block::nx() - 1)) ? bxp1 : bx), bym1, ((tz == 0) ? bzm1 : bz))]);
                    }
                }
                else if (ty == (block::ny() - 1))
                { // n
                    pop[q_i<4>()] = __ldg(&fGhost.ptr<2>()[idxPopY<0, VelocitySet::QF()>(tx, tz, bx, byp1, bz)]);
                    pop[q_i<8>()] = __ldg(&fGhost.ptr<2>()[idxPopY<1, VelocitySet::QF()>(txp1, tz, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, bz)]);
                    pop[q_i<12>()] = __ldg(&fGhost.ptr<2>()[idxPopY<2, VelocitySet::QF()>(tx, tzp1, bx, byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    pop[q_i<13>()] = __ldg(&fGhost.ptr<2>()[idxPopY<3, VelocitySet::QF()>(txm1, tz, ((tx == 0) ? bxm1 : bx), byp1, bz)]);
                    pop[q_i<18>()] = __ldg(&fGhost.ptr<2>()[idxPopY<4, VelocitySet::QF()>(tx, tzm1, bx, byp1, ((tz == 0) ? bzm1 : bz))]);
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        pop[q_i<20>()] = __ldg(&fGhost.ptr<2>()[idxPopY<5, VelocitySet::QF()>(txp1, tzp1, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                        pop[q_i<22>()] = __ldg(&fGhost.ptr<2>()[idxPopY<6, VelocitySet::QF()>(txp1, tzm1, ((tx == (block::nx() - 1)) ? bxp1 : bx), byp1, ((tz == 0) ? bzm1 : bz))]);
                        pop[q_i<23>()] = __ldg(&fGhost.ptr<2>()[idxPopY<7, VelocitySet::QF()>(txm1, tzm1, ((tx == 0) ? bxm1 : bx), byp1, ((tz == 0) ? bzm1 : bz))]);
                        pop[q_i<26>()] = __ldg(&fGhost.ptr<2>()[idxPopY<8, VelocitySet::QF()>(txm1, tzp1, ((tx == 0) ? bxm1 : bx), byp1, ((tz == (block::nz() - 1)) ? bzp1 : bz))]);
                    }
                }

                if (tz == 0)
                { // b
                    pop[q_i<5>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bzm1)]);
                    pop[q_i<9>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<1, VelocitySet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzm1)]);
                    pop[q_i<11>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<2, VelocitySet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzm1)]);
                    pop[q_i<16>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<3, VelocitySet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzm1)]);
                    pop[q_i<18>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<4, VelocitySet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)]);
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        pop[q_i<19>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<5, VelocitySet::QF()>(txm1, tym1, ((tx == 0) ? bxm1 : bx), ((ty == 0) ? bym1 : by), bzm1)]);
                        pop[q_i<22>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<6, VelocitySet::QF()>(txp1, typ1, ((tx == (block::nx() - 1)) ? bxp1 : bx), ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)]);
                        pop[q_i<23>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<7, VelocitySet::QF()>(txm1, typ1, ((tx == 0) ? bxm1 : bx), ((ty == (block::ny() - 1)) ? byp1 : by), bzm1)]);
                        pop[q_i<25>()] = __ldg(&fGhost.ptr<5>()[idxPopZ<8, VelocitySet::QF()>(txp1, tym1, ((tx == (block::nx() - 1)) ? bxp1 : bx), ((ty == 0) ? bym1 : by), bzm1)]);
                    }
                }
                else if (tz == (block::nz() - 1))
                { // f
                    pop[q_i<6>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<0, VelocitySet::QF()>(tx, ty, bx, by, bzp1)]);
                    pop[q_i<10>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<1, VelocitySet::QF()>(txp1, ty, ((tx == (block::nx() - 1)) ? bxp1 : bx), by, bzp1)]);
                    pop[q_i<12>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<2, VelocitySet::QF()>(tx, typ1, bx, ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)]);
                    pop[q_i<15>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<3, VelocitySet::QF()>(txm1, ty, ((tx == 0) ? bxm1 : bx), by, bzp1)]);
                    pop[q_i<17>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<4, VelocitySet::QF()>(tx, tym1, bx, ((ty == 0) ? bym1 : by), bzp1)]);
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        pop[q_i<20>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<5, VelocitySet::QF()>(txp1, typ1, ((tx == (block::nx() - 1)) ? bxp1 : bx), ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)]);
                        pop[q_i<21>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<6, VelocitySet::QF()>(txm1, tym1, ((tx == 0) ? bxm1 : bx), ((ty == 0) ? bym1 : by), bzp1)]);
                        pop[q_i<24>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<7, VelocitySet::QF()>(txp1, tym1, ((tx == (block::nx() - 1)) ? bxp1 : bx), ((ty == 0) ? bym1 : by), bzp1)]);
                        pop[q_i<26>()] = __ldg(&fGhost.ptr<4>()[idxPopZ<8, VelocitySet::QF()>(txm1, typ1, ((tx == 0) ? bxm1 : bx), ((ty == (block::ny() - 1)) ? byp1 : by), bzp1)]);
                    }
                }
            }

            /**
             * @brief Transposes the block halo into the shared memory
             * @param[in] pop Array containing the populations for the particular thread
             * @param[out] s_buffer Shared array containing the packed population halos
             *
             * This device function saves population values to halo regions for
             * neighboring blocks to read.
             **/
            template <const label_t N>
            __device__ static inline void transpose_to_shared(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                thread::array<scalar_t, N> &s_buffer) noexcept
            {
                const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
                const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
                const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

                // Calculate base indices for each boundary type
                constexpr label_t x_size = block::ny() * block::nz();
                constexpr label_t y_size = block::nx() * block::nz();
                constexpr label_t z_size = block::nx() * block::ny();

                // West boundary (5 populations)
                if (West(x))
                {
                    const label_t base_idx = threadIdx.y + threadIdx.z * block::ny();
                    s_buffer[base_idx + (0 * x_size) + 0] = pop[q_i<2>()];
                    s_buffer[base_idx + (1 * x_size) + 0] = pop[q_i<8>()];
                    s_buffer[base_idx + (2 * x_size) + 0] = pop[q_i<10>()];
                    s_buffer[base_idx + (3 * x_size) + 0] = pop[q_i<14>()];
                    s_buffer[base_idx + (4 * x_size) + 0] = pop[q_i<16>()];
                }

                // East boundary (5 populations)
                if (East(x))
                {
                    const label_t base_idx = threadIdx.y + threadIdx.z * block::ny();
                    constexpr label_t east_offset = 5 * x_size;
                    s_buffer[east_offset + base_idx + (0 * x_size) + 0] = pop[q_i<1>()];
                    s_buffer[east_offset + base_idx + (1 * x_size) + 0] = pop[q_i<7>()];
                    s_buffer[east_offset + base_idx + (2 * x_size) + 0] = pop[q_i<9>()];
                    s_buffer[east_offset + base_idx + (3 * x_size) + 1] = pop[q_i<13>()];
                    s_buffer[east_offset + base_idx + (4 * x_size) + 1] = pop[q_i<15>()];
                }

                // South boundary (5 populations)
                if (South(y))
                {
                    const label_t base_idx = threadIdx.x + threadIdx.z * block::nx();
                    constexpr label_t south_offset = 10 * x_size;
                    s_buffer[south_offset + base_idx + (0 * y_size) + 1] = pop[q_i<4>()];
                    s_buffer[south_offset + base_idx + (1 * y_size) + 1] = pop[q_i<8>()];
                    s_buffer[south_offset + base_idx + (2 * y_size) + 1] = pop[q_i<12>()];
                    s_buffer[south_offset + base_idx + (3 * y_size) + 1] = pop[q_i<13>()];
                    s_buffer[south_offset + base_idx + (4 * y_size) + 1] = pop[q_i<18>()];
                }

                // North boundary (5 populations)
                if (North(y))
                {
                    const label_t base_idx = threadIdx.x + threadIdx.z * block::nx();
                    constexpr label_t north_offset = 10 * x_size + 5 * y_size;
                    s_buffer[north_offset + base_idx + (0 * y_size) + 1] = pop[q_i<3>()];
                    s_buffer[north_offset + base_idx + (1 * y_size) + 2] = pop[q_i<7>()];
                    s_buffer[north_offset + base_idx + (2 * y_size) + 2] = pop[q_i<11>()];
                    s_buffer[north_offset + base_idx + (3 * y_size) + 2] = pop[q_i<14>()];
                    s_buffer[north_offset + base_idx + (4 * y_size) + 2] = pop[q_i<17>()];
                }

                // Back boundary (5 populations)
                if (Back(z))
                {
                    const label_t base_idx = threadIdx.x + threadIdx.y * block::nx();
                    constexpr label_t back_offset = 10 * x_size + 10 * y_size;
                    s_buffer[back_offset + base_idx + (0 * z_size) + 2] = pop[q_i<6>()];
                    s_buffer[back_offset + base_idx + (1 * z_size) + 2] = pop[q_i<10>()];
                    s_buffer[back_offset + base_idx + (2 * z_size) + 2] = pop[q_i<12>()];
                    s_buffer[back_offset + base_idx + (3 * z_size) + 2] = pop[q_i<15>()];
                    s_buffer[back_offset + base_idx + (4 * z_size) + 3] = pop[q_i<17>()];
                }

                // Front boundary (5 populations)
                if (Front(z))
                {
                    const label_t base_idx = threadIdx.x + threadIdx.y * block::nx();
                    constexpr label_t front_offset = 10 * x_size + 10 * y_size + 5 * z_size;
                    s_buffer[front_offset + base_idx + (0 * z_size) + 3] = pop[q_i<5>()];
                    s_buffer[front_offset + base_idx + (1 * z_size) + 3] = pop[q_i<9>()];
                    s_buffer[front_offset + base_idx + (2 * z_size) + 3] = pop[q_i<11>()];
                    s_buffer[front_offset + base_idx + (3 * z_size) + 3] = pop[q_i<16>()];
                    s_buffer[front_offset + base_idx + (4 * z_size) + 3] = pop[q_i<18>()];
                }

                __syncthreads();
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @param[in] s_buffer Shared array containing the packed population halos
             * @param[out] gGhost Collection of pointers to the halo faces
             *
             * This device function saves population values to halo regions for
             * neighboring blocks to read.
             **/
            template <const label_t N>
            __device__ static inline void save_from_shared(
                const thread::array<scalar_t, N> &s_buffer,
                const device::ptrCollection<6, scalar_t> &gGhost) noexcept
            {
                const label_t warpId = warpID(threadIdx.x, threadIdx.y, threadIdx.z);
                const label_t offset = block::warp_size() * (warpId % 2);
                const label_t idx_in_warp = idxWarp(threadIdx.x, threadIdx.y, threadIdx.z);

                // Equivalent of threadIdx.alpha, threadIdx.beta
                const dim2 xy = ij<axis::X, axis::Y>(idx_in_warp + offset);
                const dim2 xz = ij<axis::X, axis::Z>(idx_in_warp + offset);
                const dim2 yz = ij<axis::Y, axis::Z>(idx_in_warp + offset);

                const label_t ID = idx_block(threadIdx.x, threadIdx.y, threadIdx.z);

                constexpr label_t padded_stride = block::size() + 1; // 513 instead of 512
                const scalar_t val0 = s_buffer[ID];
                const scalar_t val1 = s_buffer[ID + padded_stride];
                const scalar_t val2 = s_buffer[ID + (2 * padded_stride)];
                const scalar_t val3 = s_buffer[ID + (3 * padded_stride)];

                switch (warpId / 2)
                {
                case 0:
                {
                    gGhost.ptr<0>()[idxPopX<0, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<1>()[idxPopX<3, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val1;
                    gGhost.ptr<3>()[idxPopY<1, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
                    gGhost.ptr<4>()[idxPopZ<4, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

                    break;
                }
                case 1:
                {
                    gGhost.ptr<0>()[idxPopX<1, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<1>()[idxPopX<4, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val1;
                    gGhost.ptr<3>()[idxPopY<2, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
                    gGhost.ptr<5>()[idxPopZ<0, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

                    break;
                }
                case 2:
                {
                    gGhost.ptr<0>()[idxPopX<2, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<2>()[idxPopY<0, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
                    gGhost.ptr<3>()[idxPopY<3, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
                    gGhost.ptr<5>()[idxPopZ<1, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

                    break;
                }
                case 3:
                {
                    gGhost.ptr<0>()[idxPopX<3, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<2>()[idxPopY<1, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
                    gGhost.ptr<3>()[idxPopY<4, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val2;
                    gGhost.ptr<5>()[idxPopZ<2, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

                    break;
                }
                case 4:
                {
                    gGhost.ptr<0>()[idxPopX<4, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<2>()[idxPopY<2, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
                    gGhost.ptr<4>()[idxPopZ<0, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;
                    gGhost.ptr<5>()[idxPopZ<3, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

                    break;
                }
                case 5:
                {
                    gGhost.ptr<1>()[idxPopX<0, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<2>()[idxPopY<3, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
                    gGhost.ptr<4>()[idxPopZ<1, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;
                    gGhost.ptr<5>()[idxPopZ<4, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val3;

                    break;
                }
                case 6:
                {
                    gGhost.ptr<1>()[idxPopX<1, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<2>()[idxPopY<4, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
                    gGhost.ptr<4>()[idxPopZ<2, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;

                    break;
                }
                case 7:
                {
                    gGhost.ptr<1>()[idxPopX<2, VelocitySet::QF()>(yz.i, yz.j, blockIdx)] = val0;
                    gGhost.ptr<3>()[idxPopY<0, VelocitySet::QF()>(xz.i, xz.j, blockIdx)] = val1;
                    gGhost.ptr<4>()[idxPopZ<3, VelocitySet::QF()>(xy.i, xy.j, blockIdx)] = val2;

                    break;
                }
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
            __device__ static inline void save(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, scalar_t> &gGhost) noexcept
            {
                const label_t x = threadIdx.x + blockDim.x * blockIdx.x;
                const label_t y = threadIdx.y + blockDim.y * blockIdx.y;
                const label_t z = threadIdx.z + blockDim.z * blockIdx.z;

                /* write to global pop **/
                if (West(x))
                { // w
                    gGhost.ptr<0>()[idxPopX<0, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<2>()];
                    gGhost.ptr<0>()[idxPopX<1, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<8>()];
                    gGhost.ptr<0>()[idxPopX<2, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<10>()];
                    gGhost.ptr<0>()[idxPopX<3, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<14>()];
                    gGhost.ptr<0>()[idxPopX<4, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<16>()];
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        gGhost.ptr<0>()[idxPopX<5, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<20>()];
                        gGhost.ptr<0>()[idxPopX<6, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<22>()];
                        gGhost.ptr<0>()[idxPopX<7, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<24>()];
                        gGhost.ptr<0>()[idxPopX<8, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<25>()];
                    }
                }
                if (East(x))
                { // e
                    gGhost.ptr<1>()[idxPopX<0, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<1>()];
                    gGhost.ptr<1>()[idxPopX<1, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<7>()];
                    gGhost.ptr<1>()[idxPopX<2, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<9>()];
                    gGhost.ptr<1>()[idxPopX<3, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<13>()];
                    gGhost.ptr<1>()[idxPopX<4, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<15>()];
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        gGhost.ptr<1>()[idxPopX<5, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<19>()];
                        gGhost.ptr<1>()[idxPopX<6, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<21>()];
                        gGhost.ptr<1>()[idxPopX<7, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<23>()];
                        gGhost.ptr<1>()[idxPopX<8, VelocitySet::QF()>(threadIdx.y, threadIdx.z, blockIdx)] = pop[q_i<26>()];
                    }
                }

                if (South(y))
                { // s
                    gGhost.ptr<2>()[idxPopY<0, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<4>()];
                    gGhost.ptr<2>()[idxPopY<1, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<8>()];
                    gGhost.ptr<2>()[idxPopY<2, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<12>()];
                    gGhost.ptr<2>()[idxPopY<3, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<13>()];
                    gGhost.ptr<2>()[idxPopY<4, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<18>()];
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        gGhost.ptr<2>()[idxPopY<5, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<20>()];
                        gGhost.ptr<2>()[idxPopY<6, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<22>()];
                        gGhost.ptr<2>()[idxPopY<7, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<23>()];
                        gGhost.ptr<2>()[idxPopY<8, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<26>()];
                    }
                }
                if (North(y))
                { // n
                    gGhost.ptr<3>()[idxPopY<0, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<3>()];
                    gGhost.ptr<3>()[idxPopY<1, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<7>()];
                    gGhost.ptr<3>()[idxPopY<2, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<11>()];
                    gGhost.ptr<3>()[idxPopY<3, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<14>()];
                    gGhost.ptr<3>()[idxPopY<4, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<17>()];
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        gGhost.ptr<3>()[idxPopY<5, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<19>()];
                        gGhost.ptr<3>()[idxPopY<6, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<21>()];
                        gGhost.ptr<3>()[idxPopY<7, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<24>()];
                        gGhost.ptr<3>()[idxPopY<8, VelocitySet::QF()>(threadIdx.x, threadIdx.z, blockIdx)] = pop[q_i<25>()];
                    }
                }

                if (Back(z))
                { // b
                    gGhost.ptr<4>()[idxPopZ<0, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<6>()];
                    gGhost.ptr<4>()[idxPopZ<1, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<10>()];
                    gGhost.ptr<4>()[idxPopZ<2, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<12>()];
                    gGhost.ptr<4>()[idxPopZ<3, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<15>()];
                    gGhost.ptr<4>()[idxPopZ<4, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<17>()];
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        gGhost.ptr<4>()[idxPopZ<5, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<20>()];
                        gGhost.ptr<4>()[idxPopZ<6, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<21>()];
                        gGhost.ptr<4>()[idxPopZ<7, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<24>()];
                        gGhost.ptr<4>()[idxPopZ<8, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<26>()];
                    }
                }
                if (Front(z))
                {
                    gGhost.ptr<5>()[idxPopZ<0, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<5>()];
                    gGhost.ptr<5>()[idxPopZ<1, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<9>()];
                    gGhost.ptr<5>()[idxPopZ<2, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<11>()];
                    gGhost.ptr<5>()[idxPopZ<3, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<16>()];
                    gGhost.ptr<5>()[idxPopZ<4, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<18>()];
                    if constexpr (VelocitySet::Q() == 27)
                    {
                        gGhost.ptr<5>()[idxPopZ<5, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<19>()];
                        gGhost.ptr<5>()[idxPopZ<6, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<22>()];
                        gGhost.ptr<5>()[idxPopZ<7, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<23>()];
                        gGhost.ptr<5>()[idxPopZ<8, VelocitySet::QF()>(threadIdx.x, threadIdx.y, blockIdx)] = pop[q_i<25>()];
                    }
                }
            }

        private:
            /**
             * @brief The individual halo objects
             **/
            haloFace<VelocitySet> fGhost_;
            haloFace<VelocitySet> gGhost_;

            /**
             * @brief Check if current thread is at western block boundary
             * @param[in] x Global x-coordinate
             * @return True if at western boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool West(const label_t x) noexcept
            {
                if constexpr (xPeriodic)
                {
                    return (threadIdx.x == 0);
                }
                else
                {
                    return (threadIdx.x == 0 && x != 0);
                }
            }

            /**
             * @brief Check if current thread is at eastern block boundary
             * @param[in] x Global x-coordinate
             * @return True if at eastern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool East(const label_t x) noexcept
            {
                if constexpr (xPeriodic)
                {
                    return (threadIdx.x == (block::nx() - 1));
                }
                else
                {
                    return (threadIdx.x == (block::nx() - 1) && x != (device::nx - 1));
                }
            }

            /**
             * @brief Check if current thread is at southern block boundary
             * @param[in] y Global y-coordinate
             * @return True if at southern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool South(const label_t y) noexcept
            {
                if constexpr (yPeriodic)
                {
                    return (threadIdx.y == 0);
                }
                else
                {
                    return (threadIdx.y == 0 && y != 0);
                }
            }

            /**
             * @brief Check if current thread is at northern block boundary
             * @param[in] y Global y-coordinate
             * @return True if at northern boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool North(const label_t y) noexcept
            {
                if constexpr (yPeriodic)
                {
                    return (threadIdx.y == (block::ny() - 1));
                }
                else
                {
                    return (threadIdx.y == (block::ny() - 1) && y != (device::ny - 1));
                }
            }

            /**
             * @brief Check if current thread is at back (z-min) block boundary
             * @param[in] z Global z-coordinate
             * @return True if at back boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool Back(const label_t z) noexcept
            {
                return (threadIdx.z == 0 && z != 0);
            }

            /**
             * @brief Check if current thread is at front (z-max) block boundary
             * @param[in] z Global z-coordinate
             * @return True if at front boundary but not domain edge
             **/
            __device__ [[nodiscard]] static inline bool Front(const label_t z) noexcept
            {
                return (threadIdx.z == (block::nz() - 1) && z != (device::nz - 1));
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
            __device__ __host__ [[nodiscard]] static inline label_t idx_block(const label_t tx, const label_t ty, const label_t tz) noexcept
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
            __device__ __host__ [[nodiscard]] static inline label_t warpID(const label_t tx, const label_t ty, const label_t tz) noexcept
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
            __device__ __host__ [[nodiscard]] static inline label_t idxWarp(const label_t tx, const label_t ty, const label_t tz) noexcept
            {
                return idx_block(tx, ty, tz) % block::warp_size();
            }

            /**
             * @brief Computes the two-dimensional coordinate of a thread lying on a face
             * @tparam alpha The i-direction of the face
             * @tparam beta The j-direction of the face
             * @param[in] I The index of a thread within a warp
             * @return Two-dimensional representation of I
             **/
            template <const axis::direction alpha, const axis::direction beta>
            __device__ __host__ [[nodiscard]] static inline constexpr dim2 ij(const label_t I) noexcept
            {
                if constexpr ((alpha == axis::X) && (beta == axis::Y))
                {
                    return {I % (block::nx()), I / (block::nx())};
                }

                if constexpr ((alpha == axis::X) && (beta == axis::Z))
                {
                    return {I % (block::nx()), I / (block::nx())};
                }

                if constexpr ((alpha == axis::Y) && (beta == axis::Z))
                {
                    return {I % (block::ny()), I / (block::ny())};
                }
            }
        };
    }
}

#endif