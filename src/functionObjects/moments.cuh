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
    File containing kernels and class definitions for the kinetic energy

Namespace
    LBM::functionObjects

SourceFiles
    moments.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTS_CUH
#define __MBLBM_MOMENTS_CUH

namespace LBM
{
    namespace functionObjects
    {
        struct M
        {
            /**
             * @brief Number of moments
             **/
            static constexpr const host::label_t N = NUMBER_MOMENTS<host::label_t>();

            /**
             * @brief Reads the moments
             * @param[in] devPtrs Device pointer collection containing velocity and moment fields
             * @param[in] idx Spatial index
             * @return The moments
             **/
            __device__ [[nodiscard]] static inline constexpr const thread::array<scalar_t, N> calculate(
                const device::ptrCollection<10, const scalar_t> &devPtrs,
                const device::label_t idx) noexcept
            {
                return read_from_moments<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>(devPtrs, idx);
            }

            __host__ [[nodiscard]] static inline consteval host::label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
        };

        namespace momentsDetail
        {
            using This = M;

#include "commonKernelDefinitions.cuh"
        }

        /**
         * @brief Class for managing total kinetic energy scalar calculations in LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         **/
        template <class VelocitySet>
        class moments : public FunctionObjectBase<VelocitySet, M::N>
        {
        public:
            using BaseType = FunctionObjectBase<VelocitySet, M::N>;

            /**
             * @brief Constructs a collection of the solution variables object
             * @param[in] hostWriteBuffer Reference to the host-side write buffer
             * @param[in] mesh The lattice mesh
             * @param[in] rho, ... References to the 10 solution variables
             * @param[in] streamsLBM Stream handler for LBM operations
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] moments(
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::latticeMesh &mesh,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &rho,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &u,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &v,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &w,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxx,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxy,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxz,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myy,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myz,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mzz,
                const streamHandler &streamsLBM,
                const programControl &programCtrl) noexcept
                : BaseType(
                      "moments",
                      hostWriteBuffer, mesh, rho, u, v, w,
                      mxx, mxy, mxz, myy, myz, mzz, streamsLBM),
                  rhoMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_, programCtrl)),
                  uMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[1], mesh, calculateMean_, programCtrl)),
                  vMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[2], mesh, calculateMean_, programCtrl)),
                  wMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[3], mesh, calculateMean_, programCtrl)),
                  mxxMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[4], mesh, calculateMean_, programCtrl)),
                  mxyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[5], mesh, calculateMean_, programCtrl)),
                  mxzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[6], mesh, calculateMean_, programCtrl)),
                  myyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[7], mesh, calculateMean_, programCtrl)),
                  myzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[8], mesh, calculateMean_, programCtrl)),
                  mzzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[9], mesh, calculateMean_, programCtrl))
            {
                programCtrl.configure<0, false>(momentsDetail::instantaneous);
                programCtrl.configure<0, false>(momentsDetail::instantaneousAndMean);
                programCtrl.configure<0, false>(momentsDetail::mean);
            }

            /**
             * @brief Bring base members into scope
             **/
            using BaseType::calculate_;
            using BaseType::calculateMean_;
            using BaseType::componentNames_;
            using BaseType::componentNamesMean_;
            using BaseType::hostWriteBuffer_;
            using BaseType::mesh_;
            using BaseType::mxx_;
            using BaseType::mxy_;
            using BaseType::mxz_;
            using BaseType::myy_;
            using BaseType::myz_;
            using BaseType::mzz_;
            using BaseType::name_;
            using BaseType::nameMean_;
            using BaseType::rho_;
            using BaseType::streamsLBM_;
            using BaseType::u_;
            using BaseType::v_;
            using BaseType::w_;

            /**
             * @brief Disable copying
             **/
            ~moments() = default;
            __host__ [[nodiscard]] moments(const moments &) = delete;
            __host__ [[nodiscard]] moments &operator=(const moments &) = delete;

            /**
             * @brief Calculate the time-averaged moments
             **/
            __host__ void calculateMean() noexcept
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(rhoMean_.meanCount() + 1);
                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); ++stream)
                {
                    momentsDetail::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                        {rho_.ptr(stream), u_.ptr(stream), v_.ptr(stream), w_.ptr(stream),
                         mxx_.ptr(stream), mxy_.ptr(stream), mxz_.ptr(stream),
                         myy_.ptr(stream), myz_.ptr(stream), mzz_.ptr(stream)},
                        {rhoMean_.ptr(stream), uMean_.ptr(stream), vMean_.ptr(stream), wMean_.ptr(stream),
                         mxxMean_.ptr(stream), mxyMean_.ptr(stream), mxzMean_.ptr(stream),
                         myyMean_.ptr(stream), myzMean_.ptr(stream), mzzMean_.ptr(stream)},
                        invNewCount);
                }
                rhoMean_.meanCountRef()++;
            }

            /**
             * @brief Save the time-averaged moments to a file
             **/
            __host__ void saveMean(const host::label_t timeStep) noexcept
            {
                BaseType::saveMean(
                    timeStep,
                    nameMean_,
                    componentNamesMean_,
                    rhoMean_.programCtrl().deviceList().size(),
                    rhoMean_.meanCount(),
                    rhoMean_,
                    uMean_, vMean_, wMean_,
                    mxxMean_, mxyMean_, mxzMean_, myyMean_, myzMean_, mzzMean_);
            }

            /**
             * @brief Unused functions - this object does not calculate its instantaneous or mean
             **/
            __host__ static void calculateInstantaneous() noexcept {}
            __host__ static void calculateInstantaneousAndMean() noexcept {}
            __host__ static void saveInstantaneous([[maybe_unused]] const host::label_t timeStep) noexcept {}

        private:
            /**
             * @brief Time-averaged moments
             **/
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> rhoMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> uMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> vMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> wMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> mxxMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> mxyMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> mxzMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> myyMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> myzMean_;
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> mzzMean_;
        };
    }
}

#endif