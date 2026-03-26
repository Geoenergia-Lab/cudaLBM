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
Authors: Gustavo Choiare (Geoenergia Lab, UDESC)

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
    kineticEnergy.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_KINETICENERGY_CUH
#define __MBLBM_KINETICENERGY_CUH

namespace LBM
{
    namespace functionObjects
    {
        struct k
        {
            /**
             * @brief Number of components of the kinetic energy
             **/
            static constexpr const host::label_t N = 1;

            /**
             * @brief Calculates the total kinetic energy
             * @param[in] devPtrs Device pointer collection containing velocity and moment fields
             * @param[in] idx Spatial index
             * @return The calculated total kinetic energy
             **/
            __device__ [[nodiscard]] static inline constexpr const thread::array<scalar_t, N> calculate(
                const device::ptrCollection<10, const scalar_t> &devPtrs,
                const device::label_t idx) noexcept
            {
                const thread::array<scalar_t, 3> U = read_from_moments<index::u, index::v, index::w>(devPtrs, idx);

                if constexpr (std::is_same_v<scalar_t, float>)
                {
                    return sqrtf((U[0] * U[0]) + (U[1] * U[1]) + (U[2] * U[2])) * static_cast<scalar_t>(0.5);
                }

                if constexpr (std::is_same_v<scalar_t, double>)
                {
                    return sqrt((U[0] * U[0]) + (U[1] * U[1]) + (U[2] * U[2])) * static_cast<scalar_t>(0.5);
                }
            }

            __host__ [[nodiscard]] static inline consteval host::label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
        };

        namespace kineticEnergyDetail
        {
            using This = k;

#include "commonKernelDefinitions.cuh"
        }

        /**
         * @brief Class for managing total kinetic energy scalar calculations in LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         **/
        template <class VelocitySet>
        class kineticEnergy : public FunctionObjectBase<VelocitySet, k::N>
        {
        public:
            using BaseType = FunctionObjectBase<VelocitySet, k::N>;

            /**
             * @brief Constructs a kinetic energy object
             * @param[in] hostWriteBuffer Reference to the host-side write buffer
             * @param[in] mesh The lattice mesh
             * @param[in] rho, ... References to the 10 solution variables
             * @param[in] streamsLBM Stream handler for LBM operations
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] kineticEnergy(
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
                      "k",
                      hostWriteBuffer, mesh, rho, u, v, w,
                      mxx, mxy, mxz, myy, myz, mzz, streamsLBM),
                  k_(objectAllocator<VelocitySet, time::instantaneous>(name_, mesh, calculate_, programCtrl)),
                  kMean_(objectAllocator<VelocitySet, time::timeAverage>(nameMean_, mesh, calculateMean_, programCtrl))
            {
                programCtrl.configure<0, false>(kineticEnergyDetail::instantaneous);
                programCtrl.configure<0, false>(kineticEnergyDetail::instantaneousAndMean);
                programCtrl.configure<0, false>(kineticEnergyDetail::mean);
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
            ~kineticEnergy() {};
            __host__ [[nodiscard]] kineticEnergy(const kineticEnergy &) = delete;
            __host__ [[nodiscard]] kineticEnergy &operator=(const kineticEnergy &) = delete;

            /**
             * @brief Calculate the instantaneous kinetic energy
             **/
            __host__ void calculateInstantaneous() noexcept
            {
                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); ++stream)
                {
                    kineticEnergyDetail::instantaneous<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                        {rho_.ptr(stream), u_.ptr(stream), v_.ptr(stream), w_.ptr(stream),
                         mxx_.ptr(stream), mxy_.ptr(stream), mxz_.ptr(stream),
                         myy_.ptr(stream), myz_.ptr(stream), mzz_.ptr(stream)},
                        {k_.ptr(stream)});
                }
            }

            /**
             * @brief Calculate the time-averaged kinetic energy
             **/
            __host__ void calculateMean() noexcept
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(kMean_.meanCount() + 1);
                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); ++stream)
                {
                    kineticEnergyDetail::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                        {rho_.ptr(stream), u_.ptr(stream), v_.ptr(stream), w_.ptr(stream),
                         mxx_.ptr(stream), mxy_.ptr(stream), mxz_.ptr(stream),
                         myy_.ptr(stream), myz_.ptr(stream), mzz_.ptr(stream)},
                        {kMean_.ptr(stream)},
                        invNewCount);
                }
                kMean_.meanCountRef()++;
            }

            /**
             * @brief Calculate both the instantaneous and time-averaged kinetic energy
             **/
            __host__ void calculateInstantaneousAndMean() noexcept
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(kMean_.meanCount() + 1);
                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); ++stream)
                {
                    kineticEnergyDetail::instantaneousAndMean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                        {rho_.ptr(stream), u_.ptr(stream), v_.ptr(stream), w_.ptr(stream),
                         mxx_.ptr(stream), mxy_.ptr(stream), mxz_.ptr(stream),
                         myy_.ptr(stream), myz_.ptr(stream), mzz_.ptr(stream)},
                        {k_.ptr(stream)},
                        {kMean_.ptr(stream)},
                        invNewCount);
                }
                kMean_.meanCountRef()++;
            }

            /**
             * @brief Save the instantaneous kinetic energy to a file
             **/
            __host__ void saveInstantaneous(const host::label_t timeStep) noexcept
            {
                BaseType::saveInstantaneous(
                    timeStep,
                    name_,
                    componentNames_,
                    k_.programCtrl().deviceList().size(),
                    k_);
            }

            /**
             * @brief Save the time-averaged kinetic energy to a file
             **/
            __host__ void saveMean(const host::label_t timeStep) noexcept
            {
                BaseType::saveMean(
                    timeStep,
                    nameMean_,
                    componentNamesMean_,
                    kMean_.programCtrl().deviceList().size(),
                    kMean_.meanCount(),
                    kMean_);
            }

        private:
            /**
             * @brief Instantaneous kinetic energy
             **/
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> k_;

            /**
             * @brief Time-averaged kinetic energy
             **/
            device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> kMean_;
        };
    }
}

#endif