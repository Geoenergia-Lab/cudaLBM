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
                const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
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

            /**
             * @brief Number of blocks per streaming microprocessor
             **/
            static constexpr const host::label_t MIN_BLOCKS_PER_MP = 3;
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
            /**
             * @brief Alias for the base type
             **/
            using ObjectType = k;
            using BaseType = FunctionObjectBase<VelocitySet, ObjectType::N>;
            using Kernel = kineticEnergyDetail::kernel;

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
                  k_(name_, mesh_, calculate_, programCtrl),
                  kMean_(nameMean_, mesh, calculateMean_, programCtrl)
            {
                BaseType::template configure<Kernel>(programCtrl);
            }

            /**
             * @brief Disable copying
             **/
            ~kineticEnergy() {}
            __host__ [[nodiscard]] kineticEnergy(const kineticEnergy &) = delete;
            __host__ [[nodiscard]] kineticEnergy &operator=(const kineticEnergy &) = delete;

            /**
             * @brief Calculate the instantaneous kinetic energy
             **/
            __host__ void calculateInstantaneous() noexcept
            {
                BaseType::instantaneous(Kernel::instantaneous(), *this);
            }

            /**
             * @brief Calculate the time-averaged kinetic energy
             **/
            __host__ void calculateMean() noexcept
            {
                BaseType::mean(Kernel::mean(), *this, kMean_.meanCountRef());
            }

            /**
             * @brief Calculate both the instantaneous and time-averaged kinetic energy
             **/
            __host__ void calculateInstantaneousAndMean() noexcept
            {
                BaseType::instantaneousAndMean(Kernel::instantaneousAndMean(), *this, kMean_.meanCountRef());
            }

            /**
             * @brief Save the instantaneous kinetic energy to a file
             **/
            __host__ void saveInstantaneous(const host::label_t timeStep) noexcept
            {
                BaseType::saveInstantaneous(timeStep, name_, componentNames_, k_.self().programCtrl().deviceList().size(), k_);
            }

            /**
             * @brief Save the time-averaged kinetic energy to a file
             **/
            __host__ void saveMean(const host::label_t timeStep) noexcept
            {
                BaseType::saveMean(timeStep, nameMean_, componentNamesMean_, kMean_.self().programCtrl().deviceList().size(), kMean_.meanCount(), kMean_);
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> instantaneousPtrs(const host::label_t idx) noexcept
            {
                return k_.ptr(idx);
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> meanPtrs(const host::label_t idx) noexcept
            {
                return {kMean_.ptr(idx)};
            }

        private:
            /**
             * @brief Instantaneous kinetic energy
             **/
            device::scalarField<VelocitySet, time::instantaneous> k_;

            /**
             * @brief Time-averaged kinetic energy
             **/
            device::scalarField<VelocitySet, time::instantaneous> kMean_;
        };
    }
}

#endif