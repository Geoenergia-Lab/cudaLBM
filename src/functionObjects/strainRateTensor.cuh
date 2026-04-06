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
    File containing kernels and class definitions for the strain rate tensor

Namespace
    LBM::functionObjects

SourceFiles
    strainRateTensor.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_STRAINRATETENSOR_CUH
#define __MBLBM_STRAINRATETENSOR_CUH

namespace LBM
{
    namespace functionObjects
    {
        struct S
        {
            /**
             * @brief Name of the variable
             **/
            static constexpr const char *name = "S";

            /**
             * @brief Number of components of the strain rate tensor
             **/
            static constexpr const host::label_t N = 6;

            /**
             * @brief Calculates a component of the strain rate tensor
             * @param[in] uAlpha Velocity component in the alpha direction
             * @param[in] uBeta Velocity component in the beta direction
             * @param[in] mAlphaBeta Second-order moment in the alpha/beta direction
             * @return The calculated component of the strain rate tensor
             **/
            template <const host::label_t Index>
            __device__ [[nodiscard]] static inline constexpr scalar_t calculate(const scalar_t uAlpha, const scalar_t uBeta, const scalar_t mAlphaBeta) noexcept
            {
                static_assert((Index == index::xx || Index == index::yy || Index == index::zz || Index == index::xy || Index == index::xz || Index == index::yz), "Invalid index");

                if constexpr (Index == index::xx || Index == index::yy || Index == index::zz)
                {
                    return velocitySet::as2<scalar_t>() * ((uAlpha * uBeta) - mAlphaBeta) / (static_cast<scalar_t>(2) * velocitySet::scale_ii<scalar_t>() * device::tau);
                }
                else
                {
                    return velocitySet::as2<scalar_t>() * ((uAlpha * uBeta) - mAlphaBeta) / (static_cast<scalar_t>(2) * velocitySet::scale_ij<scalar_t>() * device::tau);
                }
            }

            /**
             * @brief Calculates the strain rate tensor
             * @param[in] devPtrs Device pointer collection containing velocity and moment fields
             * @param[in] idx Spatial index
             * @return The calculated strain rate tensor
             **/
            __device__ [[nodiscard]] static inline constexpr const thread::array<scalar_t, N> calculate(
                const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
                const device::label_t idx) noexcept
            {
                const thread::array<scalar_t, 3> U = read_from_moments<index::u, index::v, index::w>(devPtrs, idx);

                const thread::array<scalar_t, 6> M = read_from_moments<index::xx, index::xy, index::xz, index::yy, index::yz, index::zz>(devPtrs, idx);

                return {calculate<index::xx>(U[0], U[0], M[0]), calculate<index::xy>(U[0], U[1], M[1]), calculate<index::xz>(U[0], U[2], M[2]), calculate<index::yy>(U[1], U[1], M[3]), calculate<index::yz>(U[1], U[2], M[4]), calculate<index::zz>(U[2], U[2], M[5])};
            }

            /**
             * @brief Number of blocks per streaming microprocessor
             **/
            static constexpr const host::label_t MIN_BLOCKS_PER_MP = 3;
        };

        namespace strainRateTensorDetail
        {
            using This = S;

#include "commonKernelDefinitions.cuh"
        }

        /**
         * @brief Class for managing strain rate tensor calculations in LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         **/
        template <class VelocitySet>
        class strainRateTensor : public FunctionObjectBase<VelocitySet, S::N>
        {
        public:
            /**
             * @brief Alias for the base type and required namespace
             **/
            using ObjectType = S;
            using BaseType = FunctionObjectBase<VelocitySet, ObjectType::N>;
            using Kernel = strainRateTensorDetail::kernel;

            /**
             * @brief Bring base members into scope
             **/
            using BaseType::calculate_;
            using BaseType::calculateMean_;
            using BaseType::componentNames_;
            using BaseType::componentNamesMean_;
            using BaseType::hostWriteBuffer_;
            using BaseType::mesh_;
            using BaseType::name_;
            using BaseType::nameMean_;
            using BaseType::Pi_;
            using BaseType::rho_;
            using BaseType::streamsLBM_;
            using BaseType::U_;

            /**
             * @brief Constructs a strain rate tensor object
             * @param[in] hostWriteBuffer Reference to the host-side write buffer
             * @param[in] mesh The lattice mesh
             * @param[in] rho, ... References to the 10 solution variables
             * @param[in] streamsLBM Stream handler for LBM operations
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] strainRateTensor(
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::latticeMesh &mesh,
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const streamHandler &streamsLBM,
                const programControl &programCtrl) noexcept
                : BaseType(ObjectType::name, hostWriteBuffer, mesh, rho, U, Pi, streamsLBM),
                  S_(name_, mesh_, 0, programCtrl, calculate_),
                  SMean_(nameMean_, mesh_, 0, programCtrl, calculateMean_)
            {
                BaseType::template configure<Kernel>(programCtrl);
            }

            /**
             * @brief Disable copying
             **/
            ~strainRateTensor() {}
            __host__ [[nodiscard]] strainRateTensor(const strainRateTensor &) = delete;
            __host__ [[nodiscard]] strainRateTensor &operator=(const strainRateTensor &) = delete;

            /**
             * @brief Calculate the instantaneous strain rate tensor
             **/
            __host__ void calculateInstantaneous() noexcept
            {
                BaseType::instantaneous(Kernel::instantaneous(), *this);
            }

            /**
             * @brief Calculate the time-averaged strain rate tensor
             **/
            __host__ void calculateMean() noexcept
            {
                BaseType::mean(Kernel::mean(), *this, SMean_.meanCountRef());
            }

            /**
             * @brief Calculate both the instantaneous and time-averaged strain rate tensor
             **/
            __host__ void calculateInstantaneousAndMean() noexcept
            {
                BaseType::instantaneousAndMean(Kernel::instantaneousAndMean(), *this, SMean_.meanCountRef());
            }

            /**
             * @brief Save the instantaneous strain rate tensor to a file
             **/
            __host__ void saveInstantaneous(const host::label_t timeStep) noexcept
            {
                S_.template save<postProcess::LBMBin>(hostWriteBuffer_, timeStep);
            }

            /**
             * @brief Save the time-averaged strain rate tensor to a file
             **/
            __host__ void saveMean(const host::label_t timeStep) noexcept
            {
                SMean_.template save<postProcess::LBMBin>(hostWriteBuffer_, timeStep);
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> instantaneousPtrs(const host::label_t idx) noexcept
            {
                return S_.ptr(idx);
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> meanPtrs(const host::label_t idx) noexcept
            {
                return SMean_.ptr(idx);
            }

        private:
            /**
             * @brief Instantaneous strain rate tensor
             **/
            device::symmetricTensorField<VelocitySet, time::instantaneous> S_;

            /**
             * @brief Time-averaged strain rate tensor
             **/
            device::symmetricTensorField<VelocitySet, time::timeAverage> SMean_;
        };
    }
}

#endif