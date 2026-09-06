/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
                static_assert((Index == axis::index<axis::X, axis::X>() || Index == axis::index<axis::X, axis::Y>() || Index == axis::index<axis::X, axis::Z>() || Index == axis::index<axis::Y, axis::Y>() || Index == axis::index<axis::Y, axis::Z>() || Index == axis::index<axis::Z, axis::Z>()), "Invalid index");

                if constexpr (Index == axis::index<axis::X, axis::X>() || Index == axis::index<axis::Y, axis::Y>() || Index == axis::index<axis::Z, axis::Z>())
                {
                    return velocitySetBase::scale_ii<scalar_t>() * ((uAlpha * uBeta) - mAlphaBeta) / device::tau;
                }
                else
                {
                    return velocitySetBase::scale_ij<scalar_t>() * ((uAlpha * uBeta) - mAlphaBeta) / device::tau;
                }
            }

            /**
             * @brief Calculates the strain rate tensor
             * @param[in] devPtrs Device pointer collection containing velocity and moment fields
             * @param[in] idx Spatial index
             * @return The calculated strain rate tensor
             **/
            __device__ [[nodiscard]] static inline constexpr const symmetricTensor calculate(
                const device::ptrColl_t &devPtrs,
                const device::label_t idx) noexcept
            {
                const vector U = read_from_moments<axis::index<axis::X>(), axis::index<axis::Y>(), axis::index<axis::Z>()>(devPtrs, idx);

                const symmetricTensor M = read_from_moments<
                    axis::index<axis::X, axis::X>(),
                    axis::index<axis::X, axis::Y>(),
                    axis::index<axis::X, axis::Z>(),
                    axis::index<axis::Y, axis::Y>(),
                    axis::index<axis::Y, axis::Z>(),
                    axis::index<axis::Z, axis::Z>()>(devPtrs, idx);

                return {calculate<axis::index<axis::X, axis::X>()>(U[0], U[0], M[0]),
                        calculate<axis::index<axis::X, axis::Y>()>(U[0], U[1], M[1]),
                        calculate<axis::index<axis::X, axis::Z>()>(U[0], U[2], M[2]),
                        calculate<axis::index<axis::Y, axis::Y>()>(U[1], U[1], M[3]),
                        calculate<axis::index<axis::Y, axis::Z>()>(U[1], U[2], M[4]),
                        calculate<axis::index<axis::Z, axis::Z>()>(U[2], U[2], M[5])};
            }

            /**
             * @brief Number of blocks per streaming microprocessor
             **/
            static constexpr const host::label_t MIN_BLOCKS_PER_MP = 1;

            /**
             * @brief Switch that defines whether or not the class will define an instantaneous calculation
             **/
            static constexpr const bool canCalculateInstantaneous = true;
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
        class strainRateTensor : public FunctionObjectBase<S::N>
        {
        public:
            /**
             * @brief Alias for the base type and required namespace
             **/
            using ObjectType = S;
            using BaseType = FunctionObjectBase<ObjectType::N>;
            using Kernel = strainRateTensorDetail::kernel;

            /**
             * @brief Bring base members into scope
             **/
            using BaseType::calculate_;
            using BaseType::componentNames_;
            using BaseType::componentNamesMean_;
            using BaseType::mesh_;
            using BaseType::name_;
            using BaseType::nameMean_;
            using BaseType::namePrime_;
            using BaseType::namePrimeSqMean_;
            using BaseType::programCtrl_;

            /**
             * @brief Constructs a strain rate tensor object
             * @param[in] mesh The lattice mesh
             * @param[in] rho Device scalar field containing the density values on the GPU
             * @param[in] U Device vector field containing the velocity values on the GPU
             * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] strainRateTensor(
                const host::latticeMesh &mesh,
                const kernel::ptrCollection &devPtrs,
                const programControl &programCtrl) noexcept
                : BaseType(ObjectType::name, mesh, devPtrs, programCtrl),
                  S_(name_, mesh_, zeros<scalar_t, 6>(), programCtrl, calculate_),
                  SMean_(nameMean_, mesh_, zeros<scalar_t, 6>(), programCtrl, (BaseType::doMean() || BaseType::doPrime() || BaseType::doPrimeSqMean())),
                  SPrime_(namePrime_, mesh_, zeros<scalar_t, 6>(), programCtrl, BaseType::doPrime()),
                  SPrimeSqMean_(namePrimeSqMean_, mesh_, zeros<scalar_t, 6>(), programCtrl, BaseType::doPrimeSqMean())
            {
                BaseType::template configure<Kernel>(programCtrl);
            }

            /**
             * @brief Disable copying
             **/
            __host__ ~strainRateTensor() {}
            __host__ [[nodiscard]] strainRateTensor(const strainRateTensor &) = delete;
            __host__ [[nodiscard]] strainRateTensor &operator=(const strainRateTensor &) = delete;

            /**
             * @brief Calculate the instantaneous strain rate tensor
             **/
            __host__ void calculateInstantaneous() noexcept
            {
                BaseType::instantaneous(*this);
            }

            /**
             * @brief Calculate the time-averaged strain rate tensor
             **/
            __host__ void calculateMean() noexcept
            {
                BaseType::mean(*this, SMean_.meanCountRef());
            }

            /**
             * @brief Calculate the perturbation of the strain rate tensor
             **/
            __host__ void calculatePrime() noexcept
            {
                BaseType::prime(*this);
            }

            /**
             * @brief Calculate the time-averaged square of the perturbation of the strain rate tensor
             **/
            __host__ void calculatePrimeSqMean() noexcept
            {
                BaseType::primeSqMean(*this, SPrimeSqMean_.meanCountRef());
            }

            /**
             * @brief Calculate both the instantaneous and time-averaged strain rate tensor
             **/
            __host__ void calculateInstantaneousAndMean() noexcept
            {
                BaseType::instantaneousAndMean(*this, SMean_.meanCountRef());
            }

            /**
             * @brief Save the instantaneous strain rate tensor to a file
             **/
            __host__ void saveInstantaneous(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                S_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the time-averaged strain rate tensor to a file
             **/
            __host__ void saveMean(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                SMean_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the time-averaged strain rate tensor to a file
             **/
            __host__ void savePrime(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                SPrime_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the time average of the square of the perturbation of the strain rate tensor to a file
             **/
            __host__ void savePrimeSqMean(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                SPrimeSqMean_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Access to the pointers of the underlying device fields
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> instantaneousPtrs(const host::label_t idx) noexcept { return S_.ptr(idx); }
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> meanPtrs(const host::label_t idx) noexcept { return SMean_.ptr(idx); }
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> primePtrs(const host::label_t idx) noexcept { return SPrime_.ptr(idx); }
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> primeSqMeanPtrs(const host::label_t idx) noexcept { return {SPrimeSqMean_.ptr(idx)}; }

        private:
            /**
             * @brief Instantaneous strain rate tensor
             **/
            device::symmetricTensorField<VelocitySet, time::instantaneous> S_;

            /**
             * @brief Time-averaged strain rate tensor
             **/
            device::symmetricTensorField<VelocitySet, time::timeAverage> SMean_;

            /**
             * @brief Perturbation of the strain rate tensor
             **/
            device::symmetricTensorField<VelocitySet, time::instantaneous> SPrime_;

            /**
             * @brief Time average of the square of the perturbation of the strain rate tensor
             **/
            device::symmetricTensorField<VelocitySet, time::timeAverage> SPrimeSqMean_;
        };
    }
}

#endif
