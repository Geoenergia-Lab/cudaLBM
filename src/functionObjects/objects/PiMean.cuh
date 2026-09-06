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
Authors: Gustavo Choiare (Geoenergia Lab, UDESC)

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
    File containing kernels and class definitions for the velocity vector.

Namespace
    LBM::functionObjects

SourceFiles
    PiMean.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PIMEAN_CUH
#define __MBLBM_PIMEAN_CUH

namespace LBM
{
    namespace functionObjects
    {
        struct Pi
        {
            /**
             * @brief Number of components of the tensor of second order moments
             **/
            static constexpr const host::label_t N = 6;

            /**
             * @brief Name of the variable
             **/
            static constexpr const char *name = "Pi";

            /**
             * @brief Reads the moments
             * @param[in] devPtrs Device pointer collection containing velocity and moment fields
             * @param[in] idx Spatial index
             * @return The moments
             **/
            __device__ [[nodiscard]] static inline constexpr symmetricTensor calculate(
                const device::ptrColl_t &devPtrs,
                const device::label_t idx) noexcept
            {
                return read_from_moments<4, 5, 6, 7, 8, 9>(devPtrs, idx);
            }

            /**
             * @brief Number of blocks per streaming microprocessor
             **/
            static constexpr const host::label_t MIN_BLOCKS_PER_MP = 1;

            /**
             * @brief Switch that defines whether or not the class will define an instantaneous calculation
             **/
            static constexpr const bool canCalculateInstantaneous = false;
        };

        namespace PiDetail
        {
            using This = Pi;

#include "commonKernelDefinitions.cuh"
        }

        /**
         * @brief Class for managing total velocity vector calculations in LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         **/
        template <class VelocitySet>
        class secondOrderMoments : public FunctionObjectBase<Pi::N>
        {
        public:
            /**
             * @brief Alias for the base type
             **/
            using ObjectType = Pi;
            using BaseType = FunctionObjectBase<ObjectType::N>;
            using Kernel = PiDetail::kernel;

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
             * @brief Constructs a velocity vector object
             * @param[in] mesh The lattice mesh
             * @param[in] rho Device scalar field containing the density values on the GPU
             * @param[in] U Device vector field containing the velocity values on the GPU
             * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] secondOrderMoments(
                const host::latticeMesh &mesh,
                const kernel::ptrCollection &devPtrs,
                const programControl &programCtrl) noexcept
                : BaseType(ObjectType::name, mesh, devPtrs, programCtrl),
                  PiMean_(nameMean_, name_, mesh, programCtrl, (BaseType::doMean() || BaseType::doPrime() || BaseType::doPrimeSqMean())),
                  PiPrime_(namePrime_, mesh, zeros<scalar_t, 6>(), programCtrl, BaseType::doPrime()),
                  PiPrimeSqMean_(namePrimeSqMean_, mesh, zeros<scalar_t, 6>(), programCtrl, BaseType::doPrimeSqMean())
            {
                BaseType::template configure<Kernel>(programCtrl);
            }

            /**
             * @brief Disable copying
             **/
            __host__ ~secondOrderMoments() {}
            __host__ [[nodiscard]] secondOrderMoments(const secondOrderMoments &) = delete;
            __host__ [[nodiscard]] secondOrderMoments &operator=(const secondOrderMoments &) = delete;

            /**
             * @brief Calculate the time-averaged second order moments
             **/
            __host__ void calculateMean() noexcept
            {
                BaseType::mean(*this, PiMean_.meanCountRef());
            }

            /**
             * @brief Calculate the perturbation of the second order moments
             **/
            __host__ void calculatePrime() noexcept
            {
                BaseType::prime(*this);
            }

            /**
             * @brief Calculate the time-averaged square of the perturbation of the second order moments
             **/
            __host__ void calculatePrimeSqMean() noexcept
            {
                BaseType::primeSqMean(*this, PiPrimeSqMean_.meanCountRef());
            }

            /**
             * @brief Save the second order moments to a file
             **/
            __host__ void saveMean(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                PiMean_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the second order moments to a file
             **/
            __host__ void savePrime(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                PiPrime_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the time average of the square of the perturbation of the second order moments to a file
             **/
            __host__ void savePrimeSqMean(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                PiPrimeSqMean_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Access to the pointers of the underlying device fields
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> meanPtrs(const host::label_t idx) noexcept { return {PiMean_.ptr(idx)}; }
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> primePtrs(const host::label_t idx) noexcept { return {PiPrime_.ptr(idx)}; }
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> primeSqMeanPtrs(const host::label_t idx) noexcept { return {PiPrimeSqMean_.ptr(idx)}; }

        private:
            /**
             * @brief Time-averaged second order moments
             **/
            device::symmetricTensorField<VelocitySet, time::timeAverage> PiMean_;

            /**
             * @brief Time-averaged second order moments
             **/
            device::symmetricTensorField<VelocitySet, time::instantaneous> PiPrime_;

            /**
             * @brief Time average of the square of the perturbation of the second order moments
             **/
            device::symmetricTensorField<VelocitySet, time::timeAverage> PiPrimeSqMean_;
        };
    }
}

#endif