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
    rhoMean.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_RHOMEAN_CUH
#define __MBLBM_RHOMEAN_CUH

namespace LBM
{
    namespace functionObjects
    {
        struct rho
        {
            /**
             * @brief Number of components of the velocity vector
             **/
            static constexpr const host::label_t N = 1;

            /**
             * @brief Name of the variable
             **/
            static constexpr const char *name = "rho";

            /**
             * @brief Reads the moments
             * @param[in] devPtrs Device pointer collection containing velocity and moment fields
             * @param[in] idx Spatial index
             * @return The moments
             **/
            __device__ [[nodiscard]] static inline constexpr const scalar calculate(
                const device::ptrColl_t &devPtrs,
                const device::label_t idx) noexcept
            {
                return read_from_moments<0>(devPtrs, idx);
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

        namespace rhoDetail
        {
            using This = rho;

#include "commonKernelDefinitions.cuh"
        }

        /**
         * @brief Class for managing total velocity vector calculations in LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         **/
        template <class VelocitySet>
        class density : public FunctionObjectBase<rho::N>
        {
        public:
            /**
             * @brief Alias for the base type
             **/
            using ObjectType = rho;
            using BaseType = FunctionObjectBase<ObjectType::N>;
            using Kernel = rhoDetail::kernel;

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
            __host__ [[nodiscard]] density(
                const host::latticeMesh &mesh,
                const kernel::ptrCollection &devPtrs,
                const programControl &programCtrl) noexcept
                : BaseType(ObjectType::name, mesh, devPtrs, programCtrl),
                  rhoMean_(nameMean_, name_, mesh, programCtrl, (BaseType::doMean() || BaseType::doPrime() || BaseType::doPrimeSqMean())),
                  rhoPrime_(namePrime_, mesh, zeros<scalar_t, 1>(), programCtrl, BaseType::doPrime()),
                  rhoPrimeSqMean_(namePrimeSqMean_, mesh, zeros<scalar_t, 1>(), programCtrl, BaseType::doPrimeSqMean())
            {
                BaseType::template configure<Kernel>(programCtrl);
            }

            /**
             * @brief Disable copying
             **/
            __host__ ~density() {}
            __host__ [[nodiscard]] density(const density &) = delete;
            __host__ [[nodiscard]] density &operator=(const density &) = delete;

            /**
             * @brief Calculate the time-averaged density
             **/
            __host__ void calculateMean() noexcept
            {
                BaseType::mean(*this, rhoMean_.meanCountRef());
            }

            /**
             * @brief Calculate the perturbation of the density
             **/
            __host__ void calculatePrime() noexcept
            {
                BaseType::prime(*this);
            }

            /**
             * @brief Calculate the time-averaged square of the perturbation of the density
             **/
            __host__ void calculatePrimeSqMean() noexcept
            {
                BaseType::primeSqMean(*this, rhoPrimeSqMean_.meanCountRef());
            }

            /**
             * @brief Save the time-averaged density to a file
             * @param[in] hostWriteBuffer Host buffer used for copying data from the device before writing.
             * @param[in] timeStep The current time step for saving the turbulence statistics
             **/
            __host__ void saveMean(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                rhoMean_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the time-averaged density to a file
             * @param[in] hostWriteBuffer Host buffer used for copying data from the device before writing.
             * @param[in] timeStep The current time step for saving the turbulence statistics
             **/
            __host__ void savePrime(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                rhoPrime_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Save the time average of the square of the perturbation of the density to a file
             * @param[in] hostWriteBuffer Host buffer used for copying data from the device before writing.
             * @param[in] timeStep The current time step for saving the turbulence statistics
             **/
            __host__ void savePrimeSqMean(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
            {
                rhoPrimeSqMean_.template save<postProcess::LBMBin>(hostWriteBuffer, timeStep);
            }

            /**
             * @brief Access to the pointers of the time averaged field
             * @param[in] idx Memory index
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> meanPtrs(const host::label_t idx) noexcept { return {rhoMean_.ptr(idx)}; }

            /**
             * @brief Access to the pointers of the perturbation field
             * @param[in] idx Memory index
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> primePtrs(const host::label_t idx) noexcept { return {rhoPrime_.ptr(idx)}; }

            /**
             * @brief Access to the pointers of the mean of the square of the perturbation field
             * @param[in] idx Memory index
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<ObjectType::N, scalar_t> primeSqMeanPtrs(const host::label_t idx) noexcept { return {rhoPrimeSqMean_.ptr(idx)}; }

        private:
            /**
             * @brief Time-averaged density field
             **/
            device::scalarField<VelocitySet, time::timeAverage> rhoMean_;

            /**
             * @brief Perturbation of the density field
             **/
            device::scalarField<VelocitySet, time::instantaneous> rhoPrime_;

            /**
             * @brief Time average of the square of the perturbation of the density field
             **/
            device::scalarField<VelocitySet, time::timeAverage> rhoPrimeSqMean_;
        };
    }
}

#endif