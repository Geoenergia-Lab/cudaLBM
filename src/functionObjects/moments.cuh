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
        namespace moments
        {
            namespace kernel
            {
                __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
#define launchBounds __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

                /**
                 * @brief CUDA kernel for calculating time-averaged total kinetic energy
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] KMeanPtrs Device pointer collection for mean total kinetic energy
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void mean(
                    const device::ptrCollection<NUMBER_MOMENTS(), scalar_t> devPtrs,
                    const device::ptrCollection<NUMBER_MOMENTS(), scalar_t> devMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

                    // Read from global memory
                    thread::array<scalar_t, NUMBER_MOMENTS<std::size_t>()> m;
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto n)
                        {
                            m[n] = devPtrs.ptr<n>()[idx];
                        });

                    // Read the mean values from global memory
                    thread::array<scalar_t, NUMBER_MOMENTS<std::size_t>()> mMean;
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto n)
                        {
                            mMean[n] = devMeanPtrs.ptr<n>()[idx];
                        });

                    // Update the mean value and write back to global
                    const thread::array<scalar_t, NUMBER_MOMENTS<std::size_t>()> meanNew = timeAverage(mMean, m, invNewCount);
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto n)
                        {
                            devMeanPtrs.ptr<n>()[idx] = meanNew[n];
                        });
                }
            }

            /**
             * @brief Class for managing total kinetic energy scalar calculations in LBM simulations
             * @tparam VelocitySet The velocity set type used in LBM
             * @tparam N The number of streams (compile-time constant)
             **/
            template <class VelocitySet, const label_t N>
            class collection
            {
            public:
                /**
                 * @brief Constructs a total kinetic energy scalar object
                 * @param[in] mesh Reference to lattice mesh
                 * @param[in] devPtrs Device pointer collection for memory access
                 * @param[in] streamsLBM Stream handler for CUDA operations
                 **/
                __host__ [[nodiscard]] collection(
                    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                    const host::latticeMesh &mesh,
                    const device::ptrCollection<10, scalar_t> &devPtrs,
                    const streamHandler<N> &streamsLBM) noexcept
                    : hostWriteBuffer_(hostWriteBuffer),
                      mesh_(mesh),
                      devPtrs_(devPtrs),
                      streamsLBM_(streamsLBM),
                      calculateMean_(initialiserSwitch(fieldNameMean_)),
                      rhoMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_)),
                      uMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[1], mesh, calculateMean_)),
                      vMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[2], mesh, calculateMean_)),
                      wMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[3], mesh, calculateMean_)),
                      mxxMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[4], mesh, calculateMean_)),
                      mxyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[5], mesh, calculateMean_)),
                      mxzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[6], mesh, calculateMean_)),
                      myyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[7], mesh, calculateMean_)),
                      myzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[8], mesh, calculateMean_)),
                      mzzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[9], mesh, calculateMean_))
                {
                    // Set the cache config to prefer L1
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::mean, cudaFuncCachePreferL1));
                };

                /**
                 * @brief Default destructor
                 **/
                ~collection() {};

                /**
                 * @brief Check if instantaneous calculation is enabled
                 * @return True if instantaneous calculation is enabled
                 **/
                __host__ inline constexpr bool calculate() const noexcept
                {
                    return calculate_;
                }

                /**
                 * @brief Check if mean calculation is enabled
                 * @return True if mean calculation is enabled
                 **/
                __host__ inline constexpr bool calculateMean() const noexcept
                {
                    return calculateMean_;
                }

                /**
                 * @brief Calculate instantaneous total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneous([[maybe_unused]] const label_t timeStep) noexcept
                {
                    return;
                }

                /**
                 * @brief Calculate time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateMean(const label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(timeStep + 1);

                    // Calculate the mean
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            moments::kernel::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {rhoMean_.ptr(),
                                 uMean_.ptr(),
                                 vMean_.ptr(),
                                 wMean_.ptr(),
                                 mxxMean_.ptr(),
                                 mxyMean_.ptr(),
                                 mxzMean_.ptr(),
                                 myyMean_.ptr(),
                                 myzMean_.ptr(),
                                 mzzMean_.ptr()},
                                invNewCount);
                        });
                }

                /**
                 * @brief Calculate both the instantaneous and time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneousAndMean([[maybe_unused]] const label_t timeStep) noexcept
                {
                    return;
                }

                /**
                 * @brief Saves the instantaneous total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveInstantaneous([[maybe_unused]] const label_t timeStep) noexcept
                {
                    return;
                }

                /**
                 * @brief Saves the mean total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveMean(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<10, scalar_t>(
                            rhoMean_.ptr(),
                            uMean_.ptr(),
                            vMean_.ptr(),
                            wMean_.ptr(),
                            mxxMean_.ptr(),
                            mxyMean_.ptr(),
                            mxzMean_.ptr(),
                            myyMean_.ptr(),
                            myzMean_.ptr(),
                            mzzMean_.ptr()),
                        mesh_);

                    fileIO::writeFile<time::timeAverage>(
                        fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesMean_,
                        hostWriteBuffer_.data(),
                        timeStep);
                }

                /**
                 * @brief Get the field name for instantaneous moments
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::string &fieldName() const noexcept
                {
                    return fieldName_;
                }

                /**
                 * @brief Get the field name for mean moments
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::string &fieldNameMean() const noexcept
                {
                    return fieldNameMean_;
                }

                /**
                 * @brief Get the component names for instantaneous moments
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNames() const noexcept
                {
                    return componentNames_;
                }

                /**
                 * @brief Get the component names for mean moments
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNamesMean() const noexcept
                {
                    return componentNamesMean_;
                }

            private:
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

                /**
                 * @brief Field name for instantaneous scalar
                 **/
                const std::string fieldName_ = "moments";

                /**
                 * @brief Field name for mean scalar
                 **/
                const std::string fieldNameMean_ = fieldName_ + "Mean";

                /**
                 * @brief Instantaneous scalar name
                 **/
                const std::vector<std::string> componentNames_ = solutionVariableNames;

                /**
                 * @brief Mean scalar name
                 **/
                const std::vector<std::string> componentNamesMean_ = string::catenate(componentNames_, "Mean");

                /**
                 * @brief Reference to lattice mesh
                 **/
                const host::latticeMesh &mesh_;

                /**
                 * @brief Device pointer collection
                 **/
                const device::ptrCollection<10, scalar_t> &devPtrs_;

                /**
                 * @brief Stream handler for CUDA operations
                 **/
                const streamHandler<N> &streamsLBM_;

                /**
                 * @brief Flag for instantaneous calculation
                 **/
                static constexpr const bool calculate_ = false;

                /**
                 * @brief Flag for mean calculation
                 **/
                const bool calculateMean_;

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
}

#endif