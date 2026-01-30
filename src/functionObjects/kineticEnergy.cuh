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
        namespace kineticEnergy
        {
            namespace kernel
            {
                __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
#define launchBounds __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

                /**
                 * @brief Calculates the total kinetic energy
                 * @param[in] u Velocity component in x direction
                 * @param[in] v Velocity component in y direction
                 * @param[in] w Velocity component in z direction
                 * @return The calculated total kinetic energy
                 **/
                template <typename T>
                __device__ [[nodiscard]] inline constexpr T K(const T u, const T v, const T w) noexcept
                {
                    static_assert((sizeof(T) == 8) || (sizeof(T) == 4), "Bad type T");

                    if constexpr (sizeof(T) == 8)
                    {
                        return sqrt((u * u) + (v * v) + (w * w)) * static_cast<T>(0.5);
                    }

                    else if constexpr (sizeof(T) == 4)
                    {
                        return sqrtf((u * u) + (v * v) + (w * w)) * static_cast<T>(0.5);
                    }
                }

                /**
                 * @brief CUDA kernel for calculating time-averaged total kinetic energy
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] KMeanPtrs Device pointer collection for mean total kinetic energy
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void mean(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<1, scalar_t> KMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

                    // Read from global memory
                    const scalar_t u = devPtrs.ptr<1>()[idx];
                    const scalar_t v = devPtrs.ptr<2>()[idx];
                    const scalar_t w = devPtrs.ptr<3>()[idx];

                    // Calculate the instantaneous
                    const scalar_t Ke = K(u, v, w);

                    // Read the mean values from global memory
                    const scalar_t Ke_Mean = KMeanPtrs.ptr<0>()[idx];

                    // Update the mean value and write back to global
                    const scalar_t Ke_MeanNew = timeAverage(Ke_Mean, Ke, invNewCount);
                    KMeanPtrs.ptr<0>()[idx] = Ke_MeanNew;
                }

                /**
                 * @brief CUDA kernel for calculating instantaneous and mean total kinetic energy
                 * @param[in] devPtrs Device pointer collection containing velocity fields
                 * @param[in] KPtrs Device pointer collection for instantaneous total kinetic energy
                 * @param[in] KMeanPtrs Device pointer collection for mean total kinetic energy
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void instantaneousAndMean(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<1, scalar_t> KPtrs,
                    const device::ptrCollection<1, scalar_t> KMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

                    // Read from global memory
                    const scalar_t u = devPtrs.ptr<1>()[idx];
                    const scalar_t v = devPtrs.ptr<2>()[idx];
                    const scalar_t w = devPtrs.ptr<3>()[idx];

                    // Calculate the instantaneous and write back to global
                    const scalar_t Ke = K(u, v, w);
                    KPtrs.ptr<0>()[idx] = Ke;

                    // Read the mean values from global memory
                    const scalar_t Ke_Mean = KMeanPtrs.ptr<0>()[idx];

                    // Update the mean value and write back to global
                    const scalar_t Ke_MeanNew = timeAverage(Ke_Mean, Ke, invNewCount);
                    KMeanPtrs.ptr<0>()[idx] = Ke_MeanNew;
                }

                /**
                 * @brief CUDA kernel for calculating instantaneous total kinetic energy
                 * @param[in] devPtrs Device pointer collection containing velocity fields
                 * @param[in] KPtrs Device pointer collection for instantaneous total kinetic energy
                 **/
                launchBounds __global__ void instantaneous(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<1, scalar_t> KPtrs)
                {
                    // Calculate the index
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

                    // Read from global memory
                    const scalar_t u = devPtrs.ptr<1>()[idx];
                    const scalar_t v = devPtrs.ptr<2>()[idx];
                    const scalar_t w = devPtrs.ptr<3>()[idx];

                    // Calculate the instantaneous and write back to global
                    const scalar_t Ke = K(u, v, w);
                    KPtrs.ptr<0>()[idx] = Ke;
                }
            }

            /**
             * @brief Class for managing total kinetic energy scalar calculations in LBM simulations
             * @tparam VelocitySet The velocity set type used in LBM
             * @tparam N The number of streams (compile-time constant)
             **/
            template <class VelocitySet, const label_t N>
            class scalar
            {
            public:
                /**
                 * @brief Constructs a total kinetic energy scalar object
                 * @param[in] mesh Reference to lattice mesh
                 * @param[in] devPtrs Device pointer collection for memory access
                 * @param[in] streamsLBM Stream handler for CUDA operations
                 **/
                __host__ [[nodiscard]] scalar(
                    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                    const host::latticeMesh &mesh,
                    const device::ptrCollection<10, scalar_t> &devPtrs,
                    const streamHandler<N> &streamsLBM) noexcept
                    : hostWriteBuffer_(hostWriteBuffer),
                      mesh_(mesh),
                      devPtrs_(devPtrs),
                      streamsLBM_(streamsLBM),
                      calculate_(initialiserSwitch(fieldName_)),
                      calculateMean_(initialiserSwitch(fieldNameMean_)),
                      k_(objectAllocator<VelocitySet, time::instantaneous>(fieldName_, mesh)),
                      kMean_(objectAllocator<VelocitySet, time::timeAverage>(fieldNameMean_, mesh))
                {
                    // std::cout << fieldNameMean_ << std::endl;
                    // Set the cache config to prefer L1
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::instantaneous, cudaFuncCachePreferL1));
                };

                /**
                 * @brief Default destructor
                 **/
                ~scalar() {};

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
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            kineticEnergy::kernel::instantaneous<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {k_.ptr()});
                        });
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
                            kineticEnergy::kernel::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {kMean_.ptr()},
                                invNewCount);
                        });
                }

                /**
                 * @brief Calculate both the instantaneous and time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneousAndMean(const label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(timeStep + 1);

                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            kineticEnergy::kernel::instantaneousAndMean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {k_.ptr()},
                                {kMean_.ptr()},
                                invNewCount);
                        });
                }

                /**
                 * @brief Saves the instantaneous total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveInstantaneous(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(device::ptrCollection<1, scalar_t>(k_.ptr()), mesh_);

                    fileIO::writeFile<time::instantaneous>(
                        fieldName_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNames_,
                        hostWriteBuffer_.data(),
                        timeStep);
                }

                /**
                 * @brief Saves the mean total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveMean(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(device::ptrCollection<1, scalar_t>(kMean_.ptr()), mesh_);

                    fileIO::writeFile<time::timeAverage>(
                        fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesMean_,
                        hostWriteBuffer_.data(),
                        timeStep);
                }

                /**
                 * @brief Get the field name for instantaneous components
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::string &fieldName() const noexcept
                {
                    return fieldName_;
                }

                /**
                 * @brief Get the field name for mean components
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::string &fieldNameMean() const noexcept
                {
                    return fieldNameMean_;
                }

                /**
                 * @brief Get the component names for instantaneous scalar
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNames() const noexcept
                {
                    return componentNames_;
                }

                /**
                 * @brief Get the component names for mean scalar
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
                const std::string fieldName_ = "k";

                /**
                 * @brief Field name for mean scalar
                 **/
                const std::string fieldNameMean_ = fieldName_ + "Mean";

                /**
                 * @brief Instantaneous scalar name
                 **/
                const std::vector<std::string> componentNames_ = {"k"};

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
                const bool calculate_;

                /**
                 * @brief Flag for mean calculation
                 **/
                const bool calculateMean_;

                /**
                 * @brief Instantaneous total kinetic energy scalar
                 **/
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> k_;

                /**
                 * @brief Time-averaged total kinetic energy scalar
                 **/
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> kMean_;
            };
        }
    }
}

#endif