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
                __host__ [[nodiscard]] inline consteval host::label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
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
                    types::assertions::validate<T>();

                    if constexpr (std::is_same_v<T, float>)
                    {
                        return sqrtf((u * u) + (v * v) + (w * w)) * static_cast<T>(0.5);
                    }

                    if constexpr (std::is_same_v<T, double>)
                    {
                        return sqrt((u * u) + (v * v) + (w * w)) * static_cast<T>(0.5);
                    }
                }

                /**
                 * @brief CUDA kernel for calculating time-averaged total kinetic energy
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] KMeanPtrs Device pointer collection for mean total kinetic energy
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void mean(
                    const device::ptrCollection<10, const scalar_t> devPtrs,
                    const device::ptrCollection<1, scalar_t> KMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

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
                    const device::ptrCollection<10, const scalar_t> devPtrs,
                    const device::ptrCollection<1, scalar_t> KPtrs,
                    const device::ptrCollection<1, scalar_t> KMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

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
                    const device::ptrCollection<10, const scalar_t> devPtrs,
                    const device::ptrCollection<1, scalar_t> KPtrs)
                {
                    // Calculate the index
                    const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

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
             * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
             * @tparam N The number of streams (compile-time constant)
             **/
            template <class VelocitySet>
            class scalar
            {
            public:
                /**
                 * @brief Constructs a total kinetic energy scalar object
                 * @param[in] mesh The lattice mesh
                 * @param[in] devPtrs Device pointer collection for memory access
                 * @param[in] streamsLBM Stream handler for CUDA operations
                 **/
                __host__ [[nodiscard]] scalar(
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
                    : hostWriteBuffer_(hostWriteBuffer),
                      mesh_(mesh),
                      rho_(rho),
                      u_(u),
                      v_(v),
                      w_(w),
                      mxx_(mxx),
                      mxy_(mxy),
                      mxz_(mxz),
                      myy_(myy),
                      myz_(myz),
                      mzz_(mzz),
                      streamsLBM_(streamsLBM),
                      calculate_(initialiserSwitch(fieldName_)),
                      calculateMean_(initialiserSwitch(fieldNameMean_)),
                      k_(objectAllocator<VelocitySet, time::instantaneous>(fieldName_, mesh, programCtrl)),
                      kMean_(objectAllocator<VelocitySet, time::timeAverage>(fieldNameMean_, mesh, programCtrl))
                {
                    // Set the cache config to prefer L1
                    errorHandler::check(cudaFuncSetCacheConfig(kernel::instantaneous, cudaFuncCachePreferL1));
                };

                /**
                 * @brief Default destructor
                 **/
                ~scalar() {}

                /**
                 * @brief Disable copying
                 **/
                __host__ [[nodiscard]] scalar(const scalar &) = delete;
                __host__ [[nodiscard]] scalar &operator=(const scalar &) = delete;

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
                __host__ void calculateInstantaneous([[maybe_unused]] const host::label_t timeStep) noexcept
                {
                    for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                    {
                        kineticEnergy::kernel::instantaneous<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                            {rho_.ptr(stream),
                             u_.ptr(stream),
                             v_.ptr(stream),
                             w_.ptr(stream),
                             mxx_.ptr(stream),
                             mxy_.ptr(stream),
                             mxz_.ptr(stream),
                             myy_.ptr(stream),
                             myz_.ptr(stream),
                             mzz_.ptr(stream)},
                            {k_.ptr(stream)});
                    }
                }

                /**
                 * @brief Calculate time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateMean([[maybe_unused]] const host::label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(kMean_.meanCount() + 1);

                    for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                    {
                        kineticEnergy::kernel::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                            {rho_.ptr(stream),
                             u_.ptr(stream),
                             v_.ptr(stream),
                             w_.ptr(stream),
                             mxx_.ptr(stream),
                             mxy_.ptr(stream),
                             mxz_.ptr(stream),
                             myy_.ptr(stream),
                             myz_.ptr(stream),
                             mzz_.ptr(stream)},
                            {kMean_.ptr(stream)},
                            invNewCount);
                    }

                    kMean_.meanCountRef()++;
                }

                /**
                 * @brief Calculate both the instantaneous and time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneousAndMean([[maybe_unused]] const host::label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(kMean_.meanCount() + 1);

                    for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                    {
                        kineticEnergy::kernel::instantaneousAndMean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                            {rho_.ptr(stream),
                             u_.ptr(stream),
                             v_.ptr(stream),
                             w_.ptr(stream),
                             mxx_.ptr(stream),
                             mxy_.ptr(stream),
                             mxz_.ptr(stream),
                             myy_.ptr(stream),
                             myz_.ptr(stream),
                             mzz_.ptr(stream)},
                            {k_.ptr(stream)},
                            {kMean_.ptr(stream)},
                            invNewCount);
                    }

                    kMean_.meanCountRef()++;
                }

                /**
                 * @brief Saves the instantaneous total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveInstantaneous(const host::label_t timeStep) noexcept
                {
                    for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < k_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                    {
                        hostWriteBuffer_.copy_from_device(
                            device::ptrCollection<1, scalar_t>(k_.ptr(virtualDeviceIndex)),
                            mesh_,
                            virtualDeviceIndex);
                    }

                    fileIO::writeFile<time::instantaneous>(
                        fieldName_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNames_,
                        hostWriteBuffer_.data(),
                        timeStep,
                        0);
                }

                /**
                 * @brief Saves the mean total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveMean(const host::label_t timeStep) noexcept
                {
                    for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < kMean_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                    {
                        hostWriteBuffer_.copy_from_device(
                            device::ptrCollection<1, scalar_t>(kMean_.ptr(virtualDeviceIndex)),
                            mesh_,
                            virtualDeviceIndex);
                    }

                    fileIO::writeFile<time::timeAverage>(
                        fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesMean_,
                        hostWriteBuffer_.data(),
                        timeStep,
                        kMean_.meanCount());
                }

                /**
                 * @brief Get the field name for instantaneous components
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const name_t &fieldName() const noexcept
                {
                    return fieldName_;
                }

                /**
                 * @brief Get the field name for mean components
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const name_t &fieldNameMean() const noexcept
                {
                    return fieldNameMean_;
                }

                /**
                 * @brief Get the component names for instantaneous scalar
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const words_t &componentNames() const noexcept
                {
                    return componentNames_;
                }

                /**
                 * @brief Get the component names for mean scalar
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const words_t &componentNamesMean() const noexcept
                {
                    return componentNamesMean_;
                }

            private:
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

                /**
                 * @brief Field name for instantaneous scalar
                 **/
                const name_t fieldName_ = "k";

                /**
                 * @brief Field name for mean scalar
                 **/
                const name_t fieldNameMean_ = fieldName_ + "Mean";

                /**
                 * @brief Instantaneous scalar name
                 **/
                const words_t componentNames_ = {"k"};

                /**
                 * @brief Mean scalar name
                 **/
                const words_t componentNamesMean_ = string::catenate(componentNames_, "Mean");

                /**
                 * @brief Reference to lattice mesh
                 **/
                const host::latticeMesh &mesh_;

                /**
                 * @brief Device pointer collection
                 **/
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &rho_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &u_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &v_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &w_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxx_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxy_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxz_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myy_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myz_;
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mzz_;

                /**
                 * @brief Stream handler for CUDA operations
                 **/
                const streamHandler &streamsLBM_;

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