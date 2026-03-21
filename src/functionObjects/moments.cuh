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
                __host__ [[nodiscard]] inline consteval host::label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
#define launchBounds __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

                /**
                 * @brief CUDA kernel for calculating time-averaged total kinetic energy
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] KMeanPtrs Device pointer collection for mean total kinetic energy
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void mean(
                    const device::ptrCollection<NUMBER_MOMENTS(), const scalar_t> devPtrs,
                    const device::ptrCollection<NUMBER_MOMENTS(), scalar_t> devMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Index into global arrays
                    const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

                    // Read from global memory
                    thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> m;
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto n)
                        {
                            m[n] = devPtrs.ptr<n>()[idx];
                        });

                    // Read the mean values from global memory
                    thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> mMean;
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto n)
                        {
                            mMean[n] = devMeanPtrs.ptr<n>()[idx];
                        });

                    // Update the mean value and write back to global
                    const thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> meanNew = timeAverage(mMean, m, invNewCount);
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto n)
                        {
                            devMeanPtrs.ptr<n>()[idx] = meanNew[n];
                        });
                }
            }

            /**
             * @brief Class for managing total kinetic energy scalar calculations in LBM simulations
             * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
             * @tparam N The number of streams (compile-time constant)
             **/
            template <class VelocitySet>
            class collection
            {
            public:
                /**
                 * @brief Constructs a total kinetic energy scalar object
                 * @param[in] mesh The lattice mesh
                 * @param[in] devPtrs Device pointer collection for memory access
                 * @param[in] streamsLBM Stream handler for CUDA operations
                 **/
                __host__ [[nodiscard]] collection(
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
                      calculateMean_(initialiserSwitch(fieldNameMean_)),
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
                    // Set the cache config to prefer L1
                    errorHandler::check(cudaFuncSetCacheConfig(kernel::mean, cudaFuncCachePreferL1));
                };

                /**
                 * @brief Default destructor
                 **/
                ~collection() {}

                /**
                 * @brief Disable copying
                 **/
                __host__ [[nodiscard]] collection(const collection &) = delete;
                __host__ [[nodiscard]] collection &operator=(const collection &) = delete;

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
                    return;
                }

                /**
                 * @brief Calculate time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateMean([[maybe_unused]] const host::label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(rhoMean_.meanCount() + 1);

                    for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                    {
                        moments::kernel::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
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
                            {rhoMean_.ptr(stream),
                             uMean_.ptr(stream),
                             vMean_.ptr(stream),
                             wMean_.ptr(stream),
                             mxxMean_.ptr(stream),
                             mxyMean_.ptr(stream),
                             mxzMean_.ptr(stream),
                             myyMean_.ptr(stream),
                             myzMean_.ptr(stream),
                             mzzMean_.ptr(stream)},
                            invNewCount);
                    }

                    rhoMean_.meanCountRef()++;
                }

                /**
                 * @brief Calculate both the instantaneous and time-averaged total kinetic energy
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneousAndMean([[maybe_unused]] const host::label_t timeStep) noexcept
                {
                    return;
                }

                /**
                 * @brief Saves the instantaneous total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveInstantaneous([[maybe_unused]] const host::label_t timeStep) noexcept
                {
                    return;
                }

                /**
                 * @brief Saves the mean total kinetic energy to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveMean(const host::label_t timeStep) noexcept
                {
                    for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < rhoMean_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                    {
                        hostWriteBuffer_.copy_from_device(
                            device::ptrCollection<10, scalar_t>(
                                rhoMean_.ptr(virtualDeviceIndex),
                                uMean_.ptr(virtualDeviceIndex),
                                vMean_.ptr(virtualDeviceIndex),
                                wMean_.ptr(virtualDeviceIndex),
                                mxxMean_.ptr(virtualDeviceIndex),
                                mxyMean_.ptr(virtualDeviceIndex),
                                mxzMean_.ptr(virtualDeviceIndex),
                                myyMean_.ptr(virtualDeviceIndex),
                                myzMean_.ptr(virtualDeviceIndex),
                                mzzMean_.ptr(virtualDeviceIndex)),
                            mesh_,
                            virtualDeviceIndex);
                    }

                    fileIO::writeFile<time::timeAverage>(
                        fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesMean_,
                        hostWriteBuffer_.data(),
                        timeStep,
                        rhoMean_.meanCount());
                }

                /**
                 * @brief Get the field name for instantaneous moments
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const name_t &fieldName() const noexcept
                {
                    return fieldName_;
                }

                /**
                 * @brief Get the field name for mean moments
                 * @return Field name string
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const name_t &fieldNameMean() const noexcept
                {
                    return fieldNameMean_;
                }

                /**
                 * @brief Get the component names for instantaneous moments
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const words_t &componentNames() const noexcept
                {
                    return componentNames_;
                }

                /**
                 * @brief Get the component names for mean moments
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
                const name_t fieldName_ = "moments";

                /**
                 * @brief Field name for mean scalar
                 **/
                const name_t fieldNameMean_ = fieldName_ + "Mean";

                /**
                 * @brief Instantaneous scalar name
                 **/
                const words_t componentNames_ = solutionVariableNames;

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