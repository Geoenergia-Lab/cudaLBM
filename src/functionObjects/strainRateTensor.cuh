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
        namespace strainRate
        {
            namespace kernel
            {
                __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
#define launchBounds __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

                /**
                 * @brief Calculates the strain rate tensor component
                 * @param[in] uAlpha Velocity component in alpha direction
                 * @param[in] uBeta Velocity component in beta direction
                 * @param[in] mAlphaBeta Second order moment component
                 * @return The calculated strain rate tensor component
                 **/
                template <const label_t Index, typename T>
                __device__ [[nodiscard]] inline constexpr T S(const T uAlpha, const T uBeta, const T mAlphaBeta) noexcept
                {
                    static_assert((Index == index::xx() || Index == index::yy() || Index == index::zz() || Index == index::xy() || Index == index::xz() || Index == index::yz()), "Invalid index");

                    if constexpr (Index == index::xx() || Index == index::yy() || Index == index::zz())
                    {
                        return velocitySet::as2<T>() * ((uAlpha * uBeta) - mAlphaBeta) / (static_cast<T>(2) * velocitySet::scale_ii<scalar_t>() * device::tau);
                    }
                    else
                    {
                        return velocitySet::as2<T>() * ((uAlpha * uBeta) - mAlphaBeta) / (static_cast<T>(2) * velocitySet::scale_ij<scalar_t>() * device::tau);
                    }
                }

                /**
                 * @brief CUDA kernel for calculating time-averaged strain rate tensor components
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] SMeanPtrs Device pointer collection for mean strain rate tensor components
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void mean(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<6, scalar_t> SMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

                    // Read from global memory
                    const scalar_t u = devPtrs.ptr<1>()[idx];
                    const scalar_t v = devPtrs.ptr<2>()[idx];
                    const scalar_t w = devPtrs.ptr<3>()[idx];
                    const scalar_t mxx = devPtrs.ptr<4>()[idx];
                    const scalar_t mxy = devPtrs.ptr<5>()[idx];
                    const scalar_t mxz = devPtrs.ptr<6>()[idx];
                    const scalar_t myy = devPtrs.ptr<7>()[idx];
                    const scalar_t myz = devPtrs.ptr<8>()[idx];
                    const scalar_t mzz = devPtrs.ptr<9>()[idx];

                    // Calculate the instantaneous
                    const scalar_t S_xx = S<index::xx()>(u, u, mxx);
                    const scalar_t S_xy = S<index::xy()>(u, v, mxy);
                    const scalar_t S_xz = S<index::xz()>(u, w, mxz);
                    const scalar_t S_yy = S<index::yy()>(v, v, myy);
                    const scalar_t S_yz = S<index::yz()>(v, w, myz);
                    const scalar_t S_zz = S<index::zz()>(w, w, mzz);

                    // Read the mean values from global memory
                    const scalar_t S_xxMean = SMeanPtrs.ptr<0>()[idx];
                    const scalar_t S_xyMean = SMeanPtrs.ptr<1>()[idx];
                    const scalar_t S_xzMean = SMeanPtrs.ptr<2>()[idx];
                    const scalar_t S_yyMean = SMeanPtrs.ptr<3>()[idx];
                    const scalar_t S_yzMean = SMeanPtrs.ptr<4>()[idx];
                    const scalar_t S_zzMean = SMeanPtrs.ptr<5>()[idx];

                    // Update the mean value and write back to global
                    const scalar_t S_xxMeanNew = timeAverage(S_xxMean, S_xx, invNewCount);
                    const scalar_t S_xyMeanNew = timeAverage(S_xyMean, S_xy, invNewCount);
                    const scalar_t S_xzMeanNew = timeAverage(S_xzMean, S_xz, invNewCount);
                    const scalar_t S_yyMeanNew = timeAverage(S_yyMean, S_yy, invNewCount);
                    const scalar_t S_yzMeanNew = timeAverage(S_yzMean, S_yz, invNewCount);
                    const scalar_t S_zzMeanNew = timeAverage(S_zzMean, S_zz, invNewCount);
                    SMeanPtrs.ptr<0>()[idx] = S_xxMeanNew;
                    SMeanPtrs.ptr<1>()[idx] = S_xyMeanNew;
                    SMeanPtrs.ptr<2>()[idx] = S_xzMeanNew;
                    SMeanPtrs.ptr<3>()[idx] = S_yyMeanNew;
                    SMeanPtrs.ptr<4>()[idx] = S_yzMeanNew;
                    SMeanPtrs.ptr<5>()[idx] = S_zzMeanNew;
                }

                /**
                 * @brief CUDA kernel for calculating instantaneous and time-averaged strain rate tensor components
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] SPtrs Device pointer collection for instantaneous strain rate tensor components
                 * @param[in] SMeanPtrs Device pointer collection for mean strain rate tensor components
                 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
                 **/
                launchBounds __global__ void instantaneousAndMean(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<6, scalar_t> SPtrs,
                    const device::ptrCollection<6, scalar_t> SMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

                    // Read from global memory
                    const scalar_t u = devPtrs.ptr<1>()[idx];
                    const scalar_t v = devPtrs.ptr<2>()[idx];
                    const scalar_t w = devPtrs.ptr<3>()[idx];
                    const scalar_t mxx = devPtrs.ptr<4>()[idx];
                    const scalar_t mxy = devPtrs.ptr<5>()[idx];
                    const scalar_t mxz = devPtrs.ptr<6>()[idx];
                    const scalar_t myy = devPtrs.ptr<7>()[idx];
                    const scalar_t myz = devPtrs.ptr<8>()[idx];
                    const scalar_t mzz = devPtrs.ptr<9>()[idx];

                    // Calculate the instantaneous and write back to global
                    const scalar_t S_xx = S<index::xx()>(u, u, mxx);
                    const scalar_t S_xy = S<index::xy()>(u, v, mxy);
                    const scalar_t S_xz = S<index::xz()>(u, w, mxz);
                    const scalar_t S_yy = S<index::yy()>(v, v, myy);
                    const scalar_t S_yz = S<index::yz()>(v, w, myz);
                    const scalar_t S_zz = S<index::zz()>(w, w, mzz);
                    SPtrs.ptr<0>()[idx] = S_xx;
                    SPtrs.ptr<1>()[idx] = S_xy;
                    SPtrs.ptr<2>()[idx] = S_xz;
                    SPtrs.ptr<3>()[idx] = S_yy;
                    SPtrs.ptr<4>()[idx] = S_yz;
                    SPtrs.ptr<5>()[idx] = S_zz;

                    // Read the mean values from global memory
                    const scalar_t S_xxMean = SMeanPtrs.ptr<0>()[idx];
                    const scalar_t S_xyMean = SMeanPtrs.ptr<1>()[idx];
                    const scalar_t S_xzMean = SMeanPtrs.ptr<2>()[idx];
                    const scalar_t S_yyMean = SMeanPtrs.ptr<3>()[idx];
                    const scalar_t S_yzMean = SMeanPtrs.ptr<4>()[idx];
                    const scalar_t S_zzMean = SMeanPtrs.ptr<5>()[idx];

                    // Update the mean value and write back to global
                    const scalar_t S_xxMeanNew = timeAverage(S_xxMean, S_xx, invNewCount);
                    const scalar_t S_xyMeanNew = timeAverage(S_xyMean, S_xy, invNewCount);
                    const scalar_t S_xzMeanNew = timeAverage(S_xzMean, S_xz, invNewCount);
                    const scalar_t S_yyMeanNew = timeAverage(S_yyMean, S_yy, invNewCount);
                    const scalar_t S_yzMeanNew = timeAverage(S_yzMean, S_yz, invNewCount);
                    const scalar_t S_zzMeanNew = timeAverage(S_zzMean, S_zz, invNewCount);
                    SMeanPtrs.ptr<0>()[idx] = S_xxMeanNew;
                    SMeanPtrs.ptr<1>()[idx] = S_xyMeanNew;
                    SMeanPtrs.ptr<2>()[idx] = S_xzMeanNew;
                    SMeanPtrs.ptr<3>()[idx] = S_yyMeanNew;
                    SMeanPtrs.ptr<4>()[idx] = S_yzMeanNew;
                    SMeanPtrs.ptr<5>()[idx] = S_zzMeanNew;
                }

                /**
                 * @brief CUDA kernel for calculating instantaneous strain rate tensor components
                 * @param[in] devPtrs Device pointer collection containing velocity and moment fields
                 * @param[in] SPtrs Device pointer collection for instantaneous strain rate tensor components
                 **/
                launchBounds __global__ void instantaneous(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<6, scalar_t> SPtrs)
                {
                    // Calculate the index
                    const label_t idx = device::idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

                    // Read from global memory
                    const scalar_t u = devPtrs.ptr<1>()[idx];
                    const scalar_t v = devPtrs.ptr<2>()[idx];
                    const scalar_t w = devPtrs.ptr<3>()[idx];
                    const scalar_t mxx = devPtrs.ptr<4>()[idx];
                    const scalar_t mxy = devPtrs.ptr<5>()[idx];
                    const scalar_t mxz = devPtrs.ptr<6>()[idx];
                    const scalar_t myy = devPtrs.ptr<7>()[idx];
                    const scalar_t myz = devPtrs.ptr<8>()[idx];
                    const scalar_t mzz = devPtrs.ptr<9>()[idx];

                    // Calculate the instantaneous and write back to global
                    const scalar_t S_xx = S<index::xx()>(u, u, mxx);
                    const scalar_t S_xy = S<index::xy()>(u, v, mxy);
                    const scalar_t S_xz = S<index::xz()>(u, w, mxz);
                    const scalar_t S_yy = S<index::yy()>(v, v, myy);
                    const scalar_t S_yz = S<index::yz()>(v, w, myz);
                    const scalar_t S_zz = S<index::zz()>(w, w, mzz);
                    SPtrs.ptr<0>()[idx] = S_xx;
                    SPtrs.ptr<1>()[idx] = S_xy;
                    SPtrs.ptr<2>()[idx] = S_xz;
                    SPtrs.ptr<3>()[idx] = S_yy;
                    SPtrs.ptr<4>()[idx] = S_yz;
                    SPtrs.ptr<5>()[idx] = S_zz;
                }
            }

            /**
             * @brief Class for managing strain rate tensor calculations in LBM simulations
             * @tparam VelocitySet The velocity set type used in LBM
             * @tparam N The number of streams (compile-time constant)
             **/
            template <class VelocitySet, const label_t N>
            class tensor
            {
            public:
                /**
                 * @brief Constructs a strain rate tensor object
                 * @param[in] mesh Reference to lattice mesh
                 * @param[in] devPtrs Device pointer collection for memory access
                 * @param[in] streamsLBM Stream handler for CUDA operations
                 **/
                __host__ [[nodiscard]] tensor(
                    const host::latticeMesh &mesh,
                    const device::ptrCollection<10, scalar_t> &devPtrs,
                    const streamHandler<N> &streamsLBM) noexcept
                    : mesh_(mesh),
                      devPtrs_(devPtrs),
                      streamsLBM_(streamsLBM),
                      calculate_(initialiserSwitch(fieldName_)),
                      calculateMean_(initialiserSwitch(fieldNameMean_)),
                      xx_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[0], mesh, calculate_)),
                      xy_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[1], mesh, calculate_)),
                      xz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[2], mesh, calculate_)),
                      yy_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[3], mesh, calculate_)),
                      yz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[4], mesh, calculate_)),
                      zz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[5], mesh, calculate_)),
                      xxMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_)),
                      xyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_)),
                      xzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_)),
                      yyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_)),
                      yzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_)),
                      zzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_))
                {
                    // Set the cache config to prefer L1
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::instantaneous, cudaFuncCachePreferL1));
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::instantaneousAndMean, cudaFuncCachePreferL1));
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::mean, cudaFuncCachePreferL1));
                };

                /**
                 * @brief Default destructor
                 **/
                ~tensor() {};

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
                 * @brief Calculate instantaneous strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneous([[maybe_unused]] const label_t timeStep) noexcept
                {
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            strainRate::kernel::instantaneous<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {xx_.ptr(), xy_.ptr(), xz_.ptr(), yy_.ptr(), yz_.ptr(), zz_.ptr()});
                        });
                }

                /**
                 * @brief Calculate time-averaged strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateMean(const label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(timeStep + 1);

                    // Calculate the mean
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            strainRate::kernel::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {xxMean_.ptr(), xyMean_.ptr(), xzMean_.ptr(), yyMean_.ptr(), yzMean_.ptr(), zzMean_.ptr()},
                                invNewCount);
                        });
                }

                /**
                 * @brief Calculate both the instantaneous and time-averaged strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculateInstantaneousAndMean(const label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(timeStep + 1);

                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            strainRate::kernel::instantaneousAndMean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {xx_.ptr(), xy_.ptr(), xz_.ptr(), yy_.ptr(), yz_.ptr(), zz_.ptr()},
                                {xxMean_.ptr(), xyMean_.ptr(), xzMean_.ptr(), yyMean_.ptr(), yzMean_.ptr(), zzMean_.ptr()},
                                invNewCount);
                        });
                }

                /**
                 * @brief Saves the instantaneous strain rate tensor components to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveInstantaneous(const label_t timeStep) noexcept
                {
                    fileIO::writeFile<time::instantaneous>(
                        fieldName_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNames_,
                        host::to_host(
                            device::ptrCollection<6, scalar_t>(
                                xx_.ptr(), xy_.ptr(),
                                xz_.ptr(), yy_.ptr(),
                                yz_.ptr(), zz_.ptr()),
                            mesh_),
                        timeStep);
                }

                /**
                 * @brief Saves the mean strain rate tensor components to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveMean(const label_t timeStep) noexcept
                {
                    fileIO::writeFile<time::timeAverage>(
                        fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesMean_,
                        host::to_host(
                            device::ptrCollection<6, scalar_t>(
                                xxMean_.ptr(), xyMean_.ptr(),
                                xzMean_.ptr(), yyMean_.ptr(),
                                yzMean_.ptr(), zzMean_.ptr()),
                            mesh_),
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
                 * @brief Get the component names for instantaneous tensor
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNames() const noexcept
                {
                    return componentNames_;
                }

                /**
                 * @brief Get the component names for mean tensor
                 * @return Vector of component names
                 **/
                __device__ __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &componentNamesMean() const noexcept
                {
                    return componentNamesMean_;
                }

            private:
                /**
                 * @brief Field name for instantaneous components
                 **/
                const std::string fieldName_ = "S";

                /**
                 * @brief Field name for mean components
                 **/
                const std::string fieldNameMean_ = fieldName_ + "Mean";

                /**
                 * @brief Instantaneous component names
                 **/
                const std::vector<std::string> componentNames_ = {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"};

                /**
                 * @brief Mean component names
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
                 * @brief Instantaneous strain rate tensor components
                 **/
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> xx_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> xy_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> xz_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> yy_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> yz_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> zz_;

                /**
                 * @brief Time-averaged strain rate tensor components
                 **/
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> xxMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> xyMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> xzMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> yyMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> yzMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> zzMean_;
            };
        }
    }
}

#endif