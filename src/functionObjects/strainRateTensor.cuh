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
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

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
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

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
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

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

                launchBounds __global__ void prime(
                    const device::ptrCollection<6, scalar_t> SPtrs,
                    const device::ptrCollection<6, scalar_t> SMeanPtrs,
                    const device::ptrCollection<6, scalar_t> SPrimePtrs)
                {
                    const label_t idx = device::idx();

                    // Calculate the prime quantity and write back to global
                    const scalar_t Sxx = SPtrs.ptr<0>()[idx];
                    const scalar_t Sxy = SPtrs.ptr<1>()[idx];
                    const scalar_t Sxz = SPtrs.ptr<2>()[idx];
                    const scalar_t Syy = SPtrs.ptr<3>()[idx];
                    const scalar_t Syz = SPtrs.ptr<4>()[idx];
                    const scalar_t Szz = SPtrs.ptr<5>()[idx];

                    const scalar_t SxxMean = SMeanPtrs.ptr<0>()[idx];
                    const scalar_t SxyMean = SMeanPtrs.ptr<1>()[idx];
                    const scalar_t SxzMean = SMeanPtrs.ptr<2>()[idx];
                    const scalar_t SyyMean = SMeanPtrs.ptr<3>()[idx];
                    const scalar_t SyzMean = SMeanPtrs.ptr<4>()[idx];
                    const scalar_t SzzMean = SMeanPtrs.ptr<5>()[idx];

                    const scalar_t SxxPrime = Sxx - SxxMean;
                    const scalar_t SxyPrime = Sxy - SxyMean;
                    const scalar_t SxzPrime = Sxz - SxzMean;
                    const scalar_t SyyPrime = Syy - SyyMean;
                    const scalar_t SyzPrime = Syz - SyzMean;
                    const scalar_t SzzPrime = Szz - SzzMean;

                    // Write back to global memory
                    SPrimePtrs.ptr<0>()[idx] = SxxPrime;
                    SPrimePtrs.ptr<1>()[idx] = SxyPrime;
                    SPrimePtrs.ptr<2>()[idx] = SxzPrime;
                    SPrimePtrs.ptr<3>()[idx] = SyyPrime;
                    SPrimePtrs.ptr<4>()[idx] = SyzPrime;
                    SPrimePtrs.ptr<5>()[idx] = SzzPrime;
                }

                launchBounds __global__ void primeMean(
                    const device::ptrCollection<10, scalar_t> devPtrs,
                    const device::ptrCollection<6, scalar_t> SMeanPtrs,
                    const device::ptrCollection<6, scalar_t> SPrimeMeanPtrs,
                    const scalar_t invNewCount)
                {
                    // Calculate the index
                    // MODIFY FOR MULTI GPU: idx must be multi GPU aware
                    const label_t idx = device::idx();

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
                    const scalar_t Sxx = S<index::xx()>(u, u, mxx);
                    const scalar_t Sxy = S<index::xy()>(u, v, mxy);
                    const scalar_t Sxz = S<index::xz()>(u, w, mxz);
                    const scalar_t Syy = S<index::yy()>(v, v, myy);
                    const scalar_t Syz = S<index::yz()>(v, w, myz);
                    const scalar_t Szz = S<index::zz()>(w, w, mzz);

                    // Read the mean from global
                    const scalar_t SxxMean = SMeanPtrs.ptr<0>()[idx];
                    const scalar_t SxyMean = SMeanPtrs.ptr<1>()[idx];
                    const scalar_t SxzMean = SMeanPtrs.ptr<2>()[idx];
                    const scalar_t SyyMean = SMeanPtrs.ptr<3>()[idx];
                    const scalar_t SyzMean = SMeanPtrs.ptr<4>()[idx];
                    const scalar_t SzzMean = SMeanPtrs.ptr<5>()[idx];

                    // Read the prime mean from global
                    const scalar_t SxxPrimeMean = SPrimeMeanPtrs.ptr<0>()[idx];
                    const scalar_t SxyPrimeMean = SPrimeMeanPtrs.ptr<1>()[idx];
                    const scalar_t SxzPrimeMean = SPrimeMeanPtrs.ptr<2>()[idx];
                    const scalar_t SyyPrimeMean = SPrimeMeanPtrs.ptr<3>()[idx];
                    const scalar_t SyzPrimeMean = SPrimeMeanPtrs.ptr<4>()[idx];
                    const scalar_t SzzPrimeMean = SPrimeMeanPtrs.ptr<5>()[idx];

                    // Calculate the prime quantity
                    const scalar_t SxxPrime = Sxx - SxxMean;
                    const scalar_t SxyPrime = Sxy - SxyMean;
                    const scalar_t SxzPrime = Sxz - SxzMean;
                    const scalar_t SyyPrime = Syy - SyyMean;
                    const scalar_t SyzPrime = Syz - SyzMean;
                    const scalar_t SzzPrime = Szz - SzzMean;

                    // Update the prime mean value and write back to global
                    const scalar_t SxxPrimeMeanNew = timeAverage(SxxPrimeMean, SxxPrime, invNewCount);
                    const scalar_t SxyPrimeMeanNew = timeAverage(SxyPrimeMean, SxyPrime, invNewCount);
                    const scalar_t SxzPrimeMeanNew = timeAverage(SxzPrimeMean, SxzPrime, invNewCount);
                    const scalar_t SyyPrimeMeanNew = timeAverage(SyyPrimeMean, SyyPrime, invNewCount);
                    const scalar_t SyzPrimeMeanNew = timeAverage(SyzPrimeMean, SyzPrime, invNewCount);
                    const scalar_t SzzPrimeMeanNew = timeAverage(SzzPrimeMean, SzzPrime, invNewCount);

                    // Write back to global memory
                    SPrimeMeanPtrs.ptr<0>()[idx] = SxxPrimeMeanNew;
                    SPrimeMeanPtrs.ptr<1>()[idx] = SxyPrimeMeanNew;
                    SPrimeMeanPtrs.ptr<2>()[idx] = SxzPrimeMeanNew;
                    SPrimeMeanPtrs.ptr<3>()[idx] = SyyPrimeMeanNew;
                    SPrimeMeanPtrs.ptr<4>()[idx] = SyzPrimeMeanNew;
                    SPrimeMeanPtrs.ptr<5>()[idx] = SzzPrimeMeanNew;
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
                      calculatePrime_(initialiserSwitch(fieldNamePrime_)),
                      calculatePrimeMean_(initialiserSwitch(fieldNamePrimeMean_)),
                      xx_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[0], mesh, calculate_)),
                      xy_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[1], mesh, calculate_)),
                      xz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[2], mesh, calculate_)),
                      yy_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[3], mesh, calculate_)),
                      yz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[4], mesh, calculate_)),
                      zz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[5], mesh, calculate_)),
                      xxMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, (calculateMean_ || calculatePrime_ || calculatePrimeMean_))),
                      xyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[1], mesh, (calculateMean_ || calculatePrime_ || calculatePrimeMean_))),
                      xzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[2], mesh, (calculateMean_ || calculatePrime_ || calculatePrimeMean_))),
                      yyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[3], mesh, (calculateMean_ || calculatePrime_ || calculatePrimeMean_))),
                      yzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[4], mesh, (calculateMean_ || calculatePrime_ || calculatePrimeMean_))),
                      zzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[5], mesh, (calculateMean_ || calculatePrime_ || calculatePrimeMean_))),
                      xxPrime_(objectAllocator<VelocitySet, time::instantaneous>(componentNamesPrime_[0], mesh, calculatePrime_)),
                      xyPrime_(objectAllocator<VelocitySet, time::instantaneous>(componentNamesPrime_[1], mesh, calculatePrime_)),
                      xzPrime_(objectAllocator<VelocitySet, time::instantaneous>(componentNamesPrime_[2], mesh, calculatePrime_)),
                      yyPrime_(objectAllocator<VelocitySet, time::instantaneous>(componentNamesPrime_[3], mesh, calculatePrime_)),
                      yzPrime_(objectAllocator<VelocitySet, time::instantaneous>(componentNamesPrime_[4], mesh, calculatePrime_)),
                      zzPrime_(objectAllocator<VelocitySet, time::instantaneous>(componentNamesPrime_[5], mesh, calculatePrime_)),
                      xxPrimeMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesPrimeMean_[0], mesh, calculatePrimeMean_)),
                      xyPrimeMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesPrimeMean_[1], mesh, calculatePrimeMean_)),
                      xzPrimeMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesPrimeMean_[2], mesh, calculatePrimeMean_)),
                      yyPrimeMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesPrimeMean_[3], mesh, calculatePrimeMean_)),
                      yzPrimeMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesPrimeMean_[4], mesh, calculatePrimeMean_)),
                      zzPrimeMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesPrimeMean_[5], mesh, calculatePrimeMean_))
                {
                    // Set the cache config to prefer L1
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::instantaneous, cudaFuncCachePreferL1));
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::instantaneousAndMean, cudaFuncCachePreferL1));
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::mean, cudaFuncCachePreferL1));
                    checkCudaErrors(cudaFuncSetCacheConfig(kernel::prime, cudaFuncCachePreferL1));
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
                 * @brief Check if mean calculation is enabled
                 * @return True if mean calculation is enabled
                 **/
                __host__ inline constexpr bool calculatePrime() const noexcept
                {
                    return calculatePrime_;
                }

                /**
                 * @brief Check if mean calculation is enabled
                 * @return True if mean calculation is enabled
                 **/
                __host__ inline constexpr bool calculatePrimeMean() const noexcept
                {
                    return calculatePrimeMean_;
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
                 * @brief Calculate both the instantaneous and time-averaged strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculatePrime([[maybe_unused]] const label_t timeStep) noexcept
                {
                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            strainRate::kernel::prime<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                {xx_.ptr(), xy_.ptr(), xz_.ptr(), yy_.ptr(), yz_.ptr(), zz_.ptr()},
                                {xxMean_.ptr(), xyMean_.ptr(), xzMean_.ptr(), yyMean_.ptr(), yzMean_.ptr(), zzMean_.ptr()},
                                {xxPrime_.ptr(), xyPrime_.ptr(), xzPrime_.ptr(), yyPrime_.ptr(), yzPrime_.ptr(), zzPrime_.ptr()});
                        });
                }

                /**
                 * @brief Calculate both the instantaneous and time-averaged strain rate tensor components
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void calculatePrimeMean(const label_t timeStep) noexcept
                {
                    const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(timeStep + 1);

                    host::constexpr_for<0, N>(
                        [&](const auto stream)
                        {
                            strainRate::kernel::primeMean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
                                devPtrs_,
                                {xxMean_.ptr(), xyMean_.ptr(), xzMean_.ptr(), yyMean_.ptr(), yzMean_.ptr(), zzMean_.ptr()},
                                {xxPrime_.ptr(), xyPrime_.ptr(), xzPrime_.ptr(), yyPrime_.ptr(), yzPrime_.ptr(), zzPrime_.ptr()},
                                invNewCount);
                        });
                }

                /**
                 * @brief Saves the instantaneous strain rate tensor components to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveInstantaneous(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<6, scalar_t>(
                            xx_.ptr(), xy_.ptr(),
                            xz_.ptr(), yy_.ptr(),
                            yz_.ptr(), zz_.ptr()),
                        mesh_);

                    fileIO::writeFile<time::instantaneous>(
                        fieldName_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNames_,
                        hostWriteBuffer_.data(),
                        timeStep);
                }

                /**
                 * @brief Saves the mean strain rate tensor components to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void saveMean(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<6, scalar_t>(
                            xx_.ptr(), xy_.ptr(),
                            xz_.ptr(), yy_.ptr(),
                            yz_.ptr(), zz_.ptr()),
                        mesh_);

                    fileIO::writeFile<time::timeAverage>(
                        fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesMean_,
                        hostWriteBuffer_.data(),
                        timeStep);
                }

                /**
                 * @brief Saves the mean strain rate tensor components to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void savePrime(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<6, scalar_t>(
                            xxPrime_.ptr(), xyPrime_.ptr(),
                            xzPrime_.ptr(), yyPrime_.ptr(),
                            yzPrime_.ptr(), zzPrime_.ptr()),
                        mesh_);

                    fileIO::writeFile<time::instantaneous>(
                        fieldNamePrime_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesPrime_,
                        hostWriteBuffer_.data(),
                        timeStep);
                }

                /**
                 * @brief Saves the mean strain rate tensor components to file
                 * @param[in] timeStep Current simulation time step
                 **/
                __host__ void savePrimeMean(const label_t timeStep) noexcept
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<6, scalar_t>(
                            xxPrimeMean_.ptr(), xyPrimeMean_.ptr(),
                            xzPrimeMean_.ptr(), yyPrimeMean_.ptr(),
                            yzPrimeMean_.ptr(), zzPrimeMean_.ptr()),
                        mesh_);

                    fileIO::writeFile<time::timeAverage>(
                        fieldNamePrimeMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        mesh_,
                        componentNamesPrimeMean_,
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
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

                /**
                 * @brief Field name for instantaneous components
                 **/
                const std::string fieldName_ = "S";

                /**
                 * @brief Field name for mean components
                 **/
                const std::string fieldNameMean_ = fieldName_ + "Mean";

                /**
                 * @brief Field name for mean components
                 **/
                const std::string fieldNamePrime_ = fieldName_ + "Prime";

                /**
                 * @brief Field name for mean components
                 **/
                const std::string fieldNamePrimeMean_ = fieldName_ + "PrimeMean";

                /**
                 * @brief Instantaneous component names
                 **/
                const std::vector<std::string> componentNames_ = {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"};

                /**
                 * @brief Mean component names
                 **/
                const std::vector<std::string> componentNamesMean_ = string::catenate(componentNames_, "Mean");

                /**
                 * @brief Mean component names
                 **/
                const std::vector<std::string> componentNamesPrime_ = string::catenate(componentNames_, "Prime");

                /**
                 * @brief Mean component names
                 **/
                const std::vector<std::string> componentNamesPrimeMean_ = string::catenate(componentNames_, "PrimeMean");

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
                 * @brief Flag for prime calculation
                 **/
                const bool calculatePrime_;

                /**
                 * @brief Flag for prime calculation
                 **/
                const bool calculatePrimeMean_;

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

                /**
                 * @brief Fluctuating strain rate tensor components
                 **/
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> xxPrime_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> xyPrime_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> xzPrime_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> yyPrime_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> yzPrime_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> zzPrime_;

                /**
                 * @brief Fluctuating strain rate tensor components
                 **/
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> xxPrimeMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> xyPrimeMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> xzPrimeMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> yyPrimeMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> yzPrimeMean_;
                device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::timeAverage> zzPrimeMean_;
            };
        }
    }
}

#endif