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
                const device::ptrCollection<10, const scalar_t> &devPtrs,
                const device::label_t idx) noexcept
            {
                const thread::array<scalar_t, 3> U = read_from_moments<index::u, index::v, index::w>(devPtrs, idx);

                const thread::array<scalar_t, 6> M = read_from_moments<index::xx, index::xy, index::xz, index::yy, index::yz, index::zz>(devPtrs, idx);

                return {calculate<index::xx>(U[0], U[0], M[0]), calculate<index::xy>(U[0], U[1], M[1]), calculate<index::xz>(U[0], U[2], M[2]), calculate<index::yy>(U[1], U[1], M[3]), calculate<index::yz>(U[1], U[2], M[4]), calculate<index::zz>(U[2], U[2], M[5])};
            }

            __host__ [[nodiscard]] static inline consteval host::label_t MIN_BLOCKS_PER_MP() noexcept { return 3; }
        };

        namespace strainRateTensorDetail
        {
            using This = S;

#include "commonKernelDefinitions.cuh"
        }

        /**
         * @brief Class for managing strain rate tensor calculations in LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam N The number of streams (compile-time constant)
         **/
        template <class VelocitySet>
        class strainRateTensor
        {
        public:
            /**
             * @brief Constructs a strain rate tensor object
             * @param[in] mesh The lattice mesh
             * @param[in] devPtrs Device pointer collection for memory access
             * @param[in] streamsLBM Stream handler for CUDA operations
             **/
            __host__ [[nodiscard]] strainRateTensor(
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
                  xx_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[0], mesh, calculate_, programCtrl)),
                  xy_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[1], mesh, calculate_, programCtrl)),
                  xz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[2], mesh, calculate_, programCtrl)),
                  yy_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[3], mesh, calculate_, programCtrl)),
                  yz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[4], mesh, calculate_, programCtrl)),
                  zz_(objectAllocator<VelocitySet, time::instantaneous>(componentNames_[5], mesh, calculate_, programCtrl)),
                  xxMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[0], mesh, calculateMean_, programCtrl)),
                  xyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[1], mesh, calculateMean_, programCtrl)),
                  xzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[2], mesh, calculateMean_, programCtrl)),
                  yyMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[3], mesh, calculateMean_, programCtrl)),
                  yzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[4], mesh, calculateMean_, programCtrl)),
                  zzMean_(objectAllocator<VelocitySet, time::timeAverage>(componentNamesMean_[5], mesh, calculateMean_, programCtrl))
            {
                // Set the cache config to prefer L1
                programCtrl.configure<0, false>(strainRateTensorDetail::instantaneous);
                programCtrl.configure<0, false>(strainRateTensorDetail::instantaneousAndMean);
                programCtrl.configure<0, false>(strainRateTensorDetail::mean);
            };

            /**
             * @brief Default destructor
             **/
            ~strainRateTensor() {}

            /**
             * @brief Disable copying
             **/
            __host__ [[nodiscard]] strainRateTensor(const strainRateTensor &) = delete;
            __host__ [[nodiscard]] strainRateTensor &operator=(const strainRateTensor &) = delete;

            /**
             * @brief Check if instantaneous calculation is enabled
             * @return True if instantaneous calculation is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doInstantaneous() const noexcept
            {
                return calculate_;
            }

            /**
             * @brief Check if mean calculation is enabled
             * @return True if mean calculation is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doMean() const noexcept
            {
                return calculateMean_;
            }

            /**
             * @brief Calculate instantaneous strain rate tensor components
             * @param[in] timeStep Current simulation time step
             **/
            __host__ void calculateInstantaneous() noexcept
            {
                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                {
                    strainRateTensorDetail::instantaneous<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
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
                        {xx_.ptr(stream),
                         xy_.ptr(stream),
                         xz_.ptr(stream),
                         yy_.ptr(stream),
                         yz_.ptr(stream),
                         zz_.ptr(stream)});
                }
            }

            /**
             * @brief Calculate time-averaged strain rate tensor components
             * @param[in] timeStep Current simulation time step
             **/
            __host__ void calculateMean() noexcept
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(xxMean_.meanCount() + 1);

                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                {
                    strainRateTensorDetail::mean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
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
                        {xxMean_.ptr(stream),
                         xyMean_.ptr(stream),
                         xzMean_.ptr(stream),
                         yyMean_.ptr(stream),
                         yzMean_.ptr(stream),
                         zzMean_.ptr(stream)},
                        invNewCount);
                }

                xxMean_.meanCountRef()++;
            }

            /**
             * @brief Calculate both the instantaneous and time-averaged strain rate tensor components
             * @param[in] timeStep Current simulation time step
             **/
            __host__ void calculateInstantaneousAndMean() noexcept
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(xxMean_.meanCount() + 1);

                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                {
                    strainRateTensorDetail::instantaneousAndMean<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(
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
                        {xx_.ptr(stream), xy_.ptr(stream), xz_.ptr(stream), yy_.ptr(stream), yz_.ptr(stream), zz_.ptr(stream)},
                        {xxMean_.ptr(stream), xyMean_.ptr(stream), xzMean_.ptr(stream), yyMean_.ptr(stream), yzMean_.ptr(stream), zzMean_.ptr(stream)},
                        invNewCount);
                }

                xxMean_.meanCountRef()++;
            }

            /**
             * @brief Saves the instantaneous strain rate tensor components to file
             * @param[in] timeStep Current simulation time step
             **/
            __host__ void saveInstantaneous(const host::label_t timeStep) noexcept
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < xx_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<6, scalar_t>(
                            xx_.ptr(virtualDeviceIndex), xy_.ptr(virtualDeviceIndex),
                            xz_.ptr(virtualDeviceIndex), yy_.ptr(virtualDeviceIndex),
                            yz_.ptr(virtualDeviceIndex), zz_.ptr(virtualDeviceIndex)),
                        mesh_,
                        virtualDeviceIndex);
                }

                postProcess::LBMBin::write<time::instantaneous>(
                    fieldName_ + "_" + std::to_string(timeStep) + ".LBMBin",
                    mesh_,
                    componentNames_,
                    hostWriteBuffer_.data(),
                    timeStep,
                    0);
            }

            /**
             * @brief Saves the mean strain rate tensor components to file
             * @param[in] timeStep Current simulation time step
             **/
            __host__ void saveMean(const host::label_t timeStep) noexcept
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < xxMean_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                {
                    hostWriteBuffer_.copy_from_device(
                        device::ptrCollection<6, scalar_t>(
                            xxMean_.ptr(virtualDeviceIndex), xyMean_.ptr(virtualDeviceIndex),
                            xzMean_.ptr(virtualDeviceIndex), yyMean_.ptr(virtualDeviceIndex),
                            yzMean_.ptr(virtualDeviceIndex), zzMean_.ptr(virtualDeviceIndex)),
                        mesh_,
                        virtualDeviceIndex);
                }

                postProcess::LBMBin::write<time::timeAverage>(
                    fieldNameMean_ + "_" + std::to_string(timeStep) + ".LBMBin",
                    mesh_,
                    componentNamesMean_,
                    hostWriteBuffer_.data(),
                    timeStep,
                    xxMean_.meanCount());
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
             * @brief Get the component names for instantaneous tensor
             * @return Vector of component names
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const words_t &componentNames() const noexcept
            {
                return componentNames_;
            }

            /**
             * @brief Get the component names for mean tensor
             * @return Vector of component names
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const words_t &componentNamesMean() const noexcept
            {
                return componentNamesMean_;
            }

        private:
            host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

            /**
             * @brief Field name for instantaneous components
             **/
            const name_t fieldName_ = "S";

            /**
             * @brief Field name for mean components
             **/
            const name_t fieldNameMean_ = fieldName_ + "Mean";

            /**
             * @brief Instantaneous component names
             **/
            const words_t componentNames_ = {"S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"};

            /**
             * @brief Mean component names
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

#endif