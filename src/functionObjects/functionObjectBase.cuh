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
    Base class for LBM function objects, containing common data members.

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FUNCTIONOBJECTBASE_CUH
#define __MBLBM_FUNCTIONOBJECTBASE_CUH

namespace LBM
{
    namespace functionObjects
    {
        /**
         * @brief Base class for LBM function objects, providing common data members.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam N The number of spatial components of the function object
         */
        template <class VelocitySet, const host::label_t N>
        class FunctionObjectBase
        {
        protected:
            /**
             * @brief Name of the field and its time-averaged counterpart
             **/
            const name_t name_;
            const name_t nameMean_;

            /**
             * @brief Name of the field's components and their time-averaged counterpart
             **/
            const words_t componentNames_;
            const words_t componentNamesMean_;

            /**
             * @brief Switches to determine whether or not the field is to be calculated
             **/
            const bool calculate_;
            const bool calculateMean_;

            /**
             * @brief Reference to the write buffer
             **/
            host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

            /**
             * @brief Reference to lattice mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Device pointer collection
             **/
            const device::scalarField<VelocitySet, time::instantaneous> &rho_;
            const device::vectorField<VelocitySet, time::instantaneous> &U_;
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi_;

            /**
             * @brief Stream handler for CUDA operations
             **/
            const streamHandler &streamsLBM_;

            /**
             * @brief Configures the kernels to allocate no dynamic shared memory and prefer L1 cache
             * @param[in] programCtrl The program control object
             **/
            template <class Kernel>
            __host__ static inline constexpr void configure(const programControl &programCtrl) noexcept
            {
                programCtrl.configure<0, false>(Kernel::instantaneous());
                programCtrl.configure<0, false>(Kernel::instantaneousAndMean());
                programCtrl.configure<0, false>(Kernel::mean());
            }

            /**
             * @brief Construct the component names from the field name
             * @param[in] name Name of the field
             **/
            __host__ [[nodiscard]] static inline constexpr const words_t componentNames(const name_t &name)
            {
                if constexpr (N == 1)
                {
                    return {name};
                }

                if constexpr (N == 3)
                {
                    return string::catenate(name, {"_x", "_y", "_z"});
                }

                if constexpr (N == 6)
                {
                    return string::catenate(name, {"_xx", "_xy", "_xz", "_yy", "_yz", "_zz"});
                }

                if constexpr (N == 10)
                {
                    return solutionVariableNames;
                }
            }

            /**
             * @brief Return the pointers that correspond to a particular device partition
             * @param[in] idx The device index
             * @return Pointers to the 10 solution variables allocated on device idx
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> devPtrs(const host::label_t idx) const noexcept
            {
                return {rho_.self().ptr(idx),
                        U_.x().ptr(idx), U_.y().ptr(idx), U_.z().ptr(idx),
                        Pi_.xx().ptr(idx), Pi_.xy().ptr(idx), Pi_.xz().ptr(idx),
                        Pi_.yy().ptr(idx), Pi_.yz().ptr(idx), Pi_.zz().ptr(idx)};
            }

            /**
             * @brief Calculate a time-averaged quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             * @param[out] meanCount Counter of time averaging steps
             **/
            template <class FunctionObject, class F>
            __host__ inline void mean(
                F *func,
                FunctionObject &object,
                host::label_t &meanCount)
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(meanCount + 1);

                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                {
                    func<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(devPtrs(stream), object.meanPtrs(stream), invNewCount);
                }

                meanCount++;
            }

            /**
             * @brief Calculate an instantaneous quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             **/
            template <class FunctionObject, class F>
            __host__ inline void instantaneous(
                F *func,
                FunctionObject &object)
            {
                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                {
                    func<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(devPtrs(stream), object.meanPtrs(stream));
                }
            }

            /**
             * @brief Calculate both an instantaneous and a time-averaged quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             * @param[out] meanCount Counter of time averaging steps
             **/
            template <class FunctionObject, class F>
            __host__ inline void instantaneousAndMean(
                F *func,
                FunctionObject &object,
                host::label_t &meanCount)
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(meanCount + 1);

                for (host::label_t stream = 0; stream < streamsLBM_.streams().size(); stream++)
                {
                    func<<<mesh_.gridBlock(), host::latticeMesh::threadBlock(), 0, streamsLBM_.streams()[stream]>>>(devPtrs(stream), object.instantaneousPtrs(stream), object.meanPtrs(stream), invNewCount);
                }

                meanCount++;
            }

        public:
            /**
             * @brief Constructs a function object base with common input data.
             * @param[in] hostWriteBuffer Host buffer for writing output data.
             * @param[in] mesh Lattice mesh.
             * @param[in] rho Density field.
             * @param[in] u x‑velocity field.
             * @param[in] v y‑velocity field.
             * @param[in] w z‑velocity field.
             * @param[in] mxx xx‑moment field.
             * @param[in] mxy xy‑moment field.
             * @param[in] mxz xz‑moment field.
             * @param[in] myy yy‑moment field.
             * @param[in] myz yz‑moment field.
             * @param[in] mzz zz‑moment field.
             * @param[in] streamsLBM Stream handler for LBM operations
             */
            __host__ [[nodiscard]] FunctionObjectBase(
                const name_t &name,
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::latticeMesh &mesh,
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const streamHandler &streamsLBM) noexcept
                : name_(name),
                  nameMean_(name + "Mean"),
                  componentNames_(componentNames(name_)),
                  componentNamesMean_(componentNames(nameMean_)),
                  calculate_(initialiserSwitch(name_)),
                  calculateMean_(initialiserSwitch(nameMean_)),
                  hostWriteBuffer_(hostWriteBuffer),
                  mesh_(mesh),
                  rho_(rho),
                  U_(U),
                  Pi_(Pi),
                  streamsLBM_(streamsLBM) {}

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
        };
    }
}

#endif // __MBLBM_FUNCTIONOBJECTBASE_CUH