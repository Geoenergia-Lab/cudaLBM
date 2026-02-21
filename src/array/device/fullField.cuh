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
    This file defines the device array specialization for full fields. A full
    field is a standard field that carries a name and may be time‑averaged. This
    specialization manages device memory for the field data, as well as metadata
    such as the field name and averaging count. It provides methods for
    initialising boundary conditions on the device and copying data back to the
    host. Full fields are typically used for primary physical quantities like
    velocity and density, which require named access and may be averaged over
    time steps.

Namespace
    LBM

SourceFiles
    device/fullField.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEARRAY_FULLFIELD_CUH
#define __MBLBM_DEVICEARRAY_FULLFIELD_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @brief Device array for full fields (with name and optional time averaging).
         *
         * This specialization holds a field name, a reference to the mesh, and a counter
         * for time‑averaged fields. It provides methods for boundary condition
         * initialisation and copying data back to the host.
         *
         * @tparam T Fundamental type of the array.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam TimeType Type of time stepping (instantaneous or timeAverage)
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class array<field::FULL_FIELD, T, VelocitySet, TimeType> : public arrayBase<T>
        {
        private:
            /**
             * @brief Bring base members into scope
             **/
            using arrayBase<T>::ptr_;
            using arrayBase<T>::mesh_;
            using arrayBase<T>::programCtrl_;
            using arrayBase<T>::allocate_device_segment;

            /**
             * @brief Alias for the current specialization
             **/
            using This = array<field::FULL_FIELD, T, VelocitySet, TimeType>;

        public:
            /**
             * @brief Construct a device array from an existing host array.
             * @tparam MallocType Type of host memory allocation (e.g., PAGED, PINNED).
             * @param[in] hostArray Source host array.
             * @param[in] programCtrl The program control object
             * @param[in] allocate If false, the array is not allocated (ptr_ remains null).
             **/
            template <const host::mallocType MallocType>
            __host__ [[nodiscard]] array(
                const host::array<MallocType, T, VelocitySet, TimeType> &hostArray,
                const programControl &programCtrl,
                const bool allocate = true)
                : arrayBase<T>(
                      This::allocate_on_devices(
                          hostArray.mesh(), hostArray.data(), allocate, programCtrl),
                      hostArray.mesh(),
                      programCtrl),
                  name_(hostArray.name()),
                  meanCount_(initialiseMeanCount(programCtrl))
            {
                initialise_boundary_condition(name_, programCtrl.deviceList());
            }

            /**
             * @brief Construct a device array with a uniform value.
             * @param[in] name Name of the field.
             * @param[in] mesh The lattice mesh
             * @param[in] value Uniform value to initialise the array.
             * @param[in] programCtrl The program control object
             * @param[in] allocate If false, the array is not allocated.
             **/
            __host__ [[nodiscard]] array(
                const name_t &name,
                const host::latticeMesh &mesh,
                const T value,
                const programControl &programCtrl,
                const bool allocate = true)
                : arrayBase<T>(
                      This::allocate_on_devices(
                          mesh, value, allocate, programCtrl),
                      mesh,
                      programCtrl),
                  name_(name),
                  meanCount_(initialiseMeanCount(programCtrl))
            {
                initialise_boundary_condition(name_, programCtrl.deviceList());
            }

            /**
             * @brief Construct a device array from checkpoint or initial condition files.
             * @param[in] name Name of the field.
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             * @param[in] allocate If false, the array is not allocated.
             **/
            __host__ [[nodiscard]] array(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : arrayBase<T>(
                      This::allocate_on_devices(
                          host::array<host::PAGED, T, VelocitySet, TimeType>(name, mesh, programCtrl),
                          allocate, programCtrl),
                      mesh,
                      programCtrl),
                  name_(name),
                  meanCount_(initialiseMeanCount(programCtrl))
            {
                initialise_boundary_condition(name_, programCtrl.deviceList());
            }

            /**
             * @brief Get read-only pointer to device memory for a given GPU.
             * @tparam Idx Type that can be converted to label_t.
             * @param[in] idx Virtual device index.
             * @return Const pointer to device memory.
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline const T *ptr(const Idx idx) const noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get mutable pointer to device memory for a given GPU.
             * @tparam Idx Type that can be converted to label_t.
             * @param[in] idx Virtual device index.
             * @return Pointer to device memory.
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline T *ptr(const Idx idx) noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get the field name.
             * @return Const reference to the name string.
             **/
            __host__ [[nodiscard]] inline const name_t &name() const noexcept { return name_; }

            /**
             * @brief Get the associated lattice mesh.
             * @return Const reference to the mesh.
             **/
            __host__ [[nodiscard]] inline const host::latticeMesh &mesh() const noexcept { return mesh_; }

            /**
             * @brief Get the program control object.
             * @return Const reference to program control.
             **/
            __host__ [[nodiscard]] inline const programControl &programCtrl() const noexcept { return programCtrl_; }

            /**
             * @brief Get total number of elements (mesh points).
             * @tparam SizeType Desired return type (default label_t).
             * @return Number of elements.
             **/
            template <typename SizeType = label_t>
            __host__ [[nodiscard]] inline constexpr SizeType size() const noexcept
            {
                return mesh_.template nPoints<SizeType>();
            }

            /**
             * @brief Returns the time type of the array.
             * @return time::type value (instantaneous or time‑averaged).
             **/
            __host__ [[nodiscard]] static inline consteval time::type timeType() noexcept
            {
                return TimeType;
            }

            /**
             * @brief Get the current averaging count (for time‑averaged fields).
             * @return Number of time steps averaged so far.
             **/
            __host__ [[nodiscard]] inline constexpr label_t meanCount() const noexcept
            {
                return meanCount_;
            }

            /**
             * @brief Get a reference to the averaging count (for modification).
             * @return Reference to meanCount_.
             **/
            __host__ [[nodiscard]] inline constexpr label_t &meanCountRef() noexcept
            {
                return meanCount_;
            }

            /**
             * @brief Copy the device array to a user‑supplied host pointer.
             * @param[in] hostPtr Destination pointer (host memory).
             * @note Assumes hostPtr has enough space (size() elements).
             **/
            __host__ void copy_to_host(T *const ptrRestrict hostPtr)
            {
                static_assert(MULTI_GPU_ASSERTION());

                const std::size_t nPointsPerDevice = mesh_.template sizePerDevice<std::size_t>();

                const std::size_t nDevices = mesh_.template nDevices<std::size_t>();

                for (std::size_t virtualDeviceIndex = 0; virtualDeviceIndex < nDevices; ++virtualDeviceIndex)
                {
                    const label_t startIndex = virtualDeviceIndex * nPointsPerDevice;
                    errorHandler::check(cudaMemcpy(&(hostPtr[startIndex]), ptr_[virtualDeviceIndex], nPointsPerDevice * sizeof(T), cudaMemcpyDeviceToHost));
                }
            }

        private:
            /**
             * @brief Name of the field
             **/
            const name_t name_;

            /**
             * @brief Number of time steps averaged over (for time‑averaged fields)
             **/
            label_t meanCount_;

            /**
             * @brief Allocate all GPU segments for a full field from a raw host pointer.
             * @param[in] mesh The lattice mesh
             * @param[in] hostArrayGlobal Raw pointer to host data.
             * @param[in] allocate If false, returns nullptr.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers, or nullptr if not allocated.
             **/
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl,
                const label_t allocationSize)
            {
                return (allocate) ? (arrayBase<T>::allocate_on_devices(mesh, hostArrayGlobal, programCtrl, allocationSize)) : (nullptr);
            }

            /**
             * @brief Allocate GPU segments from a std::vector.
             * @param[in] mesh The lattice mesh
             * @param[in] hostArrayGlobal Source vector.
             * @param[in] allocate If false, returns nullptr.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers.
             **/
            __host__ [[nodiscard]] static T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const std::vector<T> &hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl)
            {
                return This::allocate_on_devices(mesh, hostArrayGlobal.data(), allocate, programCtrl, mesh.sizePerDevice());
            }

            /**
             * @brief Allocate GPU segments from another device array (host::array).
             * @tparam MallocType Host memory type.
             * @param[in] hostArrayGlobal Source host array.
             * @param[in] allocate If false, returns nullptr.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers.
             **/
            template <const host::mallocType MallocType>
            __host__ [[nodiscard]] T **allocate_on_devices(
                const host::array<MallocType, T, VelocitySet, TimeType> &hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl)
            {
                return This::allocate_on_devices(hostArrayGlobal.mesh(), hostArrayGlobal.data(), allocate, programCtrl, hostArrayGlobal.mesh().sizePerDevice());
            }

            /**
             * @brief Allocate GPU segments with a uniform value.
             * @param[in] mesh The lattice mesh
             * @param[in] val Uniform value.
             * @param[in] allocate If false, returns nullptr.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers.
             **/
            __host__ [[nodiscard]] T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T val,
                const bool allocate,
                const programControl &programCtrl)
            {
                const std::vector<T> toAllocate(static_cast<std::size_t>(allocate) * mesh.size<std::size_t>(), val);
                return This::allocate_on_devices(mesh, toAllocate.data(), allocate, programCtrl, mesh.sizePerDevice());
            }

            /**
             * @brief Initialise boundary condition values on all GPUs for velocity fields.
             * @param[in] name Field name ("u", "v", or "w").
             * @param[in] deviceList List of device indices.
             **/
            __host__ static void initialise_boundary_condition(
                const name_t &name,
                const std::vector<deviceIndex_t> &deviceList) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::array::initialise_boundary_condition, "Believed to be correct"));

                if ((name == "u") || (name == "v") || (name == "w"))
                {
                    const label_t i = (name == "u") ? 0 : ((name == "v") ? 1 : 2);

                    const boundaryValue<VelocitySet, false> North(name, "North");
                    const boundaryValue<VelocitySet, false> South(name, "South");
                    const boundaryValue<VelocitySet, false> East(name, "East");
                    const boundaryValue<VelocitySet, false> West(name, "West");
                    const boundaryValue<VelocitySet, false> Back(name, "Back");
                    const boundaryValue<VelocitySet, false> Front(name, "Front");

                    for (std::size_t virtualDeviceIndex = 0; virtualDeviceIndex < deviceList.size(); ++virtualDeviceIndex)
                    {
                        errorHandler::check(cudaSetDevice(deviceList[virtualDeviceIndex]));
                        device::copyToSymbol(device::U_North, North(), i);
                        device::copyToSymbol(device::U_South, South(), i);
                        device::copyToSymbol(device::U_East, East(), i);
                        device::copyToSymbol(device::U_West, West(), i);
                        device::copyToSymbol(device::U_Back, Back(), i);
                        device::copyToSymbol(device::U_Front, Front(), i);
                    }
                }
            }
        };
    }
}

#endif
