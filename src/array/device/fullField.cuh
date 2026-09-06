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
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

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
         **/
        template <typename T, class VelocitySet>
        class array : public fieldType<1>, arrayBase<T>
        {
        private:
            /**
             * @brief Bring base members into scope
             **/
            using arrayBase<T>::ptr_;
            using arrayBase<T>::allocate_device_segment;

            /**
             * @brief Alias for the current specialization
             **/
            using This = array<T, VelocitySet>;
            using FieldType = fieldType<1>;

        public:
            __host__ [[nodiscard]] array(
                [[maybe_unused]] const name_t &name,
                const name_t &componentName,
                const host::latticeMesh &mesh,
                const T value,
                const programControl &programCtrl,
                const bool allocate = true)
                : FieldType(componentName),
                  arrayBase<T>(
                      This::allocate_on_devices(
                          mesh, value, allocate, programCtrl),
                      mesh,
                      programCtrl)
            {
                initialise_boundary_condition(componentName, programCtrl.deviceList(), programCtrl.Ma() / std::sqrt(static_cast<scalar_t>(3)));
            }

            /**
             * @brief Construct a device array from checkpoint or initial condition files, with a specified component name.
             * @param[in] name Name of the field (e.g., "U" for velocity).
             * @param[in] componentName Name of the component (e.g., "U_x" for x‑velocity).
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             * @param[in] allocate If false, the array is not allocated.
             **/
            __host__ [[nodiscard]] array(
                const name_t &name,
                const name_t &componentName,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : FieldType(componentName),
                  arrayBase<T>(
                      This::allocate_on_devices(
                          from_host(name, componentName, mesh, programCtrl),
                          allocate, programCtrl),
                      mesh,
                      programCtrl)
            {
                initialise_boundary_condition(componentName, programCtrl.deviceList(), programCtrl.Ma() / std::sqrt(static_cast<scalar_t>(3)));
            }

            /**
             * @brief Default destructor
             **/
            __host__ ~array() {}

            /**
             * @brief Get read-only pointer to device memory for a given GPU.
             * @tparam Idx Type that can be converted to device::label_t.
             * @param[in] idx Virtual device index.
             * @return Const pointer to device memory.
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline const T *constPtr(const Idx idx) const noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get mutable pointer to device memory for a given GPU.
             * @tparam Idx Type that can be converted to device::label_t.
             * @param[in] idx Virtual device index.
             * @return Pointer to device memory.
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline T *ptr(const Idx idx) noexcept
            {
                return ptr_[idx];
            }

            /**
             * @brief Get mutable pointer to device memory for a given GPU.
             * @tparam Idx Type that can be converted to device::label_t.
             * @param[in] idx Virtual device index.
             * @return Pointer to device memory.
             **/
            template <typename Idx>
            __device__ __host__ [[nodiscard]] inline T *mutPtr(const Idx idx) const noexcept
            {
                return ptr_[idx];
            }

            // /**
            //  * @brief Get the field name.
            //  * @return Const reference to the name string.
            //  **/
            // __host__ [[nodiscard]] inline const name_t &name() const noexcept { return FieldType::name_; }

        private:
            /**
             * @brief Allocate all GPU segments for a full field from a raw host pointer.
             * @param[in] mesh The lattice mesh
             * @param[in] hostArrayGlobal Raw pointer to host data.
             * @param[in] allocate If false, returns nullptr.
             * @param[in] programCtrl The program control object
             * @return Host array of device pointers, or nullptr if not allocated.
             **/
            __host__ [[nodiscard]] static inline T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T *hostArrayGlobal,
                const bool allocate,
                const programControl &programCtrl,
                const host::label_t allocationSize)
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
            __host__ [[nodiscard]] static inline T **allocate_on_devices(
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
            __host__ [[nodiscard]] static inline T **allocate_on_devices(
                const host::array<MallocType, T> &hostArrayGlobal,
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
            __host__ [[nodiscard]] static inline T **allocate_on_devices(
                const host::latticeMesh &mesh,
                const T val,
                const bool allocate,
                const programControl &programCtrl)
            {
                const std::vector<T> toAllocate(static_cast<host::label_t>(allocate) * mesh.size(), val);
                return This::allocate_on_devices(mesh, toAllocate.data(), allocate, programCtrl, mesh.sizePerDevice());
            }

            /**
             * @brief Initialise boundary condition values on all GPUs for velocity fields.
             * @param[in] name Field name ("u", "v", or "w").
             * @param[in] deviceList List of device indices.
             **/
            __host__ static void initialise_boundary_condition(
                const name_t &name,
                const std::vector<deviceIndex_t> &deviceList,
                const scalar_t U_inf) noexcept
            {
                if ((name == "U_x") || (name == "U_y") || (name == "U_z"))
                {
                    const device::label_t i = (name == "U_x") ? 0 : ((name == "U_y") ? 1 : 2);

                    const boundaryValue<VelocitySet, false> North(name, "North");
                    const boundaryValue<VelocitySet, false> South(name, "South");
                    const boundaryValue<VelocitySet, false> East(name, "East");
                    const boundaryValue<VelocitySet, false> West(name, "West");
                    const boundaryValue<VelocitySet, false> Back(name, "Back");
                    const boundaryValue<VelocitySet, false> Front(name, "Front");

                    for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < deviceList.size(); ++virtualDeviceIndex)
                    {
                        errorHandler::handle(cudaSetDevice(deviceList[virtualDeviceIndex]));
                        device::copyToSymbol(device::U_North, North() * U_inf, i);
                        device::copyToSymbol(device::U_South, South() * U_inf, i);
                        device::copyToSymbol(device::U_East, East() * U_inf, i);
                        device::copyToSymbol(device::U_West, West() * U_inf, i);
                        device::copyToSymbol(device::U_Back, Back() * U_inf, i);
                        device::copyToSymbol(device::U_Front, Front() * U_inf, i);
                    }
                }
            }

            /**
             * @brief Constructs a host array with a given name
             **/
            __host__ [[nodiscard]] host::array<host::PAGED, T> from_host(
                const name_t &name,
                const name_t &componentName,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
            {
                return host::array<host::PAGED, T>(name, componentName, mesh, programCtrl, boundaryFields<VelocitySet, true>(componentName));
            }
        };
    }
}

#endif
