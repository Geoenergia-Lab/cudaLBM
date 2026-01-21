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
    A templated class for allocating arrays on the GPU

Namespace
    LBM::device

SourceFiles
    deviceArray.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEARRAY_CUH
#define __MBLBM_DEVICEARRAY_CUH

namespace LBM
{
    namespace device
    {
        template <typename T, class VelocitySet, const time::type TimeType>
        class array
        {
        public:
            /**
             * @brief Constructs a device array from host data
             * @tparam VelocitySet Template parameter for velocity set configuration
             * @param[in] hostArray Source data allocated on host memory
             * @param[in] mesh Lattice mesh defining array dimensions
             * @post Device memory is allocated and initialized with host data
             **/
            __host__ [[nodiscard]] array(const host::array<false, T, VelocitySet, TimeType> &hostArray)
                : ptr_(device::allocateArray<T>(hostArray.arr())),
                  name_(hostArray.name()),
                  mesh_(hostArray.mesh())
            {
                initialise_boundary_condition(name_);
            };

            /**
             * @brief Constructs a device array with field initialization
             * @param[in] name Name identifier for the field
             * @param[in] mesh Lattice mesh defining array dimensions
             * @param[in] programCtrl Program control parameters
             * @post Array is initialized from latest time step or initial conditions
             **/
            __host__ [[nodiscard]] array(
                const std::string &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : ptr_(toDevice(host::array<false, T, VelocitySet, TimeType>(name, mesh, programCtrl))),
                  name_(name),
                  mesh_(mesh)
            {
                initialise_boundary_condition(name_);
            };

            /**
             * @brief Constructs a device array with field initialization
             * @param[in] name Name identifier for the field
             * @param[in] mesh Lattice mesh defining array dimensions
             * @param[in] value The uniform value to initialise the array to
             * @post Array is initialized from latest time step or initial conditions
             **/
            __host__ [[nodiscard]] array(
                const std::string &name,
                const host::latticeMesh &mesh,
                const T value)
                : ptr_(device::allocateArray<T>(mesh.nPoints(), value)),
                  name_(name),
                  mesh_(mesh)
            {
                initialise_boundary_condition(name_);
            };

            /**
             * @brief Allocates no memory on the device
             **/
            __host__ [[nodiscard]] array(const std::string &name, const host::latticeMesh &mesh)
                : ptr_(nullptr),
                  name_(name),
                  mesh_(mesh)
            {
                initialise_boundary_condition(name_);
            };

            /**
             * @brief Destructor - automatically releases device memory
             * @note Noexcept guarantee: failsafe if cudaFree fails
             **/
            ~array() noexcept
            {
                checkCudaErrors(cudaFree(ptr_));
            }

            /**
             * @brief Element access operator
             * @param[in] i Index of element to access
             * @return Value at index @p i
             * @warning No bounds checking performed
             **/
            __device__ __host__ [[nodiscard]] inline T operator[](const label_t i) const noexcept
            {
                return ptr_[i];
            }

            /**
             * @brief Get read-only access to underlying data
             * @return Const pointer to device memory
             **/
            __device__ __host__ [[nodiscard]] inline const T *ptr() const noexcept
            {
                return ptr_;
            }

            /**
             * @brief Get mutable access to underlying data
             * @return Pointer to device memory
             **/
            __device__ __host__ [[nodiscard]] inline T *ptr() noexcept
            {
                return ptr_;
            }

            /**
             * @brief Get array identifier name
             * @return Const reference to name string
             **/
            __host__ [[nodiscard]] inline const std::string &name() const noexcept
            {
                return name_;
            }

            /**
             * @brief Get associated mesh object
             * @return Const reference to lattice mesh
             **/
            __host__ [[nodiscard]] inline const host::latticeMesh &mesh() const noexcept
            {
                return mesh_;
            }

            /**
             * @brief Get total number of elements
             * @return Number of elements in array
             * @note Returns mesh point count - assumes 1:1 element-to-point mapping
             **/
            __host__ [[nodiscard]] inline constexpr label_t size() const noexcept
            {
                return mesh_.nPoints();
            }

            __host__ [[nodiscard]] inline consteval time::type timeType() const noexcept
            {
                return TimeType;
            }

        private:
            /**
             * @brief Pointer to the data
             **/
            T *const ptrRestrict ptr_;

            /**
             * @brief Names of the solution variables
             **/
            const std::string &name_;

            /**
             * @brief Reference to the mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Copies the underlying std::vector of a host::array type to the device
             * @param[in] hostArray The host::array to be copied to the device
             * @return A pointer to the copied data
             **/
            __host__ [[nodiscard]] T *toDevice(const host::array<false, T, VelocitySet, TimeType> &hostArray)
            {
                return device::allocateArray<T>(hostArray.arr());
            }

            /**
             * @brief Converts a variable name to an index
             * @param[in] name The name of the variable
             * @return The index of the variable, or -1 if not found
             **/
            __host__ [[nodiscard]] static inline label_t name_to_index(const std::string &name) noexcept
            {
                if (name == "u")
                {
                    return 0;
                }
                else if (name == "v")
                {
                    return 1;
                }
                else if (name == "w")
                {
                    return 2;
                }
                return static_cast<label_t>(-1);
            }

            /**
             * @brief Initialises boundary condition values on the GPU for a given variable name
             * @param name The name of the variable to initialise boundary conditions for
             **/
            __host__ static void initialise_boundary_condition(const std::string &name) noexcept
            {
                if ((name == "u") || (name == "v") || (name == "w"))
                {
                    const label_t i = name_to_index(name);

                    const boundaryValue<VelocitySet> North(name, "North");
                    const boundaryValue<VelocitySet> South(name, "South");
                    const boundaryValue<VelocitySet> East(name, "East");
                    const boundaryValue<VelocitySet> West(name, "West");
                    const boundaryValue<VelocitySet> Front(name, "Front");
                    const boundaryValue<VelocitySet> Back(name, "Back");

                    copyToSymbol(device::U_North, North(), i);
                    copyToSymbol(device::U_South, South(), i);
                    copyToSymbol(device::U_East, East(), i);
                    copyToSymbol(device::U_West, West(), i);
                    copyToSymbol(device::U_Front, Front(), i);
                    copyToSymbol(device::U_Back, Back(), i);
                }
            }
        };
    }
}

#endif
