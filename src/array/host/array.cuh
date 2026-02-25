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
    This file defines the host array specializations for both pinned and
    pageable memory. The host arrays are designed to store field data on the
    CPU side, with different allocation strategies depending on the use case.
    The pageable specialization uses std::vector for automatic memory management,
    while the pinned specialization manages raw pointers to CUDA page-locked
    memory for faster host-device transfers. Both classes inherit from a common
    base that provides access to the field name and associated lattice mesh.

Namespace
    LBM::host

SourceFiles
    host/array.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAY_CUH
#define __MBLBM_HOSTARRAY_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @brief Forward declaration of the primary template
         **/
        template <const host::mallocType AllocationType, typename T, class VelocitySet, const time::type TimeType>
        class array;

        /**
         * @brief Base class for host-side arrays, storing field name and mesh reference.
         *
         * This class provides common members (name, mesh) and their accessors.
         * It is intended to be inherited by specializations for different
         * allocation types (PINNED, PAGED). The destructor is virtual to allow
         * proper cleanup in derived classes.
         *
         * @tparam T Fundamental type of the array elements.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam TimeType Type of time stepping (instantaneous or timeAverage)
         **/
        template <typename T, class VelocitySet, const time::type TimeType>
        class arrayBase
        {
        protected:
            /**
             * @brief Name of the field
             **/
            const name_t name_;

            /**
             * @brief Reference to the lattice mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Construct a base object with name and mesh.
             * @param[in] name Field name.
             * @param[in] mesh The lattice mesh
             **/
            __host__ [[nodiscard]] arrayBase(
                const name_t &name,
                const host::latticeMesh &mesh) noexcept
                : name_(name),
                  mesh_(mesh) {}

        public:
            /**
             * @brief Virtual destructor
             **/
            __host__ virtual ~arrayBase() {}

            /**
             * @brief Disable copying
             **/
            __host__ [[nodiscard]] arrayBase(const arrayBase &) = delete;
            __host__ [[nodiscard]] arrayBase &operator=(const arrayBase &) = delete;

            /**
             * @brief Get the field name.
             * @return Const reference to the name string.
             **/
            __host__ [[nodiscard]] inline constexpr const name_t &name() const noexcept { return name_; }

            /**
             * @brief Get the associated lattice mesh.
             * @return Const reference to the mesh.
             **/
            __host__ [[nodiscard]] inline constexpr const host::latticeMesh &mesh() const noexcept { return mesh_; }

            /**
             * @brief Returns the time type of the array.
             * @return time::type value (instantaneous or time‑averaged).
             **/
            __host__ [[nodiscard]] static inline consteval time::type timeType() noexcept { return TimeType; }
        };
    }
}

#include "paged.cuh"
#include "pinned.cuh"

#endif
