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
    A class applying boundary conditions to the lid driven cavity case

Namespace
    LBM

SourceFiles
    cases.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_CASES_CUH
#define __MBLBM_CASES_CUH

#include "jetFlow/jetFlow.cuh"
#include "lidDrivenCavity/lidDrivenCavity.cuh"

#define BOUNDARY_CONDITION JET_FLOW
// #define BOUNDARY_CONDITION LID_DRIVEN_CAVITY

namespace LBM
{
    namespace assertions
    {
        namespace boundaryConditions
        {
            __device__ __host__ inline consteval void validate() noexcept
            {
#ifndef BOUNDARY_CONDITION
                static_assert(false, "BOUNDARY_CONDITION must be defined to a valid boundaryCondition_t enumerator");
#else
                static_assert(true, "BOUNDARY_CONDITION must be defined to a valid boundaryCondition_t enumerator");
#endif
            }
        }
    }

    namespace boundaryConditions
    {
        /**
         * @enum Enum
         * @brief Unscoped enumeration listing the available boundary condition types.
         *
         * This enum is defined with an underlying type of `int` to ensure a fixed size.
         * The enumerators are accessible either directly (unscoped) or via the
         * `boundaryCondition_t` typedef (as shown in the code).
         *
         * @var Enum::JET_FLOW
         *      Represents a jet inflow boundary condition.
         * @var Enum::LID_DRIVEN_CAVITY
         *      Represents the lid‑driven cavity boundary condition.
         */
        typedef enum Enum : int
        {
            JET_FLOW,
            LID_DRIVEN_CAVITY
        } boundaryCondition_t;

        /**
         * @brief Primary template for boundary condition traits.
         *
         * This template maps a compile‑time boundary condition value (given as a
         * non‑type template parameter) to a concrete type that implements or represents
         * that boundary condition. Specializations are provided for each known enumerator.
         *
         * @tparam BC A compile‑time constant of type `boundaryCondition_t`.
         */
        template <const boundaryCondition_t BC>
        class traits;

        /**
         * @brief Specialization of traits for the JET_FLOW case.
         *
         * Provides the type alias `type` defined as `jetFlow`.
         */
        template <>
        class traits<boundaryCondition_t::JET_FLOW>
        {
        public:
            /**
             * @brief Concrete type associated with the LID_DRIVEN_CAVITY boundary condition.
             **/
            using type = jetFlow;
        };

        /**
         * @brief Specialization of traits for the LID_DRIVEN_CAVITY case.
         *
         * Provides the type alias `type` defined as `lidDrivenCavity`.
         */
        template <>
        class traits<boundaryCondition_t::LID_DRIVEN_CAVITY>
        {
        public:
            /**
             * @brief Concrete type associated with the LID_DRIVEN_CAVITY boundary condition.
             **/
            using type = lidDrivenCavity;
        };

        /**
         * @brief Compile‑time evaluation of the active boundary condition.
         *
         * This function uses preprocessor macros (`JETFLOW` or `LIDDRIVENCAVITY`) to
         * determine which boundary condition is selected at compile time. It is marked
         * `consteval`, guaranteeing that its result is a compile‑time constant usable
         * as a template argument (e.g., for `traits`).
         *
         * The function is also decorated with `__device__ __host__` for CUDA compatibility
         * and `[[nodiscard]]` to warn if the return value is ignored.
         *
         * @return The boundary condition enumerator corresponding to the defined macro.
         * @note Exactly one of `JETFLOW` or `LIDDRIVENCAVITY` should be defined;
         *       if neither is defined, the function falls back to `JET_FLOW` (or could
         *       trigger a compile‑time error). The current implementation returns
         *       `JET_FLOW` if neither macro is set – consider adding a `#error` directive
         *       to enforce a choice.
         */
        __device__ __host__ [[nodiscard]] inline consteval boundaryCondition_t caseName() noexcept
        {
            assertions::boundaryConditions::validate();

            return boundaryCondition_t::BOUNDARY_CONDITION;
        }
    }
}

#endif