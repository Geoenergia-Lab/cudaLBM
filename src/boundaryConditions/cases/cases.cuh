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

#include "invalidBoundaryCondition.cuh"
#include "jetFlow/jetFlow.cuh"
#include "lidDrivenCavity/lidDrivenCavity.cuh"
#include "multiphaseJet/multiphaseJet.cuh"
#include "subseaMechanicalDispersion/subseaMechanicalDispersion.cuh"

// Monophase defines
#define BOUNDARY_CONDITION JET_FLOW
// #define BOUNDARY_CONDITION LID_DRIVEN_CAVITY

// Multiphase defines
#define MULTIPHASE_BOUNDARY_CONDITION MULTIPHASE_JET
// #define MULTIPHASE_BOUNDARY_CONDITION SUBSEA_MECHANICAL_DISPERSION

#ifndef BOUNDARY_CONDITION
#define BOUNDARY_CONDITION INVALID
#endif

#ifndef MULTIPHASE_BOUNDARY_CONDITION
#define MULTIPHASE_BOUNDARY_CONDITION INVALID
#endif

namespace LBM
{
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
         **/
        typedef enum Enum : int
        {
            INVALID = 0,
            JET_FLOW = 1,
            LID_DRIVEN_CAVITY = 2
        } boundaryCondition_t;

        /**
         * @brief Primary template for boundary condition traits.
         *
         * This template maps a compile‑time boundary condition value (given as a
         * non‑type template parameter) to a concrete type that implements or represents
         * that boundary condition. Specializations are provided for each known enumerator.
         *
         * @tparam BoundaryCondition A compile‑time constant of type `boundaryCondition_t`.
         **/
        template <const boundaryCondition_t BoundaryCondition>
        class traits;

        /**
         * @brief Specialization of traits for the JET_FLOW case.
         *
         * Provides the type alias `type` defined as `jetFlow`.
         **/
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
         **/
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
         * @brief Specialization of traits for the JET_FLOW case.
         *
         * Provides the type alias `type` defined as `invalid`.
         **/
        template <>
        class traits<boundaryCondition_t::INVALID>
        {
        public:
            /**
             * @brief Concrete type associated with the INVALID boundary condition.
             **/
            using type = invalidBoundaryCondition;
        };

        /**
         * @brief Asserts that the boundary condition type is valid
         **/
        __device__ __host__ inline consteval void validate() noexcept
        {
            static_assert(!(BOUNDARY_CONDITION == INVALID), "BOUNDARY_CONDITION must be defined to a valid boundaryCondition_t enumerator");
        }

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
         *       trigger a compile‑time error).
         **/
        __device__ __host__ [[nodiscard]] inline consteval boundaryCondition_t caseName() noexcept
        {
            validate();

            return boundaryCondition_t::BOUNDARY_CONDITION;
        }
    }

    namespace multiphase
    {
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
             * @var Enum::MULTIPHASE_JET
             *      Represents a multiphase jet flow boundary condition.
             * @var Enum::SUBSEA_MECHANICAL_DISPERSION
             *      Represents the subsea mechanical dispersion boundary condition.
             **/
            typedef enum Enum : int
            {
                INVALID = 0,
                MULTIPHASE_JET = 1,
                SUBSEA_MECHANICAL_DISPERSION = 2
            } boundaryCondition_t;

            /**
             * @brief Primary template for boundary condition traits.
             *
             * This template maps a compile‑time boundary condition value (given as a
             * non‑type template parameter) to a concrete type that implements or represents
             * that boundary condition. Specializations are provided for each known enumerator.
             *
             * @tparam BoundaryCondition A compile‑time constant of type `boundaryCondition_t`.
             **/
            template <const boundaryCondition_t BoundaryCondition>
            class traits;

            /**
             * @brief Specialization of traits for the MULTIPHASE_JET case.
             *
             * Provides the type alias `type` defined as `multiphaseJet`.
             **/
            template <>
            class traits<boundaryCondition_t::MULTIPHASE_JET>
            {
            public:
                /**
                 * @brief Concrete type associated with the MULTIPHASE_JET boundary condition.
                 **/
                using type = multiphaseJet;
            };

            /**
             * @brief Specialization of traits for the SUBSEA_MECHANICAL_DISPERSION case.
             *
             * Provides the type alias `type` defined as `subseaMechanicalDispersion`.
             **/
            template <>
            class traits<boundaryCondition_t::SUBSEA_MECHANICAL_DISPERSION>
            {
            public:
                /**
                 * @brief Concrete type associated with the SUBSEA_MECHANICAL_DISPERSION boundary condition.
                 **/
                using type = subseaMechanicalDispersion;
            };

            /**
             * @brief Specialization of traits for the MULTIPHASE_JET case.
             *
             * Provides the type alias `type` defined as `invalid`.
             **/
            template <>
            class traits<boundaryCondition_t::INVALID>
            {
            public:
                /**
                 * @brief Concrete type associated with the INVALID boundary condition.
                 **/
                using type = invalidBoundaryCondition;
            };

            /**
             * @brief Asserts that the boundary condition type is valid
             **/
            __device__ __host__ inline consteval void validate() noexcept
            {
                static_assert(!(MULTIPHASE_BOUNDARY_CONDITION == INVALID), "MULTIPHASE_BOUNDARY_CONDITION must be defined to a valid boundaryCondition_t enumerator");
            }

            /**
             * @brief Compile‑time evaluation of the active boundary condition.
             *
             * This function uses preprocessor macros (`MULTIPHASEJET` or `SUBSEAMECHANICALDISPERSION`) to
             * determine which boundary condition is selected at compile time. It is marked
             * `consteval`, guaranteeing that its result is a compile‑time constant usable
             * as a template argument (e.g., for `traits`).
             *
             * The function is also decorated with `__device__ __host__` for CUDA compatibility
             * and `[[nodiscard]]` to warn if the return value is ignored.
             *
             * @return The boundary condition enumerator corresponding to the defined macro.
             * @note Exactly one of `MULTIPHASEJET` or `SUBSEAMECHANICALDISPERSION` should be defined;
             *       if neither is defined, the function falls back to `MULTIPHASE_JET` (or could
             *       trigger a compile‑time error).
             **/
            __device__ __host__ [[nodiscard]] inline consteval boundaryCondition_t caseName() noexcept
            {
                validate();

                return boundaryCondition_t::MULTIPHASE_BOUNDARY_CONDITION;
            }
        }
    }
}

#endif