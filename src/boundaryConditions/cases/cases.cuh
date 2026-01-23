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

// Monophase boundary conditions defines
#define MONOPHASEJET
// #define LIDDRIVENCAVITY

#include "monophaseJet/monophaseJet.cuh"
#include "lidDrivenCavity/lidDrivenCavity.cuh"

// Multiphase boundary conditions defines
// #define MULTIPHASEJET
#define SUBSEAMECHANICALDISPERSION

#include "multiphaseJet/multiphaseJet.cuh"
#include "subseaMechanicalDispersion/subseaMechanicalDispersion.cuh"

namespace LBM
{
    /**
     * @brief Monophase boundary conditions aliases
     **/
#ifdef MONOPHASEJET
    using BoundaryConditions = monophaseJet;
    __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return true; }
    __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return true; }
#endif

#ifdef LIDDRIVENCAVITY
    using BoundaryConditions = lidDrivenCavity;
    __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return false; }
    __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return false; }
#endif

    /**
     * @brief Multiphase boundary conditions aliases
     **/
    namespace multiphase
    {
#ifdef MULTIPHASEJET
        using BoundaryConditions = multiphaseJet;
        __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return true; }
        __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return true; }
#endif

#ifdef SUBSEAMECHANICALDISPERSION
        using BoundaryConditions = subseaMechanicalDispersion;
        __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return true; }
        __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return false; }
#endif
    }
}

#endif