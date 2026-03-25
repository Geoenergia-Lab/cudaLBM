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
    lidDrivenCavity.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_INVALIDBOUNDARYCONDITION_CUH
#define __MBLBM_INVALIDBOUNDARYCONDITION_CUH

namespace LBM
{
    /**
     * @class invalid
     **/
    class invalidBoundaryCondition
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval invalidBoundaryCondition() {}

        /**
         * @brief Periodic boundary definitions
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicX() noexcept { return false; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicY() noexcept { return false; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicZ() noexcept { return false; }

        /**
         * @brief Placeholder for calculate_moments
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            [[maybe_unused]] const thread::array<scalar_t, VelocitySet::Q()> &pop,
            [[maybe_unused]] thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            [[maybe_unused]] const normalVector &boundaryNormal,
            [[maybe_unused]] const scalar_t *const ptrRestrict shared_buffer,
            [[maybe_unused]] const thread::coordinate &Tx,
            [[maybe_unused]] const device::pointCoordinate &point) noexcept
        {
        }

    private:
    };
}

#endif