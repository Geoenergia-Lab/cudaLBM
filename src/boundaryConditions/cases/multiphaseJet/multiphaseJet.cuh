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
Authors: Nathan Duggins, Vinicius Czarnobay, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    A class applying boundary conditions to the multiphase jet case

Namespace
    LBM

SourceFiles
    multiphaseJet.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_multiphaseJet_CUH
#define __MBLBM_multiphaseJet_CUH

namespace LBM
{
    /**
     * @class multiphaseJet
     *
     * @brief Applies boundary conditions for multiphase jet flow simulations using moment representation
     *
     * This class implements the boundary condition treatment for multiphase jet flow simulations.
     * It handles static wall, inflow, and outflow boundaries using moment-based boundary conditions
     * derived from the regularized LBM approach.
     **/
    class multiphaseJet
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval multiphaseJet(){};

        /**
         * @brief Periodic boundary definitions
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicX() noexcept { return true; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicY() noexcept { return true; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicZ() noexcept { return false; }

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment variables array to be populated
         * @param[in] boundaryNormal Normal vector information at boundary node
         *
         * This method implements the moment-based boundary condition treatment
         * Currently, it handles both the inflow (jet) boundary located at the BACK face
         * of the domain and the outflow boundary located at the FRONT face.
         *
         * The method uses the regularized LBM approach to reconstruct boundary
         * moments from available population information, ensuring mass conservation
         * and appropriate stress conditions at boundaries.
         **/
        template <class VelocitySet, class PhaseVelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments,
            const normalVector &boundaryNormal,
            const scalar_t *const ptrRestrict shared_buffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point) noexcept
        {
#include "boundaryCondition.cuh"
        }

    private:
        __device__ [[nodiscard]] static inline scalar_t center_x() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::nx - 1);
        }

        __device__ [[nodiscard]] static inline scalar_t center_y() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::ny - 1);
        }

        __device__ [[nodiscard]] static inline scalar_t radius() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::L_char);
        }

        __device__ [[nodiscard]] static inline scalar_t r2() noexcept
        {
            return radius() * radius();
        }
    };
}

#endif