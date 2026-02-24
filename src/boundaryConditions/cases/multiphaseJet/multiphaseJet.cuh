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
        template <class VelocitySet, class PhaseVelocitySet>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments,
            const normalVector &boundaryNormal,
            const scalar_t *const ptrRestrict shared_buffer,
            const label_t step) noexcept
        {
#include "jetBoundaryCondition.cuh"
        }

        template <class VelocitySet, class PhaseVelocitySet, const label_t N>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments,
            const normalVector &boundaryNormal,
            const thread::array<scalar_t, N> &shared_buffer,
            const label_t step) noexcept
        {
#include "jetBoundaryCondition.cuh"
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

        __device__ [[nodiscard]] static inline constexpr uint32_t hash32(uint32_t x) noexcept
        {
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;

            return x;
        }

        __device__ [[nodiscard]] static inline constexpr scalar_t uniform01(const uint32_t seed) noexcept
        {
            constexpr scalar_t inv2_32 = static_cast<scalar_t>(2.3283064365386963e-10);

            return (static_cast<scalar_t>(seed) + static_cast<scalar_t>(0.5)) * inv2_32;
        }

        __device__ [[nodiscard]] static inline scalar_t box_muller(
            scalar_t rrx,
            const scalar_t rry) noexcept
        {
            rrx = fmaxf(rrx, static_cast<scalar_t>(1e-12));
            const scalar_t r = sqrtf(-static_cast<scalar_t>(2) * logf(rrx));
            const scalar_t theta = static_cast<scalar_t>(2) * static_cast<scalar_t>(3.14159265358979323846) * rry;

            return r * cosf(theta);
        }

        template <uint32_t SALT = 0u>
        __device__ [[nodiscard]] static inline constexpr scalar_t white_noise(
            const label_t x,
            const label_t y,
            const label_t t) noexcept
        {
            const uint32_t base = (0x9E3779B9u ^ SALT) ^ static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(y) * 0x85EBCA6Bu) ^ (static_cast<uint32_t>(t) * 0xC2B2AE35u);

            const scalar_t rrx = uniform01(hash32(base));
            const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));

            return box_muller(rrx, rry);
        }
    };
}

#endif