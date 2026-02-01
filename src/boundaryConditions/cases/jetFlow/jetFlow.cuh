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
    A class applying boundary conditions to the monophase jet case

Namespace
    LBM

SourceFiles
    jetFlow.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_jetFlow_CUH
#define __MBLBM_jetFlow_CUH

namespace LBM
{
    /**
     * @class jetFlow
     * @brief Applies boundary conditions for jet simulations using moment representation
     *
     * This class implements the boundary condition treatment for monophase jet flow simulations.
     * It handles a round inflow with no-slip outside of the circle, periodic laterals and outflow
     * boundaries using moment-based boundary conditions derived from the regularized LBM approach.
     **/
    class jetFlow
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval jetFlow(){};

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment variables array to be populated
         * @param[in] boundaryNormal Normal vector information at boundary node
         *
         * This method implements the moment-based boundary condition treatment
         * It handles both the round inflow boundary located at the BACK face with no-slip
         * outside of the circle, periodic sides and the outflow boundary located at the FRONT face.
         *
         * The method uses the regularized LBM approach to reconstruct boundary
         * moments from available population information, ensuring mass conservation
         * and appropriate stress conditions at boundaries.
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS<false>()> &moments,
            const normalVector &boundaryNormal,
            const scalar_t *const ptrRestrict shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: boundaryConditions::calculate_moments only supports D3Q19 and D3Q27.");

            const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            switch (boundaryNormal.nodeType())
            {
            // Round inflow + no-slip
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(1);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;                                      // rho
                moments[m_i<1>()] = static_cast<scalar_t>(0);                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                 // uy
                moments[m_i<3>()] = is_jet * device::u_inf;                   // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                 // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);                 // mxy
                moments[m_i<6>()] = mxz;                                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                 // myy
                moments[m_i<8>()] = myz;                                      // myz
                moments[m_i<9>()] = is_jet * (device::u_inf * device::u_inf); // mzz

                return;
            }

// Static back face outside of the jet
#include "include/static.cuh"

// Periodic
#include "include/periodic.cuh"

// Outflow (zero-gradient) at front face
#include "include/IRBCNeumann.cuh"
            }
        }

        template <class VelocitySet, const label_t N>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS<false>()> &moments,
            const normalVector &boundaryNormal,
            const thread::array<scalar_t, N> &shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: boundaryConditions::calculate_moments only supports D3Q19 and D3Q27.");

            const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            switch (boundaryNormal.nodeType())
            {
            // Round inflow + no-slip
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, X, Z>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, Y, Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(1);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;                                      // rho
                moments[m_i<1>()] = static_cast<scalar_t>(0);                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                 // uy
                moments[m_i<3>()] = is_jet * device::u_inf;                   // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                 // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);                 // mxy
                moments[m_i<6>()] = mxz;                                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                 // myy
                moments[m_i<8>()] = myz;                                      // myz
                moments[m_i<9>()] = is_jet * (device::u_inf * device::u_inf); // mzz

                return;
            }

// Static back face outside of the jet
#include "include/static.cuh"

// Periodic
#include "include/periodic.cuh"

// Outflow (zero-gradient) at front face
#include "include/IRBCNeumann.cuh"
            }
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