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
    Definition of second order collision

Namespace
    LBM

SourceFiles
    secondOrder.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_COLLISION_SECOND_ORDER_CUH
#define __MBLBM_COLLISION_SECOND_ORDER_CUH

#include "../array/array.cuh"

namespace LBM
{
    /**
     * @class secondOrder
     * @brief Implements second-order collision operator for LBM simulations
     * @extends collision
     *
     * This class provides a specialized collision operator that handles
     * second-order moment updates in the Lattice Boltzmann Method. It updates
     * both diagonal and off-diagonal moments using relaxation parameters and velocity components.
     *
     * The collision operation follows the standard BGK approximation with
     * specialized treatment for second-order moments in the moment space.
     **/
    class secondOrder : private collision
    {
    public:
        /**
         * @brief Default constructor (consteval)
         * @return A secondOrder collision operator instance
         **/
        __device__ __host__ [[nodiscard]] inline consteval secondOrder() noexcept {}

        /**
         * @brief Perform second-order collision operation on moments
         * @param[in,out] moments Array of 10 solution moments to be updated
         *
         * This method updates the second-order moments (both diagonal and off-diagonal)
         * using the BGK collision model with the following operations:
         * - Diagonal moments (m_xx, m_yy, m_zz): Relaxed with specialized parameter
         *   and updated with squared velocity components
         * - Off-diagonal moments (m_xy, m_xz, m_yz): Relaxed and updated with
         *   product of velocity components
         *
         * @note This implementation assumes zero force terms, so velocity updates are omitted
         * @note Uses device-level relaxation parameters (device::t_omegaVar, device::omegaVar_d2, device::omega)
         **/
        __device__ static inline void collide(thread::array<scalar_t, 10> &moments) noexcept
        {
            // Velocity updates are removed since force terms are zero
            // Diagonal moment updates (remove force terms)
            moments[m_i<4>()] = device::t_omegaVar * moments[m_i<4>()] + device::omegaVar_d2 * (moments[m_i<1>()]) * (moments[m_i<1>()]);
            moments[m_i<7>()] = device::t_omegaVar * moments[m_i<7>()] + device::omegaVar_d2 * (moments[m_i<2>()]) * (moments[m_i<2>()]);
            moments[m_i<9>()] = device::t_omegaVar * moments[m_i<9>()] + device::omegaVar_d2 * (moments[m_i<3>()]) * (moments[m_i<3>()]);

            // Off-diagonal moment updates (remove force terms)
            moments[m_i<5>()] = device::t_omegaVar * moments[m_i<5>()] + device::omega * (moments[m_i<1>()]) * (moments[m_i<2>()]);
            moments[m_i<6>()] = device::t_omegaVar * moments[m_i<6>()] + device::omega * (moments[m_i<1>()]) * (moments[m_i<3>()]);
            moments[m_i<8>()] = device::t_omegaVar * moments[m_i<8>()] + device::omega * (moments[m_i<2>()]) * (moments[m_i<3>()]);
        }

        /**
         * @brief Perform second-order collision operation on moments
         * @param[in,out] moments Array of 11 solution moments to be updated
         *
         * This method updates the second-order moments (both diagonal and off-diagonal)
         * using the BGK collision model with the following operations:
         * - Diagonal moments (m_xx, m_yy, m_zz): Relaxed with specialized parameter
         *   and updated with squared velocity components
         * - Off-diagonal moments (m_xy, m_xz, m_yz): Relaxed and updated with
         *   product of velocity components
         *
         * @note This implementation is based on the Guo forcing scheme
         * @note Uses device-level relaxation parameters (device::t_omegaVar, device::omegaVar_d2, device::omega, device::tt_omegaVar_t3)
         **/
        __device__ static inline void collide(thread::array<scalar_t, 11> &moments, const scalar_t forceX, const scalar_t forceY, const scalar_t forceZ) noexcept
        {
            const scalar_t invRho = static_cast<scalar_t>(1) / moments[m_i<0>()];

            // Mixture viscosity local relaxation parameters
            const scalar_t tau_loc = (static_cast<scalar_t>(1) - moments[m_i<10>()]) * device::tau_A + moments[m_i<10>()] * device::tau_B;
            const scalar_t omega_loc = static_cast<scalar_t>(1.0) / tau_loc;
            const scalar_t t_omegaVar_loc = static_cast<scalar_t>(1) - omega_loc;
            const scalar_t omegaVar_d2_loc = static_cast<scalar_t>(0.5) * omega_loc;
            const scalar_t tt_omegaVar_loc = static_cast<scalar_t>(1) - static_cast<scalar_t>(0.5) * omega_loc;
            const scalar_t tt_omegaVar_t3_loc = static_cast<scalar_t>(3) * tt_omegaVar_loc;

            // Half-step velocities
            const scalar_t uxEq = moments[m_i<1>()] + static_cast<scalar_t>(1.5) * invRho * forceX;
            const scalar_t uyEq = moments[m_i<2>()] + static_cast<scalar_t>(1.5) * invRho * forceY;
            const scalar_t uzEq = moments[m_i<3>()] + static_cast<scalar_t>(1.5) * invRho * forceZ;

            // Velocity updates
            moments[m_i<1>()] = moments[m_i<1>()] + static_cast<scalar_t>(3) * invRho * forceX; // ux
            moments[m_i<2>()] = moments[m_i<2>()] + static_cast<scalar_t>(3) * invRho * forceY; // uy
            moments[m_i<3>()] = moments[m_i<3>()] + static_cast<scalar_t>(3) * invRho * forceZ; // uz

            // Diagonal moment updates
            moments[m_i<4>()] = t_omegaVar_loc * moments[m_i<4>()] + omegaVar_d2_loc * uxEq * uxEq + static_cast<scalar_t>(1.5) * tt_omegaVar_loc * invRho * (forceX * uxEq + forceX * uxEq); // mxx
            moments[m_i<7>()] = t_omegaVar_loc * moments[m_i<7>()] + omegaVar_d2_loc * uyEq * uyEq + static_cast<scalar_t>(1.5) * tt_omegaVar_loc * invRho * (forceY * uyEq + forceY * uyEq); // myy
            moments[m_i<9>()] = t_omegaVar_loc * moments[m_i<9>()] + omegaVar_d2_loc * uzEq * uzEq + static_cast<scalar_t>(1.5) * tt_omegaVar_loc * invRho * (forceZ * uzEq + forceZ * uzEq); // mzz

            // Off-diagonal moment updates
            moments[m_i<5>()] = t_omegaVar_loc * moments[m_i<5>()] + omega_loc * uxEq * uyEq + tt_omegaVar_t3_loc * invRho * (forceX * uyEq + forceY * uxEq); // mxy
            moments[m_i<6>()] = t_omegaVar_loc * moments[m_i<6>()] + omega_loc * uxEq * uzEq + tt_omegaVar_t3_loc * invRho * (forceX * uzEq + forceZ * uxEq); // mxz
            moments[m_i<8>()] = t_omegaVar_loc * moments[m_i<8>()] + omega_loc * uyEq * uzEq + tt_omegaVar_t3_loc * invRho * (forceY * uzEq + forceZ * uyEq); // myz
        }

    private:
        // __device__ [[nodiscard]] static inline scalar_t sponge_ramp(const label_t z) noexcept
        // {
        //     const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1();
        //     scalar_t s = (zn - sponge::z_start()) * sponge::inv_sponge();
        //     s = fminf(fmaxf(s, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
        //     return s * s * (static_cast<scalar_t>(3) - static_cast<scalar_t>(2) * s); // cubic smoothstep
        // }

        // namespace sponge
        // {
        //     __device__ __host__ [[nodiscard]] static inline consteval scalar_t K_gain() noexcept
        //     {
        //         return static_cast<scalar_t>(100);
        //     }

        //     __device__ __host__ [[nodiscard]] static inline constexpr int sponge_cells() noexcept
        //     {
        //         return static_cast<int>(device::nz / 12);
        //     }

        //     __device__ __host__ [[nodiscard]] static inline constexpr scalar_t sponge() noexcept
        //     {
        //         return static_cast<scalar_t>(sponge_cells()) / static_cast<scalar_t>(device::nz - 1);
        //     }

        //     __device__ __host__ [[nodiscard]] static inline constexpr scalar_t z_start() noexcept
        //     {
        //         return static_cast<scalar_t>(device::nz - 1 - sponge_cells()) / static_cast<scalar_t>(device::nz - 1);
        //     }

        //     __device__ __host__ [[nodiscard]] static inline constexpr scalar_t inv_nz_m1() noexcept
        //     {
        //         return static_cast<scalar_t>(1) / static_cast<scalar_t>(device::nz - 1);
        //     }

        //     __device__ __host__ [[nodiscard]] static inline constexpr scalar_t inv_sponge() noexcept
        //     {
        //         return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(sponge()));
        //     }
        // }
    };
}

#endif