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
     * second-order moment updates in the Lattice Boltzmann Method. It assumes
     * zero force terms and updates both diagonal and off-diagonal moments
     * using relaxation parameters and velocity components.
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
        __device__ static inline void collide(thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
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

    private:
    };
}

#endif