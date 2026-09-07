/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
         * @param[out] moments Moment array (rho, U, Pi)
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
        __device__ static inline void collide(momentsArray &moments) noexcept
        {
            // Diagonal moment updates (remove force terms)
            moments[m_i<4>()] = collide(moments[m_i<1>()], moments[m_i<4>()]);
            moments[m_i<7>()] = collide(moments[m_i<2>()], moments[m_i<7>()]);
            moments[m_i<9>()] = collide(moments[m_i<3>()], moments[m_i<9>()]);

            // Off-diagonal moment updates (remove force terms)
            moments[m_i<5>()] = collide(moments[m_i<1>()], moments[m_i<2>()], moments[m_i<5>()]);
            moments[m_i<6>()] = collide(moments[m_i<1>()], moments[m_i<3>()], moments[m_i<6>()]);
            moments[m_i<8>()] = collide(moments[m_i<2>()], moments[m_i<3>()], moments[m_i<8>()]);
        }

    private:
        /**
         * @brief Collision helper for diagonal moments
         * @param u_AlphaBeta Velocity component in the alpha-beta direction
         * @param m_AlphaBeta Moment component in the alpha-beta direction
         * @return Updated moment after collision
         **/
        __device__ static inline scalar_t collide(const scalar_t u_AlphaBeta, const scalar_t m_AlphaBeta) noexcept
        {
            return (device::t_omegaVar * m_AlphaBeta) + (device::omegaVar_d2 * (u_AlphaBeta * u_AlphaBeta));
        }

        /**
         * @brief Collision helper for off-diagonal moments
         * @param u_Alpha Velocity component in the alpha direction
         * @param u_Beta Velocity component in the beta direction
         * @param m_AlphaBeta Moment component in the alpha-beta direction
         * @return Updated moment after collision
         **/
        __device__ static inline scalar_t collide(const scalar_t u_Alpha, const scalar_t u_Beta, const scalar_t m_AlphaBeta) noexcept
        {
            return (device::t_omegaVar * m_AlphaBeta) + (device::omega * (u_Alpha * u_Beta));
        }
    };
}

#endif