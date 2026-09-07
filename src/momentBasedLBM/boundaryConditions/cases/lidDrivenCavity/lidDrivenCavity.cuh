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
    A class applying boundary conditions to the lid driven cavity case

Namespace
    LBM

SourceFiles
    lidDrivenCavity.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LIDDRIVENCAVITY_CUH
#define __MBLBM_LIDDRIVENCAVITY_CUH

namespace LBM
{
    /**
     * @class lidDrivenCavity
     * @brief Applies boundary conditions for lid-driven cavity simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice model
     * in lid-driven cavity flow simulations. It handles both static wall boundaries and
     * moving lid boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
     **/
    class lidDrivenCavity : public boundaryConditionType<false, false, false>
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval lidDrivenCavity() {}

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool appliesCondition() noexcept { return true; }

        /**
         * @brief Public method to calculate the post-streaming methods and update boundary conditions
         **/
        template <class VelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            [[maybe_unused]] SharedBuffer &sharedBuffer,
            [[maybe_unused]] const thread::coordinate &Tx,
            [[maybe_unused]] const device::pointCoordinate &point,
            const device::label_t tid) noexcept
        {
            const NormalVectorType boundaryNormal(point);

            VelocitySet::template calculate_moments(moments, pop, boundaryNormal);

            if (boundaryNormal.isBoundary())
            {
                calculate_moments<VelocitySet>(pop, moments, boundaryNormal, sharedBuffer, Tx, point);
            }
        }

    private:
        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         *
         * This method implements the moment-based boundary condition treatment for
         * the D3Q19 lattice model. It handles various boundary types including:
         * - Static wall boundaries (all velocity components zero)
         * - Moving lid boundaries (prescribed tangential velocity)
         * - Corner and edge cases with specialized treatment
         *
         * The method uses the regularized LBM approach to reconstruct boundary
         * moments from available population information, ensuring mass conservation
         * and appropriate stress conditions at boundaries.
         **/
        template <class VelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            const NormalVectorType &boundaryNormal,
            [[maybe_unused]] const SharedBuffer &sharedBuffer,
            [[maybe_unused]] const thread::coordinate &Tx,
            [[maybe_unused]] const device::pointCoordinate &point) noexcept
        {
            const scalar_t rho_I = moments[m_i<0>()];
            const scalar_t mxy_I = moments[m_i<5>()];
            const scalar_t mxz_I = moments[m_i<6>()];
            const scalar_t myz_I = moments[m_i<8>()];

            // Apply Dirichlet boundary conditions
            {
                const scalar_t nBoundaries = boundaryNormal.template countBoundaries<scalar_t>();

                const symmetricTensor boundarySwitches = {
                    boundaryNormal.template isWest<scalar_t>(),
                    boundaryNormal.template isEast<scalar_t>(),
                    boundaryNormal.template isNorth<scalar_t>(),
                    boundaryNormal.template isSouth<scalar_t>(),
                    boundaryNormal.template isBack<scalar_t>(),
                    boundaryNormal.template isFront<scalar_t>()};

                moments[m_i<1>()] = U<axis::X>(boundarySwitches, nBoundaries);
                moments[m_i<2>()] = U<axis::Y>(boundarySwitches, nBoundaries);
                moments[m_i<3>()] = U<axis::Z>(boundarySwitches, nBoundaries);

                // We can make m_xx branchless very easily
                // North: Equilibrium with constant velocity boundary
                // Others: Equilibrium with zero velocity boundary
                // So, we just switch U_North[0] ^ 2 based on the North condition
                // We are applying the velocity lid to ALL North boundaries, including edges and corners
                {
                    moments[m_i<4>()] = moments[m_i<1>()] * moments[m_i<1>()];
                }
            }

            // Apply the second-order moments that are universal to this case
            {
                moments[m_i<7>()] = static_cast<scalar_t>(0);
                moments[m_i<9>()] = static_cast<scalar_t>(0);
            }

            switch (boundaryNormal.nodeType())
            {
            // Static boundaries
            case NormalVectorType::SOUTH_WEST_BACK():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH_WEST_FRONT():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH_EAST_BACK():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH_EAST_FRONT():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH_WEST():
            {
                moments[m_i<0>()] = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega); // mxx
                moments[m_i<5>()] = (static_cast<scalar_t>(36) * mxy_I * rho_I - moments[m_i<0>()]) / (static_cast<scalar_t>(9) * moments[m_i<0>()]);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH_EAST():
            {
                moments[m_i<0>()] = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                moments[m_i<5>()] = (static_cast<scalar_t>(36) * mxy_I * rho_I + moments[m_i<0>()]) / (static_cast<scalar_t>(9) * moments[m_i<0>()]);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::WEST_BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::WEST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::EAST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::EAST_FRONT():
            {
                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH_BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::SOUTH_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::WEST():
            {
                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::EAST():
            {
                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::SOUTH():
            {
                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::FRONT():
            {
                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = mxz;
                moments[m_i<8>()] = myz;

                return;
            }
            // Lid boundaries
            case NormalVectorType::NORTH():
            {
                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = (static_cast<scalar_t>(6) * mxy_I * rho_I - device::U_North[0] * rho) / (static_cast<scalar_t>(3) * rho);
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::NORTH_WEST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::NORTH_WEST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::NORTH_EAST_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::NORTH_EAST_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I / (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::NORTH_BACK():
            {
                const scalar_t rho = static_cast<scalar_t>(72) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) / (static_cast<scalar_t>(18) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::NORTH_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(72) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) / (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) / (static_cast<scalar_t>(18) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0);
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = myz;

                return;
            }
            case NormalVectorType::NORTH_EAST():
            {
                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::U_North[0] - static_cast<scalar_t>(18) * device::U_North[0] * device::U_North[0] + device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device::U_North[0] * rho - static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            case NormalVectorType::NORTH_WEST():
            {
                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::U_North[0] - static_cast<scalar_t>(18) * device::U_North[0] * device::U_North[0] + device::omega - static_cast<scalar_t>(3) * device::U_North[0] * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device::U_North[0] * rho + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;
                moments[m_i<6>()] = static_cast<scalar_t>(0);
                moments[m_i<8>()] = static_cast<scalar_t>(0);

                return;
            }
            }
        }

        /**
         * @brief Branchless computation of the velocity component based on the boundary
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] boundarySwitches Switches indicating active boundary conditions
         * @param[in] n_boundaries Number of active boundaries
         * @return Velocity component value
         **/
        template <const axis::type alpha>
        __device__ static inline constexpr scalar_t U(const symmetricTensor &boundarySwitches, const scalar_t n_boundaries) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            // Calculate the boundary velocity value
            return ((boundarySwitches[0] * device::U_West[alpha]) +
                    (boundarySwitches[1] * device::U_East[alpha]) +
                    (boundarySwitches[2] * device::U_North[alpha]) +
                    (boundarySwitches[3] * device::U_South[alpha]) +
                    (boundarySwitches[4] * device::U_Back[alpha]) +
                    (boundarySwitches[5] * device::U_Front[alpha])) /
                   n_boundaries;
        }
    };
}

#endif