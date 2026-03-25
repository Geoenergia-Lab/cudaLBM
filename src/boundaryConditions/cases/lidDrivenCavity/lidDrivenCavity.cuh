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

#ifndef __MBLBM_LIDDRIVENCAVITY_CUH
#define __MBLBM_LIDDRIVENCAVITY_CUH

__device__ __host__ [[nodiscard]] inline consteval bool check_n_boundaries() noexcept { return false; }

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
    class lidDrivenCavity
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval lidDrivenCavity() {}

        /**
         * @brief Periodic boundary definitions
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicX() noexcept { return false; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicY() noexcept { return false; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicZ() noexcept { return false; }

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment variables array to be populated
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
            thread::array<scalar_t, NUMBER_MOMENTS<false>()> &moments,
            const normalVector &boundaryNormal,
            [[maybe_unused]] const SharedBuffer &shared_buffer,
            [[maybe_unused]] const thread::coordinate &Tx,
            [[maybe_unused]] const device::pointCoordinate &point) noexcept
        {
            const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, axis::NO_DIRECTION, axis::NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            // Apply Dirichlet boundary conditions
            {
                const scalar_t nBoundaries = boundaryNormal.countBoundaries<scalar_t>();

                const thread::array<scalar_t, 6> boundarySwitches = {
                    boundaryNormal.isWest<scalar_t>(),
                    boundaryNormal.isEast<scalar_t>(),
                    boundaryNormal.isNorth<scalar_t>(),
                    boundaryNormal.isSouth<scalar_t>(),
                    boundaryNormal.isBack<scalar_t>(),
                    boundaryNormal.isFront<scalar_t>()};

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
                    // moments[m_i<4>()] = boundarySwitches[m_i<2>()] * device::U_North[0] * device::U_North[0];
                }
            }

            // Apply the second-order moments that are universal to this case
            {
                moments[m_i<7>()] = static_cast<scalar_t>(0); // m_yy
                moments[m_i<9>()] = static_cast<scalar_t>(0); // m_zz
            }

            switch (boundaryNormal.nodeType())
            {
            // Static boundaries
            case normalVector::WEST_SOUTH_BACK():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::WEST_SOUTH_FRONT():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST_SOUTH_BACK():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST_SOUTH_FRONT():
            {
                if constexpr (VelocitySet::Q() == 19)
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(12) * rho_I / static_cast<scalar_t>(7);
                }
                else
                {
                    moments[m_i<0>()] = static_cast<scalar_t>(216) * rho_I / static_cast<scalar_t>(125);
                }
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::WEST_SOUTH():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;

                moments[m_i<0>()] = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega); // uz                                                                                                       // mxx
                moments[m_i<5>()] = (static_cast<scalar_t>(36) * mxy_I * rho_I - moments[m_i<0>()]) / (static_cast<scalar_t>(9) * moments[m_i<0>()]);                  // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);                                                                                                          // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0);                                                                                                          // myz

                return;
            }
            case normalVector::EAST_SOUTH():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;

                moments[m_i<0>()] = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) / (static_cast<scalar_t>(24) + device::omega); // uz
                moments[m_i<5>()] = (static_cast<scalar_t>(36) * mxy_I * rho_I + moments[m_i<0>()]) / (static_cast<scalar_t>(9) * moments[m_i<0>()]);                    // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0);                                                                                                            // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0);                                                                                                            // myz

                return;
            }
            case normalVector::WEST_BACK():
            {
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::WEST_FRONT():
            {
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST_BACK():
            {
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST_FRONT():
            {
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t mxz = (static_cast<scalar_t>(36) * mxz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::SOUTH_BACK():
            {
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I - rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::SOUTH_FRONT():
            {
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + device::omega);
                const scalar_t myz = (static_cast<scalar_t>(36) * myz_I * rho_I + rho) / (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::WEST():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::SOUTH():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = static_cast<scalar_t>(2) * mxy_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::BACK():
            {
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::FRONT():
            {
                const scalar_t mxz_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Z>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = mxz;                      // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            // Lid boundaries
            case normalVector::NORTH():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(6) * rho_I / static_cast<scalar_t>(5);
                const scalar_t mxy = (static_cast<scalar_t>(6) * mxy_I * rho_I - device::U_North[0] * rho) / (static_cast<scalar_t>(3) * rho);
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::WEST_NORTH_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::WEST_NORTH_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) - static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST_NORTH_BACK():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::EAST_NORTH_FRONT():
            {
                const scalar_t rho = -static_cast<scalar_t>(24) * rho_I /
                                     (-static_cast<scalar_t>(14) + static_cast<scalar_t>(8) * device::U_North[0] + static_cast<scalar_t>(9) * device::U_North[0] * device::U_North[0]);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::NORTH_BACK():
            {
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(72) * (-rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I + static_cast<scalar_t>(2) * rho - static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::NORTH_FRONT():
            {
                const scalar_t myz_I = velocitySet::calculate_moment<VelocitySet, axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(72) * (rho_I - myz_I * rho_I + myz_I * rho_I * device::omega) /
                                     (-static_cast<scalar_t>(48) - static_cast<scalar_t>(2) * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);

                const scalar_t myz = (static_cast<scalar_t>(72) * myz_I * rho_I - static_cast<scalar_t>(2) * rho + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) /
                                     (static_cast<scalar_t>(18) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = static_cast<scalar_t>(0); // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = myz;                      // myz

                return;
            }
            case normalVector::EAST_NORTH():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = static_cast<scalar_t>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) - static_cast<scalar_t>(18) * device::U_North[0] - static_cast<scalar_t>(18) * device::U_North[0] * device::U_North[0] + device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I - rho - static_cast<scalar_t>(3) * device::U_North[0] * rho - static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            case normalVector::WEST_NORTH():
            {
                const scalar_t mxy_I = velocitySet::calculate_moment<VelocitySet, axis::X, axis::Y>(pop, boundaryNormal) * inv_rho_I;

                const scalar_t rho = -static_cast<scalar_t>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * device::omega) /
                                     (static_cast<scalar_t>(24) + static_cast<scalar_t>(18) * device::U_North[0] - static_cast<scalar_t>(18) * device::U_North[0] * device::U_North[0] + device::omega - static_cast<scalar_t>(3) * device::U_North[0] * device::omega + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * device::omega);
                const scalar_t mxy = (static_cast<scalar_t>(36) * mxy_I * rho_I + rho - static_cast<scalar_t>(3) * device::U_North[0] * rho + static_cast<scalar_t>(3) * device::U_North[0] * device::U_North[0] * rho) /
                                     (static_cast<scalar_t>(9) * rho);

                moments[m_i<0>()] = rho;
                moments[m_i<5>()] = mxy;                      // mxy
                moments[m_i<6>()] = static_cast<scalar_t>(0); // mxz
                moments[m_i<8>()] = static_cast<scalar_t>(0); // myz

                return;
            }
            }
        }

    private:
        /**
         * @brief Branchless computation of the velocity component based on the boundary
         * @param[in] boundarySwitches Switches indicating active boundary conditions
         * @param[in] n_boundaries Number of active boundaries
         * @tparam index Index of the velocity component to compute
         * @return Velocity component value
         **/
        template <const axis::type alpha>
        __device__ static inline constexpr scalar_t U(const thread::array<scalar_t, 6> &boundarySwitches, const scalar_t n_boundaries) noexcept
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