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
    A class applying boundary conditions to the turbulent jet case

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
     * @brief Applies boundary conditions for multiphase jet simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice
     * model in multiphase jet flow simulations. It handles static wall, inflow, and
     * outflow boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
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
         * for the D3Q19 lattice model. Currently, it handles both the inflow
         * (jet) boundary located at the BACK face of the domain and the outflow
         * boundary located at the FRONT face.
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
            const scalar_t *const ptrRestrict shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: boundaryConditions::calculate_moments only supports D3Q19 and D3Q27.");
            static_assert((PhaseVelocitySet::Q() == 7), "Error: boundaryConditions::calculate_moments only supports D3Q7 for phase field.");

            const scalar_t p_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);

            switch (boundaryNormal.nodeType())
            {
            // Round inflow + no-slip
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = BACK_mxz_I(pop);
                const scalar_t myz_I = BACK_myz_I(pop);

                const scalar_t p = static_cast<scalar_t>(0);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I;

                moments[m_i<0>()] = p;                                        // p
                moments[m_i<1>()] = static_cast<scalar_t>(0);                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                 // uy
                moments[m_i<3>()] = is_jet * device::u_inf;                   // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                 // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);                 // mxy
                moments[m_i<6>()] = mxz;                                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                 // myy
                moments[m_i<8>()] = myz;                                      // myz
                moments[m_i<9>()] = is_jet * (device::u_inf * device::u_inf); // mzz
                moments[m_i<10>()] = is_jet * static_cast<scalar_t>(1);

                return;
            }

// Periodic
#include "include/periodic.cuh"

// Outflow (zero-gradient) at front face
#include "include/IRBCNeumann.cuh"

// Static back face outside of the jet
#include "include/fallback.cuh"
            }
        }

        template <class VelocitySet, class PhaseVelocitySet, const label_t N>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS<true>()> &moments,
            const normalVector &boundaryNormal,
            const thread::array<scalar_t, N> &shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: boundaryConditions::calculate_moments only supports D3Q19 and D3Q27.");
            static_assert((PhaseVelocitySet::Q() == 7), "Error: boundaryConditions::calculate_moments only supports D3Q7 for phase field.");

            const scalar_t p_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);

            switch (boundaryNormal.nodeType())
            {
            // Round inflow + no-slip
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = BACK_mxz_I(pop);
                const scalar_t myz_I = BACK_myz_I(pop);

                const scalar_t p = static_cast<scalar_t>(0);
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I;

                moments[m_i<0>()] = p;                                        // p
                moments[m_i<1>()] = static_cast<scalar_t>(0);                 // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                 // uy
                moments[m_i<3>()] = is_jet * device::u_inf;                   // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                 // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);                 // mxy
                moments[m_i<6>()] = mxz;                                      // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                 // myy
                moments[m_i<8>()] = myz;                                      // myz
                moments[m_i<9>()] = is_jet * (device::u_inf * device::u_inf); // mzz
                moments[m_i<10>()] = is_jet * static_cast<scalar_t>(1);       // phi

                return;
            }

// Periodic
#include "include/periodic.cuh"

// Outflow (zero-gradient) at front face
#include "include/IRBCNeumann.cuh"

// Static back face outside of the jet
#include "include/fallback.cuh"
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

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_SOUTH_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return pop[q_i<8>()];
            }
            else
            {
                return (pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_SOUTH_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return -pop[q_i<13>()];
            }
            else
            {
                return -(pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_BACK_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<10>()]);
            }
            else
            {
                return (pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_FRONT_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<16>()]);
            }
            else
            {
                return -(pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_BACK_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<15>()]);
            }
            else
            {
                return -(pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_FRONT_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<9>()]);
            }
            else
            {
                return (pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_BACK_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<12>()]);
            }
            else
            {
                return (pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_FRONT_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<18>()]);
            }
            else
            {
                return -(pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<8>()]) - (pop[q_i<14>()]));
            }
            else
            {
                return ((pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) - (pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<10>()]) - (pop[q_i<16>()]));
            }
            else
            {
                return ((pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) - (pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<7>()]) - (pop[q_i<13>()]));
            }
            else
            {
                return ((pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) - (pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<9>()]) - (pop[q_i<15>()]));
            }
            else
            {
                return ((pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) - (pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<8>()]) - (pop[q_i<13>()]));
            }
            else
            {
                return ((pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) - (pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<12>()]) - (pop[q_i<18>()]));
            }
            else
            {
                return ((pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) - (pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<10>()]) - (pop[q_i<15>()]));
            }
            else
            {
                return ((pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) - (pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<12>()]) - (pop[q_i<17>()]));
            }
            else
            {
                return ((pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) - (pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_mxz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<9>()]) - (pop[q_i<16>()]));
            }
            else
            {
                return ((pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) - (pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<11>()]) - (pop[q_i<18>()]));
            }
            else
            {
                return ((pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) - (pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<7>()]) - (pop[q_i<14>()]));
            }
            else
            {
                return ((pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) - (pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<11>()]) - (pop[q_i<17>()]));
            }
            else
            {
                return ((pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) - (pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()]));
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_BACK_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<17>()]);
            }
            else
            {
                return -(pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_FRONT_myz_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<11>()]);
            }
            else
            {
                return (pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_NORTH_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<7>()]);
            }
            else
            {
                return (pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]);
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_NORTH_mxy_I(const thread::array<scalar_t, Q> &pop) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<14>()]);
            }
            else
            {
                return -(pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()]);
            }
        }
    };
}

#endif