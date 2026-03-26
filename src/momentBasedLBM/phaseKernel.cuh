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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Definition of the main phase field GPU kernels

Namespace
    LBM::host, LBM::device

SourceFiles
    phaseKernel.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PHASEKERNEL_CUH
#define __MBLBM_PHASEKERNEL_CUH

namespace LBM
{
    /**
     * @brief Determines the amount of shared memory required for a kernel based on the velocity set
     **/
    template <class VelocitySet>
    __device__ __host__ [[nodiscard]] inline consteval device::label_t smem_alloc_size() noexcept
    {
        if constexpr (true)
        {
            return block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS<true, host::label_t>()>(sizeof(scalar_t));
        }
        else
        {
            return 0;
        };
    }

    /**
     * @brief Minimum number of blocks per streaming microprocessor
     **/
    __host__ [[nodiscard]] inline consteval device::label_t MIN_BLOCKS_PER_MP() noexcept { return 1; }

    /**
     * @brief Load a neighboring phase value using shared-memory fast path and scalar halo fallback
     * @tparam dx Neighbor offset in x-direction
     * @tparam dy Neighbor offset in y-direction
     * @tparam dz Neighbor offset in z-direction
     * @tparam PhaseHalo Halo type used to fetch off-block neighbors
     **/
    template <const int dx, const int dy, const int dz, class PhaseHalo, class HydroShared>
    __device__ [[nodiscard]] inline scalar_t load_phase_neighbor(
        const scalar_t *const ptrRestrict phi,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        HydroShared &hydroShared,
        const thread::coordinate &Tx,
        const block::coordinate &Bx,
        const device::pointCoordinate &point) noexcept
    {
        velocityCoefficient::assertions::validate<dx, velocityCoefficient::CAN_BE_NULL>();
        velocityCoefficient::assertions::validate<dy, velocityCoefficient::CAN_BE_NULL>();
        velocityCoefficient::assertions::validate<dz, velocityCoefficient::CAN_BE_NULL>();
        static_assert(!(dx == 0 && dy == 0 && dz == 0), "Neighbor offset must be non-zero.");

        bool xInBlock = true;
        bool yInBlock = true;
        bool zInBlock = true;

        if constexpr (dx == -1)
        {
            xInBlock = Tx.value<axis::X>() > static_cast<device::label_t>(0);
        }
        else if constexpr (dx == +1)
        {
            xInBlock = Tx.value<axis::X>() < (block::n<axis::X>() - static_cast<device::label_t>(1));
        }

        if constexpr (dy == -1)
        {
            yInBlock = Tx.value<axis::Y>() > static_cast<device::label_t>(0);
        }
        else if constexpr (dy == +1)
        {
            yInBlock = Tx.value<axis::Y>() < (block::n<axis::Y>() - static_cast<device::label_t>(1));
        }

        if constexpr (dz == -1)
        {
            zInBlock = Tx.value<axis::Z>() > static_cast<device::label_t>(0);
        }
        else if constexpr (dz == +1)
        {
            zInBlock = Tx.value<axis::Z>() < (block::n<axis::Z>() - static_cast<device::label_t>(1));
        }

        if (xInBlock && yInBlock && zInBlock)
        {
            constexpr device::label_t sharedStride = label_constant<NUMBER_MOMENTS<true>() + 1>();
            constexpr device::label_t phiSharedOffset = label_constant<NUMBER_MOMENTS<true>()>();

            const device::label_t tx = Tx.shifted_coordinate<axis::X, dx>();
            const device::label_t ty = Tx.shifted_coordinate<axis::Y, dy>();
            const device::label_t tz = Tx.shifted_coordinate<axis::Z, dz>();
            const device::label_t tid = block::idx(tx, ty, tz);

            return hydroShared[tid * sharedStride + phiSharedOffset];
        }

        return PhaseHalo::template pull_scalar<dx, dy, dz>(phi, phiBuffer, Tx, Bx, point);
    }

    /**
     * @brief Implements the streaming step of the phase-field lattice Boltzmann method using the moment representation and a chosen velocity set
     * @tparam BoundaryConditions The boundary conditions of the solver
     * @tparam VelocitySet The hydrodynamic velocity set
     * @tparam PhaseVelocitySet The phase-field velocity set
     * @tparam Collision The collision model
     * @tparam HydroHalo The class handling hydrodynamic inter-block streaming
     * @tparam PhaseHalo The class handling phase-field inter-block streaming
     * @param[in] devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param[in] hydroBuffer Collection of pointers to the block halo faces used during hydrodynamic streaming
     * @param[in] phaseBuffer Collection of pointers to the block halo faces used during phase-field streaming
     * @param[in] phiBuffer Collection of pointers to scalar phase-field halo faces used by normal calculation
     * @param[in] hydroShared Inline or externally stored shared memory buffer
     **/
    template <class BoundaryConditions, class VelocitySet, class PhaseVelocitySet, class HydroHalo, class PhaseHalo, class HydroShared, class PhaseShared>
    __device__ inline void phaseStream(
        const device::ptrCollection<11, scalar_t> &devPtrs,
        const device::ptrCollection<6, const scalar_t> &hydroBuffer,
        const device::ptrCollection<6, const scalar_t> &phaseBuffer,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        HydroShared &hydroShared,
        PhaseShared &phaseShared)
    {
        static_assert(std::is_same_v<HydroHalo, device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>>);
        static_assert(std::is_same_v<PhaseHalo, device::halo<PhaseVelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>>);

        const thread::coordinate Tx;

        const block::coordinate Bx;

        const device::pointCoordinate point(Tx, Bx);

        // Index into global arrays
        const device::label_t idx = device::idx(Tx, Bx);

        // Into block arrays
        const device::label_t tid = block::idx(Tx);

        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds(point))
            {
                return;
            }
        }

        scalar_t normx_ = static_cast<scalar_t>(0);
        scalar_t normy_ = static_cast<scalar_t>(0);
        scalar_t normz_ = static_cast<scalar_t>(0);

        const scalar_t *const ptrRestrict phi = devPtrs.ptr<10>();

        constexpr device::label_t sharedStride = label_constant<NUMBER_MOMENTS<true>() + 1>();
        constexpr device::label_t phiSharedOffset = label_constant<NUMBER_MOMENTS<true>()>();

        hydroShared[tid * sharedStride + phiSharedOffset] = phi[idx];

        block::sync();

        const bool isInterior =
            (point.value<axis::X>() > static_cast<device::label_t>(0)) &&
            (point.value<axis::X>() < (device::n<axis::X>() - static_cast<device::label_t>(1))) &&
            (point.value<axis::Y>() > static_cast<device::label_t>(0)) &&
            (point.value<axis::Y>() < (device::n<axis::Y>() - static_cast<device::label_t>(1))) &&
            (point.value<axis::Z>() > static_cast<device::label_t>(0)) &&
            (point.value<axis::Z>() < (device::n<axis::Z>() - static_cast<device::label_t>(1)));

        if (isInterior)
        {
            // Load the neighbor phi values
            const scalar_t phi_xp = load_phase_neighbor<+1, 0, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xm = load_phase_neighbor<-1, 0, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_yp = load_phase_neighbor<0, +1, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_ym = load_phase_neighbor<0, -1, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_zp = load_phase_neighbor<0, 0, +1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_zm = load_phase_neighbor<0, 0, -1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            const scalar_t phi_xp1_yp1_z = load_phase_neighbor<+1, +1, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xp1_ym1_z = load_phase_neighbor<+1, -1, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xm1_yp1_z = load_phase_neighbor<-1, +1, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xm1_ym1_z = load_phase_neighbor<-1, -1, 0, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xp1_y_zp1 = load_phase_neighbor<+1, 0, +1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xp1_y_zm1 = load_phase_neighbor<+1, 0, -1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xm1_y_zp1 = load_phase_neighbor<-1, 0, +1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_xm1_y_zm1 = load_phase_neighbor<-1, 0, -1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_x_yp1_zp1 = load_phase_neighbor<0, +1, +1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_x_yp1_zm1 = load_phase_neighbor<0, +1, -1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_x_ym1_zp1 = load_phase_neighbor<0, -1, +1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phi_x_ym1_zm1 = load_phase_neighbor<0, -1, -1, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            // Compute gradients
            const scalar_t sgx =
                VelocitySet::w_1<scalar_t>() * (phi_xp - phi_xm) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                phi_xp1_ym1_z - phi_xm1_yp1_z +
                                                phi_xp1_y_zm1 - phi_xm1_y_zp1);

            const scalar_t sgy =
                VelocitySet::w_1<scalar_t>() * (phi_yp - phi_ym) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                phi_xm1_yp1_z - phi_xp1_ym1_z +
                                                phi_x_yp1_zm1 - phi_x_ym1_zp1);

            const scalar_t sgz =
                VelocitySet::w_1<scalar_t>() * (phi_zp - phi_zm) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                                phi_x_ym1_zp1 - phi_x_yp1_zm1);

            const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
            const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
            const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

            // Interface indicator
            const scalar_t ind_ = sqrtf(gx * gx + gy * gy + gz * gz);

            // Compute normals
            const scalar_t invInd = static_cast<scalar_t>(1) / (ind_ + static_cast<scalar_t>(1e-9));
            normx_ = gx * invInd;
            normy_ = gy * invInd;
            normz_ = gz * invInd;
        }

        block::sync();

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS<true>()> moments;
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                const device::label_t ID = tid * m_i<NUMBER_MOMENTS<true>() + 1>() + m_i<moment>();
                hydroShared[ID] = devPtrs.ptr<moment>()[idx];
                if constexpr (moment == index::rho)
                {
                    moments[moment] = hydroShared[ID] + rho0();
                }
                else
                {
                    moments[moment] = hydroShared[ID];
                }
            });

        block::sync();

        // Reconstruct the populations from the moments
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g = PhaseVelocitySet::reconstruct(moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term) on g-populations
        PhaseVelocitySet::sharpen(pop_g, phi_, normx_, normy_, normz_);

        // Save/pull from shared memory
        {
            // Save populations in shared memory
            streaming::save<VelocitySet>(pop, hydroShared, tid);
            streaming::save<PhaseVelocitySet>(pop_g, phaseShared, tid);

            block::sync();

            // Pull from shared memory
            streaming::pull<VelocitySet>(pop, hydroShared, Tx);
            streaming::pull<PhaseVelocitySet>(pop_g, phaseShared, Tx);

            // Pull pop from global memory in cover nodes
            HydroHalo::pull(pop, hydroBuffer, Tx, Bx, point);
            PhaseHalo::pull(pop_g, phaseBuffer, Tx, Bx, point);

            block::sync();
        }

        // Compute post-stream moments
        velocitySet::calculate_moments<VelocitySet>(pop, moments);
        PhaseVelocitySet::calculate_phi(pop_g, moments);

        // Update the shared buffer with the refreshed moments
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                const device::label_t ID = tid * label_constant<NUMBER_MOMENTS<true>() + 1>() + label_constant<moment>();
                hydroShared[ID] = moments[moment];
            });

        block::sync();

        // Calculate the moments at the boundary
        {
            const normalVector boundaryNormal(point);

            if (boundaryNormal.isBoundary())
            {
                BoundaryConditions::template calculate_moments<VelocitySet, PhaseVelocitySet>(pop, moments, boundaryNormal, hydroShared, Tx, point);
            }
        }

        // Coalesced write to global memory
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                if constexpr (moment == index::rho)
                {
                    devPtrs.ptr<moment>()[idx] = moments[moment] - rho0();
                }
                else
                {
                    devPtrs.ptr<moment>()[idx] = moments[moment];
                }
            });
    }

    /**
     * @brief Implements the collision step of the phase-field lattice Boltzmann method using the moment representation and a chosen velocity set
     * @tparam BoundaryConditions The boundary conditions of the solver
     * @tparam VelocitySet The hydrodynamic velocity set
     * @tparam PhaseVelocitySet The phase-field velocity set
     * @tparam HydroHalo The class handling hydrodynamic inter-block streaming
     * @tparam PhaseHalo The class handling phase-field inter-block streaming
     * @param[in] devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param[in] hydroBuffer Collection of pointers to the block halo faces used during hydrodynamic streaming
     * @param[in] phaseBuffer Collection of pointers to the block halo faces used during phase-field streaming
     **/
    template <class BoundaryConditions, class VelocitySet, class PhaseVelocitySet, class Collision, class HydroHalo, class PhaseHalo>
    __device__ inline void phaseCollide(
        const device::ptrCollection<11, scalar_t> &devPtrs,
        const device::ptrCollection<6, const scalar_t> &hydroBuffer,
        const device::ptrCollection<6, const scalar_t> &phaseBuffer)
    {
        // Not implemented yet
    }
}

#endif
