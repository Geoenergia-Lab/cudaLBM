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
        constexpr int maxOffset = 2;
        static_assert((dx >= -maxOffset) && (dx <= maxOffset), "dx must be in [-2, 2].");
        static_assert((dy >= -maxOffset) && (dy <= maxOffset), "dy must be in [-2, 2].");
        static_assert((dz >= -maxOffset) && (dz <= maxOffset), "dz must be in [-2, 2].");

        const auto inBlock = [](const device::label_t t, const int d, const device::label_t n) noexcept -> bool
        {
            const int shifted = static_cast<int>(t) + d;
            return (shifted >= 0) && (shifted < static_cast<int>(n));
        };

        const bool xInBlock = inBlock(Tx.value<axis::X>(), dx, block::n<axis::X>());
        const bool yInBlock = inBlock(Tx.value<axis::Y>(), dy, block::n<axis::Y>());
        const bool zInBlock = inBlock(Tx.value<axis::Z>(), dz, block::n<axis::Z>());

        if (xInBlock && yInBlock && zInBlock)
        {
            constexpr device::label_t sharedStride = label_constant<NUMBER_MOMENTS<true>() + 1>();
            constexpr device::label_t phiSharedOffset = label_constant<NUMBER_MOMENTS<true>()>();

            const device::label_t tx = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + dx);
            const device::label_t ty = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + dy);
            const device::label_t tz = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + dz);
            const device::label_t tid = block::idx(tx, ty, tz);

            return hydroShared[tid * sharedStride + phiSharedOffset];
        }

#ifdef MULTI_GPU
        return PhaseHalo::template pull_scalar<dx, dy, dz>(phi, phiBuffer, Tx, Bx, point);
#else
        (void)phiBuffer;
        return PhaseHalo::template pull_scalar_local<dx, dy, dz>(phi, Tx, Bx, point);
#endif
    }

    /**
     * @brief Load a neighboring phase value directly from global/halo storage (no shared-memory fast path)
     * @tparam dx Neighbor offset in x-direction
     * @tparam dy Neighbor offset in y-direction
     * @tparam dz Neighbor offset in z-direction
     * @tparam PhaseHalo Halo type used to fetch off-block neighbors
     **/
    template <const int dx, const int dy, const int dz, class PhaseHalo>
    __device__ [[nodiscard]] inline scalar_t load_phase_neighbor_direct(
        const scalar_t *const ptrRestrict phi,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        const thread::coordinate &Tx,
        const block::coordinate &Bx,
        const device::pointCoordinate &point) noexcept
    {
        constexpr int maxOffset = 2;
        static_assert((dx >= -maxOffset) && (dx <= maxOffset), "dx must be in [-2, 2].");
        static_assert((dy >= -maxOffset) && (dy <= maxOffset), "dy must be in [-2, 2].");
        static_assert((dz >= -maxOffset) && (dz <= maxOffset), "dz must be in [-2, 2].");

#ifdef MULTI_GPU
        return PhaseHalo::template pull_scalar<dx, dy, dz>(phi, phiBuffer, Tx, Bx, point);
#else
        (void)phiBuffer;
        return PhaseHalo::template pull_scalar_local<dx, dy, dz>(phi, Tx, Bx, point);
#endif
    }

    /**
     * @brief Compute isotropic phase gradient at a shifted lattice point
     * @tparam ox,oy,oz Shift of the evaluation point relative to current thread point
     **/
    template <const int ox, const int oy, const int oz, class VelocitySet, class PhaseHalo, class HydroShared>
    __device__ inline void compute_phase_gradient(
        const scalar_t *const ptrRestrict phi,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        HydroShared &hydroShared,
        const thread::coordinate &Tx,
        const block::coordinate &Bx,
        const device::pointCoordinate &point,
        scalar_t &gx,
        scalar_t &gy,
        scalar_t &gz) noexcept
    {
        scalar_t sgx = static_cast<scalar_t>(0);
        scalar_t sgy = static_cast<scalar_t>(0);
        scalar_t sgz = static_cast<scalar_t>(0);

        device::constexpr_for<1, VelocitySet::Q()>(
            [&](const auto q)
            {
                constexpr device::label_t qi = q();
                constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                const scalar_t phi_q = load_phase_neighbor<ox + cx, oy + cy, oz + cz, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
                const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());

                sgx += wq * static_cast<scalar_t>(cx) * phi_q;
                sgy += wq * static_cast<scalar_t>(cy) * phi_q;
                sgz += wq * static_cast<scalar_t>(cz) * phi_q;
            });

        gx = velocitySet::as2<scalar_t>() * sgx;
        gy = velocitySet::as2<scalar_t>() * sgy;
        gz = velocitySet::as2<scalar_t>() * sgz;
    }

    /**
     * @brief Compute interface normal and indicator at a shifted lattice point
     **/
    template <const int ox, const int oy, const int oz, class VelocitySet, class PhaseHalo, class HydroShared>
    __device__ inline void compute_phase_normal(
        const scalar_t *const ptrRestrict phi,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        HydroShared &hydroShared,
        const thread::coordinate &Tx,
        const block::coordinate &Bx,
        const device::pointCoordinate &point,
        scalar_t &normx,
        scalar_t &normy,
        scalar_t &normz,
        scalar_t &ind) noexcept
    {
        scalar_t gx = static_cast<scalar_t>(0);
        scalar_t gy = static_cast<scalar_t>(0);
        scalar_t gz = static_cast<scalar_t>(0);
        compute_phase_gradient<ox, oy, oz, VelocitySet, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point, gx, gy, gz);

        ind = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

        normx = gx * invInd;
        normy = gy * invInd;
        normz = gz * invInd;
    }

    /**
     * @brief Compute curvature from divergence of normals using the current VelocitySet stencil
     **/
    template <class VelocitySet, class PhaseHalo, class HydroShared>
    __device__ [[nodiscard]] inline scalar_t compute_phase_curvature(
        const scalar_t *const ptrRestrict phi,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        HydroShared &hydroShared,
        const thread::coordinate &Tx,
        const block::coordinate &Bx,
        const device::pointCoordinate &point) noexcept
    {
        scalar_t scx = static_cast<scalar_t>(0);
        scalar_t scy = static_cast<scalar_t>(0);
        scalar_t scz = static_cast<scalar_t>(0);

        device::constexpr_for<1, VelocitySet::Q()>(
            [&](const auto q)
            {
                constexpr device::label_t qi = q();
                constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                scalar_t nx_q = static_cast<scalar_t>(0);
                scalar_t ny_q = static_cast<scalar_t>(0);
                scalar_t nz_q = static_cast<scalar_t>(0);
                scalar_t ind_q = static_cast<scalar_t>(0);

                compute_phase_normal<cx, cy, cz, VelocitySet, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point, nx_q, ny_q, nz_q, ind_q);

                const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());
                scx += wq * static_cast<scalar_t>(cx) * nx_q;
                scy += wq * static_cast<scalar_t>(cy) * ny_q;
                scz += wq * static_cast<scalar_t>(cz) * nz_q;
            });

        return velocitySet::as2<scalar_t>() * (scx + scy + scz);
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
     * @param[in] phiWriteBuffer Collection of writable pointers to scalar phase-field halo faces for next-step neighbor stencils
     * @param[in] hydroShared Inline or externally stored shared memory buffer
     **/
    template <class BoundaryConditions, class VelocitySet, class PhaseVelocitySet, class HydroHalo, class PhaseHalo, class HydroShared, class PhaseShared>
    __device__ inline void phaseStream(
        const device::ptrCollection<11, scalar_t> &devPtrs,
        const device::ptrCollection<6, const scalar_t> &hydroBuffer,
        const device::ptrCollection<6, const scalar_t> &phaseBuffer,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        const device::ptrCollection<6, scalar_t> &phiWriteBuffer,
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
            scalar_t indDummy = static_cast<scalar_t>(0);
            compute_phase_normal<0, 0, 0, VelocitySet, PhaseHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point, normx_, normy_, normz_, indDummy);
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
                const scalar_t *const hydroSharedPtr = hydroShared;

                if constexpr (requires {
                                  BoundaryConditions::template calculate_moments<VelocitySet, PhaseVelocitySet, const scalar_t *>(
                                      pop,
                                      moments,
                                      boundaryNormal,
                                      hydroSharedPtr,
                                      Tx,
                                      point);
                              })
                {
                    BoundaryConditions::template calculate_moments<VelocitySet, PhaseVelocitySet, const scalar_t *>(
                        pop,
                        moments,
                        boundaryNormal,
                        hydroSharedPtr,
                        Tx,
                        point);
                }
                else if constexpr (requires {
                                       BoundaryConditions::template calculate_moments<VelocitySet, PhaseVelocitySet>(
                                           pop,
                                           moments,
                                           boundaryNormal,
                                           hydroShared,
                                           Tx,
                                           point);
                                   })
                {
                    BoundaryConditions::template calculate_moments<VelocitySet, PhaseVelocitySet>(
                        pop,
                        moments,
                        boundaryNormal,
                        hydroShared,
                        Tx,
                        point);
                }
                else
                {
                    thread::array<scalar_t, NUMBER_MOMENTS<false>()> hydroMoments;

                    device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
                        [&](const auto moment)
                        {
                            hydroMoments[moment] = moments[moment];
                        });

                    BoundaryConditions::template calculate_moments<VelocitySet>(
                        pop,
                        hydroMoments,
                        boundaryNormal,
                        hydroShared,
                        Tx,
                        point);

                    device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
                        [&](const auto moment)
                        {
                            moments[moment] = hydroMoments[moment];
                        });
                }
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

        // Save scalar phi for neighbor-gradient stencils used by the next collide/stream step
#ifdef MULTI_GPU
        PhaseHalo::save_scalar(moments[m_i<10>()], phiWriteBuffer, Tx, Bx, point);
#else
        (void)phiWriteBuffer;
#endif
    }

    /**
     * @brief Implements the collision step of the phase-field lattice Boltzmann method using the moment representation and a chosen velocity set
     * @tparam BoundaryConditions The boundary conditions of the solver
     * @tparam VelocitySet The hydrodynamic velocity set
     * @tparam PhaseVelocitySet The phase-field velocity set
     * @tparam HydroHalo The class handling hydrodynamic inter-block streaming
     * @tparam PhaseHalo The class handling phase-field inter-block streaming
     * @param[in] devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param[in] hydroBuffer Collection of writable pointers to hydrodynamic halo faces
     * @param[in] phaseBuffer Collection of writable pointers to phase-population halo faces
     * @param[in] phiBuffer Collection of read-only pointers to scalar phase-field halo faces (freshly written in phaseStream)
     **/
    template <class BoundaryConditions, class VelocitySet, class PhaseVelocitySet, class Collision, class HydroHalo, class PhaseHalo>
    __device__ inline void phaseCollide(
        const device::ptrCollection<11, scalar_t> &devPtrs,
        const device::ptrCollection<6, scalar_t> &hydroBuffer,
        const device::ptrCollection<6, scalar_t> &phaseBuffer,
        const device::ptrCollection<6, const scalar_t> &phiBuffer)
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
                if constexpr (moment == index::rho)
                {
                    moments[moment] = devPtrs.ptr<moment>()[idx] + rho0();
                }
                else
                {
                    moments[moment] = devPtrs.ptr<moment>()[idx];
                }
            });

        // Zero forces, normals and indicator at bulk
        scalar_t Fsx = static_cast<scalar_t>(0);
        scalar_t Fsy = static_cast<scalar_t>(0);
        scalar_t Fsz = static_cast<scalar_t>(0);
        scalar_t normx_ = static_cast<scalar_t>(0);
        scalar_t normy_ = static_cast<scalar_t>(0);
        scalar_t normz_ = static_cast<scalar_t>(0);
        scalar_t ind_ = static_cast<scalar_t>(0);
        scalar_t centerNormx = static_cast<scalar_t>(0);
        scalar_t centerNormy = static_cast<scalar_t>(0);
        scalar_t centerNormz = static_cast<scalar_t>(0);
        scalar_t centerInd = static_cast<scalar_t>(0);

        const scalar_t *const ptrRestrict phi = devPtrs.ptr<10>();

        constexpr device::label_t phiTileNx = block::n<axis::X>() + static_cast<device::label_t>(4);
        constexpr device::label_t phiTileNy = block::n<axis::Y>() + static_cast<device::label_t>(4);
        constexpr device::label_t phiTileNz = block::n<axis::Z>() + static_cast<device::label_t>(4);
        constexpr device::label_t phiTileSize = phiTileNx * phiTileNy * phiTileNz;
        constexpr device::label_t normalNx = block::n<axis::X>() + static_cast<device::label_t>(2);
        constexpr device::label_t normalNy = block::n<axis::Y>() + static_cast<device::label_t>(2);
        constexpr device::label_t normalNz = block::n<axis::Z>() + static_cast<device::label_t>(2);
        constexpr device::label_t normalTileSize = normalNx * normalNy * normalNz;

        __shared__ scalar_t phiShared[phiTileSize];
        __shared__ scalar_t normSharedX[normalTileSize];
        __shared__ scalar_t normSharedY[normalTileSize];
        __shared__ scalar_t normSharedZ[normalTileSize];

        const auto phiSharedIdx = [](const device::label_t sx, const device::label_t sy, const device::label_t sz) noexcept -> device::label_t
        {
            return (sz * phiTileNy + sy) * phiTileNx + sx;
        };

        const auto normalSharedIdx = [](const device::label_t sx, const device::label_t sy, const device::label_t sz) noexcept -> device::label_t
        {
            return (sz * normalNy + sy) * normalNx + sx;
        };

        const auto ownsPhiOffsetX = [&Tx](const int ox) noexcept -> bool
        {
            return (ox == 0) ||
                   (((ox == -1) || (ox == -2)) && (Tx.value<axis::X>() == static_cast<device::label_t>(0))) ||
                   (((ox == 1) || (ox == 2)) && (Tx.value<axis::X>() == (block::n<axis::X>() - static_cast<device::label_t>(1))));
        };

        const auto ownsPhiOffsetY = [&Tx](const int oy) noexcept -> bool
        {
            return (oy == 0) ||
                   (((oy == -1) || (oy == -2)) && (Tx.value<axis::Y>() == static_cast<device::label_t>(0))) ||
                   (((oy == 1) || (oy == 2)) && (Tx.value<axis::Y>() == (block::n<axis::Y>() - static_cast<device::label_t>(1))));
        };

        const auto ownsPhiOffsetZ = [&Tx](const int oz) noexcept -> bool
        {
            return (oz == 0) ||
                   (((oz == -1) || (oz == -2)) && (Tx.value<axis::Z>() == static_cast<device::label_t>(0))) ||
                   (((oz == 1) || (oz == 2)) && (Tx.value<axis::Z>() == (block::n<axis::Z>() - static_cast<device::label_t>(1))));
        };

        const auto ownsNormalOffsetX = [&Tx](const int ox) noexcept -> bool
        {
            return (ox == 0) ||
                   ((ox == -1) && (Tx.value<axis::X>() == static_cast<device::label_t>(0))) ||
                   ((ox == 1) && (Tx.value<axis::X>() == (block::n<axis::X>() - static_cast<device::label_t>(1))));
        };

        const auto ownsNormalOffsetY = [&Tx](const int oy) noexcept -> bool
        {
            return (oy == 0) ||
                   ((oy == -1) && (Tx.value<axis::Y>() == static_cast<device::label_t>(0))) ||
                   ((oy == 1) && (Tx.value<axis::Y>() == (block::n<axis::Y>() - static_cast<device::label_t>(1))));
        };

        const auto ownsNormalOffsetZ = [&Tx](const int oz) noexcept -> bool
        {
            return (oz == 0) ||
                   ((oz == -1) && (Tx.value<axis::Z>() == static_cast<device::label_t>(0))) ||
                   ((oz == 1) && (Tx.value<axis::Z>() == (block::n<axis::Z>() - static_cast<device::label_t>(1))));
        };

        // Load phi in a (block + 2-cell halo) shared tile so each stencil point is fetched only once.
        device::constexpr_for<0, 5>(
            [&](const auto oxIdx)
            {
                constexpr int ox = static_cast<int>(oxIdx()) - 2;

                device::constexpr_for<0, 5>(
                    [&](const auto oyIdx)
                    {
                        constexpr int oy = static_cast<int>(oyIdx()) - 2;

                        device::constexpr_for<0, 5>(
                            [&](const auto ozIdx)
                            {
                                constexpr int oz = static_cast<int>(ozIdx()) - 2;

                                if (!(ownsPhiOffsetX(ox) && ownsPhiOffsetY(oy) && ownsPhiOffsetZ(oz)))
                                {
                                    return;
                                }

                                const scalar_t phiCandidate = load_phase_neighbor_direct<ox, oy, oz, PhaseHalo>(
                                    phi,
                                    phiBuffer,
                                    Tx,
                                    Bx,
                                    point);

                                const device::label_t sx = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 2 + ox);
                                const device::label_t sy = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 2 + oy);
                                const device::label_t sz = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 2 + oz);
                                const device::label_t sid = phiSharedIdx(sx, sy, sz);

                                phiShared[sid] = phiCandidate;
                            });
                    });
            });

        block::sync();

        // Precompute interface normals in a shared tile (block + 1-cell halo).
        device::constexpr_for<0, 3>(
            [&](const auto oxIdx)
            {
                constexpr int ox = static_cast<int>(oxIdx()) - 1;

                device::constexpr_for<0, 3>(
                    [&](const auto oyIdx)
                    {
                        constexpr int oy = static_cast<int>(oyIdx()) - 1;

                        device::constexpr_for<0, 3>(
                            [&](const auto ozIdx)
                            {
                                constexpr int oz = static_cast<int>(ozIdx()) - 1;

                                if (!(ownsNormalOffsetX(ox) && ownsNormalOffsetY(oy) && ownsNormalOffsetZ(oz)))
                                {
                                    return;
                                }

                                const device::label_t sxPhi = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 2 + ox);
                                const device::label_t syPhi = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 2 + oy);
                                const device::label_t szPhi = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 2 + oz);

                                scalar_t sgx = static_cast<scalar_t>(0);
                                scalar_t sgy = static_cast<scalar_t>(0);
                                scalar_t sgz = static_cast<scalar_t>(0);

                                device::constexpr_for<1, VelocitySet::Q()>(
                                    [&](const auto q)
                                    {
                                        constexpr device::label_t qi = q();
                                        constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                                        constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                                        constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                                        const device::label_t sqx = static_cast<device::label_t>(static_cast<int>(sxPhi) + cx);
                                        const device::label_t sqy = static_cast<device::label_t>(static_cast<int>(syPhi) + cy);
                                        const device::label_t sqz = static_cast<device::label_t>(static_cast<int>(szPhi) + cz);
                                        const scalar_t phi_q = phiShared[phiSharedIdx(sqx, sqy, sqz)];
                                        const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());

                                        sgx += wq * static_cast<scalar_t>(cx) * phi_q;
                                        sgy += wq * static_cast<scalar_t>(cy) * phi_q;
                                        sgz += wq * static_cast<scalar_t>(cz) * phi_q;
                                    });

                                const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
                                const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
                                const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;
                                const scalar_t indCandidate = sqrtf(gx * gx + gy * gy + gz * gz);
                                const scalar_t invInd = static_cast<scalar_t>(1) / (indCandidate + static_cast<scalar_t>(1e-9));
                                const scalar_t nxCandidate = gx * invInd;
                                const scalar_t nyCandidate = gy * invInd;
                                const scalar_t nzCandidate = gz * invInd;

                                const device::label_t sx = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 1 + ox);
                                const device::label_t sy = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 1 + oy);
                                const device::label_t sz = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 1 + oz);
                                const device::label_t sid = normalSharedIdx(sx, sy, sz);

                                normSharedX[sid] = nxCandidate;
                                normSharedY[sid] = nyCandidate;
                                normSharedZ[sid] = nzCandidate;

                                if constexpr ((ox == 0) && (oy == 0) && (oz == 0))
                                {
                                    centerNormx = nxCandidate;
                                    centerNormy = nyCandidate;
                                    centerNormz = nzCandidate;
                                    centerInd = indCandidate;
                                }
                            });
                    });
            });

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
            normx_ = centerNormx;
            normy_ = centerNormy;
            normz_ = centerNormz;
            ind_ = centerInd;

            scalar_t scx = static_cast<scalar_t>(0);
            scalar_t scy = static_cast<scalar_t>(0);
            scalar_t scz = static_cast<scalar_t>(0);

            device::constexpr_for<1, VelocitySet::Q()>(
                [&](const auto q)
                {
                    constexpr device::label_t qi = q();
                    constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                    constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                    constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                    const device::label_t sx = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 1 + cx);
                    const device::label_t sy = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 1 + cy);
                    const device::label_t sz = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 1 + cz);
                    const device::label_t sid = normalSharedIdx(sx, sy, sz);

                    const scalar_t nx_q = normSharedX[sid];
                    const scalar_t ny_q = normSharedY[sid];
                    const scalar_t nz_q = normSharedZ[sid];

                    const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());
                    scx += wq * static_cast<scalar_t>(cx) * nx_q;
                    scy += wq * static_cast<scalar_t>(cy) * ny_q;
                    scz += wq * static_cast<scalar_t>(cz) * nz_q;
                });

            const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);
            const scalar_t stCurv = -device::sigma * curvature * ind_;

            Fsx = stCurv * normx_;
            Fsy = stCurv * normy_;
            Fsz = stCurv * normz_;
        }

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments, Fsx, Fsy, Fsz);

        // Calculate post collision populations
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g = PhaseVelocitySet::reconstruct(moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term) on g-populations
        PhaseVelocitySet::sharpen(pop_g, phi_, normx_, normy_, normz_);

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

        thread::array<scalar_t, NUMBER_MOMENTS<false>()> hydroMoments;
        device::constexpr_for<0, NUMBER_MOMENTS<false>()>(
            [&](const auto moment)
            {
                hydroMoments[moment] = moments[moment];
            });

        // Save the hydro populations to the block halo
        HydroHalo::save(pop, hydroMoments, hydroBuffer, Tx, Bx, point);

        // Save the phase populations to the block halo
        PhaseHalo::save(pop_g, hydroMoments, phaseBuffer, Tx, Bx, point);
    }
}

#endif
