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
    Main kernels for the multiphase moment representation with the D3Q27
    velocity set for hydrodynamics and D3Q7 for phase field evolution

Namespace
    LBM

SourceFiles
    phaseFieldD3Q27shared.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PHASEFIELDD3Q27_CUH
#define __MBLBM_PHASEFIELDD3Q27_CUH

#include "../../src/LBMIncludes.cuh"
#include "../../src/LBMTypedefs.cuh"
#include "../../src/streaming/streaming.cuh"
#include "../../src/collision/collision.cuh"
#include "../../src/blockHalo/blockHalo.cuh"
#include "../../src/fileIO/fileIO.cuh"
#include "../../src/runTimeIO/runTimeIO.cuh"
#include "../../src/functionObjects/objectRegistry.cuh"
#include "../../src/array/array.cuh"
#include "../../src/boundaryConditions/boundaryConditions.cuh"

namespace LBM
{
    /**
     * @brief Boundary conditions aliases
     **/
#ifdef MULTIPHASEJET
    using BoundaryConditions = multiphaseJet;
    __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return true; }
    __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return true; }
#endif

#ifdef SUBSEAMECHANICALDISPERSION
    using BoundaryConditions = subseaMechanicalDispersion;
    __device__ __host__ [[nodiscard]] inline consteval bool periodicX() noexcept { return false; }
    __device__ __host__ [[nodiscard]] inline consteval bool periodicY() noexcept { return false; }
#endif

    using VelocitySet = D3Q27;
    using PhaseVelocitySet = D3Q7;
    using Collision = secondOrder;

    // Aliases use the standard halo methods
    using HydroHalo = device::halo<VelocitySet, periodicX(), periodicY()>;
    using PhaseHalo = device::halo<PhaseVelocitySet, periodicX(), periodicY()>;

    __device__ __host__ [[nodiscard]] inline consteval label_t smem_alloc_size() noexcept
    {
        return block::sharedMemoryBufferSize<VelocitySet, 11>(sizeof(scalar_t));
    }

    __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBoundsD3Q27 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

    /**
     * @brief Performs the streaming step of the lattice Boltzmann method using the multiphase moment representation (D3Q27 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
     * @param fBlockHalo Object containing pointers to the individual block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the individual block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q27 __global__ void phaseFieldStream(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const device::ptrCollection<6, const scalar_t> ghostHydro,
        const device::ptrCollection<6, const scalar_t> ghostPhase,
        const label_t step)
    {
        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds())
            {
                return;
            }
        }

        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        const bool isInterior =
            (x > 0) & (x < device::nx - 1) &
            (y > 0) & (y < device::ny - 1) &
            (z > 0) & (z < device::nz - 1);

        const label_t idx = device::idx();

        scalar_t normx_ = static_cast<scalar_t>(0);
        scalar_t normy_ = static_cast<scalar_t>(0);
        scalar_t normz_ = static_cast<scalar_t>(0);

        const scalar_t *const ptrRestrict phi = devPtrs.ptr<10>();

        if (isInterior)
        {
            // Block volume and block-to-block strides
            const label_t stride_by = block::size() * gridDim.x;
            const label_t stride_bz = block::size() * gridDim.x * gridDim.y;

            // Wraps for when crossing a block face
            const label_t wrap_x = block::size() - (block::nx() - static_cast<label_t>(1)) * static_cast<label_t>(1);
            const label_t wrap_y = stride_by - (block::ny() - static_cast<label_t>(1)) * block::nx();
            const label_t wrap_z = stride_bz - (block::nz() - static_cast<label_t>(1)) * block::stride_z();

            // +/-1 deltas in each direction, corrected when crossing block boundaries
            const label_t dxp = (threadIdx.x == (block::nx() - static_cast<label_t>(1))) ? wrap_x : static_cast<label_t>(1);
            const label_t dxm = (threadIdx.x == static_cast<label_t>(0)) ? wrap_x : static_cast<label_t>(1);

            const label_t dyp = (threadIdx.y == (block::ny() - static_cast<label_t>(1))) ? wrap_y : block::nx();
            const label_t dym = (threadIdx.y == static_cast<label_t>(0)) ? wrap_y : block::nx();

            const label_t dzp = (threadIdx.z == (block::nz() - static_cast<label_t>(1))) ? wrap_z : block::stride_z();
            const label_t dzm = (threadIdx.z == static_cast<label_t>(0)) ? wrap_z : block::stride_z();

            // Axis neighbors (diagonals can be constructed from them)
            const label_t i_xp = idx + dxp;
            const label_t i_xm = idx - dxm;
            const label_t i_yp = idx + dyp;
            const label_t i_ym = idx - dym;
            const label_t i_zp = idx + dzp;
            const label_t i_zm = idx - dzm;

            // Load the neighbor phi values
            const scalar_t phi_xp1_yp1_z = phi[i_xp + dyp];
            const scalar_t phi_xp1_ym1_z = phi[i_xp - dym];
            const scalar_t phi_xm1_yp1_z = phi[i_xm + dyp];
            const scalar_t phi_xm1_ym1_z = phi[i_xm - dym];
            const scalar_t phi_xp1_y_zp1 = phi[i_xp + dzp];
            const scalar_t phi_xp1_y_zm1 = phi[i_xp - dzm];
            const scalar_t phi_xm1_y_zp1 = phi[i_xm + dzp];
            const scalar_t phi_xm1_y_zm1 = phi[i_xm - dzm];
            const scalar_t phi_x_yp1_zp1 = phi[i_yp + dzp];
            const scalar_t phi_x_yp1_zm1 = phi[i_yp - dzm];
            const scalar_t phi_x_ym1_zp1 = phi[i_ym + dzp];
            const scalar_t phi_x_ym1_zm1 = phi[i_ym - dzm];
            const scalar_t phi_xp1_yp1_zp1 = phi[i_xp + dyp + dzp];
            const scalar_t phi_xp1_yp1_zm1 = phi[i_xp + dyp - dzm];
            const scalar_t phi_xp1_ym1_zp1 = phi[i_xp - dym + dzp];
            const scalar_t phi_xp1_ym1_zm1 = phi[i_xp - dym - dzm];
            const scalar_t phi_xm1_yp1_zp1 = phi[i_xm + dyp + dzp];
            const scalar_t phi_xm1_yp1_zm1 = phi[i_xm + dyp - dzm];
            const scalar_t phi_xm1_ym1_zp1 = phi[i_xm - dym + dzp];
            const scalar_t phi_xm1_ym1_zm1 = phi[i_xm - dym - dzm];

            // Compute gradients
            const scalar_t sgx =
                VelocitySet::w_1<scalar_t>() * (phi[i_xp] - phi[i_xm]) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                phi_xp1_ym1_z - phi_xm1_yp1_z +
                                                phi_xp1_y_zm1 - phi_xm1_y_zp1) +
                VelocitySet::w_3<scalar_t>() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                                phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                                phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                                phi_xp1_ym1_zm1 - phi_xm1_yp1_zp1);
            ;

            const scalar_t sgy =
                VelocitySet::w_1<scalar_t>() * (phi[i_yp] - phi[i_ym]) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                phi_xm1_yp1_z - phi_xp1_ym1_z +
                                                phi_x_yp1_zm1 - phi_x_ym1_zp1) +
                VelocitySet::w_3<scalar_t>() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                                phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                                phi_xm1_yp1_zm1 - phi_xp1_ym1_zp1 +
                                                phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);

            const scalar_t sgz =
                VelocitySet::w_1<scalar_t>() * (phi[i_zp] - phi[i_zm]) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                                phi_x_ym1_zp1 - phi_x_yp1_zm1) +
                VelocitySet::w_3<scalar_t>() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                                phi_xm1_ym1_zp1 - phi_xp1_yp1_zm1 +
                                                phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                                phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);

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

        // Prefetch devPtrs into L2
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
            });

        // Declare shared memory
        extern __shared__ scalar_t shared_buffer[];
        __shared__ scalar_t shared_buffer_g[(PhaseVelocitySet::Q() - 1) * block::stride()];

        const label_t tid = device::idxBlock();

        // Coalesced read from global memory
        thread::array<scalar_t, NUMBER_MOMENTS<true>()> moments;
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                const label_t ID = tid * m_i<NUMBER_MOMENTS<true>() + 1>() + m_i<moment>();
                shared_buffer[ID] = devPtrs.ptr<moment>()[idx];
                if constexpr (moment == index::rho())
                {
                    moments[moment] = shared_buffer[ID] + rho0<scalar_t>();
                }
                else
                {
                    moments[moment] = shared_buffer[ID];
                }
            });

        __syncthreads();

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
            streaming::save<VelocitySet>(pop, shared_buffer, tid);
            streaming::save<PhaseVelocitySet>(pop_g, shared_buffer_g, tid);

            __syncthreads();

            // Pull from shared memory
            streaming::pull<VelocitySet>(pop, shared_buffer);
            streaming::phase_pull(pop_g, shared_buffer_g);
        }

        // Load hydro pop from global memory in cover nodes
        HydroHalo::load(pop, ghostHydro);

        // Load phase pop from global memory in cover nodes
        PhaseHalo::load(pop_g, ghostPhase);

        // Compute post-stream moments
        velocitySet::calculate_moments<VelocitySet>(pop, moments);
        PhaseVelocitySet::calculate_phi(pop_g, moments);

        // Update the shared buffer with the refreshed moments
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                const label_t ID = tid * label_constant<NUMBER_MOMENTS<true>() + 1>() + label_constant<moment>();
                shared_buffer[ID] = moments[moment];
            });

        __syncthreads();

        // Calculate the moments at the boundary
        {
            const normalVector boundaryNormal;

            if (boundaryNormal.isBoundary())
            {
                BoundaryConditions::calculate_moments<VelocitySet, PhaseVelocitySet>(pop, moments, boundaryNormal, shared_buffer, step);
            }
        }

        // Coalesced write to global memory
        moments[m_i<0>()] = moments[m_i<0>()] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });
    }

    /**
     * @brief Performs the collision step of the lattice Boltzmann method using the multiphase moment representation (D3Q27 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param fBlockHalo Object containing pointers to the individual block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the individual block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q27 __global__ void phaseFieldCollide(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const device::ptrCollection<6, scalar_t> ghostHydro,
        const device::ptrCollection<6, scalar_t> ghostPhase)
    {
        // Always a multiple of 32, so no need to check this(I think)
        if constexpr (out_of_bounds_check())
        {
            if (device::out_of_bounds())
            {
                return;
            }
        }

        const label_t idx = device::idx();

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
                if constexpr (moment == index::rho())
                {
                    moments[moment] = devPtrs.ptr<moment>()[idx] + rho0<scalar_t>();
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
        {
            // Tiled static shared memory allocation
            __shared__ scalar_t sh_phi[block::nz() + 4][block::ny() + 4][block::nx() + 4];
            __shared__ scalar_t sh_nx[block::nz() + 2][block::ny() + 2][block::nx() + 2];
            __shared__ scalar_t sh_ny[block::nz() + 2][block::ny() + 2][block::nx() + 2];
            __shared__ scalar_t sh_nz[block::nz() + 2][block::ny() + 2][block::nx() + 2];

            // Global indexes
            const label_t x = threadIdx.x + block::nx() * blockIdx.x;
            const label_t y = threadIdx.y + block::ny() * blockIdx.y;
            const label_t z = threadIdx.z + block::nz() * blockIdx.z;
            const label_t x0 = blockIdx.x * block::nx();
            const label_t y0 = blockIdx.y * block::ny();
            const label_t z0 = blockIdx.z * block::nz();

            // Explicit domain guard: avoids OOB reads when filling halos
            auto in_domain = [&](label_t global_x, label_t global_y, label_t global_z) -> bool
            {
                return global_x < device::nx &&
                       global_y < device::ny &&
                       global_z < device::nz;
            };

            // Global linear index helper (kept inline for clarity and reuse)
            auto gidx = [&](label_t global_x, label_t global_y, label_t global_z) -> label_t
            {
                return device::idxGlobalFromIdx(global_x, global_y, global_z);
            };

            // Load phi interior + halo into shared memory (zero outside bulk)
            const scalar_t *const ptrRestrict phi = devPtrs.ptr<10>();
            for (label_t pz = threadIdx.z; pz < block::nz() + 4; pz += block::nz())
            {
                const label_t global_z = z0 + pz - 2;

                for (label_t py = threadIdx.y; py < block::ny() + 4; py += block::ny())
                {
                    const label_t global_y = y0 + py - 2;

                    for (label_t px = threadIdx.x; px < block::nx() + 4; px += block::nx())
                    {
                        const label_t global_x = x0 + px - 2;

                        sh_phi[pz][py][px] = in_domain(global_x, global_y, global_z) ? phi[gidx(global_x, global_y, global_z)] : static_cast<scalar_t>(0);
                    }
                }
            }

            __syncthreads();

            for (label_t iz = threadIdx.z; iz < block::nz() + 2; iz += block::nz())
            {
                const label_t global_z = z0 + iz - 1;
                const label_t pz = iz + 1;

                for (label_t iy = threadIdx.y; iy < block::ny() + 2; iy += block::ny())
                {
                    const label_t global_y = y0 + iy - 1;
                    const label_t py = iy + 1;

                    for (label_t ix = threadIdx.x; ix < block::nx() + 2; ix += block::nx())
                    {
                        const label_t global_x = x0 + ix - 1;
                        const label_t px = ix + 1;

                        // Outside of domain
                        const bool outside = !in_domain(global_x, global_y, global_z);

                        // Physical boundaries: normals suppressed to avoid spurious wall forcing
                        const bool isBoundary =
                            (global_x == 0) || (global_x == device::nx - 1) ||
                            (global_y == 0) || (global_y == device::ny - 1) ||
                            (global_z == 0) || (global_z == device::nz - 1);

                        if (outside || isBoundary)
                        {
                            sh_nx[iz][iy][ix] = static_cast<scalar_t>(0);
                            sh_ny[iz][iy][ix] = static_cast<scalar_t>(0);
                            sh_nz[iz][iy][ix] = static_cast<scalar_t>(0);
                        }
                        else
                        {
                            // Isotropic discrete gradient (D3Q27-consistent stencil)
                            const scalar_t sgx =
                                VelocitySet::w_1<scalar_t>() * (sh_phi[pz][py][px + 1] - sh_phi[pz][py][px - 1]) +
                                VelocitySet::w_2<scalar_t>() * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                                                sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                                                sh_phi[pz][py - 1][px + 1] - sh_phi[pz][py + 1][px - 1] +
                                                                sh_phi[pz - 1][py][px + 1] - sh_phi[pz + 1][py][px - 1]) +
                                VelocitySet::w_3<scalar_t>() * (sh_phi[pz + 1][py + 1][px + 1] - sh_phi[pz - 1][py - 1][px - 1] +
                                                                sh_phi[pz - 1][py + 1][px + 1] - sh_phi[pz + 1][py - 1][px - 1] +
                                                                sh_phi[pz + 1][py - 1][px + 1] - sh_phi[pz - 1][py + 1][px - 1] +
                                                                sh_phi[pz - 1][py - 1][px + 1] - sh_phi[pz + 1][py + 1][px - 1]);

                            const scalar_t sgy =
                                VelocitySet::w_1<scalar_t>() * (sh_phi[pz][py + 1][px] - sh_phi[pz][py - 1][px]) +
                                VelocitySet::w_2<scalar_t>() * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                                                sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                                                sh_phi[pz][py + 1][px - 1] - sh_phi[pz][py - 1][px + 1] +
                                                                sh_phi[pz - 1][py + 1][px] - sh_phi[pz + 1][py - 1][px]) +
                                VelocitySet::w_3<scalar_t>() * (sh_phi[pz + 1][py + 1][px + 1] - sh_phi[pz - 1][py - 1][px - 1] +
                                                                sh_phi[pz - 1][py + 1][px + 1] - sh_phi[pz + 1][py - 1][px - 1] +
                                                                sh_phi[pz - 1][py + 1][px - 1] - sh_phi[pz + 1][py - 1][px + 1] +
                                                                sh_phi[pz + 1][py + 1][px - 1] - sh_phi[pz - 1][py - 1][px + 1]);

                            const scalar_t sgz =
                                VelocitySet::w_1<scalar_t>() * (sh_phi[pz + 1][py][px] - sh_phi[pz - 1][py][px]) +
                                VelocitySet::w_2<scalar_t>() * (sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                                                sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                                                sh_phi[pz + 1][py][px - 1] - sh_phi[pz - 1][py][px + 1] +
                                                                sh_phi[pz + 1][py - 1][px] - sh_phi[pz - 1][py + 1][px]) +
                                VelocitySet::w_3<scalar_t>() * (sh_phi[pz + 1][py + 1][px + 1] - sh_phi[pz - 1][py - 1][px - 1] +
                                                                sh_phi[pz + 1][py - 1][px - 1] - sh_phi[pz - 1][py + 1][px + 1] +
                                                                sh_phi[pz + 1][py - 1][px + 1] - sh_phi[pz - 1][py + 1][px - 1] +
                                                                sh_phi[pz + 1][py + 1][px - 1] - sh_phi[pz - 1][py - 1][px + 1]);

                            // Convert lattice-gradient to physical gradient
                            const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
                            const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
                            const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

                            // Normalization with epsilon to avoid division by zero in bulk
                            const scalar_t ind2 = gx * gx + gy * gy + gz * gz;
                            const scalar_t ind = sqrtf(ind2);
                            const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

                            sh_nx[iz][iy][ix] = gx * invInd;
                            sh_ny[iz][iy][ix] = gy * invInd;
                            sh_nz[iz][iy][ix] = gz * invInd;
                        }
                    }
                }
            }

            __syncthreads();

            // Curvature and surface-tension terms are only well-defined away from physical boundaries
            const bool curvInterior =
                (x >= 1 && x <= device::nx - 2) &&
                (y >= 1 && y <= device::ny - 2) &&
                (z >= 1 && z <= device::nz - 2);

            if (curvInterior)
            {
                // Shared-memory indices for precomputed normals (1-cell halo)
                const label_t ix = threadIdx.x + 1;
                const label_t iy = threadIdx.y + 1;
                const label_t iz = threadIdx.z + 1;

                normx_ = sh_nx[iz][iy][ix];
                normy_ = sh_ny[iz][iy][ix];
                normz_ = sh_nz[iz][iy][ix];

                // Shared-memory indices for phi (2-cell halo, isotropic gradient)
                const label_t px = threadIdx.x + 2;
                const label_t py = threadIdx.y + 2;
                const label_t pz = threadIdx.z + 2;

                // Isotropic discrete gradient (D3Q27-consistent stencil)
                const scalar_t sgx =
                    VelocitySet::w_1<scalar_t>() * (sh_phi[pz][py][px + 1] - sh_phi[pz][py][px - 1]) +
                    VelocitySet::w_2<scalar_t>() * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                                    sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                                    sh_phi[pz][py - 1][px + 1] - sh_phi[pz][py + 1][px - 1] +
                                                    sh_phi[pz - 1][py][px + 1] - sh_phi[pz + 1][py][px - 1]) +
                    VelocitySet::w_3<scalar_t>() * (sh_phi[pz + 1][py + 1][px + 1] - sh_phi[pz - 1][py - 1][px - 1] +
                                                    sh_phi[pz - 1][py + 1][px + 1] - sh_phi[pz + 1][py - 1][px - 1] +
                                                    sh_phi[pz + 1][py - 1][px + 1] - sh_phi[pz - 1][py + 1][px - 1] +
                                                    sh_phi[pz - 1][py - 1][px + 1] - sh_phi[pz + 1][py + 1][px - 1]);

                const scalar_t sgy =
                    VelocitySet::w_1<scalar_t>() * (sh_phi[pz][py + 1][px] - sh_phi[pz][py - 1][px]) +
                    VelocitySet::w_2<scalar_t>() * (sh_phi[pz][py + 1][px + 1] - sh_phi[pz][py - 1][px - 1] +
                                                    sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                                    sh_phi[pz][py + 1][px - 1] - sh_phi[pz][py - 1][px + 1] +
                                                    sh_phi[pz - 1][py + 1][px] - sh_phi[pz + 1][py - 1][px]) +
                    VelocitySet::w_3<scalar_t>() * (sh_phi[pz + 1][py + 1][px + 1] - sh_phi[pz - 1][py - 1][px - 1] +
                                                    sh_phi[pz - 1][py + 1][px + 1] - sh_phi[pz + 1][py - 1][px - 1] +
                                                    sh_phi[pz - 1][py + 1][px - 1] - sh_phi[pz + 1][py - 1][px + 1] +
                                                    sh_phi[pz + 1][py + 1][px - 1] - sh_phi[pz - 1][py - 1][px + 1]);

                const scalar_t sgz =
                    VelocitySet::w_1<scalar_t>() * (sh_phi[pz + 1][py][px] - sh_phi[pz - 1][py][px]) +
                    VelocitySet::w_2<scalar_t>() * (sh_phi[pz + 1][py][px + 1] - sh_phi[pz - 1][py][px - 1] +
                                                    sh_phi[pz + 1][py + 1][px] - sh_phi[pz - 1][py - 1][px] +
                                                    sh_phi[pz + 1][py][px - 1] - sh_phi[pz - 1][py][px + 1] +
                                                    sh_phi[pz + 1][py - 1][px] - sh_phi[pz - 1][py + 1][px]) +
                    VelocitySet::w_3<scalar_t>() * (sh_phi[pz + 1][py + 1][px + 1] - sh_phi[pz - 1][py - 1][px - 1] +
                                                    sh_phi[pz + 1][py - 1][px - 1] - sh_phi[pz - 1][py + 1][px + 1] +
                                                    sh_phi[pz + 1][py - 1][px + 1] - sh_phi[pz - 1][py + 1][px - 1] +
                                                    sh_phi[pz + 1][py + 1][px - 1] - sh_phi[pz - 1][py - 1][px + 1]);

                // Convert lattice-gradient to physical gradient
                const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
                const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
                const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

                // Interface indicator
                ind_ = sqrtf(gx * gx + gy * gy + gz * gz);

                // Compute curvature
                const scalar_t scx =
                    VelocitySet::w_1<scalar_t>() * (sh_nx[iz][iy][ix + 1] - sh_nx[iz][iy][ix - 1]) +
                    VelocitySet::w_2<scalar_t>() * (sh_nx[iz][iy + 1][ix + 1] - sh_nx[iz][iy - 1][ix - 1] +
                                                    sh_nx[iz + 1][iy][ix + 1] - sh_nx[iz - 1][iy][ix - 1] +
                                                    sh_nx[iz][iy - 1][ix + 1] - sh_nx[iz][iy + 1][ix - 1] +
                                                    sh_nx[iz - 1][iy][ix + 1] - sh_nx[iz + 1][iy][ix - 1]) +
                    VelocitySet::w_3<scalar_t>() * (sh_nx[iz + 1][iy + 1][ix + 1] - sh_nx[iz - 1][iy - 1][ix - 1] +
                                                    sh_nx[iz - 1][iy - 1][ix + 1] - sh_nx[iz + 1][iy + 1][ix - 1] +
                                                    sh_nx[iz + 1][iy - 1][ix + 1] - sh_nx[iz - 1][iy + 1][ix - 1] +
                                                    sh_nx[iz - 1][iy + 1][ix + 1] - sh_nx[iz + 1][iy - 1][ix - 1]);

                const scalar_t scy =
                    VelocitySet::w_1<scalar_t>() * (sh_ny[iz][iy + 1][ix] - sh_ny[iz][iy - 1][ix]) +
                    VelocitySet::w_2<scalar_t>() * (sh_ny[iz][iy + 1][ix + 1] - sh_ny[iz][iy - 1][ix - 1] +
                                                    sh_ny[iz + 1][iy + 1][ix] - sh_ny[iz - 1][iy - 1][ix] +
                                                    sh_ny[iz][iy + 1][ix - 1] - sh_ny[iz][iy - 1][ix + 1] +
                                                    sh_ny[iz - 1][iy + 1][ix] - sh_ny[iz + 1][iy - 1][ix]) +
                    VelocitySet::w_3<scalar_t>() * (sh_ny[iz + 1][iy + 1][ix + 1] - sh_ny[iz - 1][iy - 1][ix - 1] +
                                                    sh_ny[iz - 1][iy + 1][ix + 1] - sh_ny[iz + 1][iy - 1][ix - 1] +
                                                    sh_ny[iz - 1][iy + 1][ix - 1] - sh_ny[iz + 1][iy - 1][ix + 1] +
                                                    sh_ny[iz + 1][iy + 1][ix - 1] - sh_ny[iz - 1][iy - 1][ix + 1]);

                const scalar_t scz =
                    VelocitySet::w_1<scalar_t>() * (sh_nz[iz + 1][iy][ix] - sh_nz[iz - 1][iy][ix]) +
                    VelocitySet::w_2<scalar_t>() * (sh_nz[iz + 1][iy][ix + 1] - sh_nz[iz - 1][iy][ix - 1] +
                                                    sh_nz[iz + 1][iy + 1][ix] - sh_nz[iz - 1][iy - 1][ix] +
                                                    sh_nz[iz + 1][iy][ix - 1] - sh_nz[iz - 1][iy][ix + 1] +
                                                    sh_nz[iz + 1][iy - 1][ix] - sh_nz[iz - 1][iy + 1][ix]) +
                    VelocitySet::w_3<scalar_t>() * (sh_nz[iz + 1][iy + 1][ix + 1] - sh_nz[iz - 1][iy - 1][ix - 1] +
                                                    sh_nz[iz + 1][iy - 1][ix - 1] - sh_nz[iz - 1][iy + 1][ix + 1] +
                                                    sh_nz[iz + 1][iy - 1][ix + 1] - sh_nz[iz - 1][iy + 1][ix - 1] +
                                                    sh_nz[iz + 1][iy + 1][ix - 1] - sh_nz[iz - 1][iy - 1][ix + 1]);

                const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);
                const scalar_t stCurv = -device::sigma * curvature * ind_;

                // Assemble surface tension forces
                Fsx = stCurv * normx_;
                Fsy = stCurv * normy_;
                Fsz = stCurv * normz_;
            }
            else
            {
                // Outside curvature-valid region: force and geometry suppressed
                Fsx = static_cast<scalar_t>(0);
                Fsy = static_cast<scalar_t>(0);
                Fsz = static_cast<scalar_t>(0);
                normx_ = static_cast<scalar_t>(0);
                normy_ = static_cast<scalar_t>(0);
                normz_ = static_cast<scalar_t>(0);
                ind_ = static_cast<scalar_t>(0);
            }
        }

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments, Fsx, Fsy, Fsz);

        // Calculate post collision populations
        thread::array<scalar_t, VelocitySet::Q()> pop;
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g;
        VelocitySet::reconstruct(pop, moments);
        PhaseVelocitySet::reconstruct(pop_g, moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term) on g-populations
        PhaseVelocitySet::sharpen(pop_g, phi_, normx_, normy_, normz_);

        // Coalesced write to global memory
        moments[m_i<0>()] = moments[m_i<0>()] - rho0<scalar_t>();
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });

        // Save the hydro populations to the block halo
        HydroHalo::save(pop, ghostHydro);

        // Save the phase populations to the block halo
        PhaseHalo::save(pop_g, ghostPhase);
    }
}

#endif