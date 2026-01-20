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
    Main kernels for the multiphase moment representation with the D3Q19 velocity set

Namespace
    LBM

SourceFiles
    multiphaseD3Q19global.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MULTIPHASED3Q19_CUH
#define __MBLBM_MULTIPHASED3Q19_CUH

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

#include <assert.h>

namespace LBM
{

    using VelocitySet = D3Q19;
    using PhaseVelocitySet = D3Q7;
    using Collision = secondOrder;

    // Aliases use the standard halo methods
    using HydroHalo = device::halo<VelocitySet, multiphase::periodicX(), multiphase::periodicY()>;
    using PhaseHalo = device::halo<PhaseVelocitySet, multiphase::periodicX(), multiphase::periodicY()>;

    __device__ __host__ [[nodiscard]] inline consteval scalar_t rho_oil() noexcept { return static_cast<scalar_t>(0.852); }
    __device__ __host__ [[nodiscard]] inline consteval scalar_t rho_water() noexcept { return static_cast<scalar_t>(1); }
    __device__ __host__ [[nodiscard]] inline consteval scalar_t delta_rho() noexcept { return rho_oil() - rho_water(); }

    __device__ __host__ [[nodiscard]] inline consteval label_t smem_alloc_size() noexcept { return block::sharedMemoryBufferSize<VelocitySet, 11>(sizeof(scalar_t)); }

    __device__ __host__ [[nodiscard]] inline consteval bool out_of_bounds_check() noexcept
    {
#ifdef OOB_CHECK
        return true;
#else
        return false;
#endif
    }

    __host__ [[nodiscard]] inline consteval label_t MIN_BLOCKS_PER_MP() noexcept { return 2; }
#define launchBoundsD3Q19 __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP())

    /**
     * @brief Performs the streaming step of the lattice Boltzmann method using the multiphase moment representation (D3Q19 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
     * @param fBlockHalo Object containing pointers to the block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q19 __global__ void multiphaseStream(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const scalar_t *const ptrRestrict normx,
        const scalar_t *const ptrRestrict normy,
        const scalar_t *const ptrRestrict normz,
        const device::ptrCollection<6, const scalar_t> ghostHydro,
        const device::ptrCollection<6, const scalar_t> ghostPhase)
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

        const scalar_t normx_ = normx[idx];
        const scalar_t normy_ = normy[idx];
        const scalar_t normz_ = normz[idx];

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
                moments[moment] = shared_buffer[ID];
            });

        __syncthreads();

        // Reconstruct the populations from the moments
        thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g = PhaseVelocitySet::reconstruct(moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term)
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

        // if constexpr (std::is_same<ultiphase::BoundaryConditions, multiphaseJet>::value)
        {
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
                    multiphase::BoundaryConditions::calculate_moments<VelocitySet, PhaseVelocitySet>(pop, moments, boundaryNormal, shared_buffer);
                }
            }
        }

        // Coalesced write to global memory
        device::constexpr_for<0, NUMBER_MOMENTS<true>()>(
            [&](const auto moment)
            {
                devPtrs.ptr<moment>()[idx] = moments[moment];
            });
    }

    /**
     * @brief Compute phase field interface normals and indicator
     * @param phi Pointer to phase field scalar
     * @param normx Pointer to x-component of the unit interface normal
     * @param normy Pointer to y-component of the unit interface normal
     * @param normz Pointer to z-component of the unit interface normal
     * @param ind Pointer to interface indicator
     **/
    launchBoundsD3Q19 __global__ void computeNormals(
        const scalar_t *const ptrRestrict phi,
        scalar_t *const ptrRestrict normx,
        scalar_t *const ptrRestrict normy,
        scalar_t *const ptrRestrict normz,
        scalar_t *const ptrRestrict ind)
    {
        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        if (x == 0 || x == device::nx - 1 ||
            y == 0 || y == device::ny - 1 ||
            z == 0 || z == device::nz - 1)
        {
            return;
        }

        const label_t idx = device::idx();

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

        const scalar_t sgx =
            VelocitySet::w_1<scalar_t>() * (phi[i_xp] - phi[i_xm]) +
            VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                            phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                            phi_xp1_ym1_z - phi_xm1_yp1_z +
                                            phi_xp1_y_zm1 - phi_xm1_y_zp1);

        const scalar_t sgy =
            VelocitySet::w_1<scalar_t>() * (phi[i_yp] - phi[i_ym]) +
            VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                            phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                            phi_xm1_yp1_z - phi_xp1_ym1_z +
                                            phi_x_yp1_zm1 - phi_x_ym1_zp1);

        const scalar_t sgz =
            VelocitySet::w_1<scalar_t>() * (phi[i_zp] - phi[i_zm]) +
            VelocitySet::w_2<scalar_t>() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                            phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                            phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                            phi_x_ym1_zp1 - phi_x_yp1_zm1);

        const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
        const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
        const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

        const scalar_t ind_ = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind_ + static_cast<scalar_t>(1e-9));

        const scalar_t normx_ = gx * invInd;
        const scalar_t normy_ = gy * invInd;
        const scalar_t normz_ = gz * invInd;

        ind[idx] = ind_;
        normx[idx] = normx_;
        normy[idx] = normy_;
        normz[idx] = normz_;
    }

    /**
     * @brief Performs the collision step of the lattice Boltzmann method using the multiphase moment representation (D3Q19 hydrodynamics + D3Q7 phase field)
     * @param devPtrs Collection of 11 pointers to device arrays on the GPU
     * @param fBlockHalo Object containing pointers to the block halo faces used to exchange the hydrodynamic population densities
     * @param gBlockHalo Object containing pointers to the block halo faces used to exchange the phase population densities
     * @note Currently only immutable halos are used due to kernel split
     **/
    launchBoundsD3Q19 __global__ void multiphaseCollide(
        const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs,
        const scalar_t *const ptrRestrict normx,
        const scalar_t *const ptrRestrict normy,
        const scalar_t *const ptrRestrict normz,
        const scalar_t *const ptrRestrict ind,
        const scalar_t *const ptrRestrict phi,
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

        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        const bool isInterior =
            (x > 0) & (x < device::nx - 1) &
            (y > 0) & (y < device::ny - 1) &
            (z > 0) & (z < device::nz - 1);

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
                moments[moment] = devPtrs.ptr<moment>()[idx];
            });

        const scalar_t normx_ = normx[idx];
        const scalar_t normy_ = normy[idx];
        const scalar_t normz_ = normz[idx];

        scalar_t Fsx = static_cast<scalar_t>(0);
        scalar_t Fsy = static_cast<scalar_t>(0);
        scalar_t Fsz = static_cast<scalar_t>(0);

        scalar_t Fpx = static_cast<scalar_t>(0);
        scalar_t Fpy = static_cast<scalar_t>(0);
        scalar_t Fpz = static_cast<scalar_t>(0);

        scalar_t Fnx = static_cast<scalar_t>(0);
        scalar_t Fny = static_cast<scalar_t>(0);
        scalar_t Fnz = static_cast<scalar_t>(0);

        scalar_t Fx = static_cast<scalar_t>(0);
        scalar_t Fy = static_cast<scalar_t>(0);
        scalar_t Fz = static_cast<scalar_t>(0);

        scalar_t rho_ = static_cast<scalar_t>(1);

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

            const label_t xp1_yp1_z = i_xp + dyp;
            const label_t xp1_ym1_z = i_xp - dym;
            const label_t xm1_yp1_z = i_xm + dyp;
            const label_t xm1_ym1_z = i_xm - dym;
            const label_t xp1_y_zp1 = i_xp + dzp;
            const label_t xp1_y_zm1 = i_xp - dzm;
            const label_t xm1_y_zp1 = i_xm + dzp;
            const label_t xm1_y_zm1 = i_xm - dzm;
            const label_t x_yp1_zp1 = i_yp + dzp;
            const label_t x_yp1_zm1 = i_yp - dzm;
            const label_t x_ym1_zp1 = i_ym + dzp;
            const label_t x_ym1_zm1 = i_ym - dzm;

            const scalar_t scx = VelocitySet::w_1<scalar_t>() * (normx[i_xp] - normx[i_xm]) +
                                 VelocitySet::w_2<scalar_t>() * (normx[xp1_yp1_z] - normx[xm1_ym1_z] +
                                                                 normx[xp1_y_zp1] - normx[xm1_y_zm1] +
                                                                 normx[xp1_ym1_z] - normx[xm1_yp1_z] +
                                                                 normx[xp1_y_zm1] - normx[xm1_y_zp1]);

            const scalar_t scy = VelocitySet::w_1<scalar_t>() * (normy[i_yp] - normy[i_ym]) +
                                 VelocitySet::w_2<scalar_t>() * (normy[xp1_yp1_z] - normy[xm1_ym1_z] +
                                                                 normy[x_yp1_zp1] - normy[x_ym1_zm1] +
                                                                 normy[xm1_yp1_z] - normy[xp1_ym1_z] +
                                                                 normy[x_yp1_zm1] - normy[x_ym1_zp1]);

            const scalar_t scz = VelocitySet::w_1<scalar_t>() * (normz[i_zp] - normz[i_zm]) +
                                 VelocitySet::w_2<scalar_t>() * (normz[xp1_y_zp1] - normz[xm1_y_zm1] +
                                                                 normz[x_yp1_zp1] - normz[x_ym1_zm1] +
                                                                 normz[xm1_y_zp1] - normz[xp1_y_zm1] +
                                                                 normz[x_ym1_zp1] - normz[x_yp1_zm1]);

            const scalar_t curvature = velocitySet::as2<scalar_t>() * (scx + scy + scz);

            const scalar_t stCurv = -device::sigma * curvature * ind[idx];

            Fsx = stCurv * normx[idx];
            Fsy = stCurv * normy[idx];
            Fsz = stCurv * normz[idx];

            // Build stress tensor
            const scalar_t pxx = moments[m_i<4>()] - moments[m_i<1>()] * moments[m_i<1>()];
            const scalar_t pxy = moments[m_i<5>()] - moments[m_i<1>()] * moments[m_i<2>()];
            const scalar_t pxz = moments[m_i<6>()] - moments[m_i<1>()] * moments[m_i<3>()];
            const scalar_t pyy = moments[m_i<7>()] - moments[m_i<2>()] * moments[m_i<2>()];
            const scalar_t pyz = moments[m_i<8>()] - moments[m_i<2>()] * moments[m_i<3>()];
            const scalar_t pzz = moments[m_i<9>()] - moments[m_i<3>()] * moments[m_i<3>()];

            // Load the neighbor phi values
            const scalar_t phi_xp1_yp1_z = phi[xp1_yp1_z];
            const scalar_t phi_xp1_ym1_z = phi[xp1_ym1_z];
            const scalar_t phi_xm1_yp1_z = phi[xm1_yp1_z];
            const scalar_t phi_xm1_ym1_z = phi[xm1_ym1_z];
            const scalar_t phi_xp1_y_zp1 = phi[xp1_y_zp1];
            const scalar_t phi_xp1_y_zm1 = phi[xp1_y_zm1];
            const scalar_t phi_xm1_y_zp1 = phi[xm1_y_zp1];
            const scalar_t phi_xm1_y_zm1 = phi[xm1_y_zm1];
            const scalar_t phi_x_yp1_zp1 = phi[x_yp1_zp1];
            const scalar_t phi_x_yp1_zm1 = phi[x_yp1_zm1];
            const scalar_t phi_x_ym1_zp1 = phi[x_ym1_zp1];
            const scalar_t phi_x_ym1_zm1 = phi[x_ym1_zm1];

            // Compute gradients
            const scalar_t sgx =
                VelocitySet::w_1<scalar_t>() * (phi[i_xp] - phi[i_xm]) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                phi_xp1_ym1_z - phi_xm1_yp1_z +
                                                phi_xp1_y_zm1 - phi_xm1_y_zp1);

            const scalar_t sgy =
                VelocitySet::w_1<scalar_t>() * (phi[i_yp] - phi[i_ym]) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                phi_xm1_yp1_z - phi_xp1_ym1_z +
                                                phi_x_yp1_zm1 - phi_x_ym1_zp1);

            const scalar_t sgz =
                VelocitySet::w_1<scalar_t>() * (phi[i_zp] - phi[i_zm]) +
                VelocitySet::w_2<scalar_t>() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                                phi_x_ym1_zp1 - phi_x_yp1_zm1);

            rho_ = static_cast<scalar_t>(1) + delta_rho() * moments[m_i<10>()];

            const scalar_t drhox = delta_rho() * velocitySet::as2<scalar_t>() * sgx;
            const scalar_t drhoy = delta_rho() * velocitySet::as2<scalar_t>() * sgy;
            const scalar_t drhoz = delta_rho() * velocitySet::as2<scalar_t>() * sgz;

            // Compute pressure force
            Fpx = -moments[m_i<0>()] * (velocitySet::cs2<scalar_t>() * drhox);
            Fpy = -moments[m_i<0>()] * (velocitySet::cs2<scalar_t>() * drhoy);
            Fpz = -moments[m_i<0>()] * (velocitySet::cs2<scalar_t>() * drhoz);

            // Compute viscous correction force
            Fnx = -device::tt_omegaVar * (pxx * drhox + pxy * drhoy + pxz * drhoz);
            Fny = -device::tt_omegaVar * (pxy * drhox + pyy * drhoy + pyz * drhoz);
            Fnz = -device::tt_omegaVar * (pxz * drhox + pyz * drhoy + pzz * drhoz);

            // Build force density
            Fx = Fsx + Fpx + Fnx;
            Fy = Fsy + Fpy + Fny;
            Fz = Fsz + Fpz + Fnz;
        }

        // Perform velocity half-step
        moments[m_i<1>()] += 0.5 * Fx / rho_;
        moments[m_i<2>()] += 0.5 * Fy / rho_;
        moments[m_i<3>()] += 0.5 * Fz / rho_;

        // Scale the moments correctly
        velocitySet::scale(moments);

        // Collide
        Collision::collide(moments, Fx, Fy, Fz, rho_);

        // Calculate post collision populations
        thread::array<scalar_t, VelocitySet::Q()> pop;
        thread::array<scalar_t, PhaseVelocitySet::Q()> pop_g;
        VelocitySet::reconstruct(pop, moments);
        PhaseVelocitySet::reconstruct(pop_g, moments);

        // Gather current phase field state
        const scalar_t phi_ = moments[m_i<10>()];

        // Add sharpening (compressive term)
        PhaseVelocitySet::sharpen(pop_g, phi_, normx_, normy_, normz_);

        // Coalesced write to global memory
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