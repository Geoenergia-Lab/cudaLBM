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
    template <const int dx, const int dy, const int dz, class PhaseHalo, const bool useScalarHalo = true, class HydroShared>
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

        if constexpr (useScalarHalo)
        {
            return PhaseHalo::template pull_scalar<dx, dy, dz>(phi, phiBuffer, Tx, Bx, point);
        }
        else
        {
            (void)phiBuffer;
            return PhaseHalo::template pull_scalar_local<dx, dy, dz>(phi, Tx, Bx, point);
        }
    }

    /**
     * @brief Load a neighboring phase value directly from global/halo storage (no shared-memory fast path)
     * @tparam dx Neighbor offset in x-direction
     * @tparam dy Neighbor offset in y-direction
     * @tparam dz Neighbor offset in z-direction
     * @tparam PhaseHalo Halo type used to fetch off-block neighbors
     **/
    template <const int dx, const int dy, const int dz, class PhaseHalo, const bool useScalarHalo = true>
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

        if constexpr (useScalarHalo)
        {
            return PhaseHalo::template pull_scalar<dx, dy, dz>(phi, phiBuffer, Tx, Bx, point);
        }
        else
        {
            (void)phiBuffer;
            return PhaseHalo::template pull_scalar_local<dx, dy, dz>(phi, Tx, Bx, point);
        }
    }

    /**
     * @brief Runtime scalar neighbor fetch (offsets in [-2,2]) using local field and optional scalar halo faces
     **/
    template <class PhaseHalo, const bool useScalarHalo = true>
    __device__ [[nodiscard]] inline scalar_t load_phase_neighbor_direct_runtime(
        const scalar_t *const ptrRestrict phi,
        const device::ptrCollection<6, const scalar_t> &phiBuffer,
        const thread::coordinate &Tx,
        const block::coordinate &Bx,
        const device::pointCoordinate &point,
        const int dx,
        const int dy,
        const int dz) noexcept
    {
        constexpr int scalarHaloDepth = 2;

        if ((dx == 0) && (dy == 0) && (dz == 0))
        {
            return __ldg(&(phi[device::idx(Tx, Bx)]));
        }

        if ((dx < -scalarHaloDepth) || (dx > scalarHaloDepth) ||
            (dy < -scalarHaloDepth) || (dy > scalarHaloDepth) ||
            (dz < -scalarHaloDepth) || (dz > scalarHaloDepth))
        {
            return static_cast<scalar_t>(0);
        }

        const int nX = static_cast<int>(block::n<axis::X>());
        const int nY = static_cast<int>(block::n<axis::Y>());
        const int nZ = static_cast<int>(block::n<axis::Z>());

        const int shiftedX = static_cast<int>(Tx.value<axis::X>()) + dx;
        const int shiftedY = static_cast<int>(Tx.value<axis::Y>()) + dy;
        const int shiftedZ = static_cast<int>(Tx.value<axis::Z>()) + dz;

        if constexpr (!PhaseHalo::periodicX())
        {
            const int gx = static_cast<int>(point.value<axis::X>()) + dx;
            if ((gx < 0) || (gx >= static_cast<int>(device::n<axis::X>())))
            {
                return static_cast<scalar_t>(0);
            }
        }

        if constexpr (!PhaseHalo::periodicY())
        {
            const int gy = static_cast<int>(point.value<axis::Y>()) + dy;
            if ((gy < 0) || (gy >= static_cast<int>(device::n<axis::Y>())))
            {
                return static_cast<scalar_t>(0);
            }
        }

        if constexpr (!PhaseHalo::periodicZ())
        {
            const int gz = static_cast<int>(point.value<axis::Z>()) + dz;
            if ((gz < 0) || (gz >= static_cast<int>(device::n<axis::Z>())))
            {
                return static_cast<scalar_t>(0);
            }
        }

        const bool crossMinusX = shiftedX < 0;
        const bool crossMinusY = shiftedY < 0;
        const bool crossMinusZ = shiftedZ < 0;

        const bool crossPlusX = shiftedX >= nX;
        const bool crossPlusY = shiftedY >= nY;
        const bool crossPlusZ = shiftedZ >= nZ;

        const device::label_t tx = static_cast<device::label_t>(crossMinusX ? (shiftedX + nX) : (crossPlusX ? (shiftedX - nX) : shiftedX));
        const device::label_t ty = static_cast<device::label_t>(crossMinusY ? (shiftedY + nY) : (crossPlusY ? (shiftedY - nY) : shiftedY));
        const device::label_t tz = static_cast<device::label_t>(crossMinusZ ? (shiftedZ + nZ) : (crossPlusZ ? (shiftedZ - nZ) : shiftedZ));

        const bool haloX = crossMinusX || crossPlusX;
        const bool haloY = crossMinusY || crossPlusY;
        const int bxShift = crossMinusX ? -1 : (crossPlusX ? +1 : 0);
        const int byShift = crossMinusY ? -1 : (crossPlusY ? +1 : 0);
        const int bzShift = crossMinusZ ? -1 : (crossPlusZ ? +1 : 0);

        const int nBlockX = static_cast<int>(device::NUM_BLOCK<axis::X>());
        const int nBlockY = static_cast<int>(device::NUM_BLOCK<axis::Y>());
        const int nBlockZ = static_cast<int>(device::NUM_BLOCK<axis::Z>());

        const device::label_t bx = static_cast<device::label_t>((static_cast<int>(Bx.value<axis::X>()) + nBlockX + bxShift) % nBlockX);
        const device::label_t by = static_cast<device::label_t>((static_cast<int>(Bx.value<axis::Y>()) + nBlockY + byShift) % nBlockY);
        const device::label_t bz = static_cast<device::label_t>((static_cast<int>(Bx.value<axis::Z>()) + nBlockZ + bzShift) % nBlockZ);

        int pointShiftX = static_cast<int>(point.value<axis::X>()) + dx;
        int pointShiftY = static_cast<int>(point.value<axis::Y>()) + dy;
        int pointShiftZ = static_cast<int>(point.value<axis::Z>()) + dz;

        if constexpr (PhaseHalo::periodicX())
        {
            if (pointShiftX < 0)
            {
                pointShiftX += static_cast<int>(device::n<axis::X>());
            }
            else if (pointShiftX >= static_cast<int>(device::n<axis::X>()))
            {
                pointShiftX -= static_cast<int>(device::n<axis::X>());
            }
        }

        if constexpr (PhaseHalo::periodicY())
        {
            if (pointShiftY < 0)
            {
                pointShiftY += static_cast<int>(device::n<axis::Y>());
            }
            else if (pointShiftY >= static_cast<int>(device::n<axis::Y>()))
            {
                pointShiftY -= static_cast<int>(device::n<axis::Y>());
            }
        }

        if constexpr (PhaseHalo::periodicZ())
        {
            if (pointShiftZ < 0)
            {
                pointShiftZ += static_cast<int>(device::n<axis::Z>());
            }
            else if (pointShiftZ >= static_cast<int>(device::n<axis::Z>()))
            {
                pointShiftZ -= static_cast<int>(device::n<axis::Z>());
            }
        }

#ifndef FORCE_MULTI_GPU_SCALAR_HALO_TEST
        const int localSizeX = static_cast<int>(block::n<axis::X>() * device::NUM_BLOCK<axis::X>());
        const int localSizeY = static_cast<int>(block::n<axis::Y>() * device::NUM_BLOCK<axis::Y>());
        const int localSizeZ = static_cast<int>(block::n<axis::Z>() * device::NUM_BLOCK<axis::Z>());

        const int localStartX = static_cast<int>(block::n<axis::X>() * device::BLOCK_OFFSET_X);
        const int localStartY = static_cast<int>(block::n<axis::Y>() * device::BLOCK_OFFSET_Y);
        const int localStartZ = static_cast<int>(block::n<axis::Z>() * device::BLOCK_OFFSET_Z);

        const bool withinLocalSubdomain =
            (pointShiftX >= localStartX) && (pointShiftX < (localStartX + localSizeX)) &&
            (pointShiftY >= localStartY) && (pointShiftY < (localStartY + localSizeY)) &&
            (pointShiftZ >= localStartZ) && (pointShiftZ < (localStartZ + localSizeZ));

        if (withinLocalSubdomain)
        {
            return __ldg(&(phi[device::idx(tx, ty, tz, bx, by, bz)]));
        }
#endif

        if constexpr (useScalarHalo)
        {
            if (haloX)
            {
                const device::label_t layer = static_cast<device::label_t>(crossMinusX ? (-shiftedX - 1) : (shiftedX - nX));
                const device::label_t faceIdx =
                    (layer == static_cast<device::label_t>(0))
                        ? device::idxPop<axis::X, 0, scalarHaloDepth>(ty, tz, bx, by, bz)
                        : device::idxPop<axis::X, 1, scalarHaloDepth>(ty, tz, bx, by, bz);

                if (crossMinusX)
                {
                    return __ldg(&(phiBuffer.ptr<1>()[faceIdx]));
                }

                return __ldg(&(phiBuffer.ptr<0>()[faceIdx]));
            }

            if (haloY)
            {
                const device::label_t layer = static_cast<device::label_t>(crossMinusY ? (-shiftedY - 1) : (shiftedY - nY));
                const device::label_t faceIdx =
                    (layer == static_cast<device::label_t>(0))
                        ? device::idxPop<axis::Y, 0, scalarHaloDepth>(tx, tz, bx, by, bz)
                        : device::idxPop<axis::Y, 1, scalarHaloDepth>(tx, tz, bx, by, bz);

                if (crossMinusY)
                {
                    return __ldg(&(phiBuffer.ptr<3>()[faceIdx]));
                }

                return __ldg(&(phiBuffer.ptr<2>()[faceIdx]));
            }

            const device::label_t layer = static_cast<device::label_t>(crossMinusZ ? (-shiftedZ - 1) : (shiftedZ - nZ));
            const device::label_t faceIdx =
                (layer == static_cast<device::label_t>(0))
                    ? device::idxPop<axis::Z, 0, scalarHaloDepth>(tx, ty, bx, by, bz)
                    : device::idxPop<axis::Z, 1, scalarHaloDepth>(tx, ty, bx, by, bz);

            if (crossMinusZ)
            {
                return __ldg(&(phiBuffer.ptr<5>()[faceIdx]));
            }

            return __ldg(&(phiBuffer.ptr<4>()[faceIdx]));
        }
        else
        {
            (void)phiBuffer;
            return __ldg(&(phi[device::idx(tx, ty, tz, bx, by, bz)]));
        }
    }

    /**
     * @brief Isotropic first derivative along axis @p alpha for D3Q19/D3Q27 stencils
     * @param[in] sample Callable sample(dx,dy,dz) returning field value at offset
     **/
    template <const axis::type alpha, class VelocitySet, class Sampler>
    __device__ [[nodiscard]] inline scalar_t stencil_derivative(const Sampler &sample) noexcept
    {
        axis::assertions::validate<alpha, axis::NOT_NULL>();

        if constexpr (VelocitySet::Q() == 19)
        {
            if constexpr (alpha == axis::X)
            {
                return VelocitySet::template w_1<scalar_t>() * (sample(+1, 0, 0) - sample(-1, 0, 0)) +
                       VelocitySet::template w_2<scalar_t>() * (sample(+1, +1, 0) - sample(-1, -1, 0) +
                                                                sample(+1, 0, +1) - sample(-1, 0, -1) +
                                                                sample(+1, -1, 0) - sample(-1, +1, 0) +
                                                                sample(+1, 0, -1) - sample(-1, 0, +1));
            }

            if constexpr (alpha == axis::Y)
            {
                return VelocitySet::template w_1<scalar_t>() * (sample(0, +1, 0) - sample(0, -1, 0)) +
                       VelocitySet::template w_2<scalar_t>() * (sample(+1, +1, 0) - sample(-1, -1, 0) +
                                                                sample(0, +1, +1) - sample(0, -1, -1) +
                                                                sample(-1, +1, 0) - sample(+1, -1, 0) +
                                                                sample(0, +1, -1) - sample(0, -1, +1));
            }

            return VelocitySet::template w_1<scalar_t>() * (sample(0, 0, +1) - sample(0, 0, -1)) +
                   VelocitySet::template w_2<scalar_t>() * (sample(+1, 0, +1) - sample(-1, 0, -1) +
                                                            sample(0, +1, +1) - sample(0, -1, -1) +
                                                            sample(-1, 0, +1) - sample(+1, 0, -1) +
                                                            sample(0, -1, +1) - sample(0, +1, -1));
        }

        if constexpr (VelocitySet::Q() == 27)
        {
            if constexpr (alpha == axis::X)
            {
                return VelocitySet::template w_1<scalar_t>() * (sample(+1, 0, 0) - sample(-1, 0, 0)) +
                       VelocitySet::template w_2<scalar_t>() * (sample(+1, +1, 0) - sample(-1, -1, 0) +
                                                                sample(+1, 0, +1) - sample(-1, 0, -1) +
                                                                sample(+1, -1, 0) - sample(-1, +1, 0) +
                                                                sample(+1, 0, -1) - sample(-1, 0, +1)) +
                       VelocitySet::template w_3<scalar_t>() * (sample(+1, +1, +1) - sample(-1, -1, -1) +
                                                                sample(+1, +1, -1) - sample(-1, -1, +1) +
                                                                sample(+1, -1, +1) - sample(-1, +1, -1) +
                                                                sample(+1, -1, -1) - sample(-1, +1, +1));
            }

            if constexpr (alpha == axis::Y)
            {
                return VelocitySet::template w_1<scalar_t>() * (sample(0, +1, 0) - sample(0, -1, 0)) +
                       VelocitySet::template w_2<scalar_t>() * (sample(+1, +1, 0) - sample(-1, -1, 0) +
                                                                sample(0, +1, +1) - sample(0, -1, -1) +
                                                                sample(-1, +1, 0) - sample(+1, -1, 0) +
                                                                sample(0, +1, -1) - sample(0, -1, +1)) +
                       VelocitySet::template w_3<scalar_t>() * (sample(+1, +1, +1) - sample(-1, -1, -1) +
                                                                sample(+1, +1, -1) - sample(-1, -1, +1) +
                                                                sample(-1, +1, -1) - sample(+1, -1, +1) +
                                                                sample(-1, +1, +1) - sample(+1, -1, -1));
            }

            return VelocitySet::template w_1<scalar_t>() * (sample(0, 0, +1) - sample(0, 0, -1)) +
                   VelocitySet::template w_2<scalar_t>() * (sample(+1, 0, +1) - sample(-1, 0, -1) +
                                                            sample(0, +1, +1) - sample(0, -1, -1) +
                                                            sample(-1, 0, +1) - sample(+1, 0, -1) +
                                                            sample(0, -1, +1) - sample(0, +1, -1)) +
                   VelocitySet::template w_3<scalar_t>() * (sample(+1, +1, +1) - sample(-1, -1, -1) +
                                                            sample(-1, -1, +1) - sample(+1, +1, -1) +
                                                            sample(+1, -1, +1) - sample(-1, +1, -1) +
                                                            sample(-1, +1, +1) - sample(+1, -1, -1));
        }

        scalar_t derivative = static_cast<scalar_t>(0);
        device::constexpr_for<1, VelocitySet::Q()>(
            [&](const auto q)
            {
                constexpr device::label_t qi = q();
                constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];
                const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());
                const scalar_t value = sample(cx, cy, cz);

                if constexpr (alpha == axis::X)
                {
                    derivative += wq * static_cast<scalar_t>(cx) * value;
                }
                else if constexpr (alpha == axis::Y)
                {
                    derivative += wq * static_cast<scalar_t>(cy) * value;
                }
                else
                {
                    derivative += wq * static_cast<scalar_t>(cz) * value;
                }
            });

        return derivative;
    }

    /**
     * @brief Compute normalized interface normal from sampled phi neighbors
     **/
    template <class VelocitySet, class Sampler>
    __device__ inline void compute_phase_normal_from_samples(
        const Sampler &samplePhi,
        scalar_t &normx,
        scalar_t &normy,
        scalar_t &normz,
        scalar_t &ind) noexcept
    {
        const scalar_t sgx = stencil_derivative<axis::X, VelocitySet>(samplePhi);
        const scalar_t sgy = stencil_derivative<axis::Y, VelocitySet>(samplePhi);
        const scalar_t sgz = stencil_derivative<axis::Z, VelocitySet>(samplePhi);

        const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
        const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
        const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

        ind = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

        normx = gx * invInd;
        normy = gy * invInd;
        normz = gz * invInd;
    }

    /**
     * @brief Compute curvature (divergence of normals) from sampled normal components
     **/
    template <class VelocitySet, class SampleNx, class SampleNy, class SampleNz>
    __device__ [[nodiscard]] inline scalar_t compute_phase_curvature_from_samples(
        const SampleNx &sampleNx,
        const SampleNy &sampleNy,
        const SampleNz &sampleNz) noexcept
    {
        const scalar_t scx = stencil_derivative<axis::X, VelocitySet>(sampleNx);
        const scalar_t scy = stencil_derivative<axis::Y, VelocitySet>(sampleNy);
        const scalar_t scz = stencil_derivative<axis::Z, VelocitySet>(sampleNz);

        return velocitySet::as2<scalar_t>() * (scx + scy + scz);
    }

    /**
     * @brief Flatten 3D tile coordinates into a linear shared-memory index
     **/
    template <device::label_t tileNx, device::label_t tileNy>
    __device__ [[nodiscard]] inline device::label_t tile_idx(
        const device::label_t sx,
        const device::label_t sy,
        const device::label_t sz) noexcept
    {
        return (sz * tileNy + sy) * tileNx + sx;
    }

    /**
     * @brief Compute normalized interface normal from an explicit phi shared-memory tile center
     **/
    template <class VelocitySet, device::label_t tileNx, device::label_t tileNy>
    __device__ inline void compute_phase_normal_from_tile(
        const scalar_t *const ptrRestrict phiTile,
        const device::label_t sx,
        const device::label_t sy,
        const device::label_t sz,
        scalar_t &normx,
        scalar_t &normy,
        scalar_t &normz,
        scalar_t &ind) noexcept
    {
        const device::label_t sxp = sx + static_cast<device::label_t>(1);
        const device::label_t sxm = sx - static_cast<device::label_t>(1);
        const device::label_t syp = sy + static_cast<device::label_t>(1);
        const device::label_t sym = sy - static_cast<device::label_t>(1);
        const device::label_t szp = sz + static_cast<device::label_t>(1);
        const device::label_t szm = sz - static_cast<device::label_t>(1);

        scalar_t sgx = static_cast<scalar_t>(0);
        scalar_t sgy = static_cast<scalar_t>(0);
        scalar_t sgz = static_cast<scalar_t>(0);

        if constexpr (VelocitySet::Q() == 19)
        {
            const scalar_t phiXpYpZ = phiTile[tile_idx<tileNx, tileNy>(sxp, syp, sz)];
            const scalar_t phiXpYmZ = phiTile[tile_idx<tileNx, tileNy>(sxp, sym, sz)];
            const scalar_t phiXmYpZ = phiTile[tile_idx<tileNx, tileNy>(sxm, syp, sz)];
            const scalar_t phiXmYmZ = phiTile[tile_idx<tileNx, tileNy>(sxm, sym, sz)];

            const scalar_t phiXpYZp = phiTile[tile_idx<tileNx, tileNy>(sxp, sy, szp)];
            const scalar_t phiXpYZm = phiTile[tile_idx<tileNx, tileNy>(sxp, sy, szm)];
            const scalar_t phiXmYZp = phiTile[tile_idx<tileNx, tileNy>(sxm, sy, szp)];
            const scalar_t phiXmYZm = phiTile[tile_idx<tileNx, tileNy>(sxm, sy, szm)];

            const scalar_t phiXYpZp = phiTile[tile_idx<tileNx, tileNy>(sx, syp, szp)];
            const scalar_t phiXYpZm = phiTile[tile_idx<tileNx, tileNy>(sx, syp, szm)];
            const scalar_t phiXYmZp = phiTile[tile_idx<tileNx, tileNy>(sx, sym, szp)];
            const scalar_t phiXYmZm = phiTile[tile_idx<tileNx, tileNy>(sx, sym, szm)];

            sgx =
                VelocitySet::template w_1<scalar_t>() * (phiTile[tile_idx<tileNx, tileNy>(sxp, sy, sz)] -
                                                         phiTile[tile_idx<tileNx, tileNy>(sxm, sy, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXpYZp - phiXmYZm +
                                                         phiXpYmZ - phiXmYpZ +
                                                         phiXpYZm - phiXmYZp);

            sgy =
                VelocitySet::template w_1<scalar_t>() * (phiTile[tile_idx<tileNx, tileNy>(sx, syp, sz)] -
                                                         phiTile[tile_idx<tileNx, tileNy>(sx, sym, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYpZ - phiXpYmZ +
                                                         phiXYpZm - phiXYmZp);

            sgz =
                VelocitySet::template w_1<scalar_t>() * (phiTile[tile_idx<tileNx, tileNy>(sx, sy, szp)] -
                                                         phiTile[tile_idx<tileNx, tileNy>(sx, sy, szm)]) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYZp - phiXpYZm +
                                                         phiXYmZp - phiXYpZm);
        }
        else if constexpr (VelocitySet::Q() == 27)
        {
            const scalar_t phiXpYpZ = phiTile[tile_idx<tileNx, tileNy>(sxp, syp, sz)];
            const scalar_t phiXpYmZ = phiTile[tile_idx<tileNx, tileNy>(sxp, sym, sz)];
            const scalar_t phiXmYpZ = phiTile[tile_idx<tileNx, tileNy>(sxm, syp, sz)];
            const scalar_t phiXmYmZ = phiTile[tile_idx<tileNx, tileNy>(sxm, sym, sz)];

            const scalar_t phiXpYZp = phiTile[tile_idx<tileNx, tileNy>(sxp, sy, szp)];
            const scalar_t phiXpYZm = phiTile[tile_idx<tileNx, tileNy>(sxp, sy, szm)];
            const scalar_t phiXmYZp = phiTile[tile_idx<tileNx, tileNy>(sxm, sy, szp)];
            const scalar_t phiXmYZm = phiTile[tile_idx<tileNx, tileNy>(sxm, sy, szm)];

            const scalar_t phiXYpZp = phiTile[tile_idx<tileNx, tileNy>(sx, syp, szp)];
            const scalar_t phiXYpZm = phiTile[tile_idx<tileNx, tileNy>(sx, syp, szm)];
            const scalar_t phiXYmZp = phiTile[tile_idx<tileNx, tileNy>(sx, sym, szp)];
            const scalar_t phiXYmZm = phiTile[tile_idx<tileNx, tileNy>(sx, sym, szm)];

            const scalar_t phiXpYpZp = phiTile[tile_idx<tileNx, tileNy>(sxp, syp, szp)];
            const scalar_t phiXpYpZm = phiTile[tile_idx<tileNx, tileNy>(sxp, syp, szm)];
            const scalar_t phiXpYmZp = phiTile[tile_idx<tileNx, tileNy>(sxp, sym, szp)];
            const scalar_t phiXpYmZm = phiTile[tile_idx<tileNx, tileNy>(sxp, sym, szm)];
            const scalar_t phiXmYpZp = phiTile[tile_idx<tileNx, tileNy>(sxm, syp, szp)];
            const scalar_t phiXmYpZm = phiTile[tile_idx<tileNx, tileNy>(sxm, syp, szm)];
            const scalar_t phiXmYmZp = phiTile[tile_idx<tileNx, tileNy>(sxm, sym, szp)];
            const scalar_t phiXmYmZm = phiTile[tile_idx<tileNx, tileNy>(sxm, sym, szm)];

            sgx =
                VelocitySet::template w_1<scalar_t>() * (phiTile[tile_idx<tileNx, tileNy>(sxp, sy, sz)] -
                                                         phiTile[tile_idx<tileNx, tileNy>(sxm, sy, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXpYZp - phiXmYZm +
                                                         phiXpYmZ - phiXmYpZ +
                                                         phiXpYZm - phiXmYZp) +
                VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                         phiXpYpZm - phiXmYmZp +
                                                         phiXpYmZp - phiXmYpZm +
                                                         phiXpYmZm - phiXmYpZp);

            sgy =
                VelocitySet::template w_1<scalar_t>() * (phiTile[tile_idx<tileNx, tileNy>(sx, syp, sz)] -
                                                         phiTile[tile_idx<tileNx, tileNy>(sx, sym, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYpZ - phiXpYmZ +
                                                         phiXYpZm - phiXYmZp) +
                VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                         phiXpYpZm - phiXmYmZp +
                                                         phiXmYpZm - phiXpYmZp +
                                                         phiXmYpZp - phiXpYmZm);

            sgz =
                VelocitySet::template w_1<scalar_t>() * (phiTile[tile_idx<tileNx, tileNy>(sx, sy, szp)] -
                                                         phiTile[tile_idx<tileNx, tileNy>(sx, sy, szm)]) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYZp - phiXpYZm +
                                                         phiXYmZp - phiXYpZm) +
                VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                         phiXmYmZp - phiXpYpZm +
                                                         phiXpYmZp - phiXmYpZm +
                                                         phiXmYpZp - phiXpYmZm);
        }
        else
        {
            device::constexpr_for<1, VelocitySet::Q()>(
                [&](const auto q)
                {
                    constexpr device::label_t qi = q();
                    constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                    constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                    constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                    const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());
                    const scalar_t phi_q = phiTile[tile_idx<tileNx, tileNy>(
                        static_cast<device::label_t>(static_cast<int>(sx) + cx),
                        static_cast<device::label_t>(static_cast<int>(sy) + cy),
                        static_cast<device::label_t>(static_cast<int>(sz) + cz))];

                    sgx += wq * static_cast<scalar_t>(cx) * phi_q;
                    sgy += wq * static_cast<scalar_t>(cy) * phi_q;
                    sgz += wq * static_cast<scalar_t>(cz) * phi_q;
                });
        }

        const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
        const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
        const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

        ind = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));
        normx = gx * invInd;
        normy = gy * invInd;
        normz = gz * invInd;
    }

    /**
     * @brief Compute curvature (divergence of normals) from explicit normal shared-memory tiles
     **/
    template <class VelocitySet, device::label_t tileNx, device::label_t tileNy>
    __device__ [[nodiscard]] inline scalar_t compute_phase_curvature_from_tiles(
        const scalar_t *const ptrRestrict normTileX,
        const scalar_t *const ptrRestrict normTileY,
        const scalar_t *const ptrRestrict normTileZ,
        const device::label_t sx,
        const device::label_t sy,
        const device::label_t sz) noexcept
    {
        const device::label_t sxp = sx + static_cast<device::label_t>(1);
        const device::label_t sxm = sx - static_cast<device::label_t>(1);
        const device::label_t syp = sy + static_cast<device::label_t>(1);
        const device::label_t sym = sy - static_cast<device::label_t>(1);
        const device::label_t szp = sz + static_cast<device::label_t>(1);
        const device::label_t szm = sz - static_cast<device::label_t>(1);

        scalar_t scx = static_cast<scalar_t>(0);
        scalar_t scy = static_cast<scalar_t>(0);
        scalar_t scz = static_cast<scalar_t>(0);

        if constexpr (VelocitySet::Q() == 19)
        {
            scx =
                VelocitySet::template w_1<scalar_t>() * (normTileX[tile_idx<tileNx, tileNy>(sxp, sy, sz)] -
                                                         normTileX[tile_idx<tileNx, tileNy>(sxm, sy, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (normTileX[tile_idx<tileNx, tileNy>(sxp, syp, sz)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sym, sz)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sy, szp)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sy, szm)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sym, sz)] - normTileX[tile_idx<tileNx, tileNy>(sxm, syp, sz)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sy, szm)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sy, szp)]);

            scy =
                VelocitySet::template w_1<scalar_t>() * (normTileY[tile_idx<tileNx, tileNy>(sx, syp, sz)] -
                                                         normTileY[tile_idx<tileNx, tileNy>(sx, sym, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (normTileY[tile_idx<tileNx, tileNy>(sxp, syp, sz)] - normTileY[tile_idx<tileNx, tileNy>(sxm, sym, sz)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sx, syp, szp)] - normTileY[tile_idx<tileNx, tileNy>(sx, sym, szm)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sxm, syp, sz)] - normTileY[tile_idx<tileNx, tileNy>(sxp, sym, sz)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sx, syp, szm)] - normTileY[tile_idx<tileNx, tileNy>(sx, sym, szp)]);

            scz =
                VelocitySet::template w_1<scalar_t>() * (normTileZ[tile_idx<tileNx, tileNy>(sx, sy, szp)] -
                                                         normTileZ[tile_idx<tileNx, tileNy>(sx, sy, szm)]) +
                VelocitySet::template w_2<scalar_t>() * (normTileZ[tile_idx<tileNx, tileNy>(sxp, sy, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxm, sy, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sx, syp, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sx, sym, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sxm, sy, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxp, sy, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sx, sym, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sx, syp, szm)]);
        }
        else if constexpr (VelocitySet::Q() == 27)
        {
            scx =
                VelocitySet::template w_1<scalar_t>() * (normTileX[tile_idx<tileNx, tileNy>(sxp, sy, sz)] -
                                                         normTileX[tile_idx<tileNx, tileNy>(sxm, sy, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (normTileX[tile_idx<tileNx, tileNy>(sxp, syp, sz)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sym, sz)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sy, szp)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sy, szm)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sym, sz)] - normTileX[tile_idx<tileNx, tileNy>(sxm, syp, sz)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sy, szm)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sy, szp)]) +
                VelocitySet::template w_3<scalar_t>() * (normTileX[tile_idx<tileNx, tileNy>(sxp, syp, szp)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sym, szm)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, syp, szm)] - normTileX[tile_idx<tileNx, tileNy>(sxm, sym, szp)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sym, szp)] - normTileX[tile_idx<tileNx, tileNy>(sxm, syp, szm)] +
                                                         normTileX[tile_idx<tileNx, tileNy>(sxp, sym, szm)] - normTileX[tile_idx<tileNx, tileNy>(sxm, syp, szp)]);

            scy =
                VelocitySet::template w_1<scalar_t>() * (normTileY[tile_idx<tileNx, tileNy>(sx, syp, sz)] -
                                                         normTileY[tile_idx<tileNx, tileNy>(sx, sym, sz)]) +
                VelocitySet::template w_2<scalar_t>() * (normTileY[tile_idx<tileNx, tileNy>(sxp, syp, sz)] - normTileY[tile_idx<tileNx, tileNy>(sxm, sym, sz)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sx, syp, szp)] - normTileY[tile_idx<tileNx, tileNy>(sx, sym, szm)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sxm, syp, sz)] - normTileY[tile_idx<tileNx, tileNy>(sxp, sym, sz)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sx, syp, szm)] - normTileY[tile_idx<tileNx, tileNy>(sx, sym, szp)]) +
                VelocitySet::template w_3<scalar_t>() * (normTileY[tile_idx<tileNx, tileNy>(sxp, syp, szp)] - normTileY[tile_idx<tileNx, tileNy>(sxm, sym, szm)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sxp, syp, szm)] - normTileY[tile_idx<tileNx, tileNy>(sxm, sym, szp)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sxm, syp, szm)] - normTileY[tile_idx<tileNx, tileNy>(sxp, sym, szp)] +
                                                         normTileY[tile_idx<tileNx, tileNy>(sxm, syp, szp)] - normTileY[tile_idx<tileNx, tileNy>(sxp, sym, szm)]);

            scz =
                VelocitySet::template w_1<scalar_t>() * (normTileZ[tile_idx<tileNx, tileNy>(sx, sy, szp)] -
                                                         normTileZ[tile_idx<tileNx, tileNy>(sx, sy, szm)]) +
                VelocitySet::template w_2<scalar_t>() * (normTileZ[tile_idx<tileNx, tileNy>(sxp, sy, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxm, sy, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sx, syp, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sx, sym, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sxm, sy, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxp, sy, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sx, sym, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sx, syp, szm)]) +
                VelocitySet::template w_3<scalar_t>() * (normTileZ[tile_idx<tileNx, tileNy>(sxp, syp, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxm, sym, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sxm, sym, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxp, syp, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sxp, sym, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxm, syp, szm)] +
                                                         normTileZ[tile_idx<tileNx, tileNy>(sxm, syp, szp)] - normTileZ[tile_idx<tileNx, tileNy>(sxp, sym, szm)]);
        }
        else
        {
            device::constexpr_for<1, VelocitySet::Q()>(
                [&](const auto q)
                {
                    constexpr device::label_t qi = q();
                    constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                    constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                    constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                    const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());
                    const scalar_t nx_q = normTileX[tile_idx<tileNx, tileNy>(
                        static_cast<device::label_t>(static_cast<int>(sx) + cx),
                        static_cast<device::label_t>(static_cast<int>(sy) + cy),
                        static_cast<device::label_t>(static_cast<int>(sz) + cz))];
                    const scalar_t ny_q = normTileY[tile_idx<tileNx, tileNy>(
                        static_cast<device::label_t>(static_cast<int>(sx) + cx),
                        static_cast<device::label_t>(static_cast<int>(sy) + cy),
                        static_cast<device::label_t>(static_cast<int>(sz) + cz))];
                    const scalar_t nz_q = normTileZ[tile_idx<tileNx, tileNy>(
                        static_cast<device::label_t>(static_cast<int>(sx) + cx),
                        static_cast<device::label_t>(static_cast<int>(sy) + cy),
                        static_cast<device::label_t>(static_cast<int>(sz) + cz))];

                    scx += wq * static_cast<scalar_t>(cx) * nx_q;
                    scy += wq * static_cast<scalar_t>(cy) * ny_q;
                    scz += wq * static_cast<scalar_t>(cz) * nz_q;
                });
        }

        return velocitySet::as2<scalar_t>() * (scx + scy + scz);
    }

    /**
     * @brief Compute isotropic phase gradient at a shifted lattice point
     * @tparam ox,oy,oz Shift of the evaluation point relative to current thread point
     **/
    template <const int ox, const int oy, const int oz, class VelocitySet, class PhaseHalo, const bool useScalarHalo = true, class HydroShared>
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

        if constexpr (VelocitySet::Q() == 19)
        {
            const scalar_t phiXpYpZ = load_phase_neighbor<ox + 1, oy + 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYmZ = load_phase_neighbor<ox + 1, oy - 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYpZ = load_phase_neighbor<ox - 1, oy + 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYmZ = load_phase_neighbor<ox - 1, oy - 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            const scalar_t phiXpYZp = load_phase_neighbor<ox + 1, oy + 0, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYZm = load_phase_neighbor<ox + 1, oy + 0, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYZp = load_phase_neighbor<ox - 1, oy + 0, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYZm = load_phase_neighbor<ox - 1, oy + 0, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            const scalar_t phiXYpZp = load_phase_neighbor<ox + 0, oy + 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXYpZm = load_phase_neighbor<ox + 0, oy + 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXYmZp = load_phase_neighbor<ox + 0, oy - 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXYmZm = load_phase_neighbor<ox + 0, oy - 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            sgx =
                VelocitySet::template w_1<scalar_t>() * (load_phase_neighbor<ox + 1, oy + 0, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point) -
                                                         load_phase_neighbor<ox - 1, oy + 0, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point)) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXpYZp - phiXmYZm +
                                                         phiXpYmZ - phiXmYpZ +
                                                         phiXpYZm - phiXmYZp);

            sgy =
                VelocitySet::template w_1<scalar_t>() * (load_phase_neighbor<ox + 0, oy + 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point) -
                                                         load_phase_neighbor<ox + 0, oy - 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point)) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYpZ - phiXpYmZ +
                                                         phiXYpZm - phiXYmZp);

            sgz =
                VelocitySet::template w_1<scalar_t>() * (load_phase_neighbor<ox + 0, oy + 0, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point) -
                                                         load_phase_neighbor<ox + 0, oy + 0, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point)) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYZp - phiXpYZm +
                                                         phiXYmZp - phiXYpZm);
        }
        else if constexpr (VelocitySet::Q() == 27)
        {
            const scalar_t phiXpYpZ = load_phase_neighbor<ox + 1, oy + 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYmZ = load_phase_neighbor<ox + 1, oy - 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYpZ = load_phase_neighbor<ox - 1, oy + 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYmZ = load_phase_neighbor<ox - 1, oy - 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            const scalar_t phiXpYZp = load_phase_neighbor<ox + 1, oy + 0, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYZm = load_phase_neighbor<ox + 1, oy + 0, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYZp = load_phase_neighbor<ox - 1, oy + 0, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYZm = load_phase_neighbor<ox - 1, oy + 0, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            const scalar_t phiXYpZp = load_phase_neighbor<ox + 0, oy + 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXYpZm = load_phase_neighbor<ox + 0, oy + 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXYmZp = load_phase_neighbor<ox + 0, oy - 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXYmZm = load_phase_neighbor<ox + 0, oy - 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            const scalar_t phiXpYpZp = load_phase_neighbor<ox + 1, oy + 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYpZm = load_phase_neighbor<ox + 1, oy + 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYmZp = load_phase_neighbor<ox + 1, oy - 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXpYmZm = load_phase_neighbor<ox + 1, oy - 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYpZp = load_phase_neighbor<ox - 1, oy + 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYpZm = load_phase_neighbor<ox - 1, oy + 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYmZp = load_phase_neighbor<ox - 1, oy - 1, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
            const scalar_t phiXmYmZm = load_phase_neighbor<ox - 1, oy - 1, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);

            sgx =
                VelocitySet::template w_1<scalar_t>() * (load_phase_neighbor<ox + 1, oy + 0, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point) -
                                                         load_phase_neighbor<ox - 1, oy + 0, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point)) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXpYZp - phiXmYZm +
                                                         phiXpYmZ - phiXmYpZ +
                                                         phiXpYZm - phiXmYZp) +
                VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                         phiXpYpZm - phiXmYmZp +
                                                         phiXpYmZp - phiXmYpZm +
                                                         phiXpYmZm - phiXmYpZp);

            sgy =
                VelocitySet::template w_1<scalar_t>() * (load_phase_neighbor<ox + 0, oy + 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point) -
                                                         load_phase_neighbor<ox + 0, oy - 1, oz + 0, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point)) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYpZ - phiXpYmZ +
                                                         phiXYpZm - phiXYmZp) +
                VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                         phiXpYpZm - phiXmYmZp +
                                                         phiXmYpZm - phiXpYmZp +
                                                         phiXmYpZp - phiXpYmZm);

            sgz =
                VelocitySet::template w_1<scalar_t>() * (load_phase_neighbor<ox + 0, oy + 0, oz + 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point) -
                                                         load_phase_neighbor<ox + 0, oy + 0, oz - 1, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point)) +
                VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                         phiXYpZp - phiXYmZm +
                                                         phiXmYZp - phiXpYZm +
                                                         phiXYmZp - phiXYpZm) +
                VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                         phiXmYmZp - phiXpYpZm +
                                                         phiXpYmZp - phiXmYpZm +
                                                         phiXmYpZp - phiXpYmZm);
        }
        else
        {
            device::constexpr_for<1, VelocitySet::Q()>(
                [&](const auto q)
                {
                    constexpr device::label_t qi = q();
                    constexpr int cx = VelocitySet::template c<int, axis::X>()[qi];
                    constexpr int cy = VelocitySet::template c<int, axis::Y>()[qi];
                    constexpr int cz = VelocitySet::template c<int, axis::Z>()[qi];

                    const scalar_t phi_q = load_phase_neighbor<ox + cx, oy + cy, oz + cz, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point);
                    const scalar_t wq = VelocitySet::template w_q<scalar_t>(q_i<qi>());

                    sgx += wq * static_cast<scalar_t>(cx) * phi_q;
                    sgy += wq * static_cast<scalar_t>(cy) * phi_q;
                    sgz += wq * static_cast<scalar_t>(cz) * phi_q;
                });
        }

        gx = velocitySet::as2<scalar_t>() * sgx;
        gy = velocitySet::as2<scalar_t>() * sgy;
        gz = velocitySet::as2<scalar_t>() * sgz;
    }

    /**
     * @brief Compute interface normal and indicator at a shifted lattice point
     **/
    template <const int ox, const int oy, const int oz, class VelocitySet, class PhaseHalo, const bool useScalarHalo = true, class HydroShared>
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
        compute_phase_gradient<ox, oy, oz, VelocitySet, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point, gx, gy, gz);

        ind = sqrtf(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

        normx = gx * invInd;
        normy = gy * invInd;
        normz = gz * invInd;
    }

    /**
     * @brief Compute curvature from divergence of normals using the current VelocitySet stencil
     **/
    template <class VelocitySet, class PhaseHalo, const bool useScalarHalo = true, class HydroShared>
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

                compute_phase_normal<cx, cy, cz, VelocitySet, PhaseHalo, useScalarHalo>(phi, phiBuffer, hydroShared, Tx, Bx, point, nx_q, ny_q, nz_q, ind_q);

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
    template <class BoundaryConditions, class VelocitySet, class PhaseVelocitySet, class HydroHalo, class PhaseHalo, const bool useScalarHalo = true, class HydroShared, class PhaseShared>
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

        if constexpr (useScalarHalo)
        {
            constexpr device::label_t sharedStride = label_constant<NUMBER_MOMENTS<true>() + 1>();
            constexpr device::label_t phiSharedOffset = label_constant<NUMBER_MOMENTS<true>()>();

            hydroShared[tid * sharedStride + phiSharedOffset] = phi[idx];

            block::sync();
        }

        const bool isInterior =
            (point.value<axis::X>() > static_cast<device::label_t>(0)) &&
            (point.value<axis::X>() < (device::n<axis::X>() - static_cast<device::label_t>(1))) &&
            (point.value<axis::Y>() > static_cast<device::label_t>(0)) &&
            (point.value<axis::Y>() < (device::n<axis::Y>() - static_cast<device::label_t>(1))) &&
            (point.value<axis::Z>() > static_cast<device::label_t>(0)) &&
            (point.value<axis::Z>() < (device::n<axis::Z>() - static_cast<device::label_t>(1)));

        if (isInterior)
        {
            if constexpr (useScalarHalo)
            {
                constexpr device::label_t sharedStride = label_constant<NUMBER_MOMENTS<true>() + 1>();
                constexpr device::label_t phiSharedOffset = label_constant<NUMBER_MOMENTS<true>()>();

                const bool isBlockInterior =
                    (Tx.value<axis::X>() > static_cast<device::label_t>(0)) &&
                    (Tx.value<axis::X>() < (block::n<axis::X>() - static_cast<device::label_t>(1))) &&
                    (Tx.value<axis::Y>() > static_cast<device::label_t>(0)) &&
                    (Tx.value<axis::Y>() < (block::n<axis::Y>() - static_cast<device::label_t>(1))) &&
                    (Tx.value<axis::Z>() > static_cast<device::label_t>(0)) &&
                    (Tx.value<axis::Z>() < (block::n<axis::Z>() - static_cast<device::label_t>(1)));

                if (isBlockInterior)
                {
                    constexpr device::label_t dy = block::n<axis::X>();
                    constexpr device::label_t dz = block::stride_z();

                    const device::label_t iXp = tid + static_cast<device::label_t>(1);
                    const device::label_t iXm = tid - static_cast<device::label_t>(1);
                    const device::label_t iYp = tid + dy;
                    const device::label_t iYm = tid - dy;
                    const device::label_t iZp = tid + dz;
                    const device::label_t iZm = tid - dz;

                    scalar_t sgx = static_cast<scalar_t>(0);
                    scalar_t sgy = static_cast<scalar_t>(0);
                    scalar_t sgz = static_cast<scalar_t>(0);

                    if constexpr (VelocitySet::Q() == 19)
                    {
                        const scalar_t phiXp = hydroShared[iXp * sharedStride + phiSharedOffset];
                        const scalar_t phiXm = hydroShared[iXm * sharedStride + phiSharedOffset];
                        const scalar_t phiYp = hydroShared[iYp * sharedStride + phiSharedOffset];
                        const scalar_t phiYm = hydroShared[iYm * sharedStride + phiSharedOffset];
                        const scalar_t phiZp = hydroShared[iZp * sharedStride + phiSharedOffset];
                        const scalar_t phiZm = hydroShared[iZm * sharedStride + phiSharedOffset];

                        sgx =
                            VelocitySet::template w_1<scalar_t>() * (phiXp - phiXm) +
                            VelocitySet::template w_2<scalar_t>() * ((hydroShared[(iXp + dy) * sharedStride + phiSharedOffset] - hydroShared[(iXm - dy) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iXp + dz) * sharedStride + phiSharedOffset] - hydroShared[(iXm - dz) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iXp - dy) * sharedStride + phiSharedOffset] - hydroShared[(iXm + dy) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iXp - dz) * sharedStride + phiSharedOffset] - hydroShared[(iXm + dz) * sharedStride + phiSharedOffset]));

                        sgy =
                            VelocitySet::template w_1<scalar_t>() * (phiYp - phiYm) +
                            VelocitySet::template w_2<scalar_t>() * ((hydroShared[(iXp + dy) * sharedStride + phiSharedOffset] - hydroShared[(iXm - dy) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iYp + dz) * sharedStride + phiSharedOffset] - hydroShared[(iYm - dz) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iXm + dy) * sharedStride + phiSharedOffset] - hydroShared[(iXp - dy) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iYp - dz) * sharedStride + phiSharedOffset] - hydroShared[(iYm + dz) * sharedStride + phiSharedOffset]));

                        sgz =
                            VelocitySet::template w_1<scalar_t>() * (phiZp - phiZm) +
                            VelocitySet::template w_2<scalar_t>() * ((hydroShared[(iXp + dz) * sharedStride + phiSharedOffset] - hydroShared[(iXm - dz) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iYp + dz) * sharedStride + phiSharedOffset] - hydroShared[(iYm - dz) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iXm + dz) * sharedStride + phiSharedOffset] - hydroShared[(iXp - dz) * sharedStride + phiSharedOffset]) +
                                                                     (hydroShared[(iYm + dz) * sharedStride + phiSharedOffset] - hydroShared[(iYp - dz) * sharedStride + phiSharedOffset]));
                    }
                    else if constexpr (VelocitySet::Q() == 27)
                    {
                        const scalar_t phiXpYpZ = hydroShared[(iXp + dy) * sharedStride + phiSharedOffset];
                        const scalar_t phiXpYmZ = hydroShared[(iXp - dy) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYpZ = hydroShared[(iXm + dy) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYmZ = hydroShared[(iXm - dy) * sharedStride + phiSharedOffset];

                        const scalar_t phiXpYZp = hydroShared[(iXp + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXpYZm = hydroShared[(iXp - dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYZp = hydroShared[(iXm + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYZm = hydroShared[(iXm - dz) * sharedStride + phiSharedOffset];

                        const scalar_t phiXYpZp = hydroShared[(iYp + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXYpZm = hydroShared[(iYp - dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXYmZp = hydroShared[(iYm + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXYmZm = hydroShared[(iYm - dz) * sharedStride + phiSharedOffset];

                        const scalar_t phiXpYpZp = hydroShared[(iXp + dy + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXpYpZm = hydroShared[(iXp + dy - dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXpYmZp = hydroShared[(iXp - dy + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXpYmZm = hydroShared[(iXp - dy - dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYpZp = hydroShared[(iXm + dy + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYpZm = hydroShared[(iXm + dy - dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYmZp = hydroShared[(iXm - dy + dz) * sharedStride + phiSharedOffset];
                        const scalar_t phiXmYmZm = hydroShared[(iXm - dy - dz) * sharedStride + phiSharedOffset];

                        sgx =
                            VelocitySet::template w_1<scalar_t>() * (hydroShared[iXp * sharedStride + phiSharedOffset] -
                                                                     hydroShared[iXm * sharedStride + phiSharedOffset]) +
                            VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                                     phiXpYZp - phiXmYZm +
                                                                     phiXpYmZ - phiXmYpZ +
                                                                     phiXpYZm - phiXmYZp) +
                            VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                                     phiXpYpZm - phiXmYmZp +
                                                                     phiXpYmZp - phiXmYpZm +
                                                                     phiXpYmZm - phiXmYpZp);

                        sgy =
                            VelocitySet::template w_1<scalar_t>() * (hydroShared[iYp * sharedStride + phiSharedOffset] -
                                                                     hydroShared[iYm * sharedStride + phiSharedOffset]) +
                            VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                                     phiXYpZp - phiXYmZm +
                                                                     phiXmYpZ - phiXpYmZ +
                                                                     phiXYpZm - phiXYmZp) +
                            VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                                     phiXpYpZm - phiXmYmZp +
                                                                     phiXmYpZm - phiXpYmZp +
                                                                     phiXmYpZp - phiXpYmZm);

                        sgz =
                            VelocitySet::template w_1<scalar_t>() * (hydroShared[iZp * sharedStride + phiSharedOffset] -
                                                                     hydroShared[iZm * sharedStride + phiSharedOffset]) +
                            VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                                     phiXYpZp - phiXYmZm +
                                                                     phiXmYZp - phiXpYZm +
                                                                     phiXYmZp - phiXYpZm) +
                            VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                                     phiXmYmZp - phiXpYpZm +
                                                                     phiXpYmZp - phiXmYpZm +
                                                                     phiXmYpZp - phiXpYmZm);
                    }
                    else
                    {
                        scalar_t indDummy = static_cast<scalar_t>(0);
                        compute_phase_normal<0, 0, 0, VelocitySet, PhaseHalo, useScalarHalo>(
                            phi,
                            phiBuffer,
                            hydroShared,
                            Tx,
                            Bx,
                            point,
                            normx_,
                            normy_,
                            normz_,
                            indDummy);
                    }

                    if constexpr ((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27))
                    {
                        const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
                        const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
                        const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

                        const scalar_t ind_ = sqrtf(gx * gx + gy * gy + gz * gz);
                        const scalar_t invInd = static_cast<scalar_t>(1) / (ind_ + static_cast<scalar_t>(1e-9));
                        normx_ = gx * invInd;
                        normy_ = gy * invInd;
                        normz_ = gz * invInd;
                    }
                }
                else
                {
                    scalar_t indDummy = static_cast<scalar_t>(0);
                    compute_phase_normal<0, 0, 0, VelocitySet, PhaseHalo, useScalarHalo>(
                        phi,
                        phiBuffer,
                        hydroShared,
                        Tx,
                        Bx,
                        point,
                        normx_,
                        normy_,
                        normz_,
                        indDummy);
                }
            }
            else
            {
                // Single-GPU fast path: explicit stencil with direct local indexing (no scalar-halo checks)
                const device::label_t strideBy = block::size() * device::NUM_BLOCK<axis::X>();
                const device::label_t strideBz = block::size() * device::NUM_BLOCK<axis::X>() * device::NUM_BLOCK<axis::Y>();

                const device::label_t wrapX = block::size() - (block::n<axis::X>() - static_cast<device::label_t>(1)) * static_cast<device::label_t>(1);
                const device::label_t wrapY = strideBy - (block::n<axis::Y>() - static_cast<device::label_t>(1)) * block::n<axis::X>();
                const device::label_t wrapZ = strideBz - (block::n<axis::Z>() - static_cast<device::label_t>(1)) * block::stride_z();

                const device::label_t dxp = (Tx.value<axis::X>() == (block::n<axis::X>() - static_cast<device::label_t>(1))) ? wrapX : static_cast<device::label_t>(1);
                const device::label_t dxm = (Tx.value<axis::X>() == static_cast<device::label_t>(0)) ? wrapX : static_cast<device::label_t>(1);

                const device::label_t dyp = (Tx.value<axis::Y>() == (block::n<axis::Y>() - static_cast<device::label_t>(1))) ? wrapY : block::n<axis::X>();
                const device::label_t dym = (Tx.value<axis::Y>() == static_cast<device::label_t>(0)) ? wrapY : block::n<axis::X>();

                const device::label_t dzp = (Tx.value<axis::Z>() == (block::n<axis::Z>() - static_cast<device::label_t>(1))) ? wrapZ : block::stride_z();
                const device::label_t dzm = (Tx.value<axis::Z>() == static_cast<device::label_t>(0)) ? wrapZ : block::stride_z();

                const device::label_t iXp = idx + dxp;
                const device::label_t iXm = idx - dxm;
                const device::label_t iYp = idx + dyp;
                const device::label_t iYm = idx - dym;
                const device::label_t iZp = idx + dzp;
                const device::label_t iZm = idx - dzm;

                const scalar_t phiXpYpZ = phi[iXp + dyp];
                const scalar_t phiXpYmZ = phi[iXp - dym];
                const scalar_t phiXmYpZ = phi[iXm + dyp];
                const scalar_t phiXmYmZ = phi[iXm - dym];

                const scalar_t phiXpYZp = phi[iXp + dzp];
                const scalar_t phiXpYZm = phi[iXp - dzm];
                const scalar_t phiXmYZp = phi[iXm + dzp];
                const scalar_t phiXmYZm = phi[iXm - dzm];

                const scalar_t phiXYpZp = phi[iYp + dzp];
                const scalar_t phiXYpZm = phi[iYp - dzm];
                const scalar_t phiXYmZp = phi[iYm + dzp];
                const scalar_t phiXYmZm = phi[iYm - dzm];

                scalar_t sgx = static_cast<scalar_t>(0);
                scalar_t sgy = static_cast<scalar_t>(0);
                scalar_t sgz = static_cast<scalar_t>(0);

                if constexpr (VelocitySet::Q() == 19)
                {
                    sgx =
                        VelocitySet::template w_1<scalar_t>() * (phi[iXp] - phi[iXm]) +
                        VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                                 phiXpYZp - phiXmYZm +
                                                                 phiXpYmZ - phiXmYpZ +
                                                                 phiXpYZm - phiXmYZp);

                    sgy =
                        VelocitySet::template w_1<scalar_t>() * (phi[iYp] - phi[iYm]) +
                        VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                                 phiXYpZp - phiXYmZm +
                                                                 phiXmYpZ - phiXpYmZ +
                                                                 phiXYpZm - phiXYmZp);

                    sgz =
                        VelocitySet::template w_1<scalar_t>() * (phi[iZp] - phi[iZm]) +
                        VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                                 phiXYpZp - phiXYmZm +
                                                                 phiXmYZp - phiXpYZm +
                                                                 phiXYmZp - phiXYpZm);
                }
                else if constexpr (VelocitySet::Q() == 27)
                {
                    const scalar_t phiXpYpZp = phi[iXp + dyp + dzp];
                    const scalar_t phiXpYpZm = phi[iXp + dyp - dzm];
                    const scalar_t phiXpYmZp = phi[iXp - dym + dzp];
                    const scalar_t phiXpYmZm = phi[iXp - dym - dzm];
                    const scalar_t phiXmYpZp = phi[iXm + dyp + dzp];
                    const scalar_t phiXmYpZm = phi[iXm + dyp - dzm];
                    const scalar_t phiXmYmZp = phi[iXm - dym + dzp];
                    const scalar_t phiXmYmZm = phi[iXm - dym - dzm];

                    sgx =
                        VelocitySet::template w_1<scalar_t>() * (phi[iXp] - phi[iXm]) +
                        VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                                 phiXpYZp - phiXmYZm +
                                                                 phiXpYmZ - phiXmYpZ +
                                                                 phiXpYZm - phiXmYZp) +
                        VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                                 phiXpYpZm - phiXmYmZp +
                                                                 phiXpYmZp - phiXmYpZm +
                                                                 phiXpYmZm - phiXmYpZp);

                    sgy =
                        VelocitySet::template w_1<scalar_t>() * (phi[iYp] - phi[iYm]) +
                        VelocitySet::template w_2<scalar_t>() * (phiXpYpZ - phiXmYmZ +
                                                                 phiXYpZp - phiXYmZm +
                                                                 phiXmYpZ - phiXpYmZ +
                                                                 phiXYpZm - phiXYmZp) +
                        VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                                 phiXpYpZm - phiXmYmZp +
                                                                 phiXmYpZm - phiXpYmZp +
                                                                 phiXmYpZp - phiXpYmZm);

                    sgz =
                        VelocitySet::template w_1<scalar_t>() * (phi[iZp] - phi[iZm]) +
                        VelocitySet::template w_2<scalar_t>() * (phiXpYZp - phiXmYZm +
                                                                 phiXYpZp - phiXYmZm +
                                                                 phiXmYZp - phiXpYZm +
                                                                 phiXYmZp - phiXYpZm) +
                        VelocitySet::template w_3<scalar_t>() * (phiXpYpZp - phiXmYmZm +
                                                                 phiXmYmZp - phiXpYpZm +
                                                                 phiXpYmZp - phiXmYpZm +
                                                                 phiXmYpZp - phiXpYmZm);
                }
                else
                {
                    scalar_t indDummy = static_cast<scalar_t>(0);
                    compute_phase_normal<0, 0, 0, VelocitySet, PhaseHalo, useScalarHalo>(
                        phi,
                        phiBuffer,
                        hydroShared,
                        Tx,
                        Bx,
                        point,
                        normx_,
                        normy_,
                        normz_,
                        indDummy);
                }

                if constexpr ((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27))
                {
                    const scalar_t gx = velocitySet::as2<scalar_t>() * sgx;
                    const scalar_t gy = velocitySet::as2<scalar_t>() * sgy;
                    const scalar_t gz = velocitySet::as2<scalar_t>() * sgz;

                    const scalar_t ind_ = sqrtf(gx * gx + gy * gy + gz * gz);
                    const scalar_t invInd = static_cast<scalar_t>(1) / (ind_ + static_cast<scalar_t>(1e-9));
                    normx_ = gx * invInd;
                    normy_ = gy * invInd;
                    normz_ = gz * invInd;
                }
            }
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
        if constexpr (useScalarHalo)
        {
            PhaseHalo::save_scalar(moments[m_i<10>()], phiWriteBuffer, Tx, Bx, point);
        }
        else
        {
            (void)phiWriteBuffer;
        }
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
    template <class BoundaryConditions, class VelocitySet, class PhaseVelocitySet, class Collision, class HydroHalo, class PhaseHalo, const bool useScalarHalo = true>
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
        __shared__ scalar_t indShared[normalTileSize];

        const auto phiSharedIdx = [](const device::label_t sx, const device::label_t sy, const device::label_t sz) noexcept -> device::label_t
        {
            return (sz * phiTileNy + sy) * phiTileNx + sx;
        };

        const auto normalSharedIdx = [](const device::label_t sx, const device::label_t sy, const device::label_t sz) noexcept -> device::label_t
        {
            return (sz * normalNy + sy) * normalNx + sx;
        };

        // Build shared tiles for phi (+/-2 halo) and normals (+/-1 halo).
        // Single-GPU uses direct strided loads (fast path); multi-GPU uses halo-aware scalar pulls.
        if constexpr (useScalarHalo)
        {
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

            for (int ox = -2; ox <= 2; ++ox)
            {
                for (int oy = -2; oy <= 2; ++oy)
                {
                    for (int oz = -2; oz <= 2; ++oz)
                    {
                        if (!(ownsPhiOffsetX(ox) && ownsPhiOffsetY(oy) && ownsPhiOffsetZ(oz)))
                        {
                            continue;
                        }

                        const scalar_t phiCandidate = load_phase_neighbor_direct_runtime<PhaseHalo, useScalarHalo>(
                            phi,
                            phiBuffer,
                            Tx,
                            Bx,
                            point,
                            ox,
                            oy,
                            oz);

                        const device::label_t sx = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 2 + ox);
                        const device::label_t sy = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 2 + oy);
                        const device::label_t sz = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 2 + oz);
                        phiShared[phiSharedIdx(sx, sy, sz)] = phiCandidate;
                    }
                }
            }

            block::sync();

            for (int ox = -1; ox <= 1; ++ox)
            {
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int oz = -1; oz <= 1; ++oz)
                    {
                        if (!(ownsNormalOffsetX(ox) && ownsNormalOffsetY(oy) && ownsNormalOffsetZ(oz)))
                        {
                            continue;
                        }

                        const device::label_t sxPhi = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 2 + ox);
                        const device::label_t syPhi = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 2 + oy);
                        const device::label_t szPhi = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 2 + oz);

                        scalar_t nxCandidate = static_cast<scalar_t>(0);
                        scalar_t nyCandidate = static_cast<scalar_t>(0);
                        scalar_t nzCandidate = static_cast<scalar_t>(0);
                        scalar_t indCandidate = static_cast<scalar_t>(0);

                        compute_phase_normal_from_tile<VelocitySet, phiTileNx, phiTileNy>(
                            phiShared,
                            sxPhi,
                            syPhi,
                            szPhi,
                            nxCandidate,
                            nyCandidate,
                            nzCandidate,
                            indCandidate);

                        const device::label_t sx = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 1 + ox);
                        const device::label_t sy = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 1 + oy);
                        const device::label_t sz = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 1 + oz);
                        const device::label_t sid = normalSharedIdx(sx, sy, sz);

                        normSharedX[sid] = nxCandidate;
                        normSharedY[sid] = nyCandidate;
                        normSharedZ[sid] = nzCandidate;
                        indShared[sid] = indCandidate;
                    }
                }
            }

            block::sync();
        }
        else
        {
            const int nX = static_cast<int>(device::n<axis::X>());
            const int nY = static_cast<int>(device::n<axis::Y>());
            const int nZ = static_cast<int>(device::n<axis::Z>());

            const int bnx = static_cast<int>(block::n<axis::X>());
            const int bny = static_cast<int>(block::n<axis::Y>());
            const int bnz = static_cast<int>(block::n<axis::Z>());

            const int x0 = static_cast<int>(Bx.value<axis::X>()) * bnx;
            const int y0 = static_cast<int>(Bx.value<axis::Y>()) * bny;
            const int z0 = static_cast<int>(Bx.value<axis::Z>()) * bnz;

            for (int sz = static_cast<int>(Tx.value<axis::Z>()); sz < static_cast<int>(phiTileNz); sz += bnz)
            {
                const int gz = z0 + sz - 2;
                for (int sy = static_cast<int>(Tx.value<axis::Y>()); sy < static_cast<int>(phiTileNy); sy += bny)
                {
                    const int gy = y0 + sy - 2;
                    for (int sx = static_cast<int>(Tx.value<axis::X>()); sx < static_cast<int>(phiTileNx); sx += bnx)
                    {
                        int gx = x0 + sx - 2;
                        int gyWrapped = gy;
                        int gzWrapped = gz;
                        bool validPoint = true;

                        if constexpr (PhaseHalo::periodicX())
                        {
                            if (gx < 0)
                            {
                                gx += nX;
                            }
                            else if (gx >= nX)
                            {
                                gx -= nX;
                            }
                        }
                        else if ((gx < 0) || (gx >= nX))
                        {
                            validPoint = false;
                        }

                        if constexpr (PhaseHalo::periodicY())
                        {
                            if (gyWrapped < 0)
                            {
                                gyWrapped += nY;
                            }
                            else if (gyWrapped >= nY)
                            {
                                gyWrapped -= nY;
                            }
                        }
                        else if ((gyWrapped < 0) || (gyWrapped >= nY))
                        {
                            validPoint = false;
                        }

                        if constexpr (PhaseHalo::periodicZ())
                        {
                            if (gzWrapped < 0)
                            {
                                gzWrapped += nZ;
                            }
                            else if (gzWrapped >= nZ)
                            {
                                gzWrapped -= nZ;
                            }
                        }
                        else if ((gzWrapped < 0) || (gzWrapped >= nZ))
                        {
                            validPoint = false;
                        }

                        scalar_t phiCandidate = static_cast<scalar_t>(0);
                        if (validPoint)
                        {
                            phiCandidate = __ldg(&(phi[GPU::idxGlobalFromIdx(
                                static_cast<device::label_t>(gx),
                                static_cast<device::label_t>(gyWrapped),
                                static_cast<device::label_t>(gzWrapped))]));
                        }

                        phiShared[phiSharedIdx(
                            static_cast<device::label_t>(sx),
                            static_cast<device::label_t>(sy),
                            static_cast<device::label_t>(sz))] = phiCandidate;
                    }
                }
            }

            block::sync();

            for (int sz = static_cast<int>(Tx.value<axis::Z>()); sz < static_cast<int>(normalNz); sz += bnz)
            {
                const device::label_t szLabel = static_cast<device::label_t>(sz);
                const device::label_t szPhi = static_cast<device::label_t>(sz + 1);
                for (int sy = static_cast<int>(Tx.value<axis::Y>()); sy < static_cast<int>(normalNy); sy += bny)
                {
                    const device::label_t syLabel = static_cast<device::label_t>(sy);
                    const device::label_t syPhi = static_cast<device::label_t>(sy + 1);
                    for (int sx = static_cast<int>(Tx.value<axis::X>()); sx < static_cast<int>(normalNx); sx += bnx)
                    {
                        const device::label_t sxLabel = static_cast<device::label_t>(sx);
                        const device::label_t sxPhi = static_cast<device::label_t>(sx + 1);

                        scalar_t nxCandidate = static_cast<scalar_t>(0);
                        scalar_t nyCandidate = static_cast<scalar_t>(0);
                        scalar_t nzCandidate = static_cast<scalar_t>(0);
                        scalar_t indCandidate = static_cast<scalar_t>(0);

                        compute_phase_normal_from_tile<VelocitySet, phiTileNx, phiTileNy>(
                            phiShared,
                            sxPhi,
                            syPhi,
                            szPhi,
                            nxCandidate,
                            nyCandidate,
                            nzCandidate,
                            indCandidate);

                        const device::label_t sid = normalSharedIdx(sxLabel, syLabel, szLabel);
                        normSharedX[sid] = nxCandidate;
                        normSharedY[sid] = nyCandidate;
                        normSharedZ[sid] = nzCandidate;
                        indShared[sid] = indCandidate;
                    }
                }
            }

            block::sync();
        }

        const device::label_t sxCenter = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::X>()) + 1);
        const device::label_t syCenter = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Y>()) + 1);
        const device::label_t szCenter = static_cast<device::label_t>(static_cast<int>(Tx.value<axis::Z>()) + 1);
        const device::label_t centerSid = normalSharedIdx(sxCenter, syCenter, szCenter);

        centerNormx = normSharedX[centerSid];
        centerNormy = normSharedY[centerSid];
        centerNormz = normSharedZ[centerSid];
        centerInd = indShared[centerSid];

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

            const scalar_t curvature = compute_phase_curvature_from_tiles<VelocitySet, normalNx, normalNy>(
                normSharedX,
                normSharedY,
                normSharedZ,
                sxCenter,
                syCenter,
                szCenter);
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

        // Save the hydro populations to the block halo
        HydroHalo::save(pop, moments, hydroBuffer, Tx, Bx, point);

        // Save the phase populations to the block halo
        PhaseHalo::save(pop_g, moments, phaseBuffer, Tx, Bx, point);
    }
}

#endif
