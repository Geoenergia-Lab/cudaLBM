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
    A class handling the device halo. This class is used to exchange the
    microscopic velocity components at the edge of a CUDA block

Namespace
    LBM::device

SourceFiles
    haloSharedMemoryOperations.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALOSHAREDMEMORYOPERATIONS_CUH
#define __MBLBM_HALOSHAREDMEMORYOPERATIONS_CUH

/**
 * @brief Computes linear index for a thread within a block
 * @param[in] tx Thread x-coordinate within block
 * @param[in] ty Thread y-coordinate within block
 * @param[in] tz Thread z-coordinate within block
 * @return Linearized index in shared memory
 *
 * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idx_block(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
{
    return tx + block::nx<device::label_t>() * (ty + block::ny<device::label_t>() * tz);
}

/**
 * @overload Passes a pre-constructed thread coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idx_block(const thread::coordinate &Tx) noexcept
{
    return idx_block(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
}

/**
 * @brief Computes the warp number of a particular thread within a block
 * @param[in] tx Thread x-coordinate within block
 * @param[in] ty Thread y-coordinate within block
 * @param[in] tz Thread z-coordinate within block
 * @return The unique ID of the warp corresponding to a particular thread
 *
 * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t warpID(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
{
    return idx_block(tx, ty, tz) / block::warp_size();
}

/**
 * @overload Passes a pre-constructed thread coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t warpID(const thread::coordinate &Tx) noexcept
{
    return warpID(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
}

/**
 * @brief Computes the linear index of a thread within a warp
 * @param[in] tx Thread x-coordinate within block
 * @param[in] ty Thread y-coordinate within block
 * @param[in] tz Thread z-coordinate within block
 * @return The unique ID of a thread within a warp, in the range [0, warp_size]
 *
 * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxWarp(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
{
    return idx_block(tx, ty, tz) % block::warp_size();
}

/**
 * @overload Passes a pre-constructed thread coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxWarp(const thread::coordinate &Tx) noexcept
{
    return idxWarp(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
}

/**
 * @brief Total area of lattice units on a block face
 * @tparam alpha The axis direction (X, Y or Z)
 * @tparam T The return type
 * @return Linearized face index>
 **/
template <const axis::type alpha, typename T = device::label_t>
__device__ __host__ [[nodiscard]] static inline consteval T faceArea() noexcept
{
    return block::n<axis::orthogonal<alpha, 0>(), T>() * block::n<axis::orthogonal<alpha, 1>(), T>();
}

/**
 * @brief Index for points located on a block face
 * @tparam alpha The axis direction (X, Y or Z)
 * @param[in] Tx Three-dimensional thread coordinates
 * @return Linearized face index>
 **/
template <const axis::type alpha>
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxFace(const thread::coordinate &Tx) noexcept
{
    return Tx.value<axis::orthogonal<alpha, 0>()>() + (Tx.value<axis::orthogonal<alpha, 1>()>() * block::n<axis::orthogonal<alpha, 0>()>());
}

/**
 * @brief Transposes an individual face into the shared memory
 * @tparam alpha The axis direction (X, Y or Z)
 * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
 * @tparam idxOffset The constant offset into the shared memory for the particular block configuration
 * @tparam SharedBuffer Type of the shared memory buffer
 * @param[in] Tx Three-dimensional thread coordinates
 * @param[in] pop Array to store loaded population values
 * @param[in] sharedBuffer Inline or externally stored shared memory buffer
 **/
template <const axis::type alpha, const int coeff, const device::label_t idxOffset, class SharedBuffer>
__device__ static inline constexpr void transpose(const thread::coordinate &Tx, const thread::array<scalar_t, VelocitySet::Q()> &pop, SharedBuffer &sharedBuffer) noexcept
{
    axis::assertions::validate<alpha, axis::NOT_NULL>();

    velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

    const device::label_t base_idx = idxFace<alpha>(Tx);
    device::constexpr_for<0, VelocitySet::template QF<device::label_t>()>(
        [&](const auto i)
        {
            sharedBuffer[idxOffset + base_idx + (static_cast<device::label_t>(i) * faceArea<alpha>())] = pop[q_i<streaming_index<alpha, coeff>(i)>()];
        });
}

/**
 * @brief Helper function for smemOffset
 * @tparam FaceIdx Index of the face (0-5) for which the preceding areas are summed
 * @return Sum of the face areas for all faces before @p FaceIdx
 **/
template <const int FaceIdx>
__device__ __host__ [[nodiscard]] static inline consteval device::label_t sumFaceAreasBefore() noexcept
{
    return []<host::label_t... I>(std::index_sequence<I...>)
    {
        return (device::label_t{0} + ... + ((I < FaceIdx) ? faceArea<static_cast<axis::type>(I / 2)>() : device::label_t{0}));
    }(std::make_index_sequence<6>{});
}

/**
 * @brief Calculates the offset into shared memory for a particular halo transpose operation
 * @tparam alpha The axis direction (X, Y or Z)
 * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
 * @return Offset (in elements) to the beginning of the halo data for this axis/direction
 **/
template <const axis::type alpha, const int coeff>
__device__ __host__ [[nodiscard]] static inline consteval device::label_t smemOffset() noexcept
{
    axis::assertions::validate<alpha, axis::NOT_NULL>();

    velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

    return VelocitySet::template QF<device::label_t>() * sumFaceAreasBefore<static_cast<int>(alpha) * 2 + (coeff == -1 ? 0 : 1)>();
}

/**
 * @brief Transposes population data in halo regions via the shared memory
 * @tparam alpha The axis direction (X, Y or Z)
 * @param[in] pop Array containing population values to save
 * @param[out] sharedBuffer Inline or externally stored shared memory buffer
 * @param[in] point The global point coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
template <const axis::type alpha, class SharedBuffer>
__device__ static inline constexpr void transpose_direction(
    const thread::array<scalar_t, VelocitySet::Q()> &pop,
    SharedBuffer &sharedBuffer,
    const device::pointCoordinate &point,
    const thread::coordinate &Tx) noexcept
{
    axis::assertions::validate<alpha, axis::NOT_NULL>();

    if (boundaryCheck<alpha, -1, BoundaryConditions::periodic<alpha>()>(point.value<alpha>(), Tx))
    {
        transpose<alpha, -1, smemOffset<alpha, -1>()>(Tx, pop, sharedBuffer);
    }
    else if (boundaryCheck<alpha, +1, BoundaryConditions::periodic<alpha>()>(point.value<alpha>(), Tx))
    {
        transpose<alpha, +1, smemOffset<alpha, +1>()>(Tx, pop, sharedBuffer);
    }
}

/**
 * @brief Transposes the block halo into the shared memory for X and Y axes, saves the Z halo
 * @param[in] pop Array containing the populations for the particular thread
 * @param[out] writeBuffer Collection of pointers to the halo faces
 * @param[out] sharedBuffer Inline or externally stored shared memory buffer
 * @param[in] Tx Three-dimensional thread coordinates
 * @param[in] Bx Three-dimensional block coordinates
 * @param[in] point The global point coordinate
 **/
template <class SharedBuffer>
__device__ static inline constexpr void transpose_to_shared(
    const thread::array<scalar_t, VelocitySet::Q()> &pop,
    const device::ptrCollection<6, scalar_t> &writeBuffer,
    SharedBuffer &sharedBuffer,
    const thread::coordinate &Tx,
    const block::coordinate &Bx,
    const device::pointCoordinate &point) noexcept
{
    // X axis halo transposition
    transpose_direction<axis::X>(pop, sharedBuffer, point, Tx);

    // Y axis halo transposition
    transpose_direction<axis::Y>(pop, sharedBuffer, point, Tx);

    block::sync();

    // Z halos: these halos coalesce naturally, so no transposition is needed
    save_direction<axis::Z>(pop, writeBuffer, Tx, Bx, point);
}

/**
 * @brief Compute the number of channels for the given block set
 **/
__device__ __host__ [[nodiscard]] static inline consteval device::label_t n_channels() noexcept
{
    return block::n_warps() / warps_per_face();
}

/**
 * @brief Precomputes the axis from warp index and warp cycle
 * @tparam warpIdx The warp index within the block
 * @tparam warpCycle The current warp cycle in the save-from-shared procedure
 * @return The axis corresponding to this warp/cycle combination
 **/
template <const host::label_t warpIdx, const host::label_t warpCycle>
__device__ __host__ [[nodiscard]] static inline consteval axis::type precompute_axis() noexcept
{
    constexpr const host::label_t result = (warpIdx + (warpCycle * n_channels())) / (VelocitySet::template QF<host::label_t>() * warps_per_face());
    axis::assertions::validate<static_cast<axis::type>(result), axis::NOT_NULL>();
    return static_cast<axis::type>(result);
}

/**
 * @brief Precomputes the population index (q) from lane index and warp cycle
 * @tparam idx The lane index within the warp cycle (0-7)
 * @tparam warpCycle The current warp cycle in the save-from-shared procedure
 * @return The population index for this lane/cycle pair
 **/
template <const host::label_t idx, const host::label_t warpCycle>
__device__ __host__ [[nodiscard]] static inline consteval host::label_t precompute_q() noexcept
{
    constexpr const host::label_t result = (idx + (warpCycle * n_channels())) % VelocitySet::template QF<host::label_t>();
    static_assert(result < VelocitySet::template QF<host::label_t>());
    return result;
}

/**
 * @brief Picks the coordinate pair based on the axis
 * @tparam alpha The axis direction (X or Y)
 * @param[in] x The coordinate pair on the X-face
 * @param[in] y The coordinate pair on the Y-face
 * @return The coordinate pair corresponding to axis @p alpha
 **/
template <const axis::type alpha>
__device__ [[nodiscard]] static inline constexpr const dim2 &choose_axis(const dim2 &x, const dim2 &y) noexcept
{
    if constexpr (alpha == axis::X)
    {
        return x;
    }

    if constexpr (alpha == axis::Y)
    {
        return y;
    }
}

/**
 * @brief Determines the buffer index (which halo face) to write to
 * @tparam warpIdx The warp index within the block
 * @param[in] c The shared memory channel (warp group ID)
 * @return Index into the write buffer for this warp/channel combination
 **/
template <const device::label_t warpIdx>
__device__ [[nodiscard]] static inline constexpr device::label_t bufferIdx(const device::label_t c) noexcept
{
    return (c + (warpIdx * n_channels())) / VelocitySet::template QF<device::label_t>();
}

/**
 * @brief Calculates the shared memory stride (padded to avoid bank conflicts)
 * @return Padded stride in elements
 **/
__device__ [[nodiscard]] static inline consteval device::label_t padded_stride() noexcept
{
    return block::size() + static_cast<device::label_t>(0);
}

/**
 * @brief Perform a save cycle from the shared memory
 * @tparam i Warp cycle index
 * @tparam SharedBuffer Type of the shared memory buffer
 * @param[in] yz Coordinates on the X-face
 * @param[in] xz Coordinates on the Y-face
 * @param[in] Bx Three-dimensional block coordinates
 * @param[out] writeBuffer Collection of pointers to the halo faces
 * @param[in] sharedBuffer Shared array containing the packed population halos
 * @param[in] ID Linear index of the thread within the block
 * @param[in] c Shared memory channel (warp group ID)
 **/
template <const device::label_t i, class SharedBuffer>
__device__ static inline void store_lane(
    const dim2 &yz,
    const dim2 &xz,
    const block::coordinate &Bx,
    const device::ptrCollection<6, scalar_t> &writeBuffer,
    const SharedBuffer &sharedBuffer,
    const device::label_t ID,
    const device::label_t c) noexcept
{
    const thread::array<const device::label_t, n_channels()> lane{
        idxPop<precompute_axis<0, i>(), precompute_q<0, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<0, i>()>(yz, xz), Bx), // case 0
        idxPop<precompute_axis<1, i>(), precompute_q<1, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<1, i>()>(yz, xz), Bx), // case 1
        idxPop<precompute_axis<2, i>(), precompute_q<2, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<2, i>()>(yz, xz), Bx), // case 2
        idxPop<precompute_axis<3, i>(), precompute_q<3, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<3, i>()>(yz, xz), Bx), // case 3
        idxPop<precompute_axis<4, i>(), precompute_q<4, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<4, i>()>(yz, xz), Bx), // case 4
        idxPop<precompute_axis<5, i>(), precompute_q<5, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<5, i>()>(yz, xz), Bx), // case 5
        idxPop<precompute_axis<6, i>(), precompute_q<6, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<6, i>()>(yz, xz), Bx), // case 6
        idxPop<precompute_axis<7, i>(), precompute_q<7, i>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<7, i>()>(yz, xz), Bx)  // case 7
    };

    writeBuffer.ptr(bufferIdx<i>(c))[lane[c]] = sharedBuffer[ID + (i * padded_stride())];
}

template <class SharedBuffer>
__device__ static inline void store_final_lane(
    const dim2 &yz,
    const dim2 &xz,
    const block::coordinate &Bx,
    const device::ptrCollection<6, scalar_t> &writeBuffer,
    const SharedBuffer &sharedBuffer,
    const device::label_t ID,
    const device::label_t c) noexcept
{
    const thread::array<const device::label_t, 4> lane{
        idxPop<precompute_axis<0, n_cycles() - static_cast<device::label_t>(1)>(), precompute_q<0, n_cycles() - static_cast<device::label_t>(1)>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<0, n_cycles() - static_cast<device::label_t>(1)>()>(yz, xz), Bx), // case 0
        idxPop<precompute_axis<1, n_cycles() - static_cast<device::label_t>(1)>(), precompute_q<1, n_cycles() - static_cast<device::label_t>(1)>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<1, n_cycles() - static_cast<device::label_t>(1)>()>(yz, xz), Bx), // case 1
        idxPop<precompute_axis<2, n_cycles() - static_cast<device::label_t>(1)>(), precompute_q<2, n_cycles() - static_cast<device::label_t>(1)>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<2, n_cycles() - static_cast<device::label_t>(1)>()>(yz, xz), Bx), // case 2
        idxPop<precompute_axis<3, n_cycles() - static_cast<device::label_t>(1)>(), precompute_q<3, n_cycles() - static_cast<device::label_t>(1)>(), VelocitySet::template QF<device::label_t>()>(choose_axis<precompute_axis<3, n_cycles() - static_cast<device::label_t>(1)>()>(yz, xz), Bx), // case 3
    };

    writeBuffer.ptr(bufferIdx<n_cycles() - static_cast<device::label_t>(1)>(c))[lane[c]] = sharedBuffer[ID + (static_cast<device::label_t>(n_cycles() - static_cast<device::label_t>(1)) * padded_stride())];
}

/**
 * @brief Calculate the number of shared memory loading cycles for the given velocity set
 * @return Number of cycles needed to save all populations per warp group
 **/
__device__ [[nodiscard]] static inline consteval device::label_t warps_per_face() noexcept
{
    return static_cast<device::label_t>(2);
}

/**
 * @brief Calculate the number of shared memory loading cycles for the given velocity set
 * @return Number of cycles needed to save all populations per warp group
 **/
__device__ [[nodiscard]] static inline consteval device::label_t n_cycles() noexcept
{
    return (static_cast<device::label_t>(4) * VelocitySet::template QF<device::label_t>() + n_channels() - static_cast<device::label_t>(1)) / n_channels();
}

/**
 * @brief Saves population data to halo regions for neighboring blocks
 * @tparam SharedBuffer Type of the shared memory buffer
 * @param[in] sharedBuffer Shared array containing the packed population halos
 * @param[out] writeBuffer Collection of pointers to the halo faces
 * @param[in] Tx Three-dimensional thread coordinates
 * @param[in] Bx Three-dimensional block coordinates
 * @note This device function saves population values to halo regions for neighboring blocks to read.
 **/
template <class SharedBuffer>
__device__ static inline constexpr void save_from_shared(
    const SharedBuffer &sharedBuffer,
    const device::ptrCollection<6, scalar_t> &writeBuffer,
    const thread::coordinate &Tx,
    const block::coordinate &Bx) noexcept
{
    const device::label_t warpId = warpID(Tx);
    const device::label_t offset = block::warp_size() * (warpId % warps_per_face());
    const device::label_t idx_in_warp = idxWarp(Tx);

    // Equivalent of threadIdx.alpha, threadIdx.beta
    const dim2 yz(dim2::i<axis::X>(idx_in_warp + offset), dim2::j<axis::X>(idx_in_warp + offset));
    const dim2 xz(dim2::i<axis::Y>(idx_in_warp + offset), dim2::j<axis::Y>(idx_in_warp + offset));

    // Get the address into the shared memory
    const device::label_t ID = idx_block(Tx);

    // Calculate the channel
    const device::label_t c = warpId / warps_per_face();

    // Store all the full cycles
    device::constexpr_for<0, n_cycles() - static_cast<device::label_t>(1)>(
        [&](const auto cycle)
        {
            store_lane<cycle>(yz, xz, Bx, writeBuffer, sharedBuffer, ID, c);
        });

    // Early return for the second half of the last cycle
    if (c >= 4)
    {
        return;
    }
    else
    {
        store_final_lane(yz, xz, Bx, writeBuffer, sharedBuffer, ID, c);
    }
}

#endif