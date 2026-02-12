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
    A class used to compute the normal vector of a boundary lattice

Namespace
    LBM

SourceFiles
    normalVector.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_NORMALVECTOR_CUH
#define __MBLBM_NORMALVECTOR_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"

namespace LBM
{
    /**
     * @class normalVector
     * @brief Represents boundary orientation using a bitmask encoding
     *
     * This class uses a compact bitmask representation to encode the position
     * of lattice nodes relative to domain boundaries. It supports detection of:
     * - Individual boundary faces (West, East, South, North, Back, Front)
     * - Edge configurations (12 possible combinations)
     * - Corner configurations (8 possible combinations)
     * - Interior points (no boundaries)
     *
     * The bitmask uses a 7-bit representation where:
     * - Bits 0-5: Individual boundary flags
     * - Bit 6: General boundary indicator (any boundary)
     **/
    class normalVector
    {
    public:
        /**
         * @brief Constructs a normalVector from current thread indices
         * @return normalVector for the current thread's position
         * @param[in] Tx Thread coordinates
         * @param[in] Bx Block coordinates
         **/
        __device__ [[nodiscard]] inline constexpr normalVector(const device::pointCoordinate &point) noexcept
            : bitmask_(computeBitmask(point)){};

        /**
         * @name Basic Boundary Flags
         * @brief Bitmask values for individual boundary faces
         * @return Bitmask value for the specified boundary face
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t WEST() noexcept
        {
            return 0x01;
        } // 1 << 0
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t EAST() noexcept
        {
            return 0x02;
        } // 1 << 1
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH() noexcept
        {
            return 0x04;
        } // 1 << 2
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH() noexcept
        {
            return 0x08;
        } // 1 << 3
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t BACK() noexcept
        {
            return 0x10;
        } // 1 << 4
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t FRONT() noexcept
        {
            return 0x20;
        } // 1 << 5

        /**
         * @name Corner Boundary Types
         * @brief Bitmask values for corner configurations (8 types)
         * @return Bitmask value for the specified corner configuration
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST_BACK() noexcept
        {
            return SOUTH() | WEST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST_FRONT() noexcept
        {
            return SOUTH() | WEST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST_BACK() noexcept
        {
            return SOUTH() | EAST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST_FRONT() noexcept
        {
            return SOUTH() | EAST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST_BACK() noexcept
        {
            return NORTH() | WEST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST_FRONT() noexcept
        {
            return NORTH() | WEST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST_BACK() noexcept
        {
            return NORTH() | EAST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST_FRONT() noexcept
        {
            return NORTH() | EAST() | FRONT();
        }

        /**
         * @name Edge Boundary Types
         * @brief Bitmask values for edge configurations (12 types)
         * @return Bitmask value for the specified edge configuration
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_WEST() noexcept
        {
            return SOUTH() | WEST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_EAST() noexcept
        {
            return SOUTH() | EAST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_WEST() noexcept
        {
            return NORTH() | WEST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_EAST() noexcept
        {
            return NORTH() | EAST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t WEST_BACK() noexcept
        {
            return WEST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t WEST_FRONT() noexcept
        {
            return WEST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t EAST_BACK() noexcept
        {
            return EAST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t EAST_FRONT() noexcept
        {
            return EAST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_BACK() noexcept
        {
            return SOUTH() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t SOUTH_FRONT() noexcept
        {
            return SOUTH() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_BACK() noexcept
        {
            return NORTH() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t NORTH_FRONT() noexcept
        {
            return NORTH() | FRONT();
        }

        /**
         * @brief Special type for interior points
         * @return Bitmask value for interior points (no boundaries)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval uint8_t INTERIOR() noexcept
        {
            return 0x00;
        }

        /**
         * @name Boundary Detection Methods
         * @brief Check if the point lies on specific boundaries
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isWest() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & WEST()));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isEast() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & EAST()));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isSouth() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & SOUTH()));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isNorth() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & NORTH()));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isBack() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & BACK()));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isFront() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & FRONT()));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isBoundary() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & 0x40));
        }
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline T isInterior() const noexcept
        {
            return static_cast<T>(!isBoundary<bool>());
        }
        template <typename T = uint8_t>
        __device__ __host__ [[nodiscard]] inline T countBoundaries() const noexcept
        {
            // Count set bits in 6-bit value using parallel addition
            // This is known as the "popcount" algorithm for small integers
            uint8_t x = bitmask_ & 0x3F;
            x = (x & 0x55) + ((x >> 1) & 0x55); // Count bits in pairs
            x = (x & 0x33) + ((x >> 2) & 0x33); // Count bits in nibbles
            x = (x & 0x0F) + ((x >> 4) & 0x0F); // Add the two nibbles
            return static_cast<T>(x);
        }

        /**
         * @brief Get the node type bitmask
         * @return The bitmask representing the node type (bits 0-5)
         **/
        __device__ [[nodiscard]] inline uint8_t nodeType() const noexcept
        {
            return bitmask_ & 0x3F;
        }

    private:
        /**
         * @brief The underlying bit mask
         **/
        const uint8_t bitmask_;

        /**
         * @brief Compute bitmask from current thread indices
         * @return Bitmask representing boundary configuration
         **/
        __device__ [[nodiscard]] static inline constexpr uint8_t computeBitmask(const device::pointCoordinate &point) noexcept
        {
            return computeBitmask(point.value<axis::X>(), point.value<axis::Y>(), point.value<axis::Z>());
        }

        /**
         * @brief Compute bitmask from specific coordinates
         * @param[in] x X-coordinate in the lattice
         * @param[in] y Y-coordinate in the lattice
         * @param[in] z Z-coordinate in the lattice
         * @return Bitmask representing boundary configuration
         *
         * The bitmask is constructed as follows:
         * - Bit 0: West boundary (x == 0)
         * - Bit 1: East boundary (x == device::nx - 1)
         * - Bit 2: South boundary (y == 0)
         * - Bit 3: North boundary (y == device::ny - 1)
         * - Bit 4: Back boundary (z == 0)
         * - Bit 5: Front boundary (z == device::nz - 1)
         * - Bit 6: Any boundary (logical OR of bits 0-5)
         **/
        __device__ [[nodiscard]] static inline constexpr uint8_t computeBitmask(const label_t x, const label_t y, const label_t z) noexcept
        {
            static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(normalVector::computeBitmask, "Believed correct"));
            return static_cast<uint8_t>(
                (x == 0) << 0 |                                // West (bit0)
                (x == device::n<axis::X>() - 1) << 1 |         // East (bit1)
                (y == 0) << 2 |                                // South (bit2)
                (y == device::n<axis::Y>() - 1) << 3 |         // North (bit3)
                (z == 0) << 4 |                                // Back (bit4)
                (z == device::n<axis::Z>() - 1) << 5 |         // Front (bit5)
                (!!(x == 0 || x == device::n<axis::X>() - 1 || //
                    y == 0 || y == device::n<axis::Y>() - 1 || //
                    z == 0 || z == device::n<axis::Z>() - 1))  //
                    << 6);                                     // Any boundary (bit6)
        }
    };
}

#endif