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
    A class used to compute the normal vector of a boundary lattice

Namespace
    LBM

SourceFiles
    normalVector.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_NORMALVECTOR_CUH
#define __MBLBM_NORMALVECTOR_CUH

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
    template <const bool periodicX, const bool periodicY, const bool periodicZ>
    class normalVector
    {
    public:
        /**
         * @brief Constructs a normalVector from current thread indices
         * @param[in] point The spatial coordinate of the point
         * @return normalVector for the current thread's position
         **/
        __device__ [[nodiscard]] inline constexpr normalVector(const device::pointCoordinate &point) noexcept
            : bitmask_(computeBitmask(point)) {}

        /**
         * @name Basic Boundary Flags
         * @brief Bitmask values for individual boundary faces
         * @return Bitmask value for the specified boundary face
         **/
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t WEST() noexcept
        {
            return 0x01;
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t EAST() noexcept
        {
            return 0x02;
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH() noexcept
        {
            return 0x04;
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH() noexcept
        {
            return 0x08;
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t BACK() noexcept
        {
            return 0x10;
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t FRONT() noexcept
        {
            return 0x20;
        }

        /**
         * @name Corner Boundary Types
         * @brief Bitmask values for corner configurations (8 types)
         * @return Bitmask value for the specified corner configuration
         **/
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_WEST_BACK() noexcept
        {
            return SOUTH() | WEST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_WEST_FRONT() noexcept
        {
            return SOUTH() | WEST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_EAST_BACK() noexcept
        {
            return SOUTH() | EAST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_EAST_FRONT() noexcept
        {
            return SOUTH() | EAST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_WEST_BACK() noexcept
        {
            return NORTH() | WEST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_WEST_FRONT() noexcept
        {
            return NORTH() | WEST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_EAST_BACK() noexcept
        {
            return NORTH() | EAST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_EAST_FRONT() noexcept
        {
            return NORTH() | EAST() | FRONT();
        }

        /**
         * @name Edge Boundary Types
         * @brief Bitmask values for edge configurations (12 types)
         * @return Bitmask value for the specified edge configuration
         **/
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_WEST() noexcept
        {
            return SOUTH() | WEST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_EAST() noexcept
        {
            return SOUTH() | EAST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_WEST() noexcept
        {
            return NORTH() | WEST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_EAST() noexcept
        {
            return NORTH() | EAST();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t WEST_BACK() noexcept
        {
            return WEST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t WEST_FRONT() noexcept
        {
            return WEST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t EAST_BACK() noexcept
        {
            return EAST() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t EAST_FRONT() noexcept
        {
            return EAST() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_BACK() noexcept
        {
            return SOUTH() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t SOUTH_FRONT() noexcept
        {
            return SOUTH() | FRONT();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_BACK() noexcept
        {
            return NORTH() | BACK();
        }
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t NORTH_FRONT() noexcept
        {
            return NORTH() | FRONT();
        }

        /**
         * @brief Special type for interior points
         * @return Bitmask value for interior points (no boundaries)
         **/
        __device__ __host__ [[nodiscard]] static inline consteval nodeType_t INTERIOR() noexcept
        {
            return 0x00;
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isWest() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & WEST()));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isEast() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & EAST()));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isSouth() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & SOUTH()));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isNorth() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & NORTH()));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isBack() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & BACK()));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isFront() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & FRONT()));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isBoundary() const noexcept
        {
            return static_cast<T>(static_cast<bool>(bitmask_ & 0x40));
        }

        /**
         * @name Boundary detection
         * @tparam T The return type
         * @return True if the point lies on the specified boundary
         **/
        template <typename T = bool>
        __device__ __host__ [[nodiscard]] inline constexpr T isInterior() const noexcept
        {
            return static_cast<T>(!isBoundary<bool>());
        }

        /**
         * @name Count the number of intersecting boundary planes at a point
         * @tparam T The return type
         * @return Number of boundary planes that intersect a point
         **/
        template <typename T = nodeType_t>
        __device__ __host__ [[nodiscard]] inline constexpr T countBoundaries() const noexcept
        {
            // Count set bits in 6-bit value using parallel addition
            // This is known as the "popcount" algorithm for small integers
            nodeType_t x = bitmask_ & 0x3F;
            x = (x & 0x55) + ((x >> 1) & 0x55); // Count bits in pairs
            x = (x & 0x33) + ((x >> 2) & 0x33); // Count bits in nibbles
            x = (x & 0x0F) + ((x >> 4) & 0x0F); // Add the two nibbles
            return static_cast<T>(x);
        }

        /**
         * @brief Get the node type bitmask
         * @return The bitmask representing the node type (bits 0-5)
         **/
        __device__ [[nodiscard]] inline constexpr nodeType_t nodeType() const noexcept
        {
            return bitmask_ & 0x3F;
        }

    private:
        /**
         * @brief The underlying bit mask
         **/
        const nodeType_t bitmask_;

        /**
         * @brief Compute bitmask from current thread indices
         * @param[in] point The spatial coordinate of the point
         * @return Bitmask representing boundary configuration
         **/
        __device__ [[nodiscard]] static inline constexpr nodeType_t computeBitmask(const device::pointCoordinate &point) noexcept
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
        __device__ [[nodiscard]] static inline constexpr nodeType_t computeBitmask(const device::label_t x, const device::label_t y, const device::label_t z) noexcept
        {
            const bool west = isBoundary<axis::X, -1, periodicX>(x);
            const bool east = isBoundary<axis::X, +1, periodicX>(x);
            const bool south = isBoundary<axis::Y, -1, periodicY>(y);
            const bool north = isBoundary<axis::Y, +1, periodicY>(y);
            const bool back = isBoundary<axis::Z, -1, periodicZ>(z);
            const bool front = isBoundary<axis::Z, +1, periodicZ>(z);
            const bool anyBoundary = west || east || south || north || back || front;

            return static_cast<nodeType_t>((west << 0) | (east << 1) | (south << 2) | (north << 3) | (back << 4) | (front << 5) | (anyBoundary << 6));
        }

        template <const axis::type alpha, const int coeff, const bool periodic>
        __device__ [[nodiscard]] static inline constexpr bool isBoundary(const device::label_t i) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();
            if constexpr (periodic)
            {
                return false;
            }
            else
            {
                if constexpr (coeff == -1)
                {
                    return (i == static_cast<device::label_t>(0));
                }
                else if constexpr (coeff == 1)
                {
                    return (i == device::n<alpha>() - static_cast<device::label_t>(1));
                }
            }
        }
    };
}

#endif