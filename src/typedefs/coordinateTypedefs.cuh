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
    A list of integral typedefs used throughout the HermiteLBM source code

Namespace
    LBM

SourceFiles
    coordinateTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_COORDINATETYPEDEFS_CUH
#define __MBLBM_COORDINATETYPEDEFS_CUH

#include "../blockConfig.cuh"
#include "../globalConstants.cuh"

namespace LBM
{
    namespace device
    {
        /**
         * @brief Returns the global mesh size in a particular axis direction
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <const axis::type alpha>
        __device__ [[nodiscard]] inline constexpr device::label_t n() noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            if constexpr (alpha == axis::X)
            {
                return nx;
            }

            if constexpr (alpha == axis::Y)
            {
                return ny;
            }

            if constexpr (alpha == axis::Z)
            {
                return nz;
            }
        }

        /**
         * @brief Returns the number of mesh blocks per GPU in a particular axis direction
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <const axis::type alpha>
        __device__ [[nodiscard]] inline constexpr device::label_t NUM_BLOCK() noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            if constexpr (alpha == axis::X)
            {
                return NUM_BLOCK_X;
            }

            if constexpr (alpha == axis::Y)
            {
                return NUM_BLOCK_Y;
            }

            if constexpr (alpha == axis::Z)
            {
                return NUM_BLOCK_Z;
            }
        }
    }

    namespace thread
    {
        /**
         * @brief Returns the thread that lies on a particular boundary
         * @tparam alpha The axis direction (X, Y or Z)
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @tparam ValueType The return type (defualt device::label_t)
         * @returns One of two thread coordinates that lie on the extremities of alpha within the block
         **/
        template <const axis::type alpha, const int coeff, typename ValueType = device::label_t>
        __host__ [[nodiscard]] inline consteval ValueType boundary() noexcept
        {
            if constexpr (coeff == -1)
            {
                return 0;
            }

            if constexpr (coeff == 1)
            {
                return block::n<alpha, ValueType>() - 1;
            }
        }

        /**
         * @brief Thread coordinate in a 3D grid.
         *
         * Stores the three thread indices (x, y, z) and provides access per axis
         * as well as a method to compute neighbour coordinates with periodic wrap‑around.
         **/
        struct coordinate : public var3<device::label_t>
        {
        public:
            /**
             * @brief Constructs from threadIdx
             **/
            __device__ [[nodiscard]] inline explicit coordinate() noexcept
                : var3<device::label_t>(
                      static_cast<device::label_t>(threadIdx.x),
                      static_cast<device::label_t>(threadIdx.y),
                      static_cast<device::label_t>(threadIdx.z)) {}

            /**
             * @brief Shifts the coordinate along a particular axis by a coefficient
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1, 0 or 1)
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] inline constexpr device::label_t shifted_coordinate() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();
                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                if constexpr (coeff == -1)
                {
                    return (value<alpha>() - 1 + block::n<alpha>()) % block::n<alpha>();
                }

                if constexpr (coeff == 0)
                {
                    return value<alpha>();
                }

                if constexpr (coeff == 1)
                {
                    return (value<alpha>() + 1 + block::n<alpha>()) % block::n<alpha>();
                }
            }
        };
    }

    namespace block
    {
        /**
         * @brief Block coordinate in a 3D grid.
         *
         * Stores the three block indices (x, y, z) and provides access per axis
         * as well as a method to compute neighbour block indices with periodic wrap‑around.
         **/
        struct coordinate : public var3<device::label_t>
        {
        public:
            /**
             * @brief Constructs from blockIdx
             **/
            __device__ [[nodiscard]] inline explicit coordinate() noexcept
                : var3<device::label_t>(
                      static_cast<device::label_t>(blockIdx.x),
                      static_cast<device::label_t>(blockIdx.y),
                      static_cast<device::label_t>(blockIdx.z)) {}

            /**
             * @brief Constructs from an arbitrary input
             **/
            __device__ [[nodiscard]] inline explicit coordinate(
                const device::label_t bx,
                const device::label_t by,
                const device::label_t bz) noexcept
                : var3<device::label_t>(bx, by, bz) {}

            /**
             * @brief Shifts the coordinate along a particular axis by a coefficient
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1, 0 or 1)
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] inline constexpr device::label_t shifted_block() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();
                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                if constexpr (coeff == -1)
                {
                    return (value<alpha>() + device::NUM_BLOCK<alpha>() - static_cast<device::label_t>(1)) % (device::NUM_BLOCK<alpha>());
                }

                if constexpr (coeff == 0)
                {
                    return value<alpha>();
                }

                if constexpr (coeff == +1)
                {
                    return (value<alpha>() + device::NUM_BLOCK<alpha>() + +static_cast<device::label_t>(1)) % (device::NUM_BLOCK<alpha>());
                }
            }
        };
    }

    namespace device
    {
        /**
         * @brief Global point coordinate (lattice site) combining thread and block positions.
         *
         * Stores the absolute x, y, z indices of a lattice cell.
         * The calculation includes block offsets and thread indices scaled by block dimensions.
         **/
        struct pointCoordinate : public var3<device::label_t>
        {
        public:
            /**
             * @brief Constructs from thread and block coordinates
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            __device__ [[nodiscard]] inline explicit pointCoordinate(
                const thread::coordinate &Tx,
                const block::coordinate &Bx) noexcept
                : var3<device::label_t>(
                      Tx.value<axis::X>() + block::nx<device::label_t>() * (Bx.value<axis::X>() + device::BLOCK_OFFSET_X),
                      Tx.value<axis::Y>() + block::ny<device::label_t>() * (Bx.value<axis::Y>() + device::BLOCK_OFFSET_Y),
                      Tx.value<axis::Z>() + block::nz<device::label_t>() * (Bx.value<axis::Z>() + device::BLOCK_OFFSET_Z)) {}
        };
    }

    /**
     * @brief Struct used to represent 2D indices in a more readable way
     **/
    class dim2
    {
    public:
        __device__ __host__ [[nodiscard]] inline constexpr dim2(const device::label_t a, const device::label_t b) noexcept
            : i_(a),
              j_(b){};

        __device__ __host__ [[nodiscard]] dim2(const dim2 &) = delete;
        __device__ __host__ [[nodiscard]] dim2 &operator=(const dim2 &) = delete;

        __device__ __host__ [[nodiscard]] inline constexpr device::label_t i() const noexcept
        {
            return i_;
        }

        template <const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline constexpr device::label_t i(const device::label_t linearIdx) noexcept
        {
            return linearIdx % (block::n<axis::orthogonal<alpha, 0>()>());
        }

        template <const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline constexpr device::label_t j(const device::label_t linearIdx) noexcept
        {
            return linearIdx / (block::n<axis::orthogonal<alpha, 0>()>());
        }

        __device__ __host__ [[nodiscard]] inline constexpr device::label_t j() const noexcept
        {
            return j_;
        }

    private:
        const device::label_t i_;
        const device::label_t j_;
    };
} // namespace LBM

#endif