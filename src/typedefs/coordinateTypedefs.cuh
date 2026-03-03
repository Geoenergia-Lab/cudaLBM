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
    A list of integral typedefs used throughout the cudaLBM source code

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
         * @tparam alpha The axis
         **/
        template <axis::type alpha>
        __device__ [[nodiscard]] inline constexpr label_t n() noexcept
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
         * @tparam alpha The axis
         **/
        template <axis::type alpha>
        __device__ [[nodiscard]] inline constexpr label_t NUM_BLOCK() noexcept
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
         * @tparam alpha The axis direction
         * @tparam coeff The axis normal coefficient
         * @tparam ValueType The return type (defualt label_t)
         * @returns One of two thread coordinates that lie on the extremities of alpha within the block
         **/
        template <const axis::type alpha, const int coeff, typename ValueType = label_t>
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
         */
        struct coordinate : public var3<label_t>
        {
        public:
            /**
             * @brief Constructs from threadIdx
             **/
            __device__ [[nodiscard]] inline explicit coordinate() noexcept
                : var3<label_t>(
                      static_cast<label_t>(threadIdx.x),
                      static_cast<label_t>(threadIdx.y),
                      static_cast<label_t>(threadIdx.z)) {}

            /**
             * @brief Shifts the coordinate along a particular axis by a coefficient
             * @tparam alpha The axis
             * @tparam coeff The coefficient to shift by (-1, 0 or +1)
             **/
            template <axis::type alpha, const int coeff>
            __device__ [[nodiscard]] inline constexpr label_t shifted_coordinate() const noexcept
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
        struct coordinate : public var3<label_t>
        {
        public:
            /**
             * @brief Constructs from blockIdx
             **/
            __device__ [[nodiscard]] inline explicit coordinate() noexcept
                : var3<label_t>(
                      static_cast<label_t>(blockIdx.x),
                      static_cast<label_t>(blockIdx.y),
                      static_cast<label_t>(blockIdx.z)) {}

            /**
             * @brief Shifts the coordinate along a particular axis by a coefficient
             * @tparam alpha The axis
             * @tparam coeff The coefficient to shift by (-1, 0 or +1)
             **/
            template <axis::type alpha, const int coeff>
            __device__ [[nodiscard]] inline constexpr label_t shifted_block() const noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();
                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                if constexpr (coeff == -1)
                {
                    return (value<alpha>() - 1 + device::NUM_BLOCK<alpha>()) % device::NUM_BLOCK<alpha>();
                }

                if constexpr (coeff == 0)
                {
                    return value<alpha>();
                }

                if constexpr (coeff == 1)
                {
                    return (value<alpha>() + 1 + device::NUM_BLOCK<alpha>()) % device::NUM_BLOCK<alpha>();
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
        struct pointCoordinate : public var3<label_t>
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
                : var3<label_t>(
                      Tx.value<axis::X>() + block::nx<label_t>() * (Bx.value<axis::X>() + device::BLOCK_OFFSET_X),
                      Tx.value<axis::Y>() + block::ny<label_t>() * (Bx.value<axis::Y>() + device::BLOCK_OFFSET_Y),
                      Tx.value<axis::Z>() + block::nz<label_t>() * (Bx.value<axis::Z>() + device::BLOCK_OFFSET_Z)) {}
        };
    }

    /**
     * @brief Struct used to represent 2D indices in a more readable way
     **/
    template <const axis::type alpha>
    class dim2
    {
    public:
        /**
         * @brief Constructs from a linear index of a flattened 2D array with dimensions (block::n<alpha>(), block::n<beta>())
         * @param[in] linearIdx The linear index to convert to 2D indices
         **/
        __device__ __host__ [[nodiscard]] inline constexpr dim2(const label_t linearIdx) noexcept
            : i_(linearIdx % (block::n<axis::orthogonal<alpha, 0>()>())),
              j_(linearIdx / (block::n<axis::orthogonal<alpha, 0>()>()))
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();
        };

        __device__ __host__ [[nodiscard]] inline constexpr dim2(const label_t a, const label_t b) noexcept
            : i_(a),
              j_(b)
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();
        };

        __device__ __host__ [[nodiscard]] inline constexpr label_t i() const noexcept
        {
            return i_;
        }

        __device__ __host__ [[nodiscard]] inline constexpr label_t j() const noexcept
        {
            return j_;
        }

    private:
        const label_t i_;
        const label_t j_;
    };
} // namespace LBM

#endif