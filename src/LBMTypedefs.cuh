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
    A list of typedefs used throughout the cudaLBM source code

Namespace
    LBM

SourceFiles
    LBMTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TYPEDEFS_CUH
#define __MBLBM_TYPEDEFS_CUH

#include "LBMIncludes.cuh"
#include "errorHandler.cuh"

namespace LBM
{
    /**
     * @brief Shorthand for __restrict__
     **/
#define ptrRestrict __restrict__

    /**
     * @brief Verbose logging
     **/
    __device__ __host__ [[nodiscard]] inline consteval bool verbose() noexcept
    {
#ifdef VERBOSE
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief Runtime bounds checking for GPU kernels
     **/
    __device__ __host__ [[nodiscard]] inline consteval bool out_of_bounds_check() noexcept
    {
#ifdef OOB_CHECK
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief CUDA implementation of a std::integral constant
     * @param T The type of integral value
     * @param v The value
     **/
    template <typename T, T v>
    struct integralConstant
    {
        static constexpr const T value = v;
        using value_type = T;
        using type = integralConstant;
        __device__ __host__ [[nodiscard]] inline consteval operator value_type() const noexcept { return value; }
        __device__ __host__ [[nodiscard]] inline consteval value_type operator()() const noexcept { return value; }
    };

    /**
     * @brief Floating point type used for scalar types
     * @note Types are either 32 bit or 64 bit floating point numbers
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef SCALAR_PRECISION_32
    typedef float scalar_t;
#elif SCALAR_PRECISION_64
    typedef double scalar_t;
#endif

    /**
     * @brief Struct used to hold three-dimensional coordinates of a point
     **/
    struct pointVector
    {
        const scalar_t x;
        const scalar_t y;
        const scalar_t z;
    };

    /**
     * @brief Label type used for scalar types
     * @note Types are either 32 bit or 64 bit unsigned integers
     * @note These types are supplied via command line defines during compilation
     **/
#ifdef LABEL_SIZE_32
    typedef uint32_t label_t;
#elif LABEL_SIZE_64
    typedef uint64_t label_t;
#endif

    /**
     * @brief Block dimensions descriptor
     * @details Stores lattice dimensions in 3D space
     **/
    struct blockLabel_t
    {
        const label_t nx; // < Lattice points in x-direction
        const label_t ny; // < Lattice points in y-direction
        const label_t nz; // < Lattice points in z-direction

        void print(const std::string &name) const noexcept
        {
            std::cout << "    " << name << " = [" << nx << ", " << ny << ", " << nz << "];" << std::endl;
        }
    };

    /**
     * @brief Point indices descriptor
     * @details Stores point indices in 3D space
     **/
    struct pointLabel_t
    {
        const label_t x; // < Lattice point in x-direction
        const label_t y; // < Lattice point in y-direction
        const label_t z; // < Lattice point in z-direction
    };

    /**
     * @brief 1D range descriptor [begin, end)
     **/
    struct blockPartitionRange_t
    {
        const label_t begin; // < Inclusive start index
        const label_t end;   // < Exclusive end index
    };

    /**
     * @brief 3D block range descriptor
     * @details Defines a rectangular region in lattice space
     **/
    struct blockRange_t
    {
        const blockPartitionRange_t xRange; // < X-dimension range
        const blockPartitionRange_t yRange; // < Y-dimension range
        const blockPartitionRange_t zRange; // < Z-dimension range
    };

    /**
     * @brief Type used to contain a variable or variables over a block of shared memory
     * @param T The type of variable
     * @param N The number of variables
     * @param blockSize The number of lattice points in the shared memory block
     **/
    template <typename T, const label_t N, const label_t blockSize>
    struct sharedArray
    {
        T arr[N][blockSize];
    };

    /**
     * @brief Type used for compile-time indices
     **/
    template <const label_t label>
    using label_constant = const integralConstant<label_t, label>;
    template <const std::size_t label>
    using size_constant = const integralConstant<std::size_t, label>;
    template <const label_t label>
    using q_i = const integralConstant<label_t, label>;
    template <const label_t label>
    using m_i = const integralConstant<label_t, label>;

    /**
     * @brief Type used for lattice indices
     **/
    template <const label_t q_>
    using lattice_constant = const std::integral_constant<label_t, q_>;

    /**
     * @brief Label type used for GPU indices
     * @note Has to be int because cudaSetDevice operates on int
     **/
    typedef int deviceIndex_t;

    /**
     * @brief Label type used for MPI ranks
     * @note Has to be int because MPI_Comm_rank and etc take &int
     **/
    typedef int mpiRank_t;

    /**
     * @brief Constructor read types
     * @note Has to be enumerated because there are only so many possible read configurations
     **/
    namespace ctorType
    {
        typedef enum Enum : int
        {
            NO_READ = 0,
            MUST_READ = 1,
            READ_IF_PRESENT = 2
        } type;
    }

    /**
     * @brief Time stepping types: instantaneous or time-averaged
     */
    namespace time
    {
        typedef enum Enum : int
        {
            instantaneous = 0,
            timeAverage = 1
        } type;
    }

    namespace axis
    {
        /**
         * @brief Cardinal axis directions: X, Y, Z or NO_DIRECTION
         */
        typedef enum Enum : label_t
        {
            X = 0,
            Y = 1,
            Z = 2,
            NO_DIRECTION = static_cast<label_t>(-1)
        } type;

        /**
         * @brief Enumerated type for axes: The axis either can or cannot be null
         */
        typedef enum nullEnum : bool
        {
            NOT_NULL = false,
            CAN_BE_NULL = true
        } null;
    }

    namespace field
    {
        /**
         * @brief Type of field to be allocated
         * @note The skeleton type contains only a pointer;
         * FULL_FIELD contains a pointer, name and a reference to the mesh
         */
        typedef enum Enum : bool
        {
            SKELETON = 0,
            FULL_FIELD = 1
        } type;
    }

    namespace host
    {
        /**
         * @brief Type of memory allocation on the host:
         * The memory is either pageable or pinned
         */
        typedef enum Enum : bool
        {
            PAGED = 0,
            PINNED = 1
        } mallocType;
    }

    namespace assertions
    {
        namespace axis
        {
            /**
             * @brief Asserts that the direction alpha is a valid axis direction
             * @tparam alpha The axis direction
             * @tparam potentialNull Switch that determines whether alpha is allowed to be NO_DIRECTION or not
             **/
            template <const LBM::axis::type alpha, const LBM::axis::null null>
            __device__ __host__ inline consteval void validate() noexcept
            {
                if constexpr (null == LBM::axis::CAN_BE_NULL)
                {
                    static_assert(((alpha == LBM::axis::X) || (alpha == LBM::axis::Y) || (alpha == LBM::axis::Z) || (alpha == LBM::axis::NO_DIRECTION)), "Axis direction must be X, Y or Z");
                }
                else
                {
                    static_assert(((alpha == LBM::axis::X) || (alpha == LBM::axis::Y) || (alpha == LBM::axis::Z)), "Axis direction must be X, Y, Z or NO_DIRECTION");
                }
            }
        }
    }

    struct dim2
    {
        const label_t i;
        const label_t j;
    };

    namespace device
    {
        /**
         * @brief Device constant variables
         * @note These variables MUST be initialised on the GPU at program start with cudaMemcpyToSymbol
         **/
        __device__ __constant__ label_t nx;
        __device__ __constant__ label_t ny;
        __device__ __constant__ label_t nz;
        __device__ __constant__ scalar_t tau;
        __device__ __constant__ scalar_t L_char;

        __device__ __constant__ scalar_t U_North[3];
        __device__ __constant__ scalar_t U_South[3];
        __device__ __constant__ scalar_t U_East[3];
        __device__ __constant__ scalar_t U_West[3];
        __device__ __constant__ scalar_t U_Back[3];
        __device__ __constant__ scalar_t U_Front[3];

        __device__ __constant__ scalar_t omega;
        __device__ __constant__ scalar_t t_omegaVar;
        __device__ __constant__ scalar_t omegaVar_d2;

        __device__ __constant__ label_t NUM_BLOCK_X;
        __device__ __constant__ label_t NUM_BLOCK_Y;
        __device__ __constant__ label_t NUM_BLOCK_Z;

        // Extra for multiphase
        __device__ __constant__ scalar_t tauA;
        __device__ __constant__ scalar_t tauB;
        __device__ __constant__ scalar_t sigma;
        __device__ __constant__ scalar_t gamma;

        /**
         * @brief Class holding N device pointers of type T
         **/
        template <label_t N, typename T>
        class ptrCollection
        {
        public:
            static_assert(N > 0, "N must be positive"); // Ensure N is valid

            /**
             * @brief Variadic constructor: construct from an arbitrary number of pointers
             * @return A pointer collection object constructed from args
             * @param args An arbitrary number N of pointers of type T
             **/
            template <typename... Args>
            __device__ __host__ constexpr ptrCollection(const Args... args)
                : ptrs_{args...} // Initialize array with arguments
            {
                static_assert(sizeof...(Args) == N, "Incorrect number of arguments");

                static_assert((std::is_convertible_v<Args, T *> && ...), "All arguments must be convertible to T*");
            }

            /**
             * @brief Provides access to the GPU pointer
             * @param i The index of the pointer
             **/
            template <const label_t i>
            __device__ __host__ [[nodiscard]] inline constexpr T *ptr() const noexcept
            {
                static_assert(i < N, "Invalid pointer access");

                return ptrs_[i];
            }

            /**
             * @brief Element access operator
             * @param[in] i Index of element to access
             * @return Value at index @p i
             * @warning No bounds checking performed
             **/
            __device__ __host__ [[nodiscard]] inline const T *operator[](const label_t i) const noexcept
            {
                return ptrs_[i];
            }

        private:
            /**
             * @brief The underlying pointers
             **/
            T *const ptrRestrict ptrs_[N];
        };
    }
}

#include "hardwareConfig.cuh"

#endif