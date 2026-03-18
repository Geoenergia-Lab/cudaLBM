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
    Class handling the run-time IO of the solver

Namespace
    LBM::runTimeIO

SourceFiles
    runTimeIO.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_RUNTIMEIO_CUH
#define __MBLBM_RUNTIMEIO_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"

namespace LBM
{
    /**
     * @class runTimeIO
     * @brief Handles runtime input/output operations and performance monitoring
     *
     * This class manages runtime operations including timing, performance metrics,
     * and output formatting. It tracks simulation duration and calculates
     * performance metrics like MLUPS (Million Lattice Updates Per Second).
     **/
    class runTimeIO
    {
    public:
        /**
         * @brief Constructs a runTimeIO object and starts timing
         * @param[in] mesh The lattice mesh
         * @param[in] programCtrl The program control object
         **/
        __host__ [[nodiscard]] runTimeIO(
            const host::latticeMesh &mesh,
            const programControl &programCtrl)
            : mesh_(mesh),
              programCtrl_(programCtrl),
              start_(std::chrono::high_resolution_clock::now())
        {
            std::cout << "Time loop start" << std::endl;
            std::cout << std::endl;
        };

        /**
         * @brief Destructor - calculates and outputs performance metrics
         *
         * Upon destruction, this class calculates and displays:
         * - Total elapsed time in HH:MM:SS format
         * - MLUPS (Million Lattice Updates Per Second) performance metric
         **/
        inline ~runTimeIO() noexcept
        {
            const std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            std::cout << std::endl;
            std::cout << "Elapsed time: " << runTimeIO::duration(std::chrono::duration_cast<std::chrono::seconds>(end - start_).count()) << std::endl;
            std::cout << std::endl;
            std::cout << "MLUPS: " << runTimeIO::MLUPS<double>(mesh_, programCtrl_, start_, end) << std::endl;
        };

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] runTimeIO(const runTimeIO &) = delete;
        __host__ [[nodiscard]] runTimeIO &operator=(const runTimeIO &) = delete;

        /**
         * @brief Formats a duration in seconds into HH:MM:SS string format
         * @param[in] totalSeconds Total number of seconds to format
         * @return String formatted as HH:MM:SS (supports negative durations)
         **/
        __host__ [[nodiscard]] static const name_t duration(const long long totalSeconds) noexcept
        {
            // Handle sign and absolute value conversion
            const bool isNegative = (totalSeconds < 0);
            const unsigned long long abs_seconds = isNegative ? -static_cast<unsigned long long>(totalSeconds) : static_cast<unsigned long long>(totalSeconds);

            // Calculate time components
            const unsigned long long hours = abs_seconds / 3600;
            const unsigned long long secondsRemaining = abs_seconds % 3600;
            const unsigned long long minutes = secondsRemaining / 60;
            const unsigned long long seconds = secondsRemaining % 60;

            // Format components to HH:MM:SS
            std::ostringstream oss;
            if (isNegative)
            {
                oss << "-";
            }
            oss << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":" << std::setfill('0') << std::setw(2) << seconds;

            return oss.str();
        }

        /**
         * @brief Calculates Million Lattice Updates Per Second (MLUPS) performance metric
         * @tparam T Data type for the MLUPS calculation (typically double)
         * @param[in] mesh The lattice mesh
         * @param[in] programCtrl The program control object
         * @param[in] start Simulation start time point
         * @param[in] end Simulation end time point
         * @return MLUPS value as type T, or 0 if calculation is not applicable
         *
         * MLUPS is calculated as: (total lattice points × time steps) / (execution time in seconds × 10⁶)
         * This metric provides a standardized way to compare LBM implementation performance.
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline constexpr T MLUPS(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const std::chrono::high_resolution_clock::time_point &start,
            const std::chrono::high_resolution_clock::time_point &end) noexcept
        {
            if ((programCtrl.nt() == (programCtrl.latestTime() - 1)) | mesh.size() == 0)
            {
                return 0;
            }

            const host::label_t nPoints = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * mesh.dimension<axis::Z>();

            const host::label_t nTime = programCtrl.nt() - programCtrl.latestTime() - 1;

            const host::label_t numerator = nPoints * nTime;

            const host::label_t denominator = static_cast<host::label_t>(1000000) * static_cast<host::label_t>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

            return static_cast<T>(numerator) / static_cast<T>(denominator);
        }

    private:
        /**
         * @brief Reference to the mesh
         **/
        const host::latticeMesh &mesh_;

        /**
         * @brief Reference to the program control
         **/
        const programControl &programCtrl_;

        /**
         * @brief Start point
         **/
        const std::chrono::high_resolution_clock::time_point start_;
    };
}

#endif