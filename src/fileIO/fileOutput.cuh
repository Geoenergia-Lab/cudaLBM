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
    Implementation of writing solution variables encoded in binary format

Namespace
    LBM::fileIO

SourceFiles
    fileOutput.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FILEOUTPUT_CUH
#define __MBLBM_FILEOUTPUT_CUH

namespace LBM
{
    namespace fileIO
    {
        /**
         * @brief Returns a string based on the type of time stepping
         **/
        template <const time::type TimeType>
        __host__ [[nodiscard]] const name_t timeTypeString() noexcept
        {
            if constexpr (TimeType == time::instantaneous)
            {
                return "instantaneous";
            }

            if constexpr (TimeType == time::timeAverage)
            {
                return "timeAverage";
            }
        }

        /**
         * @brief Write a pointer of type T to an ofstream object
         * @tparam T The type of the pointer
         * @param[in] ptr The pointer to write from
         * @param[in] size The length of the data to write
         * @param[out] out The output ofstream object
         **/
        template <typename T>
        __host__ void writeBinaryBlock(
            const T *const ptrRestrict ptr,
            const host::label_t size,
            std::ofstream &out)
        {
            out.write(reinterpret_cast<const char *>(ptr), to_streamsize(size));
        }

        /**
         * @brief Write a std::vector of type T to an ofstream object
         * @tparam T The type of the vector
         * @param[in] vec The vector to write from
         * @param[out] out The output ofstream object
         **/
        template <typename T>
        __host__ void writeBinaryBlock(const std::vector<T> &vec, std::ofstream &out)
        {
            const host::label_t blockSize = vec.size() * static_cast<host::label_t>(sizeof(T));

            writeBinaryBlock(&blockSize, static_cast<host::label_t>(sizeof(host::label_t)), out);

            writeBinaryBlock(vec.data(), blockSize, out);
        }
    }
}

#endif