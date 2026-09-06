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
    Implementation of scalar, vector and tensor fields on the device

Namespace
    LBM

SourceFiles
    deviceField.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTFIELD_CUH
#define __MBLBM_HOSTFIELD_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @brief Class representing a scalar field on the host, derived from fieldBase with N=1 component.
         **/
        template <const host::mallocType AllocationType, const time::type TimeType>
        class scalarField : public fieldType<1>, timeType<TimeType>
        {
            using ComponentType = host::array<AllocationType, scalar_t>;
            using FieldType = fieldType<1>;

        public:
            __host__ [[nodiscard]] scalarField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : FieldType(name),
                  self_(name, name, mesh, programCtrl) {}

            /**
             * @brief Default destructor
             **/
            __host__ ~scalarField() {}

            __host__ [[nodiscard]] inline constexpr ComponentType &self() noexcept { return self_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &self() const noexcept { return self_; }

        private:
            /**
             * @brief Components of the vector field
             **/
            ComponentType self_;
        };

        /**
         * @brief Class representing a vector field on the host, derived from fieldBase with N=3 components.
         **/
        template <const host::mallocType AllocationType, const time::type TimeType>
        class vectorField : public fieldType<3>, timeType<TimeType>
        {
            using ComponentType = host::array<AllocationType, scalar_t>;
            using FieldType = fieldType<3>;

        public:
            __host__ [[nodiscard]] vectorField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : FieldType(name),
                  x_(name, fieldType<3>::makeComponentNames<words_t>(name)[0], mesh, programCtrl),
                  y_(name, fieldType<3>::makeComponentNames<words_t>(name)[1], mesh, programCtrl),
                  z_(name, fieldType<3>::makeComponentNames<words_t>(name)[2], mesh, programCtrl) {}

            /**
             * @brief Default destructor
             **/
            __host__ ~vectorField() {}

            __host__ [[nodiscard]] inline constexpr ComponentType &x() noexcept { return x_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &y() noexcept { return y_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &z() noexcept { return z_; }

            __host__ [[nodiscard]] inline constexpr const ComponentType &x() const noexcept { return x_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &y() const noexcept { return y_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &z() const noexcept { return z_; }

        private:
            /**
             * @brief Components of the vector field
             **/
            ComponentType x_;
            ComponentType y_;
            ComponentType z_;
        };

        /**
         * @brief Class representing a symmetric tensor field on the host, derived from fieldBase with N=6 components.
         **/
        template <const host::mallocType AllocationType, const time::type TimeType>
        class symmetricTensorField : public fieldType<6>, timeType<TimeType>
        {
            using ComponentType = host::array<AllocationType, scalar_t>;
            using FieldType = fieldType<6>;

        public:
            __host__ [[nodiscard]] symmetricTensorField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl)
                : FieldType(name),
                  xx_(name, fieldType<6>::makeComponentNames<words_t>(name)[0], mesh, programCtrl),
                  xy_(name, fieldType<6>::makeComponentNames<words_t>(name)[1], mesh, programCtrl),
                  xz_(name, fieldType<6>::makeComponentNames<words_t>(name)[2], mesh, programCtrl),
                  yy_(name, fieldType<6>::makeComponentNames<words_t>(name)[3], mesh, programCtrl),
                  yz_(name, fieldType<6>::makeComponentNames<words_t>(name)[4], mesh, programCtrl),
                  zz_(name, fieldType<6>::makeComponentNames<words_t>(name)[5], mesh, programCtrl) {}

            /**
             * @brief Default destructor
             **/
            __host__ ~symmetricTensorField() {}

            __host__ [[nodiscard]] inline constexpr ComponentType &xx() noexcept { return xx_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &xy() noexcept { return xy_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &xz() noexcept { return xz_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &yy() noexcept { return yy_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &yz() noexcept { return yz_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &zz() noexcept { return zz_; }

            __host__ [[nodiscard]] inline constexpr const ComponentType &xx() const noexcept { return xx_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &xy() const noexcept { return xy_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &xz() const noexcept { return xz_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &yy() const noexcept { return yy_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &yz() const noexcept { return yz_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &zz() const noexcept { return zz_; }

        private:
            /**
             * @brief Components of the symmetric tensor field
             **/
            ComponentType xx_;
            ComponentType xy_;
            ComponentType xz_;
            ComponentType yy_;
            ComponentType yz_;
            ComponentType zz_;
        };
    }
}

#endif
