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
    Implementation of scalar, vector and tensor fields on the device

Namespace
    LBM

SourceFiles
    deviceField.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEFIELD_CUH
#define __MBLBM_DEVICEFIELD_CUH

namespace LBM
{
    namespace device
    {
        template <class VelocitySet, const time::type TimeType>
        class scalarField
        {
            using ComponentType = device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>;

        public:
            __host__ [[nodiscard]] scalarField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const scalar_t value,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  self_(name, mesh, value, programCtrl, allocate) {}

            __host__ [[nodiscard]] scalarField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  self_(name, name, mesh, programCtrl, allocate) {}

            ~scalarField() {}

            template <class Writer>
            __host__ void save(
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::label_t timeStep)
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < self_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                {
                    hostWriteBuffer.copyFromDevice(
                        constPtr(virtualDeviceIndex),
                        self_.mesh(),
                        virtualDeviceIndex);
                }

                if constexpr (TimeType == time::instantaneous)
                {
                    Writer::write(
                        name_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        self_.mesh(),
                        {self_.name()},
                        hostWriteBuffer.data(),
                        timeStep);
                }
                else
                {
                    Writer::write(
                        name_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        self_.mesh(),
                        {self_.name()},
                        hostWriteBuffer.data(),
                        timeStep,
                        meanCount());
                }
            }

            __host__ [[nodiscard]] inline constexpr ComponentType &self() noexcept { return self_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &self() const noexcept { return self_; }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<1, const scalar_t> constPtr(const host::label_t idx) const noexcept
            {
                return {self_.ptr(idx)};
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<1, scalar_t> ptr(const host::label_t idx) noexcept
            {
                return {self_.ptr(idx)};
            }

            __host__ [[nodiscard]] inline constexpr host::label_t meanCount() const noexcept
            {
                return self_.meanCount();
            }

            __host__ [[nodiscard]] inline constexpr host::label_t &meanCountRef() noexcept
            {
                return self_.meanCountRef();
            }

            __host__ [[nodiscard]] inline constexpr const name_t &name() const noexcept
            {
                return name_;
            }

        private:
            const name_t name_;

            /**
             * @brief Components of the vector field
             **/
            ComponentType self_;
        };

        template <class VelocitySet, const time::type TimeType>
        class vectorField
        {
            using ComponentType = device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>;

        public:
            __host__ [[nodiscard]] vectorField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const scalar_t value,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  x_(name, name + "_x", mesh, value, programCtrl, allocate),
                  y_(name, name + "_y", mesh, value, programCtrl, allocate),
                  z_(name, name + "_z", mesh, value, programCtrl, allocate) {}

            __host__ [[nodiscard]] vectorField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  x_(name, name + "_x", mesh, programCtrl, allocate),
                  y_(name, name + "_y", mesh, programCtrl, allocate),
                  z_(name, name + "_z", mesh, programCtrl, allocate) {}

            ~vectorField() {}

            template <class Writer>
            __host__ void save(
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::label_t timeStep)
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < x_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                {
                    hostWriteBuffer.copyFromDevice(
                        constPtr(virtualDeviceIndex),
                        x_.mesh(),
                        virtualDeviceIndex);
                }

                if constexpr (TimeType == time::instantaneous)
                {
                    Writer::write(
                        name_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        x_.mesh(),
                        {x_.name(), y_.name(), z_.name()},
                        hostWriteBuffer.data(),
                        timeStep);
                }
                else
                {
                    Writer::write(
                        name_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        x_.mesh(),
                        {x_.name(), y_.name(), z_.name()},
                        hostWriteBuffer.data(),
                        timeStep,
                        meanCount());
                }
            }

            __host__ [[nodiscard]] inline constexpr ComponentType &x() noexcept { return x_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &y() noexcept { return y_; }
            __host__ [[nodiscard]] inline constexpr ComponentType &z() noexcept { return z_; }

            __host__ [[nodiscard]] inline constexpr const ComponentType &x() const noexcept { return x_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &y() const noexcept { return y_; }
            __host__ [[nodiscard]] inline constexpr const ComponentType &z() const noexcept { return z_; }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<3, const scalar_t> constPtr(const host::label_t idx) const noexcept
            {
                return {x_.ptr(idx), y_.ptr(idx), z_.ptr(idx)};
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<3, scalar_t> ptr(const host::label_t idx) noexcept
            {
                return {x_.ptr(idx), y_.ptr(idx), z_.ptr(idx)};
            }

            __host__ [[nodiscard]] inline constexpr host::label_t meanCount() const noexcept
            {
                return x_.meanCount();
            }

            __host__ [[nodiscard]] inline constexpr host::label_t &meanCountRef() noexcept
            {
                return x_.meanCountRef();
            }

            __host__ [[nodiscard]] inline constexpr const name_t &name() const noexcept
            {
                return name_;
            }

        private:
            const name_t name_;

            /**
             * @brief Components of the vector field
             **/
            ComponentType x_;
            ComponentType y_;
            ComponentType z_;
        };

        template <class VelocitySet, const time::type TimeType>
        class symmetricTensorField
        {
            using ComponentType = device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>;

        public:
            __host__ [[nodiscard]] symmetricTensorField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const scalar_t value,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  xx_(name, name + "_xx", mesh, value, programCtrl, allocate),
                  xy_(name, name + "_xy", mesh, value, programCtrl, allocate),
                  xz_(name, name + "_xz", mesh, value, programCtrl, allocate),
                  yy_(name, name + "_yy", mesh, value, programCtrl, allocate),
                  yz_(name, name + "_yz", mesh, value, programCtrl, allocate),
                  zz_(name, name + "_zz", mesh, value, programCtrl, allocate) {}

            __host__ [[nodiscard]] symmetricTensorField(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  xx_(name, name + "_xx", mesh, programCtrl, allocate),
                  xy_(name, name + "_xy", mesh, programCtrl, allocate),
                  xz_(name, name + "_xz", mesh, programCtrl, allocate),
                  yy_(name, name + "_yy", mesh, programCtrl, allocate),
                  yz_(name, name + "_yz", mesh, programCtrl, allocate),
                  zz_(name, name + "_zz", mesh, programCtrl, allocate) {}

            ~symmetricTensorField() {}

            template <class Writer>
            __host__ void save(
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::label_t timeStep)
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < xx_.programCtrl().deviceList().size(); virtualDeviceIndex++)
                {
                    hostWriteBuffer.copyFromDevice(
                        constPtr(virtualDeviceIndex),
                        xx_.mesh(),
                        virtualDeviceIndex);
                }

                if constexpr (TimeType == time::instantaneous)
                {
                    Writer::write(
                        name_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        xx_.mesh(),
                        {xx_.name(), xy_.name(), xz_.name(), yy_.name(), yz_.name(), zz_.name()},
                        hostWriteBuffer.data(),
                        timeStep);
                }
                else
                {
                    Writer::write(
                        name_ + "_" + std::to_string(timeStep) + ".LBMBin",
                        xx_.mesh(),
                        {xx_.name(), xy_.name(), xz_.name(), yy_.name(), yz_.name(), zz_.name()},
                        hostWriteBuffer.data(),
                        timeStep,
                        meanCount());
                }
            }

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

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> constPtr(const host::label_t idx) const noexcept
            {
                return {xx_.ptr(idx), xy_.ptr(idx), xz_.ptr(idx), yy_.ptr(idx), yz_.ptr(idx), zz_.ptr(idx)};
            }

            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> ptr(const host::label_t idx) noexcept
            {
                return {xx_.ptr(idx), xy_.ptr(idx), xz_.ptr(idx), yy_.ptr(idx), yz_.ptr(idx), zz_.ptr(idx)};
            }

            __host__ [[nodiscard]] inline constexpr host::label_t meanCount() const noexcept
            {
                return xx_.meanCount();
            }

            __host__ [[nodiscard]] inline constexpr host::label_t &meanCountRef() noexcept
            {
                return xx_.meanCountRef();
            }

            __host__ [[nodiscard]] inline constexpr const name_t &name() const noexcept
            {
                return name_;
            }

        private:
            const name_t name_;

            /**
             * @brief Components of the vector field
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
