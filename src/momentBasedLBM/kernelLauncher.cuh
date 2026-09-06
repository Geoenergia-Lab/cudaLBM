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
    Definition of the main GPU kernel

Namespace
    LBM::host, LBM::device

SourceFiles
    kernelLauncher.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDLBM_KERNELLAUNCHER_CUH
#define __MBLBM_MOMENTBASEDLBM_KERNELLAUNCHER_CUH

namespace LBM
{
    /**
     * @brief Launches the boundary and internal computation threads for multi‑GPU execution.
     *
     * Creates two std::thread objects: one for the boundary kernel and one for the internal
     * kernel, then joins them to synchronize computation and communication.
     *
     * @param mesh        Host-side lattice mesh description.
     * @param programCtrl Program control parameters (time step, streams, etc.).
     * @param devPtrs     Collection of device pointers for the LBM fields.
     * @param haloPtrs    Halo buffers for inter‑GPU communication.
     * @param devComm     Device communicator managing multi‑GPU transfers.
     **/
    __host__ inline void launch_multi_GPU(
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const kernel::ptrCollection &devPtrs,
        const haloBuffer<VelocitySet> &haloPtrs,
        const deviceCommunicator<VelocitySet> &devComm) noexcept
    {
        std::thread boundaryThread(
            std::addressof(kernel::launchBoundary),
            std::cref(mesh),
            std::cref(programCtrl),
            std::cref(devPtrs),
            std::cref(haloPtrs),
            std::cref(devComm),
            programCtrl.timeStep());
        std::thread internalThread(
            std::addressof(kernel::launchInternal),
            std::cref(mesh),
            std::cref(programCtrl),
            std::cref(devPtrs),
            std::cref(haloPtrs),
            programCtrl.timeStep());

        // Synchronize computation and communication
        boundaryThread.join();
        internalThread.join();
    }

    /**
     * @brief Launches the LBM kernel on a single GPU.
     *
     * Calls the moment‑based LBM kernel using the internal stream for the first device.
     *
     * @param mesh        Host-side lattice mesh description.
     * @param programCtrl Program control parameters (streams, time step, etc.).
     * @param devPtrs     Collection of device pointers for the LBM fields.
     * @param haloPtrs    Halo buffers for boundary handling (single GPU case).
     **/
    __host__ inline void launch_single_GPU(
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const kernel::ptrCollection &devPtrs,
        const haloBuffer<VelocitySet> &haloPtrs) noexcept
    {
        kernel::launch<kernel::momentBasedLBM, VelocitySet::smem_alloc_size()>(
            mesh,
            programCtrl.streams()[GPU::internalStreamID(0)],
            devPtrs[0],
            haloPtrs.readBuffer(0, programCtrl.timeStep()),
            haloPtrs.writeBuffer(0, programCtrl.timeStep()),
            static_cast<device::label_t>(0));
    }

    /**
     * @brief Concrete launcher for multi‑GPU configurations.
     *
     * Owns the device pointer collection, halo buffers, and device communicator,
     * and provides a launch() method to start the boundary and internal kernels.
     **/
    class MultiGPULauncher
    {
    public:
        /**
         * @brief Constructs a MultiGPULauncher with all required components.
         *
         * @param mesh        Host lattice mesh.
         * @param programCtrl Program control parameters.
         * @param rho         Device scalar field for density.
         * @param U           Device vector field for velocity.
         * @param Pi          Device symmetric tensor field for pressure tensor.
         **/
        MultiGPULauncher(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi) noexcept
            : mesh_(mesh),
              programCtrl_(programCtrl),
              devPtrs_(rho, U, Pi, programCtrl),
              haloPtrs_(rho, U, Pi, mesh, programCtrl),
              devComm_(mesh, programCtrl, haloPtrs_) {}

        /**
         * @brief Launches the multi‑GPU kernels.
         **/
        __host__ inline void launch() const noexcept
        {
            launch_multi_GPU(mesh_, programCtrl_, devPtrs_, haloPtrs_, devComm_);
        }

        /**
         * @brief Returns a constant reference to the device pointer collection.
         * @return const reference to kernel::ptrCollection.
         **/
        __host__ [[nodiscard]] inline constexpr const kernel::ptrCollection &devPtrs() const noexcept { return devPtrs_; }

    private:
        /**
         * @briefReference to host mesh
         **/
        const host::latticeMesh &mesh_;

        /**
         * @brief Reference to program control.
         **/
        const programControl &programCtrl_;

        /**
         * @brief Device pointer collection.
         **/
        const kernel::ptrCollection devPtrs_;

        /**
         * @brief Halo buffers for communication.
         **/
        const haloBuffer<VelocitySet> haloPtrs_;

        /**
         * @brief Device communicator.
         **/
        const deviceCommunicator<VelocitySet> devComm_;
    };

    /**
     * @brief Concrete launcher for single‑GPU configurations.
     *
     * Owns the device pointer collection and halo buffers, and provides a launch()
     * method to start the LBM kernel on one GPU.
     **/
    class SingleGPULauncher
    {
    public:
        /**
         * @brief Constructs a SingleGPULauncher with all required components.
         *
         * @param mesh        Host lattice mesh.
         * @param programCtrl Program control parameters.
         * @param rho         Device scalar field for density.
         * @param U           Device vector field for velocity.
         * @param Pi          Device symmetric tensor field for pressure tensor.
         **/
        SingleGPULauncher(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi) noexcept
            : mesh_(mesh),
              programCtrl_(programCtrl),
              devPtrs_(rho, U, Pi, programCtrl),
              haloPtrs_(rho, U, Pi, mesh, programCtrl) {}

        /**
         * @brief Launches the single‑GPU kernel.
         **/
        __host__ inline void launch() const noexcept
        {
            launch_single_GPU(mesh_, programCtrl_, devPtrs_, haloPtrs_);
        }

        /**
         * @brief Returns a constant reference to the device pointer collection.
         * @return const reference to kernel::ptrCollection.
         **/
        __host__ [[nodiscard]] inline constexpr const kernel::ptrCollection &devPtrs() const noexcept { return devPtrs_; }

    private:
        /**
         * @briefReference to host mesh
         **/
        const host::latticeMesh &mesh_;

        /**
         * @brief Reference to program control.
         **/
        const programControl &programCtrl_;

        /**
         * @brief Device pointer collection.
         **/
        const kernel::ptrCollection devPtrs_;

        /**
         * @brief Halo buffers for communication.
         **/
        const haloBuffer<VelocitySet> haloPtrs_;
    };

    /**
     * @brief Wrapper launcher that selects either MultiGPULauncher or SingleGPULauncher
     *        based on the number of available GPUs and the system configuration.
     *
     * Uses a std::variant to hold the concrete launcher chosen at construction time.
     * Provides a unified launch() interface and access to the device pointers.
     **/
    class KernelLauncher
    {
    public:
        /**
         * @brief Constructs a KernelLauncher and selects the appropriate concrete launcher.
         *
         * If the system supports multi‑GPU and the device list contains more than one
         * device, a MultiGPULauncher is created; otherwise a SingleGPULauncher is used.
         *
         * @param mesh        Host lattice mesh.
         * @param programCtrl Program control parameters.
         * @param rho         Device scalar field for density.
         * @param U           Device vector field for velocity.
         * @param Pi          Device symmetric tensor field for pressure tensor.
         **/
        KernelLauncher(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi) noexcept
            : variant_(makeVariant(mesh, programCtrl, rho, U, Pi)) {}

        /**
         * @brief Launches the underlying concrete launcher.
         **/
        __host__ inline constexpr void launch() const noexcept
        {
            std::visit(
                [](const auto &launcher)
                { launcher.launch(); }, variant_);
        }

        /**
         * @brief Returns a constant reference to the device pointer collection of the active launcher.
         * @return const reference to kernel::ptrCollection.
         **/
        __host__ [[nodiscard]] inline const kernel::ptrCollection &devPtrs() const noexcept
        {
            return std::visit(
                [](const auto &launcher) -> const kernel::ptrCollection &
                {
                    return launcher.devPtrs();
                },
                variant_);
        }

    private:
        /**
         * @brief Factory function that creates the appropriate variant type.
         * @param[in] mesh Host lattice mesh.
         * @param[in] programCtrl Program control parameters.
         * @param[in] rho Device scalar field containing the density values on the GPU
         * @param[in] U Device vector field containing the velocity values on the GPU
         * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
         * @return A std::variant holding either a MultiGPULauncher or SingleGPULauncher.
         **/
        static const std::variant<MultiGPULauncher, SingleGPULauncher> makeVariant(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi) noexcept
        {
            if constexpr (system::hasMultiGPU())
            {
                if (programCtrl.deviceList().size() > 1)
                {
                    return std::variant<MultiGPULauncher, SingleGPULauncher>(std::in_place_type<MultiGPULauncher>, mesh, programCtrl, rho, U, Pi);
                }
            }
            return std::variant<MultiGPULauncher, SingleGPULauncher>(std::in_place_type<SingleGPULauncher>, mesh, programCtrl, rho, U, Pi);
        }

        /**
         * @brief Holds the active launcher
         **/
        const std::variant<MultiGPULauncher, SingleGPULauncher> variant_;
    };
}

#endif