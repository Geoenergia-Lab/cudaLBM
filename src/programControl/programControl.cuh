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
    A class handling the setup of the solver

Namespace
    LBM

SourceFiles
    programControl.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PROGRAMCONTROL_CUH
#define __MBLBM_PROGRAMCONTROL_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../strings.cuh"
#include "inputControl.cuh"
#include "../fileIO/fileIO.cuh"

namespace LBM
{
    class programControl
    {
    public:
        /**
         * @brief Constructor for the programControl class
         * @param[in] argc First argument passed to main
         * @param[in] argv Second argument passed to main
         **/
        __host__ [[nodiscard]] programControl(const int argc, const char *const argv[]) noexcept
            : input_(inputControl(argc, argv)),
              caseName_(string::extractParameter<name_t>(string::readFile("programControl"), "caseName")),
              Re_(initialiseConst<scalar_t>("Re")),
              u_inf_(initialiseConst<scalar_t>("u_inf")),
              L_char_(initialiseConst<scalar_t>("L_char")),
              nTimeSteps_(string::extractParameter<device::label_t>(string::readFile("programControl"), "nTimeSteps")),
              saveInterval_(string::extractParameter<device::label_t>(string::readFile("programControl"), "saveInterval")),
              infoInterval_(string::extractParameter<device::label_t>(string::readFile("programControl"), "infoInterval")),
              latestTime_(fileIO::latestTime(caseName_))
        {
            types::assertions::validate<scalar_t>();
            types::assertions::validate<device::label_t>();

            // Get the launch time
            const time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            // Get current working directory
            const std::filesystem::path launchDirectory = std::filesystem::current_path();

            std::cout << "/*---------------------------------------------------------------------------*\\" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |" << std::endl;
            std::cout << "| Developed at UDESC - State University of Santa Catarina                     |" << std::endl;
            std::cout << "| Website: https://www.udesc.br                                               |" << std::endl;
            std::cout << "| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "\\*---------------------------------------------------------------------------*/" << std::endl;
            std::cout << std::endl;
            std::cout << "programControl:" << std::endl;
            std::cout << "{" << std::endl;
            std::cout << "    programName: " << input_.commandLine()[0] << ";" << std::endl;
            std::cout << "    launchTime: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << ";" << std::endl;
            std::cout << "    launchDirectory: " << launchDirectory.string() << ";" << std::endl;
            std::cout << "    deviceList: [";
            if (deviceList().size() > 1)
            {
                for (device::label_t i = 0; i < deviceList().size() - 1; i++)
                {
                    std::cout << deviceList()[i] << ", ";
                }
            }
            std::cout << deviceList()[deviceList().size() - 1] << "];" << std::endl;
            std::cout << "    caseName: " << caseName_ << ";" << std::endl;
            std::cout << "    Re = " << Re_ << ";" << std::endl;
            std::cout << "    nTimeSteps = " << nTimeSteps_ << ";" << std::endl;
            std::cout << "    saveInterval = " << saveInterval_ << ";" << std::endl;
            std::cout << "    infoInterval = " << infoInterval_ << ";" << std::endl;
            std::cout << "    latestTime = " << latestTime_ << ";" << std::endl;
            std::cout << "    scalarSize: " << sizeof(scalar_t) * 8 << ";" << std::endl;
            std::cout << "    labelType: uint" << sizeof(device::label_t) * 8 << "_t" << ";" << std::endl;
            std::cout << "};" << std::endl;
            std::cout << std::endl;

            for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < deviceList().size(); virtualDeviceIndex++)
            {
                errorHandler::check(cudaSetDevice(deviceList()[virtualDeviceIndex]));

                // Allocate symbols on the GPU
                const scalar_t viscosityTemp = u_inf() * L_char() / Re();
                const scalar_t tauTemp = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3.0) * viscosityTemp;
                const scalar_t omegaTemp = static_cast<scalar_t>(1.0) / tauTemp;
                const scalar_t t_omegaVarTemp = static_cast<scalar_t>(1) - omegaTemp;
                const scalar_t omegaVar_d2Temp = omegaTemp * static_cast<scalar_t>(0.5);

                device::copyToSymbol(device::L_char, L_char());
                device::copyToSymbol(device::Re, Re());
                device::copyToSymbol(device::tau, tauTemp);
                device::copyToSymbol(device::omega, omegaTemp);
                device::copyToSymbol(device::t_omegaVar, t_omegaVarTemp);
                device::copyToSymbol(device::omegaVar_d2, omegaVar_d2Temp);
            }

            // Make sure we synchronize and set active device to 0
            // Probably unnecessary but nice to do it anyway
            errorHandler::check(cudaDeviceSynchronize());
            errorHandler::check(cudaSetDevice(deviceList()[0]));
            errorHandler::check(cudaDeviceSynchronize());
        };

        /**
         * @brief Destructor for the programControl class
         **/
        ~programControl() noexcept
        {
            std::cout << std::endl;
            std::cout << "End" << std::endl;
            std::cout << std::endl;
        }

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] programControl(const programControl &) = delete;
        __host__ [[nodiscard]] programControl &operator=(const programControl &) = delete;

        /**
         * @brief Returns the name of the case
         * @return A const name_t
         **/
        __host__ [[nodiscard]] inline constexpr const name_t &caseName() const noexcept
        {
            return caseName_;
        }

        /**
         * @brief Returns the array of device indices
         * @return A read-only reference to deviceList_ contained within input_
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return input_.deviceList();
        }

        /**
         * @brief Returns the Reynolds number
         * @return The Reynolds number
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Re() const noexcept
        {
            return Re_;
        }

        /**
         * @brief Returns the characteristic velocity
         * @return The characteristic velocity
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t u_inf() const noexcept
        {
            return u_inf_;
        }

        /**
         * @brief Returns the characteristic velocity
         * @return The characteristic velocity
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t L_char() const noexcept
        {
            return L_char_;
        }

        /**
         * @brief Returns the total number of simulation time steps
         * @return The total number of simulation time steps
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline constexpr T nt() const noexcept
        {
            return static_cast<T>(nTimeSteps_);
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool save(const device::label_t timeStep) const noexcept
        {
            return (timeStep % saveInterval_) == 0;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool print(const device::label_t timeStep) const noexcept
        {
            return (timeStep % infoInterval_) == 0;
        }

        /**
         * @brief Returns the latest time step of the solution files contained within the current directory
         * @return The latest time step as a device::label_t
         **/
        template <typename T = device::label_t>
        __device__ __host__ [[nodiscard]] inline constexpr T latestTime() const noexcept
        {
            return static_cast<T>(latestTime_);
        }

        /**
         * @brief Provides read-only access to the input control
         * @return A const reference to an inputControl object
         **/
        __host__ [[nodiscard]] inline constexpr const inputControl &input() const noexcept
        {
            return input_;
        }

        /**
         * @brief Veriefies if the command line has the argument -type
         * @return A string representing the convertion type passed at the command line
         * @param[in] programCtrl The program control object
         **/
        __host__ [[nodiscard]] const name_t getArgument(const name_t &argument) const
        {
            if (input_.isArgPresent(argument))
            {
                for (device::label_t arg = 0; arg < commandLine().size(); arg++)
                {
                    if (commandLine()[arg] == argument)
                    {
                        if (arg + 1 == commandLine().size())
                        {
                            throw std::runtime_error("Argument " + argument + " not specified: the correct syntax is " + argument + " Arg");
                        }
                        else
                        {
                            return commandLine()[arg + 1];
                        }
                    }
                }
            }

            throw std::runtime_error("Argument " + argument + " not specified: the correct syntax is " + argument + " Arg");
        }

        /**
         * @brief Provides read-only access to the arguments supplied at the command line
         * @return The command line input as a vector of strings
         **/
        __host__ [[nodiscard]] inline constexpr const words_t &commandLine() const noexcept
        {
            return input_.commandLine();
        }

        /**
         * @brief Configures a kernel function to prefer shared memory and sets its dynamic shared memory size
         * @tparam smem_alloc_size The amount of shared memory (in bytes) to allocate for the kernel
         * @tparam T The function type (e.g., a lambda or a function pointer)
         * @param[in] func The kernel function to configure
         **/
        template <const device::label_t smem_alloc_size, class T>
        __host__ void configure(T *func) const
        {
            for (host::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < deviceList().size(); VirtualDeviceIndex++)
            {
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaSetDevice(deviceList()[VirtualDeviceIndex]));
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaFuncSetCacheConfig(func, cudaFuncCachePreferShared));
                errorHandler::check(cudaDeviceSynchronize());
                errorHandler::check(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_alloc_size));
                errorHandler::check(cudaDeviceSynchronize());
            }
        }

    private:
        /**
         * @brief A reference to the input control object
         **/
        const inputControl input_;

        /**
         * @brief The name of the simulation case
         **/
        const name_t caseName_;

        /**
         * @brief The Reynolds number
         **/
        const scalar_t Re_;

        /**
         * @brief The characteristic velocity
         **/
        const scalar_t u_inf_;

        /**
         * @brief The characteristic length scale
         **/
        const scalar_t L_char_;

        /**
         * @brief Total number of simulation time steps, the save interval, info output interval and the latest time step at program start
         **/
        const device::label_t nTimeSteps_;
        const device::label_t saveInterval_;
        const device::label_t infoInterval_;
        const device::label_t latestTime_;

        /**
         * @brief Reads a variable from the caseInfo file into a parameter of type T
         * @return The variable as type T
         * @param[in] varName The name of the variable to read
         **/
        template <typename T>
        __host__ [[nodiscard]] T initialiseConst(const name_t &varName) const noexcept
        {
            return string::extractParameter<T>(string::readFile("programControl"), varName);
        }
    };
}

#include "streamHandler.cuh"

#endif