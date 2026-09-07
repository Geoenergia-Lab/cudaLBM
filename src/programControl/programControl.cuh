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
#include "streamHandler.cuh"
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
              Ma_(initialiseConst<scalar_t>("Ma")),
              L_char_(initialiseConst<scalar_t>("L_char")),
              nTimeSteps_(string::extractParameter<host::label_t>(string::readFile("programControl"), "nTimeSteps")),
              saveInterval_(string::extractParameter<host::label_t>(string::readFile("programControl"), "saveInterval")),
              infoInterval_(string::extractParameter<host::label_t>(string::readFile("programControl"), "infoInterval")),
              latestTime_(latestSaved()),
              timeStep_(latestTime_),
              streams_(deviceList())
        {
            types::assertions::validate<scalar_t>();
            types::assertions::validate<device::label_t>();

            // Get the launch time
            const time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            // Get current working directory
            const std::filesystem::path launchDirectory = std::filesystem::current_path();

            printHeader<true>();
            IO::printBlock(
                std::cout, "programControl",
                "{", "}",
                "programName\t", input_.commandLine()[0],
                "launchTime\t", std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S"),
                "launchDirectory", launchDirectory.string(),
                "deviceList\t", deviceList(),
                "caseName\t", caseName_,
                "Re\t\t", Re_,
                "Ma\t\t", Ma_,
                "U_inf\t", Ma_ / std::sqrt(static_cast<scalar_t>(3)),
                "nTimeSteps\t", nTimeSteps_,
                "saveInterval", saveInterval_,
                "infoInterval", infoInterval_,
                "latestTime\t", latestTime_,
                "scalarSize\t", sizeof(scalar_t) * 8,
                "labelType\t", "uint" + std::to_string(sizeof(device::label_t) * 8) + "_t");

            std::cout << std::endl;
            if (deviceList().size() > 0)
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < deviceList().size(); virtualDeviceIndex++)
                {
                    errorHandler::handle(cudaSetDevice(deviceList()[virtualDeviceIndex]));

                    // Allocate symbols on the GPU
                    const scalar_t U = Ma_ / std::sqrt(static_cast<scalar_t>(3));
                    const scalar_t nuTemp = U * L_char() / Re();
                    const scalar_t tauTemp = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3) * nuTemp;
                    const scalar_t omegaTemp = static_cast<scalar_t>(1) / tauTemp;
                    const scalar_t t_omegaVarTemp = static_cast<scalar_t>(1) - omegaTemp;
                    const scalar_t omegaVar_d2Temp = omegaTemp * static_cast<scalar_t>(0.5);

                    device::copyToSymbol(device::L_char, L_char());
                    device::copyToSymbol(device::nu, nuTemp);
                    device::copyToSymbol(device::Re, Re());
                    device::copyToSymbol(device::tau, tauTemp);
                    device::copyToSymbol(device::omega, omegaTemp);
                    device::copyToSymbol(device::t_omegaVar, t_omegaVarTemp);
                    device::copyToSymbol(device::omegaVar_d2, omegaVar_d2Temp);
                }
            }

            // Make sure we synchronize and set active device to 0
            // Probably unnecessary but nice to do it anyway
            if (deviceList().size() > 0)
            {
                errorHandler::handle(cudaDeviceSynchronize());
                errorHandler::handle(cudaSetDevice(deviceList()[0]));
                errorHandler::handle(cudaDeviceSynchronize());
            }
        };

        /**
         * @brief Destructor for the programControl class
         **/
        __host__ ~programControl() noexcept
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
         * @brief Print the file header
         * @tparam LineBreak Add a line break
         **/
        template <const bool LineBreak = false>
        __host__ static void printHeader() noexcept
        {
            std::cout << "/*---------------------------------------------------------------------------*\\" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |" << std::endl;
            std::cout << "| Developed at UDESC - State University of Santa Catarina                     |" << std::endl;
            std::cout << "| Website: https://www.udesc.br                                               |" << std::endl;
            std::cout << "| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "\\*---------------------------------------------------------------------------*/" << std::endl;
            if constexpr (LineBreak)
            {
                std::cout << std::endl;
            }
        }

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
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Ma() const noexcept
        {
            return Ma_;
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
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t nt() const noexcept
        {
            return nTimeSteps_;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @param[in] timeStep The time step to check
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool save(const host::label_t timeStep) const noexcept
        {
            return (timeStep % saveInterval_) == 0;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @param[in] timeStep The time step to check
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool print(const host::label_t timeStep) const noexcept
        {
            return (timeStep % infoInterval_) == 0;
        }

        /**
         * @brief Returns the latest time step of the solution files contained within the current directory
         * @return The latest time step as a host::label_t
         **/
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t latestTime() const noexcept
        {
            return latestTime_;
        }

        /**
         * @brief Returns a reference to the current time step
         * @return The latest time step as a host::label_t
         **/
        __device__ __host__ [[nodiscard]] inline constexpr const host::label_t &timeStep() const noexcept
        {
            return timeStep_;
        }
        __device__ __host__ [[nodiscard]] inline constexpr host::label_t &timeStep() noexcept
        {
            return timeStep_;
        }

        /**
         * @brief Determines whether or not the run time loop should exit
         **/
        __host__ [[nodiscard]] inline constexpr bool end() const noexcept
        {
            return (timeStep() < nt() && (runTime::program_status.load() == runTime::GOOD));
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
         * @param[in] argument The argument to search for
         **/
        __host__ [[nodiscard]] const name_t getArgument(const name_t &argument) const
        {
            if (input_.isArgPresent(argument))
            {
                for (host::label_t arg = 0; arg < commandLine().size(); arg++)
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
         * @tparam PreferShared Controls preference of shared memory (default true)
         * @tparam T The function type (e.g., a lambda or a function pointer)
         * @param[in] func The kernel function to configure
         **/
        template <const host::label_t smem_alloc_size, const bool PreferShared = true, class T>
        __host__ void configure(T *func) const
        {
            for (host::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < deviceList().size(); VirtualDeviceIndex++)
            {
                errorHandler::handle(cudaDeviceSynchronize());
                errorHandler::handle(cudaSetDevice(deviceList()[VirtualDeviceIndex]));
                errorHandler::handle(cudaDeviceSynchronize());
                if constexpr (PreferShared)
                {
                    errorHandler::handle(cudaFuncSetCacheConfig(func, cudaFuncCachePreferShared));
                }
                else
                {
                    errorHandler::handle(cudaFuncSetCacheConfig(func, cudaFuncCachePreferL1));
                }
                errorHandler::handle(cudaDeviceSynchronize());
                errorHandler::handle(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_alloc_size));
                errorHandler::handle(cudaDeviceSynchronize());
            }
        }

        /**
         * @brief Returns a vector of time step indices corresponding to the saved time steps in the "timeStep" directory
         * @return A vector of host::label_t representing the saved time step indices
         **/
        __host__ [[nodiscard]] const std::vector<host::label_t> timeStepIndices() const
        {
            const std::vector<host::label_t> fileNameIndices = savedTimeSteps("timeStep");

            if (input().isArgPresent("-fieldName"))
            {
                if (input().isArgPresent("-latestTime"))
                {
                    return {fileNameIndices.back()};
                }
                else
                {
                    return fileNameIndices;
                }
            }
            else
            {
                return fileNameIndices;
            }
        }

        /**
         * @brief Return a reference to the streamHandler object
         * @return A const reference to the streamHandler object
         **/
        __host__ [[nodiscard]] inline constexpr const streamHandler &streams() const noexcept
        {
            return streams_;
        }

        /**
         * @brief Synchronizes all CUDA streams managed by the streamHandler object
         **/
        __host__ inline void allsync() const noexcept
        {
            for (host::label_t stream = 0; stream < streams_.streams().size(); stream++)
            {
                streams_.synchronize(stream);
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
         * @brief The characteristic Mach number
         **/
        const scalar_t Ma_;

        /**
         * @brief The characteristic length scale
         **/
        const scalar_t L_char_;

        /**
         * @brief Total number of simulation time steps, the save interval, info output interval and the latest time step at program start
         **/
        const host::label_t nTimeSteps_;
        const host::label_t saveInterval_;
        const host::label_t infoInterval_;
        const host::label_t latestTime_;

        host::label_t timeStep_;

        const streamHandler streams_;

        /**
         * @brief Reads a variable from the caseInfo file into a parameter of type T
         * @tparam T The return type
         * @return The variable as type T
         * @param[in] varName The name of the variable to read
         **/
        template <typename T>
        __host__ [[nodiscard]] static inline T initialiseConst(const name_t &varName) noexcept
        {
            return string::extractParameter<T>(string::readFile("programControl"), varName);
        }

        /**
         * @brief Reads the "timeStep" directory and returns a vector of host::label_t corresponding to the time step indices of the saved time steps
         * @return A vector of host::label_t representing the saved time step indices
         * @param[in] dir_path The path to the directory to read (e.g., "timeStep")
         **/
        __host__ [[nodiscard]] static const std::vector<host::label_t> savedTimeSteps(const std::string &dir_path)
        {
            std::vector<host::label_t> numbers;

            // Check if the given path exists and is a directory
            if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path))
            {
                return numbers; // return empty vector if invalid
            }

            for (const auto &entry : std::filesystem::directory_iterator(dir_path))
            {
                // Only consider subdirectories (ignore files, symlinks, etc.)
                if (entry.is_directory())
                {
                    const name_t name = entry.path().filename().string();
                    try
                    {
                        // Convert name to host::label_t
                        // std::stoul handles leading/trailing whitespace? No, but we assume clean names.
                        // Use std::stoul for unsigned long, then cast to host::label_t.
                        const host::label_t num = std::stoull(name);
                        numbers.push_back(num);
                    }
                    catch (const std::invalid_argument &)
                    {
                        // Name is not a number - skip
                    }
                    catch (const std::out_of_range &)
                    {
                        // Number is too large for unsigned long - skip
                    }
                }
            }

            std::sort(numbers.begin(), numbers.end());

            return numbers;
        }

        /**
         * @brief Returns the latest time step of the solution files contained within the current directory
         * @return The latest time step as a host::label_t
         **/
        __host__ [[nodiscard]] static inline host::label_t latestSaved()
        {
            return savedTimeSteps("timeStep").empty() ? 0 : savedTimeSteps("timeStep").back();
        }
    };
}

#endif