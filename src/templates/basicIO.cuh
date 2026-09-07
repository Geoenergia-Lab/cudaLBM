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
    Basic templated IO routines used throughout the code base

Namespace
    LBM, LBM::IO

SourceFiles
    basicIO.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BASICIO_CUH
#define __MBLBM_BASICIO_CUH

namespace LBM
{
    namespace IO
    {
        /**
         * @brief Compile-time generator of whitespace strings with a fixed size.
         * @tparam N Number of spaces to generate.
         **/
        template <const std::size_t N>
        struct whitespace
        {
        public:
            using ValueType = std::array<char, N + 1>;

            /**
             * @brief Returns a C-string containing N spaces followed by a null terminator.
             * @return const char* pointer to the whitespace string.
             **/
            __host__ [[nodiscard]] static inline constexpr const char *c_str() noexcept
            {
                return data_.data();
            }

        private:
            /**
             * @brief Initializes the static whitespace array with spaces.
             * @return ValueType array filled with spaces and null terminator.
             **/
            __host__ [[nodiscard]] static inline constexpr const ValueType initialise_spaces() noexcept
            {
                ValueType result;
                if constexpr (N > 0)
                {
                    for (decltype(result.size()) i = 0; i < N; i++)
                    {
                        result[i] = ' ';
                    }
                }
                result[N] = '\0';
                return result;
            }

            static constexpr const ValueType data_ = initialise_spaces();
        };

        /**
         * @brief Stream insertion operator for whitespace objects.
         * @tparam N Number of spaces.
         * @param[in] os Output stream.
         * @param[in] s Whitespace object to insert.
         * @return Reference to the output stream.
         **/
        template <const std::size_t N>
        __host__ [[nodiscard]] std::ostream inline constexpr &operator<<(std::ostream &os, const whitespace<N> &s) noexcept
        {
            os << s.c_str();
            return os;
        }

        /**
         * @brief Prints a container's elements in braces, separated by commas.
         * @tparam Container Type of the container (must have size() and operator[]).
         * @param[in] os Output stream.
         * @param[in] c Container to print.
         **/
        template <typename Container>
        __host__ void print_container(std::ostream &os, const Container &c) noexcept
        {
            os << "{";
            for (decltype(c.size()) i = 0; i < c.size(); ++i)
            {
                if (i > 0)
                {
                    os << ", ";
                }
                os << c[i];
            }
            os << "}";
        }

        /**
         * @brief Prints a pair as a range [first, second).
         * @tparam T Type of the pair elements.
         * @param[in] os Output stream.
         * @param[in] c Pair to print.
         **/
        template <typename T>
        __host__ void print_container(std::ostream &os, const std::pair<T, T> &c) noexcept
        {
            os << "{";
            for (T v = c.first; v < c.second - 1; v++)
            {
                os << v << ", ";
            }
            os << c.second;
            os << "};" << std::endl;
        }

        /**
         * @brief Prints a container's name followed by its elements on a new line.
         * @tparam Container Type of the container (must have size() and operator[]).
         * @tparam Name Type of the name (must be streamable).
         * @param[in] os Output stream.
         * @param[in] c Container to print.
         * @param[in] name Name to print before container.
         **/
        template <class Container, class Name>
        __host__ void print_container(std::ostream &os, const Container &c, const Name &name) noexcept
        {
            os << name << std::endl;
            print_container(os, c);
        }

        /**
         * @brief Templated method to insert a std::vector of type T into an ostream object
         * @tparam T Type of the underlying vector
         * @param[in] os The ostream object
         * @param[in] vec The vector to print
         **/
        template <typename T>
        __host__ [[nodiscard]] inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) noexcept
        {
            print_container(os, vec);
            return os;
        }

        /**
         * @brief Stream insertion operator for std::array objects.
         * @tparam T Type of array elements.
         * @tparam N Size of the array.
         * @param[in] os Output stream.
         * @param[in] arr Array to insert.
         * @return Reference to the output stream.
         **/
        template <typename T, const std::size_t N>
        __host__ [[nodiscard]] inline std::ostream &operator<<(std::ostream &os, const std::array<T, N> &arr) noexcept
        {
            print_container(os, arr);
            return os;
        }

        /**
         * @brief Prints a container with each element on a new line, indented.
         * @tparam BraceIndent Number of spaces before opening/closing braces.
         * @tparam ElementIndent Number of spaces before each element.
         * @tparam Container Type of the container (must have size() and operator[]).
         * @param[in] os Output stream.
         * @param[in] c Container to print.
         **/
        template <const std::size_t BraceIndent = 0, const std::size_t ElementIndent = 4, typename Container>
        __host__ void print_container_expanded(std::ostream &os, const Container &c) noexcept
        {
            os << whitespace<BraceIndent>{} << "{" << std::endl;
            for (decltype(c.size()) i = 0; i < c.size(); ++i)
            {
                os << whitespace<ElementIndent>{} << c[i] << ";" << std::endl;
            }
            os << whitespace<BraceIndent>{} << "};" << std::endl;
        }

        /**
         * @brief Prints a container's name, size, and elements in expanded form.
         * @tparam BraceIndent Number of spaces before opening/closing braces.
         * @tparam ElementIndent Number of spaces before each element.
         * @tparam Container Type of the container (must have size() and operator[]).
         * @param[in] os Output stream.
         * @param[in] c Container to print.
         * @param[in] name Name of the container.
         **/
        template <const std::size_t BraceIndent = 0, const std::size_t ElementIndent = 4, typename Container>
        __host__ void print_container_expanded(std::ostream &os, const Container &c, const char *name) noexcept
        {
            os << whitespace<BraceIndent>{} << name << "[" << c.size() << "]" << std::endl;
            print_container_expanded<BraceIndent, ElementIndent>(os, c);
        }

        /**
         * @brief Base case for recursive printing: does nothing.
         **/
        __host__ void printFields() noexcept {}

        /**
         * @brief Prints one title-value pair and recurses on the rest.
         * @tparam BraceIndent Number of spaces before opening/closing braces.
         * @tparam Value Type of the current value (must support operator<<).
         * @tparam Rest Remaining title-value pairs.
         * @param[in] os Output stream.
         * @param[in] title Field title (must be convertible to const char*).
         * @param[in] value Field value.
         * @param[in] rest Remaining arguments.
         **/
        template <const std::size_t BraceIndent = 4, class Value, class... Rest>
        __host__ void printFields(std::ostream &os, const char *title, const Value &value, const Rest &...rest) noexcept
        {
            os << whitespace<BraceIndent>{} << title << "\t" << value << ";" << std::endl;
            if constexpr (sizeof...(Rest) > 0)
            {
                printFields(os, rest...);
            }
        }

        /**
         * @brief Prints a formatted block with header, braces, and indented fields.
         * @tparam Args Variadic arguments: alternating const char* titles and printable values.
         * @param[in] os Output stream.
         * @param[in] header Header string.
         * @param[in] openBrace Opening brace string.
         * @param[in] closeBrace Closing brace string.
         * @param[in] args Title-value pairs (must be even in number).
         **/
        template <class... Args>
        __host__ void printBlock(std::ostream &os, const char *header, const char *openBrace, const char *closeBrace, const Args &...args) noexcept
        {
            static_assert(sizeof...(Args) % 2 == 0, "Number of arguments must be even (title-value pairs)");

            os << header << std::endl;
            os << openBrace << std::endl;
            printFields(os, args...);
            os << closeBrace << std::endl;
        }

        /**
         * @brief Prints a formatted block with header, curly braces, and indented fields.
         * @tparam Args Variadic arguments: alternating const char* titles and printable values.
         * @param[in] os Output stream.
         * @param[in] header Header string.
         * @param[in] args Title-value pairs (must be even in number).
         **/
        template <class... Args>
        __host__ void printBlock(std::ostream &os, const char *header, const Args &...args) noexcept
        {
            printBlock(os, header, "{", "};", args...);
        }

        /**
         * @brief Prints a formatted block to std::cout with header, curly braces, and indented fields.
         * @tparam Args Variadic arguments: alternating const char* titles and printable values.
         * @param[in] header Header string.
         * @param[in] args Title-value pairs (must be even in number).
         **/
        template <class... Args>
        __host__ void print(const char *header, const Args &...args) noexcept
        {
            printBlock(std::cout, header, args...);
        }

        /**
         * @brief Prints a formatted block to std::cerr with header, curly braces, and indented fields.
         * @tparam Args Variadic arguments: alternating const char* titles and printable values.
         * @param[in] header Header string.
         * @param[in] args Title-value pairs (must be even in number).
         **/
        template <class... Args>
        __host__ void printError(const char *header, const Args &...args) noexcept
        {
            printBlock(std::cerr, header, "{", "}", args...);
        }
    }

    using IO::operator<<;
}

#endif