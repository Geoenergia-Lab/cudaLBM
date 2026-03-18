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
    Functions employed throughout the source code to manipulate strings

Namespace
    LBM

SourceFiles
    strings.cuh

\*---------------------------------------------------------------------------*/

#include "LBMIncludes.cuh"
#include "typedefs/typedefs.cuh"

#ifndef __MBLBM_STRINGS_CUH
#define __MBLBM_STRINGS_CUH

namespace LBM
{
    namespace string
    {
        /**
         * @brief Left-side concatenates a string to each element of a vector of strings.
         * @param[in] S The vector of strings to which the string will be concatenated.
         * @param[in] s The string to concatenate to each element of S.
         * @return A new vector of strings with s concatenated to each element of S.
         * @note This function creates a new vector and does not modify the input vector S.
         **/
        __host__ [[nodiscard]] const words_t catenate(const name_t &s, const words_t &S) noexcept
        {
            words_t S_new(S.size(), "");

            for (host::label_t i = 0; i < S_new.size(); i++)
            {
                S_new[i] = s + S_new[i];
            }

            return S_new;
        }

        /**
         * @brief Right-side concatenates a string to each element of a vector of strings.
         * @param[in] S The vector of strings to which the string will be concatenated.
         * @param[in] s The string to concatenate to each element of S.
         * @return A new vector of strings with s concatenated to each element of S.
         * @note This function creates a new vector and does not modify the input vector S.
         **/
        __host__ [[nodiscard]] const words_t catenate(const words_t &S, const name_t &s) noexcept
        {
            words_t S_new(S.size(), "");

            for (host::label_t i = 0; i < S_new.size(); i++)
            {
                S_new[i] = S[i] + s;
            }

            return S_new;
        }

        /**
         * @brief Checks if a target string exists within a vector of strings.
         * @param[in] vec The vector of strings to search.
         * @param[in] target The string to search for.
         * @return true if target is found in vec, false otherwise.
         * @note Uses std::find for efficient searching.
         **/
        __host__ [[nodiscard]] inline constexpr bool containsString(const words_t &vec, const name_t &target) noexcept
        {
            return std::find(vec.begin(), vec.end(), target) != vec.end();
        }

        /**
         * @brief Finds the position of a char within a string
         * @param[in] str The string to search
         * @param[in] c The character to search for
         * @return The position of c within str
         **/
        __host__ [[nodiscard]] inline constexpr host::label_t findCharPosition(const name_t &str, const char (&c)[2])
        {
            return str.find(c[0]);
        }

        /**
         * @brief Concatenates a vector of strings into a single string with newline separators.
         * @param[in] S The vector of strings to concatenate.
         * @return A single string with each element of S separated by a newline character.
         * @note This function is useful for creating multi-line strings from a list of lines.
         **/
        __host__ [[nodiscard]] const name_t catenate(const words_t &S) noexcept
        {
            name_t s;
            for (host::label_t line = 0; line < S.size(); line++)
            {
                s = s + S[line] + "\n";
            }
            return s;
        }

        /**
         * @brief Removes the first and last lines from a vector of strings.
         * @param[in] lines The input vector of strings.
         * @return A new vector of strings with the first and last lines removed.
         * @throws std::runtime_error if the input vector has 2 or fewer lines.
         * @note This function is useful for removing enclosing braces from blocks of text.
         **/
        __host__ [[nodiscard]] const words_t eraseBraces(const words_t &lines)
        {
            // Check minimum size requirement
            if (lines.size() < 3)
            {
                throw std::runtime_error("Lines must have at least 3 entries: opening bracket, content, and closing bracket. Problematic entry: " + catenate(lines));
            }

            // Check that first element is exactly "{"
            if (lines.front() != "{")
            {
                throw std::runtime_error("First element must be opening brace '{'. Problematic entry: " + catenate(lines));
            }

            // Check that last element is exactly "};"
            if (lines.back() != "};")
            {
                throw std::runtime_error("Last element must be closing brace with semicolon '};'. Problematic entry: " + catenate(lines));
            }

            // Create new vector without the braces
            words_t newLines(lines.size() - 2);

            for (host::label_t line = 1; line < lines.size() - 1; line++)
            {
                newLines[line - 1] = lines[line];
            }

            return newLines;
        }

        __host__ [[nodiscard]] const std::vector<std::string> splitByWhitespace(const std::string &str)
        {
            std::istringstream iss(str);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token)
            {
                // operator>> skips leading whitespace and stops at whitespace
                tokens.push_back(token);
            }
            return tokens;
        }

        /**
         * @brief Trims leading and trailing whitespace from a string.
         * @param[in] str The input string to trim.
         * @return Trimmed string, or empty string if only whitespace.
         * @note Handles space, tab, newline, carriage return, form feed, and vertical tab.
         **/
        template <const bool trimSemicolon>
        __host__ [[nodiscard]] const name_t trim(const name_t &str)
        {
            const host::label_t start = str.find_first_not_of(" \t\n\r\f\v");

            if (start == name_t::npos)
            {
                return "";
            }

            if constexpr (trimSemicolon)
            {
                return str.substr(start, str.find_last_not_of(" \t\n\r\f\v;") - start + 1);
            }
            else
            {
                return str.substr(start, str.find_last_not_of(" \t\n\r\f\v") - start + 1);
            }
        }

        /**
         * @brief Trims leading and trailing whitespace from each string in a vector.
         * @param[in] str The vector of strings to trim.
         * @return A new vector with each string trimmed.
         **/
        template <const bool trimSemicolon>
        __host__ [[nodiscard]] const words_t trim(const words_t &str)
        {
            words_t strTrimmed(str.size(), "");

            for (host::label_t s = 0; s < str.size(); s++)
            {
                strTrimmed[s] = trim<trimSemicolon>(str[s]);
            }

            return strTrimmed;
        }

        /**
         * @brief Removes C++-style comments from a string.
         * @param[in] str The input string to process.
         * @return String with comments removed (everything after '//').
         * @note Only handles single-line comments starting with '//'.
         **/
        __host__ [[nodiscard]] const name_t removeComments(const name_t &str)
        {
            const host::label_t commentPos = str.find("//");
            if (commentPos != name_t::npos)
            {
                return str.substr(0, commentPos);
            }
            return str;
        }

        /**
         * @brief Checks if a string contains only whitespace characters.
         * @param[in] str The string to check.
         * @return true if string contains only whitespace, false otherwise.
         * @note Uses std::isspace for whitespace detection.
         **/
        __host__ [[nodiscard]] bool isOnlyWhitespace(const name_t &str)
        {
            for (char c : str)
            {
                if (!std::isspace(static_cast<unsigned char>(c)))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Finds the line number where a block declaration starts.
         * @param[in] lines Vector of strings representing the source lines.
         * @param[in] blockName The name of the block to find (e.g., "boundaryField").
         * @param[in] startLine Line number to start searching from (default: 0).
         * @return Line number where the block declaration was found.
         * @throws std::runtime_error if block is not found.
         * @note Handles various declaration styles including braces and semicolons.
         **/
        __host__ [[nodiscard]] host::label_t findBlockLine(const words_t &lines, const name_t &blockName, const host::label_t startLine = 0)
        {
            for (host::label_t i = startLine; i < lines.size(); ++i)
            {
                const name_t processedLine = removeComments(lines[i]);
                const name_t trimmedLine = trim<false>(processedLine);

                // Check if line starts with the block name
                if (trimmedLine.compare(0, blockName.length(), blockName) == 0)
                {
                    // Extract the rest of the line after the block name
                    const name_t rest = trim<false>(trimmedLine.substr(blockName.length()));

                    // Check if the next token is empty or a brace
                    if (rest.empty() || rest == "{" || rest[0] == '{')
                    {
                        return i;
                    }

                    // Check if the next token is a semicolon (for field declarations)
                    if (rest[0] == ';')
                    {
                        return i;
                    }

                    // For cases like "internalField" which might be followed by other content
                    std::istringstream iss(rest);
                    name_t nextToken;
                    iss >> nextToken;

                    if (nextToken.empty() || nextToken == "{" || nextToken == ";")
                    {
                        return i;
                    }
                }
            }

            throw std::runtime_error("Field block \"" + blockName + "\" not defined");
        }

        /**
         * @brief Extracts a complete block (with braces) from source lines.
         * @param[in] lines Vector of strings representing the source lines.
         * @param[in] blockName The name of the block to extract.
         * @param[in] startLine Line number to start searching from (default: 0).
         * @return Vector of strings containing the complete block including braces.
         * @throws std::runtime_error for malformed blocks or unbalanced braces.
         * @note Preserves original formatting including comments in the returned block.
         **/
        __host__ [[nodiscard]] const words_t extractBlock(const words_t &lines, const name_t &blockName, const host::label_t startLine = 0)
        {
            words_t result;

            // Find the block line using the helper function
            const host::label_t blockLine = findBlockLine(lines, blockName, startLine);

            // Check for non-whitespace content between block declaration and opening brace
            bool foundOpeningBrace = false;
            int braceCount = 0;
            host::label_t openingBraceLine = 0;

            // First, check if the opening brace is on the same line as the block declaration
            const name_t blockLineProcessed = removeComments(lines[blockLine]);
            host::label_t openBracePos = blockLineProcessed.find('{');
            if (openBracePos != name_t::npos)
            {
                // Opening brace is on the same line as block declaration
                braceCount = 1;
                result.push_back(lines[blockLine]);
                foundOpeningBrace = true;
                openingBraceLine = blockLine;
            }
            else
            {
                // Check subsequent lines for the opening brace
                for (host::label_t i = blockLine + 1; i < lines.size(); ++i)
                {
                    const name_t processedLine = removeComments(lines[i]);
                    const name_t trimmedLine = trim<false>(processedLine);

                    // Check for closing brace before opening brace
                    if (processedLine.find('}') != name_t::npos)
                    {
                        throw std::runtime_error("Found closing brace before opening brace for block '" + blockName + "'");
                    }

                    // Check for non-whitespace content
                    if (!isOnlyWhitespace(trimmedLine) && trimmedLine.find('{') == name_t::npos)
                    {
                        throw std::runtime_error("Non-whitespace content found between block declaration and opening brace for block '" + blockName + "'");
                    }

                    // Check for opening brace
                    openBracePos = processedLine.find('{');
                    if (openBracePos != name_t::npos)
                    {
                        braceCount = 1;
                        result.push_back(lines[i]);
                        foundOpeningBrace = true;
                        openingBraceLine = i;
                        break;
                    }

                    // If we reach here, the line contains only whitespace or comments
                    // We don't add these lines to the result yet
                }
            }

            if (!foundOpeningBrace)
            {
                throw std::runtime_error("Opening brace not found for block '" + blockName + "'");
            }

            // Continue processing from the line after the opening brace
            for (host::label_t i = openingBraceLine + 1; i < lines.size(); ++i)
            {
                // Process each line to count braces, but keep the original line in the result
                const name_t processedLineInner = removeComments(lines[i]);
                for (char c : processedLineInner)
                {
                    if (c == '{')
                    {
                        braceCount++;
                    }
                    else if (c == '}')
                    {
                        braceCount--;
                    }
                }
                result.push_back(lines[i]);

                if (braceCount == 0)
                {
                    return result;
                }
            }

            // If we reach here, the braces are unbalanced
            throw std::runtime_error("Unbalanced braces for block '" + blockName + "'");
        }

        /**
         * @brief Extracts a field-specific block using a combined key-field identifier.
         * @param[in] lines Vector of strings representing the source lines.
         * @param[in] fieldName The field name (e.g., "p" for pressure).
         * @param[in] key The block type key (e.g., "internalField").
         * @return Vector of strings containing the complete block.
         * @note Convenience wrapper for extractBlock(lines, key + " " + fieldName).
         **/
        __host__ [[nodiscard]] const words_t extractBlock(const words_t &lines, const name_t &fieldName, const name_t &key)
        {
            return extractBlock(lines, key + " " + fieldName);
        }

        /**
         * @brief Reads the caseInfo file in the current directory into a vector of strings
         * @return A std::vector of std::string_view objects contained within the caseInfo file
         * @note This function will cause the program to exit if caseInfo is not found in the launch directory
         **/
        __host__ [[nodiscard]] const words_t readFile(const name_t &fileName)
        {
            // Does the file even exist?
            if (!std::filesystem::exists(fileName))
            {
                throw std::runtime_error(name_t(fileName) + name_t(" file not opened"));
            }

            // Read the caseInfo file contained within the directory
            std::ifstream caseInfo(name_t(fileName).c_str());
            words_t S;
            name_t s;

            // Count the number of lines
            device::label_t nLines = 0;

            // Count the number of lines
            while (std::getline(caseInfo, s))
            {
                S.push_back(s);
                nLines = nLines + 1;
            }

            S.resize(nLines);

            return S;
        }

        /**
         * @brief Checks that the input string is numeric.
         * @param[in] s The string_view object which is to be checked.
         * @return True if s is a valid number, false otherwise.
         * @note A valid number can optionally start with a '+' or '-' sign and may contain one decimal point.
         **/
        __host__ [[nodiscard]] bool isNumber(const name_t &s) noexcept
        {
            if (s.empty())
            {
                return false;
            }

            name_t::const_iterator it = s.begin();

            // Check for optional sign
            if (*it == '+' || *it == '-')
            {
                ++it;
                // If string is just a sign, it's not a valid number
                if (it == s.end())
                {
                    return false;
                }
            }

            bool has_digits = false;
            bool has_decimal = false;

            // Process each character
            while (it != s.end())
            {
                if (std::isdigit(*it))
                {
                    has_digits = true;
                    ++it;
                }
                else if (*it == '.')
                {
                    // Only one decimal point allowed
                    if (has_decimal)
                    {
                        return false;
                    }
                    has_decimal = true;
                    ++it;
                }
                else
                {
                    // Invalid character found
                    return false;
                }
            }

            // Must have at least one digit and if there's a decimal point,
            // there must be digits after it (handled by the iteration)
            return has_digits;
        }

        /**
         * @brief Determines whether or not the number string is all digits
         * @param[in] numStr The number string
         * @return True if the string is all digits, false otherwise
         **/
        __host__ [[nodiscard]] inline bool isAllDigits(const name_t &numStr) noexcept
        {
            for (char c : numStr)
            {
                if (!std::isdigit(static_cast<unsigned char>(c)))
                {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Splits the string_view object s according to the delimiter delim
         * @param[in] s The string_view object which is to be split
         * @param[in] delim The delimiter character by which s is split, e.g. a comma, space, etc
         * @param[in] removeWhitespace Controls the removal of whitespace; removes blank spaces from the return value if true (default true)
         * @return A std::vector of std::string_view objects split from s by delim
         * @note This function can be used to, for example, split a string by commas, spaces, etc
         **/
        template <const char delim, const bool removeWhitespace>
        __host__ [[nodiscard]] const words_t split(const std::string_view &s) noexcept
        {
            words_t result;
            const char *left = s.begin();
            for (const char *it = left; it != s.end(); ++it)
            {
                if (*it == delim)
                {
                    result.emplace_back(&*left, it - left);
                    left = it + 1;
                }
            }
            if (left != s.end())
            {
                result.emplace_back(&*left, s.end() - left);
            }

            // Remove whitespace from the returned vector
            if constexpr (removeWhitespace)
            {
                result.erase(std::remove(result.begin(), result.end(), " "), result.end());
            }

            return result;
        }

        __host__ [[nodiscard]] const name_t extractParameterLine(const words_t &S, const name_t &name)
        {
            // Loop over S
            for (device::label_t i = 0; i < S.size(); i++)
            {
                // Check if S[i] contains a substring of name
                if (S[i].find(name) != name_t::npos)
                {
                    // Split by space and remove whitespace
                    const words_t s = splitByWhitespace(S[i]);
                    // const words_t s = split<" "[0], true>(S[i]);

                    // Check that the last char is ;
                    // Perform the exit here if the above string is not equal to ;

                    return name_t(s[1].begin(), s[1].end() - 1);
                }
            }

            // Otherwise return 0
            // Should theoretically never get to this point because we have checked already that the string exists

            throw std::runtime_error("Parameter " + name_t(name) + " not found");

            return "";
        }

        /**
         * @brief Searches for an entry corresponding to variableName within the vector of strings S
         * @param[in] T The type of variable returned
         * @param[in] S The vector of strings which is searched
         * @param[in] name The name of the variable which is to be found and returned as type T
         * @return The value of the variable expressed as a type T
         * @note This function can be used to, for example, read an entry of nx within caseInfo after caseInfo has been loaded into S
         * @note The line containing the definition of variableName must separate variableName and its value with a space, for instance nx 128;
         **/
        template <typename T>
        __host__ [[nodiscard]] T extractParameter(const words_t &S, const name_t &name)
        {
            // First get the parameter line string
            const name_t toReturn = extractParameterLine(S, name);

            // Is it supposed an integral value?
            if constexpr (std::is_integral_v<T>)
            {
                if (isNumber(toReturn))
                {
                    // Check if T is an unsigned integral type
                    if constexpr (std::is_unsigned_v<T>)
                    {
                        return static_cast<T>(std::stoul(toReturn));
                    }
                    // T must be a signed integral type
                    else
                    {
                        return static_cast<T>(std::stol(toReturn));
                    }
                }
            }
            // Is it supposed a floating ponit value?
            else if constexpr (std::is_floating_point_v<T>)
            {
                return static_cast<T>(std::stold(toReturn));
            }
            // Is it supposed a string?
            else if constexpr (std::is_same_v<T, name_t>)
            {
                return toReturn;
            }

            return 0;
        }

        template <typename T>
        __host__ [[nodiscard]] T extractParameter(const name_t &toReturn)
        {
            // Is it supposed an integral value?
            if constexpr (std::is_integral_v<T>)
            {
                if (isNumber(toReturn))
                {
                    // Check if T is an unsigned integral type
                    if constexpr (std::is_unsigned_v<T>)
                    {
                        return static_cast<T>(std::stoul(toReturn));
                    }
                    // T must be a signed integral type
                    else
                    {
                        return static_cast<T>(std::stol(toReturn));
                    }
                }
            }
            // Is it supposed a floating point value?
            else if constexpr (std::is_floating_point_v<T>)
            {
                return static_cast<T>(std::stold(toReturn));
            }
            // Is it supposed a string?
            else if constexpr (std::is_same_v<T, name_t>)
            {
                return toReturn;
            }

            return 0;
        }

        /**
         * @brief Extract a three‑component vector from a configuration file.
         *
         * Reads the file to obtain the x, y, z components using keys formed by appending
         * "x", "y", "z" to the prefix. Each component is extracted via
         * string::extractParameter<value_type>`
         *
         * @tparam T Vector type with nested value_type and constructible from three values.
         * @param[in] fileName Configuration file path.
         * @param[in] prefix Common prefix for component keys (e.g., "block").
         * @return T constructed from the three extracted values.
         **/
        template <typename T>
        __host__ [[nodiscard]] const T extractParameter(const name_t &fileName, const name_t &prefix) noexcept
        {
            using value_type = typename T::value_type;

            const words_t fileLines = string::readFile(fileName);

            return {string::extractParameter<value_type>(fileLines, prefix + "x"),
                    string::extractParameter<value_type>(fileLines, prefix + "y"),
                    string::extractParameter<value_type>(fileLines, prefix + "z")};
        }

        /**
         * @brief Parses a name-value pair
         * @param[in] args The list of arguments to be searched
         * @param[in] name The argument to be searched for
         * @return A std::string_view of the value argument corresponding to name
         **/
        __host__ [[nodiscard]] const name_t parseNameValuePair(const words_t &args, const name_t &name)
        {
            // Loop over the input arguments and search for name
            for (device::label_t i = 0; i < args.size(); i++)
            {
                // The name argument exists, so handle it
                if (args[i] == name)
                {
                    // First check to see that i + 1 is not out of bounds
                    if (i + 1 < args.size())
                    {
                        return args[i + 1];
                    }
                    // Otherwise it is out of bounds: the supplied argument is the last argument and no value pair has been supplied
                    else
                    {
                        throw std::runtime_error("Input argument " + name_t(name) + name_t(" has not been supplied with a value; the correct syntax is -GPU 0,1 for example"));
                        return "";
                    }
                }
            }
            throw std::runtime_error("Input argument " + name_t(name) + name_t(" has not been supplied"));
            return "";
        }

        /**
         * @brief Parses the value of the argument following name
         * @param[in] argc First argument passed to main
         * @param[in] argv Second argument passed to main
         * @return A vector of integral type T
         * @note This function can be used to parse arguments passed to the executable on the command line such as -GPU 0,1
         **/
        template <typename T>
        __host__ [[nodiscard]] const std::vector<T> parseValue(const words_t &args, const name_t &name)
        {
            const words_t s_v = string::split<","[0], true>(parseNameValuePair(args, name));

            std::vector<T> arr;
            device::label_t arrLength = 0;

            for (device::label_t i = 0; i < s_v.size(); i++)
            {
                // Should check here if the string converts to a negative number and exit
                if constexpr (std::is_signed_v<T>)
                {
                    if (isNumber(s_v[i]))
                    {
                        arr.push_back(std::stoi(name_t(s_v[i])));
                    }
                    else
                    {
                        throw std::runtime_error(name_t(name) + name_t(" is not numeric"));
                    }
                }
                else
                {
                    if (isNumber(s_v[i]))
                    {
                        arr.push_back(std::stoul(name_t(s_v[i])));
                    }
                    else
                    {
                        throw std::runtime_error(name_t("Value supplied to argument ") + name_t(name) + name_t(" is not numeric"));
                    }
                }
                arrLength = arrLength + 1;
            }

            arr.resize(arrLength);

            return arr;
        }
    }
}

#endif