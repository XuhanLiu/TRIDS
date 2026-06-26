#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
// MSVC does not treat and/or/not as operators unless /Za is enabled.
#if defined(_MSC_VER) && !defined(__cplusplus_cli)
#define and &&
#define or ||
#define not !
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// libtorch (std::byte) vs Windows SDK (global ::byte) on MSVC 2022.
#ifdef _MSC_VER
#pragma push_macro("byte")
#undef byte
#endif

// OpenBabel 3.1 still references removed C++17 std::binary_function on MSVC (host C++ only).
#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
#include <functional>
namespace std {
template <class Arg1, class Arg2, class Result>
struct binary_function {
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};
}  // namespace std
#endif

#ifdef _MSC_VER
#pragma pop_macro("byte")
#endif
