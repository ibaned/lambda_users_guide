# The Lambda User's Guide

## What is a lambda ?

In the context of the C++11 language standard, a lambda is a new syntactic feature
that essentially automatically creates a functor.
Consider the standard library's templated `std::find_if` function.
Prior to C++11, functors were the way to use this function:

```cpp
struct IsTheSameAsX {
  int x;
  IsTheSameAsX(int given_x) { x = given_x; }
  bool operator()(int const& y) const { return y == x; }
};

std::vector<int>::iterator find_equal(std::vector<int> const& v, int x) {
  return std::find_if(v.begin(), v.end(), IsTheSameAsX(x));
}
```

In C++11, the lambda syntax
```
[captures] (arguments) -> result_type { code }
```
allows automatically creating functors without needing to give them a name,
and most importantly it automatically handles the member variables of the functor.
For example, the following should be exactly equivalent to our previous usage
of `std::find_if`:

```cpp
std::vector<int>::iterator find_equal(std::vector<int> const& v, int x) {
  return std::find_if(v.begin(), v.end(), [=] (int const& y) { return y == x; });
}
```

The `[=]` capture list means two things:
 - The compiler should determine which variables are to be captured, based on the code inside the lambda 
   (alternatively one may explicitly list the variables to capture).
 - All such captures should be done by value (indicated by the `=`), as opposed to by reference
   (which would be indicated by an `&`).

One can see in comparing the examples above that lambdas offer a great advantage
in terms of the number of lines of code.
Also, because the functor created is an unnamed implicit type, there is no need for
developers to come up with names like `IsTheSameAsX` or place its definition outside
the function.
Finally, note that the result type of this lambda was automatically determined based
on the `return` statement inside of it.

## Lambdas in Kokkos

Like `std::find_if`, `Kokkos::parallel_for` accepts user-defined functors to essentially
implement a zero-overhead callback to user code.
If used properly, lambdas can provide their intended benefits to Kokkos-calling code as well.
However, there are additional restrictions imposed by CUDA and to a lesser extent OpenMP,
which are the Kokkos backends.

### CUDA support for lambdas

CUDA is currently Kokkos' only backend which can utilize GPUs, and it makes use of a custom
C++ compiler called NVCC.
This compiler has its own set of supported and non-supported C++ features, separate from
the usual host (CPU) compiler.
At the time of this writing, the two most widely used versions of CUDA are version 7.5 and version 8.0.

CUDA defines function attributes that define how a function will be compiled by NVCC:

1. `__host__` A version of this function will be compiled for execution on the CPU host.
1. `__device__` A version of this function will be compiled for execution on the GPU device.
1. `__global__` This function is a kernel dispatched from the host.
   Users of Kokkos need not know about `__global__`, only internal Kokkos functions have this attribute.

As an example, this function can be called at any point in a code
that is compiled by NVCC:

```cpp
__host__ __device__ double funny_multiply(double a, double b) {
  return a * b - 1e-6;
}

Kokkos provides convenience macros which expand to these attributes when using CUDA and
expand to nothing otherwise:

```cpp
#ifdef KOKKOS_HAVE_CUDA
#define KOKKOS_INLINE_FUNCTION __host__ __device__
#else
#define KOKKOS_INLINE_FUNCTION
#endif
```

These macros exist to satisfy the "single-source" principle
that Kokkos strives for, i.e. the following code using Kokkos
can be compiled for any machine by changing only configuration settings:

```cpp
KOKKOS_INLINE_FUNCTION double funny_multiply(double a, double b) {
  return a * b - 1e-6;
}
```

Note that by default the attributes are both `__host__` and `__device__`, which instructs
the NVCC compiler to generate both CPU and GPU versions of the annotated function.

Like functions, lambdas can have CUDA attributes.
CUDA 7.5 and CUDA 8.0 both support the `__device__` attribute on a lambda (`[=] __device__ (int x) { ... }`).
These are referred to as "device lambdas".
CUDA 8.0 supports lambdas with both attributes (`[=] __host__ __device__ (int x) { ... }`),
which are referred to as "host-device lambdas".

Note the trade-off with CUDA 7.5: one can use lambdas, but they will only be executable
on the GPU, so one cannot compile a single "kernel" which can execute on both the CPU and GPU.

### Lambdas inside classes
