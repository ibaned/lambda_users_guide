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
[captures] (arguments) -> returns { code }
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
