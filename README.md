# Unitary.jl

This package implements a differentiable parametrization of a group of unitary matrices as described in paper *Sum-Product-Transform Networks: Exploiting Symmetries using Invertible Transformations, Tomas Pevny, Vasek Smidl, Martin Trapp, Ondrej Polacek, Tomas Oberhuber, 2020* [https://arxiv.org/abs/2005.01297](https://arxiv.org/abs/2005.01297)

The actual "Dense" node implementing `f(x) = σ.(W * x .+ b)`, where `W` is in svd form has moved to [https://github.com/pevnak/SumProductTransform.jl](https://github.com/pevnak/SumProductTransform.jl) to keep this simple. Since in the paper, we have experimented with different ways, how to efficiently implement Dense matrices featuring efficient inversion and calculation of determinant, the repository contains a little bit more.

- `Givens` - representation of a unitary matrix using Givens rotations
- `UnitaryHouseholder` - representation of a unitary matrix using Householder reflections, an approach common in Machine Learning
- LU - representation of a matrix using LU decomposition
- LDU  - representation of a matrix using LDU decomposition

The usage is simple:
```
using Unitary, Flux, BenchmarkTools
using Unitary: Givens, lowup

x = randn(Float32, 50, 100)
xx = randn(Float32, 100, 50)

a = Givens(50)
@btime a * x;		
#  224.097 μs (4 allocations: 20.00 KiB)
@btime xx * a;	
#  79.517 μs (4 allocations: 20.00 KiB)

ps = Flux.params(a)
@btime gradient(() -> sum(a * x), ps);	# 890.323 μs (58 allocations: 71.52 KiB)
# 891.481 μs (60 allocations: 72.42 KiB)
@btime gradient(() -> sum(xx * a), ps);	# 473.158 μs (58 allocations: 71.52 KiB)
@ 468.794 μs (60 allocations: 72.42 KiB)

a = Givens(50)
@btime a * x;
# 646.874 μs (10154 allocations: 2.37 MiB)

@btime xx * a;
#  726.198 μs (10204 allocations: 2.39 MiB)

@btime gradient(() -> sum(a * x), ps);  
#  103.869 ms (44538 allocations: 179.60 MiB)

@btime gradient(() -> sum(xx * a), ps);
#  105.061 ms (44688 allocations: 179.67 MiB)
```

Matrices support only multiplication, because that is what they have been designed for, but you can always convert them to normal matrices using `Matrix` (but this is not at the moment differentiable). 
