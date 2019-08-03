using Unitary, Test, LinearAlgebra
using Unitary: Butterfly


function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end


@testset "Assertions in Butterfly constructors" begin
	@test_throws AssertionError Butterfly([π/2,π], [1,3], [1,4], 4)
	@test_throws AssertionError Butterfly([π/2,π], [1,3], [1,4], 3)
	@test_throws AssertionError Butterfly([π/2], [1,3], [1,4], 4)
	@test_throws AssertionError Butterfly([π/2, π], [1], [1,4], 4)
	@test_throws AssertionError Butterfly([π/2, π], [1, 3], [1], 4)
end

@testset "Conversion of Butterfly to matrix" begin 
	@test Matrix(Butterfly([π/2,π], [1,3], [2,4], 4)) ≈ [0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1]
	@test Matrix(Butterfly([π/2,π], [1,4], [2,3], 4)) ≈ [0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1]
	@test Matrix(Butterfly([π/2,π], [1,2], [3,4], 4)) ≈ [0  0 -1 0;0 -1 0 0;1 0 0 0;0 0 0 -1]
	@test Matrix(Butterfly([π/2,π], [1,2], [4,3], 4)) ≈ [0  0 0 -1;0 -1 0 0;0 0 -1 0;1 0 0 0]
	@test Matrix(transpose(Butterfly([π/2,π], [1,3], [2,4], 4))) ≈ transpose([0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1])
	@test Matrix(transpose(Butterfly([π/2,π], [1,4], [2,3], 4))) ≈ transpose([0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1])
	@test Matrix(transpose(Butterfly([π/2,π], [1,2], [3,4], 4))) ≈ transpose([0  0 -1 0;0 -1 0 0;1 0 0 0;0 0 0 -1])
	@test Matrix(transpose(Butterfly([π/2,π], [1,2], [4,3], 4))) ≈ transpose([0  0 0 -1;0 -1 0 0;0 0 -1 0;1 0 0 0])
end


@testset "Multiplication" begin
	a = Butterfly([π/2,π], [1,3], [2,4], 4)
	m = Matrix(a)
	x = randn(4,4);
	@test a * x ≈ m * x
	@test x * a ≈ x * m
	@test transpose(a) * x ≈ transpose(m) * x
	@test x * transpose(a) ≈ x * transpose(m)
end


@testset "Testing calculation of the gradient with Flux" begin
	for x in [randn(4), randn(4, 10), transpose(randn(10, 4)), transpose(randn(1, 4))]
		for a in [Butterfly(randn(2), [1,3], [2,4], 4), Butterfly(randn(2), [1,2], [3,4], 4), Butterfly(randn(2), [1,4], [3,2], 4)]
			Δ = ones(size(x))
			@test isapprox(Unitary._∇mulax(Δ, a.θ, a.i, a.j, x, 1), ngradient(_θ -> sum(Unitary._mulax(_θ, a.i, a.j, x, 1)), a.θ)[1], atol = 1e-6)
			@test isapprox(Unitary._∇mulax(Δ, a.θ, a.i, a.j, x, -1), ngradient(_θ -> sum(Unitary._mulax(_θ, a.i, a.j, x, -1)), a.θ)[1], atol = 1e-6)
		end
	end

	for x in [rand(10, 4), rand(1, 4), transpose(rand(4,10)), transpose(rand(4)), transpose(rand(4,1))]
		for a in [Butterfly(randn(2), [1,3], [2,4], 4), Butterfly(randn(2), [1,2], [3,4], 4), Butterfly(randn(2), [1,4], [3,2], 4)]
			Δ = ones(size(x))
			@test isapprox(Unitary._∇mulxa(Δ, x, a.θ, a.i, a.j, 1), ngradient(_θ -> sum(Unitary._mulxa(x, _θ, a.i, a.j, 1)), a.θ)[1], atol = 1e-6)
			@test isapprox(Unitary._∇mulxa(Δ, x, a.θ, a.i, a.j, -1), ngradient(_θ -> sum(Unitary._mulxa(x, _θ, a.i, a.j, -1)), a.θ)[1], atol = 1e-6)
		end
	end
end
