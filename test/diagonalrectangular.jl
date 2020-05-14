using Test, Unitary, Flux, Zygote, LinearAlgebra
using Unitary: DiagonalRectangular, diagmul
using FiniteDifferences


d = DiagonalRectangular([1.0,2.0], 2, 2)
ps = params(d)
gradient(() -> logabsdet(d), ps)


fdm = central_fdm(5, 1);

@testset  "DiagonalRectangular: mul" begin
	ar = [0.5 0 0; 0 2 0]
	a = DiagonalRectangular([0.5, 2], 2, 3)
	for x in  [randn(3, 5), transpose(randn(5, 3))]
		@test ar * x ≈ a * x
	end

	for x in  [randn(5, 2), transpose(randn(2,5))]
		@test x * ar ≈ x * a
	end


	ar = [0.5 0; 0 2; 0 0]
	a = DiagonalRectangular([0.5, 2], 3, 2)
	for x in  [randn(5, 3), transpose(randn(3, 5))]
		@test x * ar ≈ x * a
	end
	for x in  [randn(2, 5), transpose(randn(5,2))]
		@test ar * x ≈ a * x
	end
end

@testset  "DiagonalRectangular: Matrix" begin
	ar = [0.5 0 0; 0 2 0]
	a = DiagonalRectangular([0.5, 2], 2, 3)
	@test Matrix(a) ≈ ar

	ar = [0.5 0; 0 2; 0 0]
	a = DiagonalRectangular([0.5, 2], 3, 2)
	@test Matrix(a) ≈ ar
end


@testset  "DiagonalRectangular: transpose" begin
	a = DiagonalRectangular([0.5, 2], 2, 3)
	for x in  [randn(3, 5), transpose(randn(5, 3))]
		@test a * x ≈ transpose(transpose(x) * transpose(a))
	end

	a = DiagonalRectangular([0.5, 2], 3, 2)
	for x in  [randn(5, 3), transpose(randn(3, 5))]
		@test x * a ≈ transpose(transpose(a) * transpose(x))
	end
end

@testset  "DiagonalRectangular: inv" begin
	a = DiagonalRectangular([0.5, 2], 2, 3)
	ai = DiagonalRectangular([2, 0.5], 3, 2)
	x = randn(3, 5)
	@test ai * (a * x) ≈ vcat(x[1:2, :], zeros(1,5))	

	x = randn(5, 2)
	@test (x * a) * ai ≈ x

	inv(a) == ai
end

@testset  "DiagonalRectangular: gradient" begin
	for (n, m) in [(3,2), (2,3), (3,3)]
		a = DiagonalRectangular(rand(min(n,m)), n, m)
		x = randn(m, n)
		am = Matrix(a)

		@test Flux.gradient(x -> sum(sin.(x * am)), x)[1] ≈ Flux.gradient(x -> sum(sin.(x * a)), x)[1]
		@test isapprox(grad(fdm, d -> sum(sin.(diagmul(x, d, a.n, a.m))), a.d)[1],  Flux.gradient(a -> sum(sin.(x * a)), a)[1][1], atol = 1e-6)

		@test Flux.gradient(x -> sum(sin.(am * x)), x)[1] ≈ Flux.gradient(x -> sum(sin.(a * x)), x)[1]
		@test isapprox(grad(fdm, d -> sum(sin.(diagmul(d, a.n, a.m, x))), a.d)[1],  Flux.gradient(a -> sum(sin.(a * x)), a)[1][1], atol = 1e-6)

		ps = params(a)
		@test length(ps) == 1
		@test isapprox(grad(fdm, d -> sum(sin.(diagmul(d, a.n, a.m, x))), a.d)[1],
			gradient(() -> sum(sin.(a * x)),ps)[a.d], atol = 1e-6)
	end

	n, m = 3, 2
	a = DiagonalRectangular(rand(min(n,m)), n, m)
	x = randn(m)
	am = Matrix(a)

	@test Flux.gradient(x -> sum(sin.(am * x)), x)[1] ≈ Flux.gradient(x -> sum(sin.(a * x)), x)[1]
	@test isapprox(grad(fdm, d -> sum(sin.(diagmul(d, a.n, a.m, x))), a.d)[1],  Flux.gradient(a -> sum(sin.(a * x)), a)[1][1], atol = 1e-6)
end

@testset  "DiagonalRectangular: integration with flux" begin
	n,m = 2,3
	a = DiagonalRectangular(rand(min(n,m)), n, m)
	x = randn(m, n)
	ps = params(a)
	@test length(ps) == 1
	@test isapprox(grad(fdm, d -> sum(sin.(diagmul(d, a.n, a.m, x))), a.d)[1],
		gradient(() -> sum(sin.(a * x)),ps)[a.d], atol = 1e-6)
end

@testset "DiagonalRectangular: logabsdet" begin 
	d = DiagonalRectangular([1.0,2.0], 2, 2)
	ps = params(d)
	gradient(() -> logabsdet(d), ps)

	f, back = Zygote.pullback(() -> logabsdet(d), ps)
end

