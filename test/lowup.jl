using FiniteDifferences, LinearAlgebra, Unitary
using Unitary: lowup, mulax, ∇mulax, ∇mulxa, ∇mulax_inv, ∇mulxa_inv
using Test, Flux
using Zygote: gradient


@testset "Test gradient functions" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	x = rand(5, 5)
	Δ = ones(size(m*x))
	ax = x
	xa = x
	@test grad(cfdm, m -> sum(lowup(m) * x), m)[1] ≈
	∇mulax(Δ, m, ax)[1]
	@test grad(cfdm, x -> sum(lowup(m) * x), x)[1] ≈
	∇mulax(Δ, m, ax)[2]
	@test grad(cfdm, m -> sum(x*lowup(m)), m)[1] ≈
	∇mulxa(Δ, m, xa)[1]
	@test grad(cfdm, x -> sum(x*lowup(m)), x)[1] ≈
	∇mulxa(Δ, m, xa)[2]
end

@testset "Test gradient functions using inversions" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	x = rand(5, 5)
	Δ = ones(size(m*x))
	ax = lowup(m)*x
	xa = x*lowup(m)
	@test grad(cfdm, m -> sum(lowup(m) * x), m)[1] ≈
	∇mulax_inv(Δ, m, ax)[1]
	@test grad(cfdm, x -> sum(lowup(m) * x), x)[1] ≈
	∇mulax_inv(Δ, m, ax)[2]
	@test grad(cfdm, m -> sum(x*lowup(m)), m)[1] ≈
	∇mulxa_inv(Δ, m, xa)[1]
	@test grad(cfdm, x -> sum(x*lowup(m)), x)[1] ≈
	∇mulxa_inv(Δ, m, xa)[2]
end

@testset "Testing integration with Flux" begin
	m = rand(5, 5)
	x = rand(5, 10)
	a = lowup(m)
	ps = Flux.params(a)
	cfdm = central_fdm(5, 1)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowup(m) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(lowup(m) * x)), x)[1]
	x = rand(10, 5)
	@test gradient(() -> sum(sin.(x * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(x * lowup(m))), m)[1]
	@test gradient(x -> sum(sin.(x * a)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(x * lowup(m))), x)[1]
end
