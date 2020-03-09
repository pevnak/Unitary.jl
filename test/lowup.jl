using FiniteDifferences, LinearAlgebra, Unitary
using Unitary: lowup, ∇mulaxlu, ∇mulxalu
using Test, Flux
using Zygote: gradient

@testset "transposition and inverse" begin
	a = lowup(5)
	@test Matrix(transpose(a)) ≈ transpose(Matrix(a))
	@test Matrix(inv(a)) ≈ inv(Matrix(a))
	a = inv(a)
	@test Matrix(transpose(a)) ≈ transpose(Matrix(a))
	@test Matrix(inv(a)) ≈ inv(Matrix(a))
end

@testset "Test gradient functions" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	x = rand(5, 10)
	Δ = ones(size(m*x))
	invs = false
	ax = lowup(m, invs)*x
	xt = rand(10, 5)
	Δt = ones(size(xt*m))
	xa = xt*lowup(m, invs)
	@test grad(cfdm, m -> sum(lowup(m, invs) * x), m)[1] ≈
	∇mulaxlu(Δ, m, invs, ax)[1]
	@test grad(cfdm, x -> sum(lowup(m, invs) * x), x)[1] ≈
	∇mulaxlu(Δ, m, invs, ax)[2]
	@test grad(cfdm, m -> sum(xt*lowup(m, invs)), m)[1] ≈
	∇mulxalu(Δt, m, invs, xa)[1]
	@test grad(cfdm, x -> sum(xt*lowup(m, invs)), xt)[1] ≈
	∇mulxalu(Δt, m, invs, xa)[2]
	invs = true
	ax = lowup(m, invs)*x
	xa = xt*lowup(m, invs)
	@test grad(cfdm, m -> sum(lowup(m, invs) * x), m)[1] ≈
	∇mulaxlu(Δ, m, invs, ax)[1]
	@test grad(cfdm, x -> sum(lowup(m, invs) * x), x)[1] ≈
	∇mulaxlu(Δ, m, invs, ax)[2]
	@test grad(cfdm, m -> sum(xt*lowup(m, invs)), m)[1] ≈
	∇mulxalu(Δt, m, invs, xa)[1]
	@test grad(cfdm, x -> sum(xt*lowup(m, invs)), xt)[1] ≈
	∇mulxalu(Δt, m, invs, xa)[2]
end

@testset "Testing integration with Flux" begin
	m = rand(5, 5)
	x = rand(5, 10)
	xt = rand(10, 5)
	cfdm = central_fdm(5, 1)
	invs = false
	a = lowup(m, invs)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowup(m, invs) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(lowup(m, invs) * x)), x)[1]
	@test gradient(() -> sum(sin.(xt * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(xt * lowup(m, invs))), m)[1]
	@test gradient(xt -> sum(sin.(xt * a)), xt)[1] ≈
	grad(cfdm, xt -> sum(sin.(xt * lowup(m, invs))), xt)[1]
	invs = true
	a = lowup(m, invs)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowup(m, invs) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(lowup(m, invs) * x)), x)[1]
	@test gradient(() -> sum(sin.(xt * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(xt * lowup(m, invs))), m)[1]
	@test gradient(xt -> sum(sin.(xt * a)), xt)[1] ≈
	grad(cfdm, xt -> sum(sin.(xt * lowup(m, invs))), xt)[1]
end
