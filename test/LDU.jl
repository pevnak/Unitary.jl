using FiniteDifferences, LinearAlgebra, Unitary
using Unitary: lowdup, inverted_lowdup, ∇mulaxldu, ∇mulxaldu, ∇mulaxildu, ∇mulxaildu
using Unitary: ∇Matrixldu, ∇Matrixildu
using Test, Flux
using Zygote: gradient

@testset "transposition and inverse" begin
	a = lowdup(5)
	@test Matrix(transpose(a)) ≈ transpose(Matrix(a))
	@test Matrix(inv(a)) ≈ inv(Matrix(a))
	a = inverted_lowdup(5)
	@test Matrix(transpose(a)) ≈ transpose(Matrix(a))
	@test Matrix(inv(a)) ≈ inv(Matrix(a))
end

@testset "Test matrix gradient" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	Δ = rand(5, 5)
	@test grad(cfdm, m -> sum(Δ.*Matrix(lowdup(m))), m)[1] ≈ ∇Matrixldu(Δ, m)[1]
	@test grad(cfdm, m -> sum(Δ.*Matrix(inverted_lowdup(m))), m)[1] ≈ ∇Matrixildu(Δ, m)[1]
end

@testset "Testing matrix integration with Flux" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	a = lowdup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(Matrix(a))), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(Matrix(lowdup(m)))), m)[1]
	a = inverted_lowdup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(Matrix(a))), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(Matrix(inverted_lowdup(m)))), m)[1]
end

@testset "Testing lowdup-lowdup multiplication integration with Flux" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	n = rand(5, 5)
	a = lowdup(m)
	b = lowdup(n)
	ia = inverted_lowdup(m)
	ib = inverted_lowdup(n)
	ps = Flux.params(a, b)
	ips = Flux.params(ia, ib)
	@test gradient(() -> sum(sin.(a * b)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowdup(m) * lowdup(n))), m)[1]
	@test gradient(() -> sum(sin.(a * b)), ps)[b.m] ≈
	grad(cfdm, n -> sum(sin.(lowdup(m) * lowdup(n))), n)[1]
	@test gradient(() -> sum(sin.(ia * b)), ips)[ia.m] ≈
	grad(cfdm, m -> sum(sin.(inverted_lowdup(m) * lowdup(n))), m)[1]
	@test gradient(() -> sum(sin.(ia * b)), ps)[b.m] ≈
	grad(cfdm, n -> sum(sin.(inverted_lowdup(m) * lowdup(n))), n)[1]
	@test gradient(() -> sum(sin.(a * ib)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowdup(m) * inverted_lowdup(n))), m)[1]
	@test gradient(() -> sum(sin.(a * ib)), ips)[ib.m] ≈
	grad(cfdm, n -> sum(sin.(lowdup(m) * inverted_lowdup(n))), n)[1]
	@test gradient(() -> sum(sin.(ia * ib)), ips)[ia.m] ≈
	grad(cfdm, m -> sum(sin.(inverted_lowdup(m) * inverted_lowdup(n))), m)[1]
	@test gradient(() -> sum(sin.(ia * ib)), ips)[ib.m] ≈
	grad(cfdm, n -> sum(sin.(inverted_lowdup(m) * inverted_lowdup(n))), n)[1]
end

@testset "Test transposition integration with Flux" begin
	m = rand(5, 5)
	cfdm = central_fdm(5, 1)
	d = rand(5, 5)
	#construct lowdup in gradient
	@test gradient(m -> sum(d.*Matrix(transpose(lowdup(m)))), m)[1] ≈
	grad(cfdm, m -> sum(d.*Matrix(transpose(lowdup(m)))), m)[1]
	@test gradient(m -> sum(d.*Matrix(transpose(inverted_lowdup(m)))), m)[1] ≈
	grad(cfdm, m -> sum(d.*Matrix(transpose(inverted_lowdup(m)))), m)[1]
	#calculate grad of preconstructed lowdup
	a = lowdup(m)
	psa = Flux.params(a)
	b = inverted_lowdup(m)
	psb = Flux.params(b)
	#trans
	@test gradient(() -> sum(d.*Matrix(Unitary.trans(a))), psa)[a.m] ≈
	grad(cfdm, m -> sum(d.*Matrix(Unitary.trans(lowdup(m)))), m)[1]
	@test gradient(() -> sum(d.*Matrix(Unitary.trans(b))), psb)[b.m] ≈
	grad(cfdm, m -> sum(d.*Matrix(Unitary.trans(inverted_lowdup(m)))), m)[1]
	#transpose without Flux.params
	@test gradient(a -> sum(d.*Matrix(transpose(a))), a)[1][:m] ≈
	grad(cfdm, m -> sum(d.*Matrix(transpose(lowdup(m)))), m)[1]
	@test gradient(b -> sum(d.*Matrix(transpose(b))), b)[1][:m] ≈
	grad(cfdm, m -> sum(d.*Matrix(transpose(inverted_lowdup(m)))), m)[1]
	#transpose with Flux.params, is broken upstream
	@test_broken gradient(() -> sum(d.*Matrix(transpose(a))), psa)[a.m] ≈
	grad(cfdm, m -> sum(d.*Matrix(transpose(lowdup(m)))), m)[1]
	@test_broken gradient(() -> sum(d.*Matrix(transpose(b))), psb)[b.m] ≈
	grad(cfdm, m -> sum(d.*Matrix(transpose(inverted_lowdup(m)))), m)[1]
end

@testset "Test multiplacation gradient functions" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	x = rand(5, 10)
	Δ = ones(size(m*x))
	xt = rand(10, 5)
	Δt = ones(size(xt*m))
	@test grad(cfdm, m -> sum(lowdup(m) * x ), m )[1] ≈ ∇mulaxldu(Δ , m, x )[1]
	@test grad(cfdm, x -> sum(lowdup(m) * x ), x )[1] ≈ ∇mulaxldu(Δ , m, x )[2]
	@test grad(cfdm, m -> sum(xt * lowdup(m)), m )[1] ≈ ∇mulxaldu(Δt, m, xt)[1]
	@test grad(cfdm, x -> sum(xt * lowdup(m)), xt)[1] ≈ ∇mulxaldu(Δt, m, xt)[2]
	#inverted versions
	@test grad(cfdm, m -> sum(inverted_lowdup(m) * x ), m )[1] ≈
	∇mulaxildu(Δ , m, x )[1]
	@test grad(cfdm, x -> sum(inverted_lowdup(m) * x ), x )[1] ≈
	∇mulaxildu(Δ , m, x )[2]
	@test grad(cfdm, m -> sum(xt * inverted_lowdup(m)), m )[1] ≈
	∇mulxaildu(Δt, m, xt)[1]
	@test grad(cfdm, x -> sum(xt * inverted_lowdup(m)), xt)[1] ≈
	∇mulxaildu(Δt, m, xt)[2]
end

@testset "Testing multiplication integration with Flux" begin
	m = rand(5, 5)
	x = rand(5, 10)
	xt = rand(10, 5)
	cfdm = central_fdm(5, 1)
	a = lowdup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowdup(m) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(lowdup(m) * x)), x)[1]
	@test gradient(() -> sum(sin.(xt * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(xt * lowdup(m))), m)[1]
	@test gradient(xt -> sum(sin.(xt * a)), xt)[1] ≈
	grad(cfdm, xt -> sum(sin.(xt * lowdup(m))), xt)[1]
	#invereted versions
	a = inverted_lowdup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(inverted_lowdup(m) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(inverted_lowdup(m) * x)), x)[1]
	@test gradient(() -> sum(sin.(xt * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(xt * inverted_lowdup(m))), m)[1]
	@test gradient(xt -> sum(sin.(xt * a)), xt)[1] ≈
	grad(cfdm, xt -> sum(sin.(xt * inverted_lowdup(m))), xt)[1]
end
