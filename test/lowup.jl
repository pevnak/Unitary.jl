using FiniteDifferences, LinearAlgebra, Unitary
using Unitary: lowup, inverted_lowup, ∇mulaxlu, ∇mulxalu, ∇mulaxilu, ∇mulxailu
using Unitary: ∇Matrixlu, ∇Matrixilu, transposelu, transposeilu, ∇transposelu, ∇transposeilu
using Test, Flux
using Zygote: gradient

@testset "transposition and inverse" begin
	a = lowup(5)
	@test Matrix(transpose(a)) ≈ transpose(Matrix(a))
	@test Matrix(inv(a)) ≈ inv(Matrix(a))
	a = inverted_lowup(5)
	@test Matrix(transpose(a)) ≈ transpose(Matrix(a))
	@test Matrix(inv(a)) ≈ inv(Matrix(a))
end

@testset "Test transposition gradient functions" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5) + I
	Δ = rand(5, 5)
	@test grad(cfdm, m -> sum(Δ .* transposelu(m)), m)[1] ≈ ∇transposelu(Δ, m)[1]
	@test grad(cfdm, m -> sum(Δ .* transposeilu(m)), m)[1] ≈ ∇transposeilu(Δ, m)[1]
end

@testset "Test transposition integration with flux" begin
	m = rand(5, 5)
	cfdm = central_fdm(5, 1)
	a = lowup(m)
	b = inverted_lowup(m)
	psa = Flux.params(a)
	psb = Flux.params(b)
	@test gradient(() -> sum(Matrix(transpose(a))), psa)[a.m] ≈
	grad(cfdm, m -> sum(Matrix(transpose(lowup(m)))), m)[1]
	@test gradient(() -> sum(Matrix(transpose(b))), psb)[b.m] ≈
	grad(cfdm, m -> sum(Matrix(transpose(inverted_lowup(m)))), m)[1]
end

@testset "Test multiplacation gradient functions" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	x = rand(5, 10)
	Δ = ones(size(m*x))
	xt = rand(10, 5)
	Δt = ones(size(xt*m))
	@test grad(cfdm, m -> sum(lowup(m) * x ), m )[1] ≈ ∇mulaxlu(Δ , m, x )[1]
	@test grad(cfdm, x -> sum(lowup(m) * x ), x )[1] ≈ ∇mulaxlu(Δ , m, x )[2]
	@test grad(cfdm, m -> sum(xt * lowup(m)), m )[1] ≈ ∇mulxalu(Δt, m, xt)[1]
	@test grad(cfdm, x -> sum(xt * lowup(m)), xt)[1] ≈ ∇mulxalu(Δt, m, xt)[2]
	#inverted versions
	@test grad(cfdm, m -> sum(inverted_lowup(m) * x ), m )[1] ≈ ∇mulaxilu(Δ , m, x )[1]
	@test grad(cfdm, x -> sum(inverted_lowup(m) * x ), x )[1] ≈ ∇mulaxilu(Δ , m, x )[2]
	@test grad(cfdm, m -> sum(xt * inverted_lowup(m)), m )[1] ≈ ∇mulxailu(Δt, m, xt)[1]
	@test grad(cfdm, x -> sum(xt * inverted_lowup(m)), xt)[1] ≈ ∇mulxailu(Δt, m, xt)[2]
end

@testset "Testing multiplication integration with Flux" begin
	m = rand(5, 5)
	x = rand(5, 10)
	xt = rand(10, 5)
	cfdm = central_fdm(5, 1)
	a = lowup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowup(m) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(lowup(m) * x)), x)[1]
	@test gradient(() -> sum(sin.(xt * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(xt * lowup(m))), m)[1]
	@test gradient(xt -> sum(sin.(xt * a)), xt)[1] ≈
	grad(cfdm, xt -> sum(sin.(xt * lowup(m))), xt)[1]
	#invereted versions
	a = inverted_lowup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(a * x)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(inverted_lowup(m) * x)), m)[1]
	@test gradient(x -> sum(sin.(a * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(inverted_lowup(m) * x)), x)[1]
	@test gradient(() -> sum(sin.(xt * a)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(xt * inverted_lowup(m))), m)[1]
	@test gradient(xt -> sum(sin.(xt * a)), xt)[1] ≈
	grad(cfdm, xt -> sum(sin.(xt * inverted_lowup(m))), xt)[1]
end

@testset "Test matrix gradient" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	Δ = ones(size(m))
	@test grad(cfdm, m -> sum(Matrix(lowup(m))), m)[1] ≈ ∇Matrixlu(Δ, m)[1]
	@test grad(cfdm, m -> sum(Matrix(inverted_lowup(m))), m)[1] ≈ ∇Matrixilu(Δ, m)[1]
end

@testset "Testing matrix integration with Flux" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	a = lowup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(Matrix(a))), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(Matrix(lowup(m)))), m)[1]
	a = inverted_lowup(m)
	ps = Flux.params(a)
	@test gradient(() -> sum(sin.(Matrix(a))), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(Matrix(inverted_lowup(m)))), m)[1]
end

@testset "Testing lowup-lowup multiplication integration with Flux" begin
	cfdm = central_fdm(5, 1)
	m = rand(5, 5)
	n = rand(5, 5)
	a = lowup(m)
	b = lowup(n)
	ia = inverted_lowup(m)
	ib = inverted_lowup(n)
	ps = Flux.params(a, b)
	ips = Flux.params(ia, ib)
	@test gradient(() -> sum(sin.(a * b)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowup(m) * lowup(n))), m)[1]
	@test gradient(() -> sum(sin.(a * b)), ps)[b.m] ≈
	grad(cfdm, n -> sum(sin.(lowup(m) * lowup(n))), n)[1]
	@test gradient(() -> sum(sin.(ia * b)), ips)[ia.m] ≈
	grad(cfdm, m -> sum(sin.(inverted_lowup(m) * lowup(n))), m)[1]
	@test gradient(() -> sum(sin.(ia * b)), ps)[b.m] ≈
	grad(cfdm, n -> sum(sin.(inverted_lowup(m) * lowup(n))), n)[1]
	@test gradient(() -> sum(sin.(a * ib)), ps)[a.m] ≈
	grad(cfdm, m -> sum(sin.(lowup(m) * inverted_lowup(n))), m)[1]
	@test gradient(() -> sum(sin.(a * ib)), ips)[ib.m] ≈
	grad(cfdm, n -> sum(sin.(lowup(m) * inverted_lowup(n))), n)[1]
	@test gradient(() -> sum(sin.(ia * ib)), ips)[ia.m] ≈
	grad(cfdm, m -> sum(sin.(inverted_lowup(m) * inverted_lowup(n))), m)[1]
	@test gradient(() -> sum(sin.(ia * ib)), ips)[ib.m] ≈
	grad(cfdm, n -> sum(sin.(inverted_lowup(m) * inverted_lowup(n))), n)[1]
end
