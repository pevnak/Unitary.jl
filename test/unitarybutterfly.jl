using Unitary, Test, Flux
using Unitary: Butterfly
using FiniteDifferences
using Unitary: _mulax!, _∇mulax, _∇mulxa

@testset "UnitaryButterfly: multiplication, transposition, and, inversion" begin
	a = Butterfly(4)
	x = randn(4,4)

	am = Matrix(a)
	@testset "a * x" begin
		@test a * x ≈ am * x
		@test inv(a) * x ≈ inv(am) * x
		@test transpose(a) * x ≈ transpose(am) * x
	end

	@testset "x * a" begin
		@test x * a ≈ x * am
		@test x * transpose(a) ≈ x * transpose(am)
		@test x * inv(a) ≈ x * inv(am)
	end
end

@testset "inplace multiplication" begin 
	a = Butterfly(5)
	x = randn(5,5)
	fdm = central_fdm(5, 1)
	@testset "a * x" begin
		o = a * x;
		Δ = ones(size(x));
		∇θ = _∇mulax(Δ, a.θs, a.idxs, o, 1)[1]

		∇θr = grad(fdm, θ -> sum(Butterfly(θ, a.idxs, 5) * x), a.θs)[1]
		@test ∇θ ≈ ∇θr

		ps = Flux.params(a)
		∇θr = gradient(() -> sum(a * x), ps)[a.θs]
		@test ∇θr ≈ ∇θr
	end

	@testset "x * a" begin
		o = x * a;
		Δ = ones(size(x));

		∇θr = grad(fdm, θ -> sum(x * Butterfly(θ, a.idxs, 5)), a.θs)[1]
		∇xr = grad(fdm, x -> sum(x * a), x)[1]

		∇x, ∇θ = _∇mulxa(Δ, o, a.θs, a.idxs, 1)[1:2]
		@test ∇θ ≈ ∇θr
		@test ∇x ≈ ∇xr

		ps = Flux.params(a)
		∇θ = gradient(() -> sum(x * a), ps)[a.θs]
		∇x = gradient(x -> sum(x * a), x)[1]
		@test ∇x ≈ ∇xr
		@test ∇θr ≈ ∇θr
	end
end