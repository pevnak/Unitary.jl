using Unitary, Test, Flux
using Unitary: Givens
using FiniteDifferences
using Unitary: _mulax!, _∇mulax, _∇mulxa

@testset "Conversion of Givens to matrix" begin 
	@test Matrix(Givens([π/2,π], [(1,2), (3,4)], 4)) ≈ [0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1]
	@test Matrix(Givens([π/2,π], [(1,2), (4,3)], 4)) ≈ [0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1]
	@test Matrix(Givens([π/2,π], [(1,3), (2,4)], 4)) ≈ [0  0 -1 0;0 -1 0 0;1 0 0 0;0 0 0 -1]
	@test Matrix(Givens([π/2,π], [(1,4), (2,3)], 4)) ≈ [0  0 0 -1;0 -1 0 0;0 0 -1 0;1 0 0 0]
	@test Matrix(transpose(Givens([π/2,π], [(1,2), (3,4)], 4))) ≈ transpose([0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1])
	@test Matrix(transpose(Givens([π/2,π], [(1,2), (4,3)], 4))) ≈ transpose([0 -1 0 0;1 0 0 0;0 0 -1 0;0 0 0 -1])
	@test Matrix(transpose(Givens([π/2,π], [(1,3), (2,4)], 4))) ≈ transpose([0  0 -1 0;0 -1 0 0;1 0 0 0;0 0 0 -1])
	@test Matrix(transpose(Givens([π/2,π], [(1,4), (2,3)], 4))) ≈ transpose([0  0 0 -1;0 -1 0 0;0 0 -1 0;1 0 0 0])
end

@testset "UnitaryGivens: multiplication, transposition, and, inversion" begin
	a = Givens(4)
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
	a = Givens(5)
	x = randn(5,5)
	fdm = central_fdm(5, 1)
	@testset "a * x" begin
		o = a * x;
		Δ = ones(size(x));
		∇θr = grad(fdm, θ -> sum(Givens(θ, a.idxs, 5) * x), a.θs)[1]
		∇xr = grad(fdm, x -> sum(a * x), x)[1]

		∇θ, ∇x = _∇mulax(Δ, a.θs, a.idxs, o, 1)[[1,3]]
		@test ∇θ ≈ ∇θr
		@test ∇x ≈ ∇xr

		ps = Flux.params(a)
		∇θr = gradient(() -> sum(a * x), ps)[a.θs]
		∇x = gradient(x -> sum(a * x), x)[1]
		@test ∇θr ≈ ∇θr
		@test ∇x ≈ ∇xr
	end

	@testset "transpose(a) * x" begin
		o = transpose(a) * x;
		Δ = ones(size(x));
		∇θr = grad(fdm, θ -> sum(transpose(Givens(θ, a.idxs, 5)) * x), a.θs)[1]
		∇xr = grad(fdm, x -> sum(transpose(a)* x), x)[1]

		∇θ, ∇x = _∇mulax(Δ, a.θs, a.idxs, o, -1)[[1,3]]
		@test ∇θ ≈ ∇θr
		@test ∇x ≈ ∇xr

		ps = Flux.params(a)
		∇θr = gradient(() -> sum(transpose(a) * x), ps)[a.θs]
		∇x = gradient(x -> sum(transpose(a)* x), x)[1]
		@test ∇θr ≈ ∇θr
		@test ∇x ≈ ∇xr
	end

	@testset "x * a" begin
		o = x * a;
		Δ = ones(size(x));

		∇θr = grad(fdm, θ -> sum(x * Givens(θ, a.idxs, 5)), a.θs)[1]
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

	@testset "x * a" begin
		o = x * transpose(a);
		Δ = ones(size(x));

		∇θr = grad(fdm, θ -> sum(x * transpose(Givens(θ, a.idxs, 5))), a.θs)[1]
		∇xr = grad(fdm, x -> sum(x * transpose(a)), x)[1]

		∇x, ∇θ = _∇mulxa(Δ, o, a.θs, a.idxs, -1)[1:2]
		@test ∇θ ≈ ∇θr
		@test ∇x ≈ ∇xr

		ps = Flux.params(a)
		∇θ = gradient(() -> sum(x * transpose(a)), ps)[a.θs]
		∇x = gradient(x -> sum(x * transpose(a)), x)[1]
		@test ∇x ≈ ∇xr
		@test ∇θr ≈ ∇θr
	end
end