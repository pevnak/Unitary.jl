using Unitary, Test, Flux
using Unitary: Butterfly
using FiniteDifferences
using Unitary: _mulax!, _∇mulax

@testset "UnitaryButterfly: multiplication, transposition, and, inversion" begin
	a = Butterfly(randn(2), [1,3], [2,4], 4)
	b = Butterfly(randn(2), [1,2], [3,4], 4)
	c = Butterfly(randn(2), [1,4], [3,2], 4)
	ub = UnitaryButterfly((a,b,c), 4)
	x = randn(4,4)

	@test a*(b*(c*x)) ≈ ub * x
	@test ((x*a)*b)*c ≈ x * ub

	@test transpose(c)*(transpose(b)*(transpose(a)*x)) ≈ transpose(ub) * x
	@test ((x*transpose(c))*transpose(b))*transpose(a) ≈ x * transpose(ub)
	@test transpose(transpose(ub)) == ub
	@test inv(inv(ub)) == ub

	for x in [randn(4), randn(4, 10), transpose(randn(10, 4)), transpose(randn(1, 4))]
		@test ub * (inv(ub) * x) ≈ x
		@test inv(ub) * (ub * x) ≈ x
	end

	for x in [rand(10, 4), rand(1, 4), transpose(rand(4,10)), transpose(rand(4)), transpose(rand(4,1))]
		@test (x * ub) * inv(ub) ≈ x
		@test (x * inv(ub)) * ub ≈ x
	end
end

@testset "inplace multiplication" begin 
	a = Butterfly(5)
	x = randn(5,5)
	fdm = central_fdm(5, 1)
	o = a * x;
	Δ = ones(size(x));
	∇θ = _∇mulax(Δ, a.θs, a.idxs, o, 1)[1]

	function f(x,θs,idxs) 
		o = deepcopy(x)
		for i in 1:length(idxs)
			_mulax!(o, cos(θs[i]), sin(θs[i]), idxs[i][1], idxs[i][2], o, 1)
		end
		o
	end
	∇θr = grad(fdm, θ -> sum(f(x,θ,a.idxs)), a.θs)[1]
	@test ∇θ ≈ ∇θr

	ps = Flux.params(a)
	∇θr = gradient(() -> sum(a * x), ps)[a.θs]
	@test ∇θr ≈ ∇θr
end

@testset "UnitaryButterfly: integration with Flux" begin
	a = Butterfly(randn(2), [1,3], [2,4], 4)
	b = Butterfly(randn(2), [1,2], [3,4], 4)
	c = Butterfly(randn(2), [1,4], [3,2], 4)
	ub = UnitaryButterfly((a,b,c), 4)
	x = randn(4,4)
	ps = Flux.params(ub)
	@test length(ps) == 3
	gs = Flux.gradient(() -> sum(sin.(ub * x)),ps) 
	@test all([gs[p.θ] != nothing for p in [a,b,c]])
	gs = Flux.gradient(() -> sum(sin.(x * ub)),ps) 
	@test all([gs[p.θ] != nothing for p in [a,b,c]])
end
