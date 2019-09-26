using Unitary, Test, Flux
using Unitary: UnitaryButterfly, Butterfly, InPlaceUnitaryButterfly, givenses_column, givenses, filtergivenses

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
	a = UnitaryButterfly(5)
	ai = InPlaceUnitaryButterfly(a)
	x = randn(5,5)
	@test a*x ≈ ai * x
	@test transpose(a)*x ≈ transpose(ai) * x
	@test x*a ≈ x*ai
	@test x*transpose(a) ≈ x*transpose(ai)


	for ai in  [InPlaceUnitaryButterfly(a), transpose(InPlaceUnitaryButterfly(a))]
		Δ = ones(size(x))
		xs = Unitary.accummulax(ai.θs, ai.is, ai.js, ai.transposed, x)
		@test isapprox(Unitary._∇buttermulax(Δ, xs, ai.θs, ai.is, ai.js, ai.transposed, x)[1], ngradient(θ -> sum(Unitary.accummulax(θ, ai.is, ai.js, ai.transposed, x)[1]), ai.θs)[1], atol = 1e-2)
		@test isapprox(Unitary._∇buttermulax(Δ, xs, ai.θs, ai.is, ai.js, ai.transposed, x)[end], ngradient(x -> sum(Unitary.accummulax(ai.θs, ai.is, ai.js, ai.transposed, x)[1]), x)[1], atol = 1e-2)
	end
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


@testset "Generation of givens transformations" begin
	@test all(givenses_column(16, 8) .== ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]))
	@test all(givenses_column(16, 4) .== ([1, 2, 3, 4, 9, 10, 11, 12], [5, 6, 7, 8, 13, 14, 15, 16]))
	@test all(givenses_column(8, 4) .== ([1,2,3,4],[5,6,7,8]))
	@test all(givenses_column(8, 2) .== ([1,2,5,6],[3,4,7,8]))
	@test all(givenses_column(8, 1) .== ([1,3,5,7],[2,4,6,8]))

	@test all(filtergivenses(4, ([1, 3, 5, 7], [2, 4, 6, 8])) .== ([1, 3], [2, 4]))
	@test all(filtergivenses(6, ([1, 3, 5, 7], [2, 4, 6, 8])) .== ([1, 3, 5], [2, 4, 6]))

	@test all(givenses(8) .== 
	(([1, 3, 5, 7], [2, 4, 6, 8]), 
	 ([1, 2, 5, 6], [3, 4, 7, 8]), 
	 ([1, 3, 5, 7], [2, 4, 6, 8]), 
	 ([1, 2, 3, 4], [5, 6, 7, 8]), 
	 ([1, 3, 5, 7], [2, 4, 6, 8]), 
	 ([1, 2, 5, 6], [3, 4, 7, 8]), 
	 ([1, 3, 5, 7], [2, 4, 6, 8])))

	@test all(givenses(6) .== 
	(([1, 3, 5 ], [2, 4, 6]), 
	 ([1, 2], [3, 4]), 
	 ([1, 3, 5 ], [2, 4, 6]), 
	 ([1, 2], [5, 6]), 
	 ([1, 3, 5 ], [2, 4, 6]), 
	 ([1, 2], [3, 4]), 
	 ([1, 3, 5 ], [2, 4, 6])))
end

