using Unitary, Test, LinearAlgebra, Flux
using Unitary: SVDDense
using FiniteDifferences

@testset "Can I invert SVDDense and its chain" begin
	for d in [2,3,4]
		for m in [SVDDense(d, identity), SVDDense(d, selu), Chain(SVDDense(d, identity), SVDDense(d, identity)), Chain(SVDDense(d, selu), SVDDense(d, selu))]
			mi = inv(m)
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-4)
			end
		end
	end
end

@testset "reducing number of operations in SVDDense" begin 
	@test length(Unitary.SVDDense(10, identity, maxn = 3).u.matrices) == 3
	@test length(Unitary.SVDDense(10, identity, maxn = 3).v.matrices) == 3
end

@testset "testing the determinant" begin
	fdm = central_fdm(5, 1);
	x = randn(2)
	for m in [SVDDense(2, identity), SVDDense(2, selu), Chain(SVDDense(2, identity), SVDDense(2, selu))]
		@test isapprox(logabsdet(jacobian(fdm, m, x)[1])[1], m((x,0))[2][1], atol = 1e-4)
	end

	m = Chain(SVDDense(2, 4, identity), SVDDense(4, 2, identity))
	logabsdet(jacobian(fdm, m, x)[1])[1] 
	m((x,0))[2][1]	
end