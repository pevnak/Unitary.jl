using Unitary, Test, LinearAlgebra, Flux
using Unitary: SVDTransform, ScaleShift
using FiniteDifferences

@testset "Can I invert SVDTransform and its chain" begin
	for d in [1, 2, 3, 4]
		for m in [SVDTransform(d, identity), SVDTransform(d, selu), Chain(SVDTransform(d, identity), SVDTransform(d, identity)), Chain(SVDTransform(d, selu), SVDTransform(d, selu))]
			mi = inv(m)
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-3)
			end
		end
	end

	for d in [1, 2, 3, 4]
		for m in [ScaleShift(d, identity), ScaleShift(d, selu), Chain(ScaleShift(d, identity), ScaleShift(d, identity)), Chain(ScaleShift(d, selu), ScaleShift(d, selu))]
			mi = inv(m)
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-3)
			end
		end
	end
end

@testset "testing the determinant" begin
	fdm = central_fdm(5, 1);
	x = randn(2)
	for m in [SVDTransform(2, identity), SVDTransform(2, selu), Chain(SVDTransform(2, identity), SVDTransform(2, selu))]
		@test isapprox(logabsdet(jacobian(fdm, m, x)[1])[1], m((x,0))[2][1], atol = 1e-3)
	end

	for m in [ScaleShift(2, identity), ScaleShift(2, selu), Chain(ScaleShift(2, identity), ScaleShift(2, selu))]
		@test isapprox(logabsdet(jacobian(fdm, m, x)[1])[1], m((x,0))[2][1], atol = 1e-3)
	end

	for m in [SVDTransform(2, identity), SVDTransform(2, selu), Chain(SVDTransform(2, identity), SVDTransform(2, selu))]
		@test m((x,0))[1] ≈ m(x)
	end

	for m in [ScaleShift(2, identity), ScaleShift(2, selu), Chain(ScaleShift(2, identity), ScaleShift(2, selu))]
		@test m((x,0))[1] ≈ m(x)
	end
end
