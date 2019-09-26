using Test, Unitary
using Unitary: DiagonalRectangular
@testset begin "DiagonalRectangular: mul"
	ar = [0.5 0 0; 0 2 0]
	a = DiagonalRectangular([0.5, 2], 2, 3)
	for x in  [randn(3, 5), transpose(randn(5, 3))]
		@test ar * x ≈ a * x
	end

	ar = [0.5 0; 0 2; 0 0]
	a = DiagonalRectangular([0.5, 2], 3, 2)
	for x in  [randn(5, 3), transpose(randn(3, 5))]
		@test x * ar ≈ x * a
	end
end

@testset begin "DiagonalRectangular: tranpose"
	a = DiagonalRectangular([0.5, 2], 2, 3)
	for x in  [randn(3, 5), transpose(randn(5, 3))]
		@test a * x ≈ transpose(transpose(x) * transpose(a))
	end

	a = DiagonalRectangular([0.5, 2], 3, 2)
	for x in  [randn(5, 3), transpose(randn(3, 5))]
		@test x * a ≈ transpose(transpose(a) * transpose(x))
	end
end

