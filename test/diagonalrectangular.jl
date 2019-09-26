using Test, Unitary
using Unitary: DiagonalRectangular
@testset begin "DiagonalRectangular: mul"
	ar = [0.5 0 0; 0 2 0]
	a = DiagonalRectangular([0.5, 2], 2, 3)
	for x in  [randn(3, 5), transpose(randn(5, 3))]
		@test ar * x ≈ a * x
	end

	for x in  [randn(5, 2), transpose(randn(2,5))]
		@test x * ar ≈ x * a
	end


	ar = [0.5 0; 0 2; 0 0]
	a = DiagonalRectangular([0.5, 2], 3, 2)
	for x in  [randn(5, 3), transpose(randn(3, 5))]
		@test x * ar ≈ x * a
	end
	for x in  [randn(2, 5), transpose(randn(5,2))]
		@test ar * x ≈ a * x
	end
end

@testset begin "DiagonalRectangular: Matrix"
	ar = [0.5 0 0; 0 2 0]
	a = DiagonalRectangular([0.5, 2], 2, 3)
	@test Matrix(a) ≈ ar

	ar = [0.5 0; 0 2; 0 0]
	a = DiagonalRectangular([0.5, 2], 3, 2)
	@test Matrix(a) ≈ ar
end


@testset begin "DiagonalRectangular: transpose"
	a = DiagonalRectangular([0.5, 2], 2, 3)
	for x in  [randn(3, 5), transpose(randn(5, 3))]
		@test a * x ≈ transpose(transpose(x) * transpose(a))
	end

	a = DiagonalRectangular([0.5, 2], 3, 2)
	for x in  [randn(5, 3), transpose(randn(3, 5))]
		@test x * a ≈ transpose(transpose(a) * transpose(x))
	end
end

@testset begin "DiagonalRectangular: inv"
	a = DiagonalRectangular([0.5, 2], 2, 3)
	ai = DiagonalRectangular([2, 0.5], 3, 2)
	x = randn(3, 5)
	@test ai * (a * x) ≈ vcat(x[1:2, :], zeros(1,5))	

	x = randn(5, 2)
	@test (x * a) * ai ≈ x
end

