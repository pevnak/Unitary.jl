using Unitary, Test, LinearAlgebra
using Unitary: UnitaryMatrix

@testset "Testing multiplication and transposed" begin
	a = UnitaryMatrix([1])
	ad = Matrix(a)

	x = rand(2,10);
	@test a * x ≈ ad * x
	@test transpose(a) * x ≈ transpose(ad) * x
	@test transpose(x) * transpose(a) ≈ transpose(x) * transpose(a)
	@test transpose(x) * a ≈ transpose(x) * ad

	x = rand(10, 2);
	@test x * a ≈ x * ad
	@test x * transpose(a) ≈ x * transpose(ad)
	@test transpose(a) * transpose(x) ≈ transpose(ad) * transpose(x)
	@test a * transpose(x) ≈ ad * transpose(x)
end


@testset "Integration with Flux" begin
	a = UnitaryMatrix(param([1]))
	ad = Matrix(a)

	x = rand(2,10)
	@test a * x ≈ ad * x
	x = rand(10, 2)
	@test x * a ≈ x * ad
end


