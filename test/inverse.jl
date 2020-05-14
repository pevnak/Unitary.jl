using Unitary, Test

@testset "Inversions of activation function" begin
	for f in [identity, selu, NNlib.σ, NNlib.leakyrelu]
		x = -10:1:10
		@test inv(f).(f.(x)) ≈ x
		@test inv(f).(f.(-x)) ≈ -x
		@test inv(inv(f)) == f
	end
	#tanh is unstable
	x = -4:1:4
	@test inv(tanh).(tanh.(x)) ≈ x
	@test inv(tanh).(tanh.(-x)) ≈ -x
	@test inv(inv(tanh)) == tanh

end
