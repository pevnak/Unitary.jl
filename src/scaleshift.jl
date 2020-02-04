struct ScaleShift{D, B, S}
	d::D
	b::B
	σ::S
end

Base.show(io::IO, m::ScaleShift) = print(io, "ScaleShift{$(size(m.d)), $(m.σ)}")

Flux.@functor ScaleShift

"""
	ScaleShift(n, σ)

	scales the input variables and shift them. Optionally, they are preprocessed by non-linearity.
	
	`σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
"""
function ScaleShift(n::Int, σ = identity)
	ScaleShift(DiagonalRectangular(rand(Float32, n), n, n),
		randn(Float32,n),
		σ)
end

(m::ScaleShift)(x::AbstractMatVec) = m.σ.(m.d * x .+ m.b)

function (m::ScaleShift)(xx::Tuple)
	x, logdet = xx
	pre = (m.d * x .+ m.b)
	g = explicitgrad.(m.σ, pre)
	(m.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(m.d))
end

struct InvertedScaleShift{D, B, S}
	d::D
	b::B
	σ::S
end
Flux.@functor InvertedScaleShift

Base.inv(m::ScaleShift) = InvertedScaleShift(inv(m.d), m.b, inv(m.σ))
Base.inv(m::InvertedScaleShift) = ScaleShift(inv(m.d), m.b, inv(m.σ))

(m::InvertedScaleShift)(x::AbstractMatVec)  = m.d * (m.σ.(x) .- m.b)