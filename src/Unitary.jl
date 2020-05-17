module Unitary

using Flux, LinearAlgebra, Zygote
using Zygote: @adjoint
import Base: *, transpose

const AbstractMatVec = Union{AbstractMatrix, AbstractVector}
const MatVec = Union{Matrix, Vector}
const TransposedMatVec = Union{Matrix, SubArray, Vector,Transpose{T,Matrix{T}} where T, Transpose{T,Vector{T}} where T}


include("givens/include.jl")
include("householder/include.jl")
include("layers/include.jl")
include("LU/include.jl")
include("LDU/include.jl")

export UnitaryHouseholder, UnitaryGivens, SVDTransform
export lowup, lowdup, LUTransform
export Svd, Transform
end # module
