module Unitary

using Flux, LinearAlgebra, Zygote
using Zygote: @adjoint
import Base: *, transpose

const AbstractMatVec = Union{AbstractMatrix, AbstractVector}
const MatVec = Union{Matrix, Vector}
const TransposedMatVec = Union{Matrix, Vector,Transpose{T,Matrix{T}} where T, Transpose{T,Vector{T}} where T}


include("givens/givens.jl")
include("householder/householder.jl")
include("layers/layers.jl")

export UnitaryHouseholder, UnitaryButterfly, SVDDense
end # module
