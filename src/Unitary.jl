module Unitary

using Flux, LinearAlgebra, Zygote
using Zygote: @adjoint
import Base.*

const AbstractMatVec = Union{AbstractMatrix, AbstractVector}
const MatVec = Union{Matrix, Vector}
const TransposedMatVec = Union{Matrix, Vector,Transpose{T,Matrix{T}} where T, Transpose{T,Vector{T}} where T}

include("unitarymatrix.jl")
include("svd.jl")

end # module
