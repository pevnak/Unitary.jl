module Unitary

using Flux, LinearAlgebra
import Base.*

const AbstractMatVec = Union{AbstractMatrix, AbstractVector}
const MatVec = Union{Matrix, Vector}
const TransposedMatVec = Union{Matrix, Vector,Transpose{T,Matrix{T}} where T, Transpose{T,Vector{T}} where T}
const TrackedMatVec = Union{TrackedArray{T,2,A} where A where T, TrackedArray{T,1,A} where A where T}

include("unitarymatrix.jl")
include("svd.jl")

end # module
