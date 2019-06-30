module Unitary

using Flux, LinearAlgebra
import Base.*

const AbstractMatVec = Union{AbstractMatrix, AbstractVector}
const TrackedMatVec = Union{TrackedArray{T,2,A} where A where T, TrackedArray{T,1,A} where A where T}

include("unitarymatrix.jl")
include("svd.jl")

end # module
