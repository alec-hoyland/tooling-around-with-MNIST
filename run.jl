cd("/home/alec/code/homework/CAS-CS-640/project/julia/")
using Pkg
Pkg.activate(".")
Pkg.instantiate()


    @info "including conv"
    include("conv.jl")



    @info "including convFuzz"
    include("convFuzz.jl")



    @info "including convAdpt"
    include("convAdpt.jl")



    @info "including convFull"
    include("convFull.jl")



    @info "including dense"
    include("dense.jl")



    @info "including denseFuzz"
    include("denseFuzz.jl")



    @info "including denseAdpt"
    include("denseAdpt.jl")



    @info "including denseFull"
    include("denseFull.jl")
