envpath = normpath(joinpath((@__DIR__, "../../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../../src/")))
datapath = normpath(joinpath((@__DIR__, "../../data/")))
using Pkg; Pkg.activate(envpath)
using JLD2

include(srcpath*"plotting.jl");
include(srcpath*"animation.jl")
include(srcpath*"feedback_loop.jl")

get_protein(cells::AbstractMatrix{MVector{2, Int64}}) = map(x -> x[1], cells)

function get_frames(nframes, niter_per_frame, ps)
    config = create_grid_FL(n, m)
    buf = Matrix{Int64}(undef, n, m)
    helpers = (MVector{6, Int}(undef),      # signal 
               MVector{2, Int}(undef))
    frames = Vector{Matrix{Int}}(undef, nframes)
    
    init_cells_FL!(config)
    frames[1] = get_protein(config)
    for i in 2:nframes
        for _ in 1:niter_per_frame
            run_iter_FL!(config, buf, ps, tau, helpers...)
        end
        frames[i] = get_protein(config)
    end

    return frames
end

# Model parameters
n = 128
m = 192
σ_u = 0.1
σ_b = 0.01
ρ_u = 0.0
ρ_b = 100.0
η = 0.08
δ = 1.0
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]

# Simulation parameters
tau = 0.01 
nframes = 500
niter_per_frame = 100

@time frames = get_frames(nframes, niter_per_frame, ps)
@time hex_animation(frames, hexsize=5, creategif=false, createmovie=true, fps=30,
                    fname=datapath*"animations/SI_Movie_1.mp4")