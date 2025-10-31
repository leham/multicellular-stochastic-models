envpath = normpath(joinpath((@__DIR__, "../../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../../src/")))
datapath = normpath(joinpath((@__DIR__, "../../data/")))
using Pkg; Pkg.activate(envpath)
using JLD2, LaTeXStrings, Printf

include(srcpath*"plotting.jl");
include(srcpath*"animation.jl")
include(srcpath*"toggle_triad.jl")

get_protein(cells) = map(x -> Tuple(view(x, 1, :)), cells)

function get_frames(nframes, niter_per_frame, ps, 
                    update_f=update_TT,
                    init_f=nothing)
    config = create_grid_TT(n, m)
    buf = Array{Int64, 3}(undef, n, m, 3)
    helpers = (MVector{3, Int}(undef),            # trans
               MVector{3, Float64}(undef),        # propensities
               ProbabilityWeights(ones(3) ./ 3),  # helper weights
               MMatrix{3, 6, Int}(undef),         # signal 
               MVector{3, Int}(undef),            # sum_signal
               MVector{3, Int}(undef),            # decay
               MMatrix{2, 3, Int}(undef))         # res
    frames = Vector{Matrix{Tuple{Int, Int, Int}}}(undef, nframes)
    
    if !isnothing(init_f)
        init_f(config)
    end

    frames[1] = get_protein(config)
    for i in 2:nframes
        for _ in 1:niter_per_frame
            run_iter_TT!(config, buf, ps, tau, helpers..., update_f)
        end
        frames[i] = get_protein(config)
    end

    return frames
end


# --- Movie 2 ---
# Model parameters
n = 400
m = 600
σ_u = [0.1, 0.1, 0.1]
σ_b = [1.0, 1.0, 1.0]
ρ_u = [0.005, 0.005, 0.005]
ρ_b = [100.0, 100.0, 100.0]
δ = ones(3) 
η = [0.6, 0.6, 0.6]
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]

# Simulation parameters
tau = 0.01 
nframes = 150
niter_per_frame = 50

@time frames = get_frames(nframes, niter_per_frame, ps)
@time fast_animation(frames, fps=15,
                     fname=datapath*"animations/SI_Movie_2.mp4")

# --- Movie 3 ---
n = 400
m = 600
σ_u = [2.2, 2.2, 2.2]
σ_b = [0.1, 0.1, 0.1]
ρ_u = [0.01, 0.01, 0.01]
ρ_b = [60.0, 60.0, 60.0]
δ = ones(3) 
η = [0.3, 0.3, 0.3]
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]

tau = 0.01 
nframes = 150
niter_per_frame = 600

@time frames = get_frames(nframes, niter_per_frame, ps)
@time fast_animation(frames, fps=15,
                     fname=datapath*"animations/SI_Movie_3.mp4")

# --- Movie 4 ---
n = 128
m = 128
σ_u = [0.6, 0.6, 0.6]
σ_b = [0.08, 0.08, 0.08]
ρ_u = [0.03, 0.03, 0.03]
ρ_b = [60.0, 60.0, 60.0]
δ = ones(3) 

tau = 0.01 
nframes = 150
niter_per_frame = 100

# initialise gene 1 in each cell in the bound (active) cell
function init_gene_1!(cells)
    for cell in cells
        cell[2, 1] = 1
    end
    return nothing
end


function animate_compare_hex(frames1::AbstractVector, frames2::AbstractVector; 
                             tempdirectory1=datapath*"animations/tmp/movie_5-1",
                             tempdirectory2=datapath*"animations/tmp/movie_5-2",
                             bckgrd_col=light_gray_col,
                             fname="animation4.gif", fps=10, 
                             η1 = 0.03, η2 = 0)
    
    @assert length(frames1) == length(frames2) "frame lengths do not agree"
    gray_line_col = RGBA(205/255, 205/255, 205/255)
    
    hex_animation(frames1, hexsize=7, creategif=false, createmovie=false, fps=15,
                  tempdirectory=tempdirectory1, bckgrd_col=gray_line_col, marginsize=7)
    hex_animation(frames2, hexsize=7, creategif=false, createmovie=false, fps=15,
                  tempdirectory=tempdirectory2, bckgrd_col=gray_line_col, marginsize=7)

    f = CairoMakie.Figure(size=(size_pt[1]*8, size_pt[1]*4), 
                          background_color=bckgrd_col,
                          fontsize=24)
    ax1 = Axis(f[1, 1],
          xticksvisible=false, yticksvisible=false,
          xticklabelsvisible=false, yticklabelsvisible=false,
          spinewidth=0, aspect=1, title=L"\eta_i = %$(η1)", titlegap = 8)
    ax2 = Axis(f[1, 2],
          xticksvisible=false, yticksvisible=false,
          xticklabelsvisible=false, yticklabelsvisible=false,
          spinewidth=0, aspect=1, title=L"\eta_i = %$(η2)", titlegap = 8)

    # NOTE: tempdirectory1 & 2 must be created beforehand for this to work
    rr = record(f, fname, eachindex(frames1); framerate=fps) do i
        fname1 = "$tempdirectory1/$(@sprintf("%.10d", i)).png"
        img1 = load(assetpath(fname1))
        fname2 = "$tempdirectory2/$(@sprintf("%.10d", i)).png"
        img2 = load(assetpath(fname2))
        image!(ax1, img1, interpolate=true)
        image!(ax2, img2, interpolate=true)
        colgap!(f.layout, 150)
        x_mid = ax2.scene.viewport[].origin[1] / 2  + (ax1.scene.viewport[].origin[1] + ax1.scene.viewport[].widths[1]) / 2
        y_1 = ax1.scene.viewport[].origin[2] + 50.0
        y_2 = ax1.scene.viewport[].origin[2] + ax1.scene.viewport[].widths[1] - 50.0
        lines!(f.scene, [x_mid, x_mid], [y_1, y_2], linewidth=6, linecap=:round, color=gray_line_col)
    end

    return rr
end

# with signalling
η = [0.03, 0.03, 0.03]
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]
@time frames1 = get_frames(nframes, niter_per_frame, ps, update_TT, init_gene_1!)

# no signalling
η = zeros(3)
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]
@time frames2 = get_frames(nframes, niter_per_frame, ps, update_TT, init_gene_1!)

@time animate_compare_hex(frames1, frames2, fps=15, η1=0.03, η2 = 0,
                          fname=datapath*"animations/SI_Movie_4.mp4")