# Generate heatmaps of the pseudo steady-state mean and Fano factor of the protein numbers for the negative feedback loop.
envpath = normpath(joinpath((@__DIR__, "../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
datapath = normpath(joinpath((@__DIR__, "../data/")))
using Pkg; Pkg.activate(envpath)
include(srcpath*"plotting.jl")
include(srcpath*"feedback_loop.jl")

using JLD2
using OhMyThreads: tmap
using Base.Threads: nthreads
using LinearAlgebra

# Tau leaping
tau = 0.01
# Dimensions of the 2D tissue
n = m = 32
# Gene switching off (unbinding) rate
σ_u = 0.1
# Gene switching on (binding) rate
σ_b = 0.0
# Translation rate (in the unbound state)
ρ_u = 100.0
# Translation rate ρ_b (in the bound state)
ρ_b = 0.0
# Decay rate
δ = 1.0
# Diffusion rate for the protein (movement from one cell to another)
η = 0.0
# Create parameter vector
# NOTE: order is fixed
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]

n_snapshots = 400
niter_per_snapshot = 500

# compute variance over all cells on the grid given the mean μ
get_var_protein_number_FL(cells::AbstractMatrix, μ::Real) = mapreduce(x -> (x[1] - μ)^2, +, cells) / length(cells)

# multithreaded code to measure protein counts (mean, variance, and Fano factor per cell and trajectories) over time in (η, σ_b) space
function get_timeseries_η_vs_σ_b(n_snapshots, niter_per_snapshot, η_values, σ_b_values, ntrajectories; ntasks=nthreads())
    lk = ReentrantLock()
    chnl_buf, chnl_ps, chnl_config, chnl_helper = init_sim_utils_FL(n, m, ps, ntasks)
    chnl_mean_mtx = Channel{MMatrix{ntrajectories, n_snapshots, Float64, ntrajectories*n_snapshots}}(ntasks)
    chnl_var_mtx = Channel{MMatrix{ntrajectories, n_snapshots, Float64, ntrajectories*n_snapshots}}(ntasks)
    foreach(1:ntasks) do _
        put!(chnl_mean_mtx, MMatrix{ntrajectories, n_snapshots, Float64}(undef))
        put!(chnl_var_mtx, MMatrix{ntrajectories, n_snapshots, Float64}(undef))
    end
    pmat = collect(Iterators.product(η_values, σ_b_values))

    all_means = tmap(pmat; ntasks) do (η_value, σ_b_value)
        buf = take!(chnl_buf)
        _ps = take!(chnl_ps) 
        config = take!(chnl_config)
        helpers = take!(chnl_helper)
        mean_mtx = take!(chnl_mean_mtx)
        var_mtx = take!(chnl_var_mtx)
        
        # update η and σ_b in the parameter vector
        _ps[5] = η_value
        _ps[2] = σ_b_value
        
        t = @elapsed for i in 1:ntrajectories
            # all genes are in the unbound state initially
            reset_grid_FL!(config)
            for j in 1:n_snapshots
                for _ in 1:niter_per_snapshot
                    run_iter_FL!(config, buf, _ps, tau, helpers...)
                end
                mean_mtx[i, j] = get_mean_protein_number_FL(config)
                var_mtx[i, j] = get_var_protein_number_FL(config, mean_mtx[i, j])
            end
        end

        lock(lk) do
            println("η = $η_value, σ_b = $σ_b_value, t = $t s")
        end

        res = vec(mean(mean_mtx, dims=1)), vec(mean(var_mtx, dims=1)), vec(mean(var_mtx ./ mean_mtx, dims=1))
        put!(chnl_buf, buf)
        put!(chnl_ps, _ps)
        put!(chnl_config, config)
        put!(chnl_helper, helpers)
        put!(chnl_mean_mtx, mean_mtx)
        put!(chnl_var_mtx, var_mtx)
        res
    end

    return all_means
end

# Plot the timeseries for the observable of interest over time and a range of η values.
# This is used to explore the dynamics for different parameter sets and ensure convergence
function plot_timeseries_η(n_snapshots, niter_per_snapshot, η_values, σ_b, all_means; ylabel = "Mean protein")
    f = CairoMakie.Figure(size = (size_pt[1]*1.7, size_pt[2]*1.1), figure_padding = 1)
    ax = Axis(f[1, 1], xlabel = "Time", ylabel = ylabel)
    
    _iters = niter_per_snapshot*tau:niter_per_snapshot*tau:n_snapshots*niter_per_snapshot*tau
    for i in 1:lastindex(η_values)
        CairoMakie.lines!(ax, _iters, all_means[i], label="η = $(η_values[i]), σ_b = $σ_b")
    end
    
    CairoMakie.Legend(f[1, 2], ax, labelsize=6, rowgap=-10, framewidth=0.2)
    return f
end

η_values = 0:0.2:1.0
σ_b_values = 0:0.2:1.0
n_snapshots = 100
niter_per_snapshot = 50
println("Simulation time = ", n_snapshots * niter_per_snapshot * tau)
ntrajectories = 2
res = @time get_timeseries_η_vs_σ_b(n_snapshots, niter_per_snapshot, η_values, σ_b_values, ntrajectories)

# NOTE: the dataset used for final analysis is generated using scripts/run_NFL.jl
# the code above similar to the code in the script, but it's left here for completeness and to enable exploration
@load datapath*"mean_timeseries/timeseries_NFL.jld2" η_values σ_b_values res

mean_timeseries = map(x -> first(x), res)
var_timeseries = map(x -> x[2], res)
ff_timeseries = map(x -> x[3], res)

σ_b_ind = 1
plot_timeseries_η(n_snapshots, niter_per_snapshot, η_values, σ_b_values[σ_b_ind], vec(mean_timeseries[:, σ_b_ind]), ylabel = "Mean")
plot_timeseries_η(n_snapshots, niter_per_snapshot, η_values, σ_b_values[σ_b_ind], vec(var_timeseries[:, σ_b_ind]), ylabel = "Variance")
plot_timeseries_η(n_snapshots, niter_per_snapshot, η_values, σ_b_values[σ_b_ind], vec(ff_timeseries[:, σ_b_ind]), ylabel = "Fano factor")

# SI Figure 2 (left)
ms = map(m -> m[end], mean_timeseries)
f = CairoMakie.Figure(size = (size_pt[1]*1.3, size_pt[2]*1.3), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "η", ylabel = "σ_b", aspect = 1.1)
hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, ms, colorscale=log10, colormap = ColorSchemes.batlow, colorrange=(1, 100))
cb = CairoMakie.Colorbar(f[1, 2], hm, 
                         ticks = LogTicks(0:2),
                         minorticksvisible=true,
                         minorticks=IntervalsBetween(9),
                         minortickwidth=0.7,
                         minorticksize=1.5,
                         ticksize=1.5,
                         ticklabelpad=1,
                         labelpadding=1,
                         tickwidth=0.7,
                         spinewidth=0.7,
                         height = Relative(3/4))
f

# SI Figure 2 (right)
ffs = map(ff -> ff[end], ff_timeseries)
f = CairoMakie.Figure(size = (size_pt[1]*1.3, size_pt[2]*1.3), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "η", ylabel = "σ_b", aspect = 1.1)
hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, ffs, colorscale=log10, colormap = ColorSchemes.batlow)

cb = CairoMakie.Colorbar(f[1, 2], hm,
                         ticks = LogTicks(0:2),
                         minorticksvisible=true,
                         minorticks=IntervalsBetween(9),
                         ticksize=1.5,
                         ticklabelpad=1,
                         labelpadding=1,
                         tickwidth=0.7,
                         minortickwidth=0.7,
                         minorticksize=1.5,
                         spinewidth=0.7,
                         height = Relative(3/4))
f