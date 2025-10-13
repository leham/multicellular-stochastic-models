#= 
The following code is used to explore and identify which points in the (η, σ_b) parameter space have not converged to the pseudo steady state.
This is relevant to the analysis of the signalling threshold for multicellular stability, presented in Figure 4(D) and SI Figure of the paper,
where we plot heatmap of the pseudo steady-state protein levels (σ_b, η) parameter space showing the signaling threshold for different values 
of σ_u and different grid sizes. 

The initial version of the heatmaps is constructed by runing simulations up to t = 10³ (100k tau leaping iterations) for each parameter set 
and measuring the mean protein level at that time. However, we observed that a considerable number of points has not yet converged to the 
pseudo steady state, and much longer simulations are needed to ensure the accuracy of the presented heatmaps. Here, we try to automatically
identify the parameter points that should be run for longer (extending the simulation time up to t = 10⁴). This is done in two parts:
- (1) Using a rough analysis based on the root mean square deviation in a sliding window over the given time series
measurements of protein counts. 
- (2) Using the initial heatmap to select a narrow ribbon in the parameter space surrounding the transition boundary between the low (zero)
and functionally stable expression.
=#

envpath = normpath(joinpath((@__DIR__, "../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
datapath = normpath(joinpath((@__DIR__, "../data/")))
using Pkg; Pkg.activate(envpath)
include(srcpath*"plotting.jl")
include(srcpath*"feedback_loop.jl")

using JLD2
using StatsBase

# Parameters used throughout
tau = 0.01
n = m = 32
σ_b = 0.03
ρ_u = 0.0
ρ_b = 60.0
δ = 1.0

# ------------------------------------------------------------------------------------------------------------
# Part 1

# These settings were used to generate the timeseries data of protein counts
tau = 0.01
n_snapshots = 400
niter_per_snapshot = 500

# Compute root mean square deviation in a sliding window over the given time series
function wrmsd_series(series::AbstractArray, w_size::Int) 
    @assert length(series) >= w_size "window size is larger than the length of the series"
    res = map(1:length(series)-w_size+1) do i
        x = @view series[i:i+w_size-1]
        x[end] < 0 ? 0 : sum(abs2.(x .- mean(x))) / (w_size - 1)
    end
    res
end


function plot_mean_timeseries_η_and_σ_b(n_snapshots, niter_per_snapshot, η_values, σ_b_values, all_means)
    ts = niter_per_snapshot*tau:niter_per_snapshot*tau:n_snapshots*niter_per_snapshot*tau
    f = CairoMakie.Figure(size = (size_pt[1]*1.7, size_pt[2]*1.1), figure_padding = 1)
    ax = Axis(f[1, 1], xlabel = "Time", ylabel="log₁₀(Protein count)")
    for i in 1:lastindex(η_values)
        CairoMakie.lines!(ax, ts, all_means[i], label="η = $(η_values[i]), σ_b = $(σ_b_values[i])", alpha=0.5)
    end
    CairoMakie.hlines!(ax, [0.0], linestyle=:dash, label=nothing, color=:gray)
    CairoMakie.Legend(f[1, 2], ax, labelsize=6, rowgap=-10, framewidth=0.2)
    return f
end


function plot_non_converged_timeseries(η_values, σ_b_values, all_means; t_ind::Int=11, w_size::Int=30)
    #this holds with default n_snapshots = 400 and niter_per_snapshot = 500
    #t_ind = 11 # 20k iters
    #t_ind = 171 # 100k iters

    convmap = map(all_means) do x
        _x = log10.(x)
        _y = wrmsd_series(_x, w_size)
        # the steady-state condition here is based only on some rough heuristics
        is_steady = (_y[t_ind] < 0.001) && ( abs((_x[end] - _x[t_ind]) / _x[t_ind]) < 0.1 || (_x[t_ind] <= 0 && _x[end] <= 0 ) )
        !is_steady
    end

    inds = findall(convmap)
    plot_mean_timeseries_η_and_σ_b(n_snapshots, niter_per_snapshot, 
                                   η_values[[first(ind.I) for ind in inds]], 
                                   σ_b_values[[last(ind.I) for ind in inds]],
                                   all_means[inds])
end

function plot_convergence_heatmap(η_values, σ_b_values, all_means; t_ind::Int=11, w_size::Int=30)
    #this holds with default n_snapshots = 400 and niter_per_snapshot = 500
    #t_ind = 11 # 20k iters
    #t_ind = 171 # 100k iters

    convmap = map(all_means) do x
        _x = log10.(x)
        _y = wrmsd_series(_x, w_size)
        is_steady = (_y[t_ind] < 0.0005) && ( abs((_x[end] - _x[t_ind]) / _x[t_ind]) < 0.1 || (_x[t_ind] <= 0 && _x[end] <= 0 ) )
        !is_steady
    end

    f = CairoMakie.Figure(size = (size_pt[1]*1.5, size_pt[2]*1.8), figure_padding = 1)
    ax = Axis(f[1, 1], title="Steady state at niters=$((t_ind + w_size - 1 ) * niter_per_snapshot)", 
              xlabel = "η", ylabel = "σ_b")

    hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, convmap, colormap=cgrad(ColorSchemes.batlow, 2, categorical=true))
    CairoMakie.Colorbar(f[1, 2], hm; ticks=([0.25, 0.75], ["Yes", "No"]))

end

# utility function to find the transition boundary from the heuristic pseudo-steady-state map
function get_boundary_pset_steady_state(η_values, σ_b_values, all_means;
                                        Δp_conv=0.002, Δp_heatmap=0.0005, sigdigits=4,
                                        t_ind::Int=171, w_size::Int=30)
    
    # Δp_conv here is the full width of the step size between parameter values in the heatmap
    # Δp_heatmap is the full width of the step size between parameter values in the convergence map computation 
    # sigdigits is necessary for rounding the floating point numbers to obtain consistent parameter values                
    # NOTE: this code is a bit rough and will only work for certain Δp_conv and Δp_heatmap combinations
    
    @show Δp_conv
    @show Δp_heatmap
    η_max = maximum(η_values)
    σ_b_max = maximum(σ_b_values)
    
    # case 1 or 2: depending on the step sizing, construct ranges differently
    cond = if isequal(rationalize(Δp_conv), rationalize(Δp_heatmap))
               cond = 1
           elseif iszero(rationalize(Δp_conv) % rationalize(Δp_heatmap))
               cond = 2
           else
               error("This function will not work")
           end

    convmap = map(all_means) do x
        _x = log10.(x)
        _y = wrmsd_series(_x, w_size)
        # the steady-state condition here is based only on some rough heuristics
        is_steady = (_y[t_ind] < 0.0005) && ( abs((_x[end] - _x[t_ind]) / _x[t_ind]) < 0.1 || (_x[t_ind] <= 0 && _x[end] <= 0 ) )
        !is_steady
    end
    
    inds = findall(convmap)
    pset = Vector{Tuple{Float64, Float64}}()
    for ind in inds
        η_value = η_values[ind[1]]
        σ_b_value = σ_b_values[ind[2]]
        if cond == 1
            η_1 = max(0.0, round(η_value - Δp_heatmap; sigdigits))
            η_2 = min(η_max, round(η_value + Δp_heatmap; sigdigits))
            η_range = collect(η_1:Δp_heatmap:η_2)
            σ_b_1 = max(0.0, round(σ_b_value - Δp_heatmap; sigdigits))
            σ_b_2 = min(σ_b_max, round(σ_b_value + Δp_heatmap; sigdigits))
            σ_b_range = collect(σ_b_1:Δp_heatmap:σ_b_2)
        else
            η_1 = max(0.0, round(η_value - Δp_conv/2 - Δp_heatmap; sigdigits))
            η_2 = min(η_max, round(η_value + Δp_conv/2 + Δp_heatmap; sigdigits))
            η_range = collect(η_1:Δp_heatmap:η_2)
            σ_b_1 = max(0.0, round(σ_b_value - Δp_conv/2 - Δp_heatmap; sigdigits))
            σ_b_2 = min(σ_b_max, round(σ_b_value + Δp_conv/2 + Δp_heatmap; sigdigits))
            σ_b_range = collect(σ_b_1:Δp_heatmap:σ_b_2)
        end
        
        ps_pairs = vec(collect(Iterators.product(η_range, σ_b_range)))
        pset = vcat(pset, ps_pairs)
    end
   
    return unique(pset)
end

# Explore some of the parameter sets that fail to reach the pseudo steady state at early times.
# This is done for different grid sizes and values of σ_u, corresponding to the six heatmaps presented in SI Figure 3 of the paper. 
# NOTE: the loaded datasets are generated using scripts/run_timeseries.jl

η_values, σ_b_values, mean_timeseries = load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.1.jld2", "η_values", "σ_b_values", "all_means")
plot_convergence_heatmap(η_values, σ_b_values, mean_timeseries, t_ind=71, w_size=10)
plot_non_converged_timeseries(η_values, σ_b_values, mean_timeseries, t_ind=71)
#pset_1 = get_boundary_pset_steady_state(η_values, σ_b_values, mean_timeseries, t_ind=71, w_size=10, Δp_conv=0.002, Δp_heatmap=0.0005)

η_values, σ_b_values, mean_timeseries = load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.5.jld2", "η_values", "σ_b_values", "all_means")
plot_convergence_heatmap(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
plot_non_converged_timeseries(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
#pset_1 = get_boundary_pset_steady_state(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)

η_values, σ_b_values, mean_timeseries = load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_1.0.jld2", "η_values", "σ_b_values", "all_means")
plot_convergence_heatmap(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
plot_non_converged_timeseries(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
#pset_1 = get_boundary_pset_steady_state(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)

η_values, σ_b_values, mean_timeseries = load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_10.0.jld2", "η_values", "σ_b_values", "all_means")
plot_convergence_heatmap(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
plot_non_converged_timeseries(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
#pset_1 = get_boundary_pset_steady_state(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10, Δp_conv=0.02, Δp_heatmap=0.02)

η_values, σ_b_values, mean_timeseries = load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.1_10x10_grid.jld2", "η_values", "σ_b_values", "all_means")
plot_convergence_heatmap(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
plot_non_converged_timeseries(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
#pset_1 = get_boundary_pset_steady_state(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)

η_values, σ_b_values, mean_timeseries = load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.1_20x20_grid.jld2", "η_values", "σ_b_values", "all_means")
plot_convergence_heatmap(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
plot_non_converged_timeseries(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10)
#pset_1 = get_boundary_pset_steady_state(η_values, σ_b_values, mean_timeseries, t_ind=91, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)

# ------------------------------------------------------------------------------------------------------------
# Part 2

get_threshold(ρ_b, σ_b, σ_u, δ, k=6) = δ / ( (k-1) * exp(ρ_b * σ_b / (1 + σ_b)) * (ρ_b * σ_b / (δ * σ_u)) - k )

function get_colormap(nmax::Real; cres::Int=1024)
    lognmax = log10(nmax)+1.0
    l1 = Int(round(1/lognmax * cres))-7
    l2 = Int(round((lognmax-1)/lognmax * cres))+7
    @assert l1+l2 == cres "color ranges are not properly assigned"
    cs = ColorSchemes.batlow
    cs_rescaled = ColorScheme(get(cgrad(cs, scale=log10), range(0, 0, length=l1))) * 
                  ColorScheme(get(cgrad(get(cgrad(cs, scale=log10), range(0.5, 1.0, length=l2)), scale=log10), range(0, 1, length=l2)))
    cs_rescaled
end


function plot_heatmap(η_values, σ_b_values, all_means, sigma_u; pset=[], strokewidth=0.05, markersize=0.8)
    
    # Colormap is constructed such that all zero values correspond to 0.1 (ensuring proper color scale)
    # Heuristic: if mean protein count is under 1, the absorbing state of zero protein will soon 
    # be reached, hence we set the spurious low values to zero for visual clarity.
    retouched_all_means = map(all_means) do m 
        m < 1 ? 0.0999 : m
    end
    nmax = maximum(retouched_all_means)+1
    cmap = get_colormap(nmax)

    f = CairoMakie.Figure(size = (size_pt[1]*1.2, size_pt[2]*1.0), figure_padding = 2)
    ax = Axis(f[1, 1], xlabel = "η", ylabel = "σ_b", aspect=1.2)

    hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, retouched_all_means, colormap=cmap, colorscale=log10)
    # TODO: find a way to relabel the "10^(-1)" tick to "0"
    cb = CairoMakie.Colorbar(f[1, 2], hm;
                            ticks = LogTicks(-1:2),
                            minorticksvisible=true,
                            minorticks=IntervalsBetween(9))
    
    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    xb = sigma_u == 10.0 ? -0.01 : -0.001
    CairoMakie.xlims!(ax, xb, x_f)

    dy = sigma_u == 10.0 ? 0.00005 : 0.00001
    ys = max(0.0, y_i):dy:y_f
    xs = get_threshold.(ρ_b, ys, sigma_u, δ) 
    idxs = 0 .<= xs .<= x_f
    CairoMakie.lines!(ax, xs[idxs], ys[idxs], color=:white, linewidth=1.5, alpha=0.8)
    colgap!(f.layout, -10)
    
    # Plot the highlighted parameter pairs
    if !isempty(pset)
        CairoMakie.scatter!(ax, pset; color=:white, strokecolor=:black, strokewidth, markersize)
    end

    return f 
end


# Separate function for the heatmap in the main text due to complicated construction
# out of three blocks of different resolutions
function plot_MAIN_heatmap(; pset=[], strokewidth=0.05, markersize=0.3)
    sigma_u = 0.1
    
    η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_block_1_100k.jld2", "η_values", "σ_b_values", "all_means")
    η_values_1 = deepcopy(vcat(η_values .- 0.00025, last(η_values) + 0.0025 / 2))
    η_values_1[1] = -0.00125
    σ_b_values_1 = deepcopy(vcat(σ_b_values .- 0.00025, last(σ_b_values) + 0.00025))
    σ_b_values_1[1] = -0.00125
    σ_b_values_1[end] = 0.10125
    lrmeans_1 = [m < 1 ? 0.099 : m for m in all_means]

    η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_block_2_100k.jld2", "η_values", "σ_b_values", "all_means")
    η_values_2 = deepcopy(vcat(η_values .- 0.00025, last(η_values) + 0.00025))
    η_values_2[end] = 0.10125
    σ_b_values_2 = deepcopy(vcat(σ_b_values .- 0.00025, last(σ_b_values) + 0.002 / 2))
    σ_b_values_2[1] = -0.001
    lrmeans_2 = [m < 1 ? 0.099 : m for m in all_means]

    η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_block_3_100k.jld2", "η_values", "σ_b_values", "all_means")
    η_values_3 = vcat(η_values[1] - 0.00125, η_values .+ 0.00125)
    σ_b_values_3 = vcat(σ_b_values[1] - 0.001, σ_b_values .+ 0.001)
    σ_b_values_3[end] = 0.10125
    lrmeans_3 = [m < 1 ? 0.099 : m for m in all_means]

    nmax = maximum(maximum.((lrmeans_1, lrmeans_2, lrmeans_3)))+1
    cmap = get_colormap(nmax)
    crange=(0.099, nmax)

    f = CairoMakie.Figure(size = (size_pt[1]*1.2, size_pt[2]*1.0), figure_padding = 2);
    ax = Axis(f[1, 1], xlabel = "η", ylabel = "σ_b", aspect=1.2,
              xticks = (0:0.05:0.1, ["0", "0.05", "0.1"]), yticks = (0:0.05:0.1, ["0", "0.05", "0.1"]),
            );
    CairoMakie.heatmap!(ax, η_values_1, σ_b_values_1, lrmeans_1, colormap=cmap, colorscale=log10, colorrange=crange)
    CairoMakie.heatmap!(ax, η_values_2, σ_b_values_2, lrmeans_2, colormap=cmap, colorscale=log10, colorrange=crange)
    hm = CairoMakie.heatmap!(ax, η_values_3, σ_b_values_3, lrmeans_3, colormap=cmap, colorscale=log10, colorrange=crange)
    
    CairoMakie.Colorbar(f[1, 2], hm;
                        ticks = LogTicks(-1:1:nmax-1),
                        minorticksvisible=true,
                        minorticks=IntervalsBetween(9));
    colgap!(f.layout, -10)

    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    xb = -0.001
    CairoMakie.xlims!(ax, xb, x_f)

    dy = 0.00001
    ys = max(0.0, y_i):dy:y_f
    xs = get_threshold.(ρ_b, ys, sigma_u, δ) 
    idxs = 0 .<= xs .<= x_f
    CairoMakie.lines!(ax, xs[idxs], ys[idxs], color=:white, linewidth=1.5, alpha=0.8)

    # Plot the highlighted parameter pairs
    if !isempty(pset)
        CairoMakie.scatter!(ax, pset, color=:white, strokecolor=:black; strokewidth, markersize)
    end

    f
end

# Utility function to select a narrow ribbon in the parameter space surrounding the transition boundary 
function get_boundary_pset_heatmap(η_values, σ_b_values, all_means, Δp::Int=1)
    pset = Vector{Tuple{Float64, Float64}}()
    for η_ind in eachindex(η_values)
        _ind = findfirst(all_means[η_ind, :] .>= 1)
        if !isnothing(_ind)
            ps_pairs = collect(Iterators.product(η_values[η_ind], σ_b_values[max(1, _ind-Δp):min(_ind+Δp, lastindex(σ_b_values))]))
            append!(pset, ps_pairs)
        end
    end

    for σ_b_ind in eachindex(σ_b_values)
        _ind = findfirst(all_means[:, σ_b_ind] .>= 1)
        if !isnothing(_ind)
            ps_pairs = collect(Iterators.product(η_values[max(1, _ind-Δp):min(_ind+Δp, lastindex(η_values))], σ_b_values[σ_b_ind]))
            append!(pset, ps_pairs)
        end
    end

    return unique(pset)
end

# Separate function for the heatmap in the main text due to its more complicated construction
function get_MAIN_boundary_pset_heatmap(Δp::Int=3)
    pset_2_1 = get_boundary_pset_heatmap(load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_block_1_100k.jld2", "η_values", "σ_b_values", "all_means")...)
    
    pset_2_2 = Vector{Tuple{Float64, Float64}}()
    η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_block_2_100k.jld2", "η_values", "σ_b_values", "all_means")
    for η_ind in eachindex(η_values)
        _ind = findfirst(all_means[η_ind, :] .>= 1)
        if !isnothing(_ind)
            ps_pairs = collect(Iterators.product(η_values[η_ind], σ_b_values[max(1, _ind-Δp):min(_ind+Δp, lastindex(σ_b_values))]))
            append!(pset_2_2, ps_pairs)
        end
    end

    return unique(vcat(pset_2_1, pset_2_2))
end

# NOTE: various "means_eta_vs_sigma_b_sigma_u ⋯" datasets are generated using "scripts/run_heatmap.jl"
# Highlight points in the pseudo-steady-state heatmaps to rerun up to t=10^4 (1M tau leaping iterations)

# σ_u = 0.1
pset_1 = get_boundary_pset_steady_state(load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.1.jld2", "η_values", "σ_b_values", "all_means")...,
                                        t_ind=71, w_size=10, Δp_conv=0.002, Δp_heatmap=0.0005)
pset_2 = get_MAIN_boundary_pset_heatmap()
pset = unique(vcat(pset_1, pset_2))
plot_MAIN_heatmap(; pset)
#@save datapath*"parameter_sets/rerun_1M_sigma_u_0.1_pset.jld2" pset

# σ_u = 0.5
pset_1 = get_boundary_pset_steady_state(load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.5.jld2", "η_values", "σ_b_values", "all_means")...,
                                        t_ind=91, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)
η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.5_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
pset_2 = get_boundary_pset_heatmap(η_values, σ_b_values, all_means)
pset = unique(vcat(pset_1, pset_2))
plot_heatmap(η_values, σ_b_values, all_means, 0.5; pset)
#@save datapath*"parameter_sets/rerun_1M_sigma_u_0.5_pset.jld2" pset

# σ_u = 1.0
pset_1 = get_boundary_pset_steady_state(load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_1.0.jld2", "η_values", "σ_b_values", "all_means")...,
                                        t_ind=91, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)
η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_1.0_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
pset_2 = get_boundary_pset_heatmap(η_values, σ_b_values, all_means)
pset = unique(vcat(pset_1, pset_2))
plot_heatmap(η_values, σ_b_values, all_means, 1.0; pset)
#@save datapath*"parameter_sets/rerun_1M_sigma_u_1.0_pset.jld2" pset

# σ_u = 10.0
pset_1 = get_boundary_pset_steady_state(load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_10.0.jld2", "η_values", "σ_b_values", "all_means")...,
                                        t_ind=91, w_size=10, Δp_conv=0.02, Δp_heatmap=0.02)
η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_10.0_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
pset_2 = get_boundary_pset_heatmap(η_values, σ_b_values, all_means)
pset = unique(vcat(pset_1, pset_2))
plot_heatmap(η_values, σ_b_values, all_means, 10.0; pset)
#@save datapath*"parameter_sets/rerun_1M_sigma_u_10.0_pset.jld2" pset

# σ_u = 0.1 on a 10 × 10 grid
pset_1 = get_boundary_pset_steady_state(load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.1_10x10_grid.jld2", "η_values", "σ_b_values", "all_means")...,
                                        t_ind=71, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)
η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k.jld2", "η_values", "σ_b_values", "all_means")
pset_2 = get_boundary_pset_heatmap(η_values, σ_b_values, all_means)
pset = unique(vcat(pset_1, pset_2))
plot_heatmap(η_values, σ_b_values, all_means, 0.1; pset)
#@save datapath*"parameter_sets/rerun_1M_sigma_u_0.1_10x10_pset.jld2" pset

# σ_u = 0.1 on a 20 × 20 grid
pset_1 = get_boundary_pset_steady_state(load(datapath*"mean_timeseries/mean_timeseries_eta_vs_sigma_b_sigma_u_0.1_20x20_grid.jld2", "η_values", "σ_b_values", "all_means")...,
                                        t_ind=71, w_size=10, Δp_conv=0.002, Δp_heatmap=0.002)
η_values, σ_b_values, all_means = load(datapath*"pss_means/means_eta_vs_sigma_b_sigma_u_0.1_20x20_grid_100k.jld2", "η_values", "σ_b_values", "all_means")
pset_2 = get_boundary_pset_heatmap(η_values, σ_b_values, all_means)
pset = unique(vcat(pset_1, pset_2))
plot_heatmap(η_values, σ_b_values, all_means, 0.1; pset)
#@save datapath*"parameter_sets/rerun_1M_sigma_u_0.1_20x20_pset.jld2" pset