envpath = normpath(joinpath((@__DIR__, "../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
datapath = normpath(joinpath((@__DIR__, "../data/pss_means/")))
using Pkg; Pkg.activate(envpath)
include(srcpath*"plotting.jl")

# The code below is used to produce Figure 4(D) and SI Figure 3, following the analysis in `fig_4_and_SI_fig_3_pseudo_steady_state.jl`. 
# The loaded datasets have been generated using `scripts/run_heatmap.jl`

using JLD2

tau = 0.01
n = m = 32
σ_b = 0.03
ρ_u = 0.0
ρ_b = 60.0
δ = 1.0

get_threshold(ρ_b, σ_b, σ_u, δ, k=6) = δ / ( (k-1) * exp(ρ_b * σ_b / (1 + σ_b)) * (ρ_b * σ_b / (δ * σ_u)) - k )

function get_colormap(nmax::Real; cres::Int=1024)
    lognmax = log10(nmax)+1.0
    l1 = Int(round(1/lognmax * cres))-7
    l2 = Int(round((lognmax-1)/lognmax * cres))+7
    @assert l1+l2 == cres "color ranges are not properly assigned"
    cs = ColorSchemes.batlow
    cmap = ColorScheme(get(cgrad(cs, scale=log10), range(0, 0, length=l1))) * 
                  ColorScheme(get(cgrad(get(cgrad(cs, scale=log10), range(0.5, 1.0, length=l2)), scale=log10), range(0, 1, length=l2)))
    cmap
end

function get_nmax()
    # Load all the mean values and find the maximum value (used to obtain a unified colorbar)
    _, _, all_means_1 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_block_1_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, _, all_means_2 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_block_2_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, _, all_means_3 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_block_3_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, all_means_4 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_1M_pset.jld2", "pvec", "all_means")
    nmax1 = max(all_means_1..., all_means_2..., all_means_3..., all_means_4...)

    _, _, all_means_1 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_20x20_grid_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, all_means_2 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_20x20_1M_pset.jld2", "pvec", "all_means")
    nmax2 = max(all_means_1..., all_means_2...)

     _, _, all_means_1 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, all_means_2 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_1M_pset.jld2", "pvec", "all_means")
    nmax3 = max(all_means_1..., all_means_2...)

    _, _, all_means_1 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.5_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, all_means_2 = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.5_1M_pset.jld2", "pvec", "all_means")
    nmax4 = max(all_means_1..., all_means_2...)

    _, _, all_means_1 = load(datapath*"means_eta_vs_sigma_b_sigma_u_1.0_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, all_means_2 = load(datapath*"means_eta_vs_sigma_b_sigma_u_1.0_1M_pset.jld2", "pvec", "all_means")
    nmax5 = max(all_means_1..., all_means_2...)

    _, _, all_means_1 = load(datapath*"means_eta_vs_sigma_b_sigma_u_10.0_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
    _, all_means_2 = load(datapath*"means_eta_vs_sigma_b_sigma_u_10.0_1M_pset.jld2", "pvec", "all_means")
    nmax6 = max(all_means_1..., all_means_2...)

    nmaxs = [nmax1, nmax2, nmax3, nmax4, nmax5, nmax6]
    #println(nmaxs)
    println(findmax(nmaxs))
    maximum(nmaxs)
end


function plot_axis_MAIN_heatmap(ax::Axis, cs::ColorScheme, nmax::Real)

    sigma_u = 0.1
    ncounter = 0

    #pvec, all_means = load("analysis/data/means_eta_vs_sigma_b_sigma_u_0.1_1M_block.jld2", "pvec", "all_means")
    pvec, all_means = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_1M_pset.jld2", "pvec", "all_means")
    lrmeans_fixed = [m < 1 ? 0.099 : m for m in all_means]

    η_values, σ_b_values, all_means = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_block_1_100k.jld2", "η_values", "σ_b_values", "all_means")
    η_values_1 = deepcopy(vcat(η_values .- 0.00025, last(η_values) + 0.0025 / 2))
    η_values_1[1] = -0.00125
    σ_b_values_1 = deepcopy(vcat(σ_b_values .- 0.00025, last(σ_b_values) + 0.00025))
    σ_b_values_1[1] = -0.00125
    σ_b_values_1[end] = 0.10125
    lrmeans_1 = [m < 1 ? 0.099 : m for m in all_means]
    
    _pmat = collect(Iterators.product(η_values, σ_b_values))
    for i in eachindex(pvec)
        _ind = findfirst(==(pvec[i]), _pmat)
        if !isnothing(_ind)
            ncounter += 1
            lrmeans_1[_ind] = lrmeans_fixed[i] 
        end
    end

    η_values, σ_b_values, all_means = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_block_2_100k.jld2", "η_values", "σ_b_values", "all_means")
    η_values_2 = deepcopy(vcat(η_values .- 0.00025, last(η_values) + 0.00025))
    η_values_2[end] = 0.10125
    σ_b_values_2 = deepcopy(vcat(σ_b_values .- 0.00025, last(σ_b_values) + 0.002 / 2))
    σ_b_values_2[1] = -0.001
    lrmeans_2 = [m < 1 ? 0.099 : m for m in all_means]
    
    _pmat = collect(Iterators.product(η_values, σ_b_values))
    for i in eachindex(pvec)
        _ind = findfirst(==(pvec[i]), _pmat)
        if !isnothing(_ind)
            ncounter += 1
            lrmeans_2[_ind] = lrmeans_fixed[i] 
        end
    end

    @show ncounter
    @assert ncounter == length(pvec) "not all updated values assigned"

    η_values, σ_b_values, all_means = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_block_3_100k.jld2", "η_values", "σ_b_values", "all_means")
    η_values_3 = vcat(η_values[1] - 0.00125, η_values .+ 0.00125)
    σ_b_values_3 = vcat(σ_b_values[1] - 0.001, σ_b_values .+ 0.001)
    σ_b_values_3[end] = 0.10125
    lrmeans_3 = [m < 1 ? 0.099 : m for m in all_means]
    
    crange = (0.099, nmax)
    CairoMakie.heatmap!(ax, η_values_1, σ_b_values_1, lrmeans_1, colormap=cs, colorscale=log10, colorrange=crange)
    CairoMakie.heatmap!(ax, η_values_2, σ_b_values_2, lrmeans_2, colormap=cs, colorscale=log10, colorrange=crange)
    hm = CairoMakie.heatmap!(ax, η_values_3, σ_b_values_3, lrmeans_3, colormap=cs, colorscale=log10, colorrange=crange)
    
    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    CairoMakie.xlims!(ax, -0.001, x_f)                         
    ys = max(0.0, y_i):0.00001:y_f
    xs = get_threshold.(ρ_b, ys, sigma_u, δ) 
    idxs = 0 .<= xs .<= x_f
    CairoMakie.lines!(ax, xs[idxs], ys[idxs], color=:white, linewidth=1.5, alpha=0.8)
    
    ax, hm
end


function plot_axis_heatmap(ax::Axis, sigma_u::Real, cs::ColorScheme, fname1::String, fname2::String)

    empty!(ax) # remove previous plots from the axis object
    η_values, σ_b_values, all_means = load(fname1, "η_values", "σ_b_values", "all_means")
    pvec, all_means_extra = load(fname2, "pvec", "all_means")

    ptable = collect(Iterators.product(η_values, σ_b_values))
    inds = [findfirst(ps == _ps for _ps in ptable) for ps in pvec]
    all_means[inds] .= all_means_extra
    retouched_all_means = map(m -> m < 1 ? 0.0999 : m, all_means)
    
    hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, retouched_all_means, colormap=cs, colorscale=log10)

    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    ys = max(0.0, x_i):0.00005:y_f

    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    xb = sigma_u == 10.0 ? -0.01 : -0.001
    CairoMakie.xlims!(ax, xb, x_f)                         
    
    dy = sigma_u == 10.0 ? 0.00005 : 0.00001
    ys = max(0.0, y_i):dy:y_f
    xs = get_threshold.(ρ_b, ys, sigma_u, δ) 
    idxs = 0 .<= xs .<= x_f
    CairoMakie.lines!(ax, xs[idxs], ys[idxs], color=:white, linewidth=1.5, alpha=0.8)
    
    return ax, hm 

end

nmax = get_nmax()
cmap = get_colormap(nmax)

f = CairoMakie.Figure(size = (size_pt[1]*3.5, size_pt[2]*2.8), figure_padding = 2)

ax11 = Axis(f[1, 1], xlabel = "η", ylabel = "σb", aspect = 1)
ax11, _ = plot_axis_MAIN_heatmap(ax11, cmap, nmax)

ax12 = Axis(f[1, 2], xlabel = "η", ylabel = "σb", aspect = 1)
ax12, hm12 = plot_axis_heatmap(ax12, 0.1, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.1_20x20_grid_100k.jld2",
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.1_20x20_1M_pset.jld2")

ax13 = Axis(f[1, 3], xlabel = "η", ylabel = "σb", aspect = 1)
ax13, _ = plot_axis_heatmap(ax13, 0.1, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k.jld2",
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_1M_pset.jld2")

ax21 = Axis(f[2, 1], xlabel = "η", ylabel = "σb", aspect = 1)
ax21, _ = plot_axis_heatmap(ax21, 0.5, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.5_single_block_100k.jld2",
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.5_1M_pset.jld2" )

ax22 = Axis(f[2, 2], xlabel = "η", ylabel = "σb", aspect = 1)
ax22, _ = plot_axis_heatmap(ax22, 1.0, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_1.0_single_block_100k.jld2",
                               datapath*"means_eta_vs_sigma_b_sigma_u_1.0_1M_pset.jld2" )

ax23 = Axis(f[2, 3], xlabel = "η", ylabel = "σb", aspect = 1)
ax23, _ = plot_axis_heatmap(ax23, 10.0, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_10.0_single_block_100k.jld2",
                               datapath*"means_eta_vs_sigma_b_sigma_u_10.0_1M_pset.jld2" )

cb = CairoMakie.Colorbar(f[1, 4], hm12;
                         ticks = LogTicks(-1:2),
                         minorticksvisible=true,
                         minorticks=IntervalsBetween(9))

f

#save("Figures/all_heatmaps.svg", f, pt_per_unit = 1)

# ------------------------------------------------------

η_values, σ_b_values, all_means_OG = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k.jld2", "η_values", "σ_b_values", "all_means")
η_values, σ_b_values, all_means_CHECK = load(datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k_CHECK.jld2", "η_values", "σ_b_values", "all_means")

all_means_OG
all_means_CHECK
diffmat = all_means_CHECK .- all_means_OG
maximum(all_means_OG)
maximum(all_means_CHECK)

function plot_axis_heatmap_mod(ax::Axis, sigma_u::Real, cs::ColorScheme, fname::String)

    empty!(ax) # remove previous plots from the axis object
    η_values, σ_b_values, all_means = load(fname, "η_values", "σ_b_values", "all_means")
    
    retouched_all_means = map(m -> m < 1 ? 0.0999 : m, all_means)
    
    hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, retouched_all_means, colormap=cs, colorscale=log10)

    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    ys = max(0.0, x_i):0.00005:y_f

    y_i, y_f = ax.yaxis.attributes.limits[]
    x_i, x_f = ax.xaxis.attributes.limits[]
    xb = sigma_u == 10.0 ? -0.01 : -0.001
    CairoMakie.xlims!(ax, xb, x_f)                         
    
    dy = sigma_u == 10.0 ? 0.00005 : 0.00001
    ys = max(0.0, y_i):dy:y_f
    xs = get_threshold.(ρ_b, ys, sigma_u, δ) 
    idxs = 0 .<= xs .<= x_f
    CairoMakie.lines!(ax, xs[idxs], ys[idxs], color=:white, linewidth=1.5, alpha=0.8)
    
    return ax, hm 

end

nmax = 59
cmap = get_colormap(nmax)

f = CairoMakie.Figure(size = (size_pt[1]*2.4, size_pt[2]*1.3), figure_padding = 2)

ax11 = Axis(f[1, 1], xlabel = "η", ylabel = "σb", aspect = 1)
ax11, hm11 = plot_axis_heatmap_mod(ax11, 0.1, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k.jld2")

ax12 = Axis(f[1, 2], xlabel = "η", ylabel = "σb", aspect = 1)
ax12, hm12 = plot_axis_heatmap_mod(ax12, 0.1, cmap,
                               datapath*"means_eta_vs_sigma_b_sigma_u_0.1_10x10_grid_100k_CHECK.jld2")
f

zzf = CairoMakie.Figure(size = (size_pt[1]*1.3, size_pt[2]*1.3), figure_padding = 2)
ax = Axis(f[1, 1])
hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, abs.(all_means_CHECK .- all_means_OG))
cb = CairoMakie.Colorbar(f[1, 2], hm)
f

rinds = sortperm(vec(all_means_CHECK .- all_means_OG))
vec(all_means_CHECK .- all_means_OG)[rinds]
vec(all_means_CHECK)[rinds][1:30]
vec(all_means_OG)[rinds][1:30]

vec(all_means_CHECK)[rinds][end-30:end]
vec(all_means_OG)[rinds][end-30:end]

# --- σᵤ = 10.0 ---

f = CairoMakie.Figure(size = (size_pt[1]*2.4, size_pt[2]*1.3), figure_padding = 2)

ax11 = Axis(f[1, 1], xlabel = "η", ylabel = "σb", aspect = 1)
ax11, hm11 = plot_axis_heatmap_mod(ax11, 0.1, cmap, 
                               datapath*"means_eta_vs_sigma_b_sigma_u_10.0_single_block_100k.jld2")

ax12 = Axis(f[1, 2], xlabel = "η", ylabel = "σb", aspect = 1)
ax12, hm12 = plot_axis_heatmap_mod(ax12, 0.1, cmap,
                               datapath*"means_eta_vs_sigma_b_sigma_u_10.0_10x10_grid_100k_sigma_u_10_CHECK.jld2")
f

f = CairoMakie.Figure(size = (size_pt[1]*1.3, size_pt[2]*1.3), figure_padding = 2)
ax = Axis(f[1, 1])
hm = CairoMakie.heatmap!(ax, η_values, σ_b_values, abs.(all_means_CHECK .- all_means_OG))
cb = CairoMakie.Colorbar(f[1, 2], hm)
f

η_values, σ_b_values, all_means_OG = load(datapath*"means_eta_vs_sigma_b_sigma_u_10.0_single_block_100k.jld2", "η_values", "σ_b_values", "all_means")
η_values, σ_b_values, all_means_CHECK = load(datapath*"means_eta_vs_sigma_b_sigma_u_10.0_10x10_grid_100k_sigma_u_10_CHECK.jld2", "η_values", "σ_b_values", "all_means")

all_means_OG
all_means_CHECK
diffmat = all_means_CHECK .- all_means_OG
maximum(all_means_OG)
maximum(all_means_CHECK)

rinds = sortperm(vec(all_means_CHECK .- all_means_OG))
vec(all_means_CHECK .- all_means_OG)[rinds]
vec(all_means_CHECK)[rinds][1:30]
vec(all_means_OG)[rinds][1:30]

vec(all_means_CHECK)[rinds][end-30:end]
vec(all_means_OG)[rinds][end-30:end]
