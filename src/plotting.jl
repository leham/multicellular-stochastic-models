### Plotting utilities ###

using Colors
using CairoMakie

fig_theme = Theme(fonts = (; regular = "Calibri", weird = "Calibri"),
                  fontsize=8,
                  Axis=(xgridvisible=false, ygridvisible=false, 
                        xticksize=1.5, yticksize=1.5,
                        xticklabelpad=1, xlabelpadding=0.5,
                        yticklabelpad=1, ylabelpadding=2,
                        xtickwidth=0.7,
                        ytickwidth=0.7,
                        spinewidth=0.7),
                  linewidth=0.7)
set_theme!(fig_theme)
CairoMakie.activate!(type = "svg")
size_inches = (2.23, 1.5)
size_pt = 72 .* size_inches

# Colours for plotting
red_col = RGBA(249/255, 61/255, 46/255, 1.0)
pastel_red_col = RGBA(254/255, 107/255, 108/255, 1.0)
purple_col = RGBA(150/255, 177/255, 249/255, 1.0)
turquoise_col = RGBA(112/255, 209/255, 214/255, 1.0)
light_gray_col = RGBA(247/255, 247/255, 247/255, 1.0)
gray_col = RGBA(174/255, 174/255, 174/255, 1.0)