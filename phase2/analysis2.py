import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from scipy import stats
from scipy.stats import binom
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from math import comb
import numpy as np

# load csv again

filename = "clean_livwell.csv"
save_plots_here = "/Users/lioba/Desktop/PIK/PikxDataProcessing/docs/phase2_plots"

livwell_all = pd.read_csv(filename)

print(livwell_all.shape)
print(livwell_all.columns)

# some reasoning:
# I will filter for regions that have a minimum of 3 years,
# because only then can you approximate anything like a direction for a regression
# across years.
# with 2 years, any variation will immediately ruin the whole slope...
# but i don't have many regions with more than 4.
# so filter:


years_per_region = (
    livwell_all
    .groupby("region_id")["year"]
    .nunique()
)

print("\n=== YEARS PER REGION SUMMARY ===")
print(years_per_region.describe())

# now filter:

min_years = 3 # can be changed here
filterr = (years_per_region >= min_years)
valid_regions = years_per_region[filterr].index

print("valid regions:")
print(valid_regions)

livwell_3plus = livwell_all[
    livwell_all["region_id"].isin(valid_regions)
].copy()

# rebuild region-year structure after filtering
region_years_3plus = (
    livwell_3plus
    .groupby(["region_id", "year"])
    .size()
    .reset_index(name="n_obs")
)

# small summary:

print("\n###FILTERED SAMPLE (≥3 YEARS)###")
print("Regions:", livwell_3plus["region_id"].nunique())
print("Observations/region-years:", len(livwell_3plus))

print("Mean years/region:", years_per_region[valid_regions].mean())
print("Median years/region:", years_per_region[valid_regions].median())


# did i delete most of the countries now? If yes, I can still do the analysis,
# but that would need to be mentioned:
# COUNTRIES IN FINAL SAMPLE
# countries after >= 3-year filter)

# get unique country list
countries_left = sorted(livwell_3plus["country"].unique())

print("\n#################################")
print("COUNTRIES IN FINAL ANALYSIS SAMPLE")
print("###################################")

print(f"Number of countries: {len(countries_left)}\n")

for c in countries_left:
    print(c)

# a nice plot to inspect this visually:

# compute country-level sample size
country_counts = (
    livwell_3plus
    .groupby("country")
    .size()
    .reset_index(name="n_obs")
)

print(country_counts.sort_values("n_obs", ascending=False))


world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

world = world.merge(
    country_counts,
    how="left",
    left_on="SOV_A3",
    right_on="country"
)

fig, ax = plt.subplots(figsize=(12, 6))

world.plot(
    column="n_obs",
    cmap="RdYlGn",
    legend=True,
    vmin=0,
    vmax=120,
    missing_kwds={"color": "lightgrey"},
    ax=ax
)

ax.set_title("Data Availability (DV Observations per Country, ≥3-year regions)")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(save_plots_here, "country_data_availability_map.png"))
plt.show()



################################################################################
################################################################################
# Now I essentially redo analysis 1, but with a different model,
# and my filtered dataset
################################################################################
################################################################################

# Analysing LivWell
# Lioba Roggendorf
# May 2nd, 2026

"""
In this file, I filter the data to regions with >= 3 years of observations,
and conduct assumption checks and analysis.

The new model is still a fixed effects regression:
region + year fixed effects and rainfall control.

what it therefore does is:
Compare regions to their own average.
-- in years when a region is hotter than its own normal, is it more violent?

so Model 4:
DV_rct = β · Heat_rct + γ · Rain_rct + α_r + τ_t + ε_rct
Heat(waves) is main independent variable, per region, country, and year
Rain is a covariate
α_r are region fixed effects; removes all time-invariant differences between regions (and absorbs country effects!)
τ_t are year fixed effects; removes global shocks shared by all regions in a given year
therefore, β is the coefficient of interest (the "within" effect).
γ is the coefficient on rain; i'll report it, but not interpret. just a control.
"""


# use the same colours:
orange = "#F8B195"
pink = "#F67280"
brighter_purple = "#C06C84"
darker_purple = "#6C5B7B"
blue = "#355C7D"
colors = [orange, pink, brighter_purple, darker_purple, blue]

################################################################################
# STEP 1: Model 4, Assumption Checks (again, this time using livwell_3plus)
################################################################################

"""
1. Linearity
2. Homoscedasticity
3. Normality of residuals
4. Multicollinearity
5. Outliers / influential observations
6. Skew of DV
"""

# I now use C(region_id) + C(year) instead of country!
# BUT I still cluster errors by country because errors within the same country
# are likely correlated (shared culture, climate patterns, etc. etc.).
the_holy_grail = smf.ols("dv ~ heat + rain + C(region_id) + C(year)", data=livwell_3plus).fit(
    cov_type="cluster", cov_kwds={"groups": livwell_3plus["region_id"]}
)

# check if heat actually varies still:
print('############# heat variance! ################')
print(livwell_3plus.groupby("region_id")["heat"].std().describe())
# let Leonie know it's quite small!
# ############# heat variance! ################
# count    120.000000
# mean       0.067391
# std        0.074298
# min        0.000000
# 25%        0.000000
# 50%        0.048113
# 75%        0.096225
# max        0.336788

# residuals vs. fitted
plt.scatter(the_holy_grail.fittedvalues, the_holy_grail.resid, alpha=0.3, color=brighter_purple)
plt.axhline(0, color=pink)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. fitted")
plt.savefig(os.path.join(save_plots_here, "residualsVsFitted2.jpg"))
plt.show()
plt.close()
# NOTE: Looks like heteroscedasticity!
# but clustered SEs handle that! MENTION THIS alongside the plot!

# heat vs. DV
plt.scatter(livwell_3plus["heat"], livwell_3plus["dv"], alpha=0.3, color=darker_purple)
plt.xlabel("Heat")
plt.ylabel("DV")
plt.title("Heat vs. DV")
plt.savefig(os.path.join(save_plots_here, "heatVsDV2.jpg"))
plt.show()
plt.close()

# AIC
the_holy_grail_two_degree_poly_version = smf.ols("dv ~ heat + I(heat**2) + rain + C(region_id) + C(year)", data=livwell_3plus).fit(
    cov_type="cluster", cov_kwds={"groups": livwell_3plus["region_id"]}
)
print(f'\n\nBase model AIC: {the_holy_grail.aic},\n Square model AIC: {the_holy_grail_two_degree_poly_version.aic}.\n\n')
# Base model AIC: 2723.5225371436723,
# Square model AIC: 2724.06070430192.
# this is good: when in doubt, assume the simpler model,
# and also, base model does a bit better than square... YAY


# Q-Q plot
qqplot(the_holy_grail.resid, line='s', markerfacecolor=blue, markeredgecolor=blue, alpha=0.5)
plt.title("Q-Q plot of residuals")
plt.savefig(os.path.join(save_plots_here, "qq2.jpg"))
plt.show()
plt.close()
# it's right-skewed, with a fat heavy thicc right tail...
# but my sample is big enough (464)
# and I check outliers to see what's happening... so all good

# Histogram of residuals
residuals = the_holy_grail.resid
plt.hist(residuals, color=blue, edgecolor=brighter_purple)
plt.xlabel("Residuals")
plt.title("Histogram of residuals")
plt.savefig(os.path.join(save_plots_here, "residualHist2.jpg"))
plt.show()
plt.close()
# looks good enough, pretty normal, yay :)

# Multicollinearity
heatAndRain = livwell_3plus[["heat", "rain"]]
heatCol = 0 # which column is that of the variable i wanna check?
print("VIF Columns:", heatAndRain.columns)
vifHeat = variance_inflation_factor(heatAndRain.values, heatCol)
print(f"VIF for heat: {vifHeat}")

# Outliers
influence = OLSInfluence(the_holy_grail)
cooks, _ = influence.cooks_distance
stdandardizedResiduals = influence.resid_studentized_internal

threshold = 4 / len(livwell_3plus)
print(f"Observations with Cook's distance > {threshold:.4f}:")
print(livwell_3plus[cooks > threshold])

print(f"\nObservations with |standardised residual| > 3:")
print(livwell_3plus[abs(stdandardizedResiduals) > 3])

cooking = livwell_3plus.index[cooks > threshold].tolist()
farfarawaydatapoints = livwell_3plus.index[abs(stdandardizedResiduals) > 3].tolist()
outliers = list(set(cooking + farfarawaydatapoints))
print(f"Outlier count: {len(outliers)}")

# DV Skew
plt.hist(livwell_3plus["dv"], color=blue, edgecolor = brighter_purple)
plt.title("DV distribution")
plt.savefig(os.path.join(save_plots_here, "dvHist2.jpg"))
plt.show()
plt.close()
# slightly skewed, but no problem with OLS etc. etc., all good

# PRINT Results

print('\nMODEL INCOMING:\n\n')
print(the_holy_grail.summary())
print('\n\n')

print('\nSIGNIFICANCE INCOMING:\n\n')
print(f"Heat coefficient: {the_holy_grail.params['heat']} (SE: {the_holy_grail.bse['heat']}, p = {the_holy_grail.pvalues['heat']})")
print('\n\n')

# save filtered csv for reanalysis with JASP (just to be sure)

livwell_3plus.to_csv("threepluslivwell.csv")

################################################################################
# Bootstrap to be safe...
################################################################################
reps = 500

# store results
booted_heat_betas = []

#okay plan:
# 1. randomly pick regions, WITH replacement (boot boot boot)
# -- (re-sample observation per region, because obviously regions are dependent, but within regions,
# -- I assume independence (so once I account for region, then within that region, independence)
# 2. select the data from those regions
# 3. run the regression on that region
# 4. check the heat coefficient
# 5. save it, at the end, calculate its variance (SE)

# so it's like:
# "What if I had observed different sets of countries,
# each with their full time histories?""

# commenting this out because it takes a long time to run

regions = livwell_3plus["region_id"].unique()
regioncount = len(regions)

# this is to save the estimates for the heat coefficients
# so I can later check the variance
booted_heat_betas = []

print("\nRunning simple cluster bootstrap...")

for boot in range(reps):
    # WITH REPLACEMENT
    boot_regions = np.random.choice(regions, size=regioncount, replace=True)

    # note: I am not re-sampling the years and values indivdually! I am re-sampling
    # whole regions!
    # I think I need this to keep the dependence structure, but ask Leonie!
    sampled = []
    for region in boot_regions:
        mask = (livwell_3plus["region_id"] == region)
        selected = livwell_3plus[mask]
        sampled.append(selected)

    sampled = pd.concat(sampled, ignore_index=True)
    # fit the same model again
    boot_model = smf.ols(
        "dv ~ heat + rain + C(region_id) + C(year)",
        data=sampled
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": sampled["region_id"]}
    )


    booted_heat_betas.append(boot_model.params["heat"])

booted_heat_betas = np.array(booted_heat_betas)


print("\nBOOTSTRAP RESULTS (heat coefficient)")
print("------------------------------------")
print(f"Mean: {booted_heat_betas.mean():.4f}")
print(f"Std (bootstrap SE): {booted_heat_betas.std():.4f}")
print(f"95% CI: [{np.percentile(booted_heat_betas, 2.5):.4f}, "
      f"{np.percentile(booted_heat_betas, 97.5):.4f}]")

fig, ax = plt.subplots()
ax.hist(booted_heat_betas, bins=30)
ax.set_title("Bootstrap distribution of heat coefficient")
fig.savefig(os.path.join(save_plots_here, "bootstrap.png"),
            dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Skew:", pd.Series(booted_heat_betas).skew())
print("Min:", booted_heat_betas.min())
print("Max:", booted_heat_betas.max())

# BOOTSTRAP RESULTS (heat coefficient)
# ------------------------------------
# Mean: 10.2750
# Std (bootstrap SE): 2.6544
# 95% CI: [5.0176, 15.9027]
# that is actually SMALLER than the one I got... so am I fine?

# To account for fat tails and outliers, I will do MM-estimation in R!
# see another file
# it did not converge... so I will trim and check

# ----------------------------
# just trim 2%

lower = livwell_3plus["dv"].quantile(0.02)
upper = livwell_3plus["dv"].quantile(0.98)

livwell_trimmed = livwell_3plus[
    (livwell_3plus["dv"] >= lower) &
    (livwell_3plus["dv"] <= upper)
].copy()

print("Original n:", len(livwell_3plus))
print("Trimmed n:", len(livwell_trimmed))

trimmed_model = smf.ols(
    "dv ~ heat + rain + C(region_id) + C(year)",
    data=livwell_trimmed
).fit(
    cov_type="cluster",
    cov_kwds={"groups": livwell_trimmed["region_id"]}
)

print(trimmed_model.summary())

print("\nHeat (original):", the_holy_grail.params["heat"])
print("Heat (trimmed): ", trimmed_model.params["heat"])

# Heat (original): 10.314538761526485
# Heat (trimmed):  8.856016542846431

# Estimate (heat): 8.86
# Std. error: 2.68
# t-statistic: 3.30
# p-value: 0.001
# 95% CI: [3.60, 14.11]
