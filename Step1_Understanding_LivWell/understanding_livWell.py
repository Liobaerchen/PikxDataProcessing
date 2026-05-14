# Understanding LivWell
# Lioba Roggendorf
# April 23rd, 2026

"""
Here, I create a first overview over the most important variables
for my primary LivWell analysis:
- Extreme heat
- Violence against women
-- interesting secondary: (women's) empowerment

Reminder: RQ:
Stehen Hitzeextreme mit Gewalt gegen Frauen in Zusammenhang?
(English translation:
Is extreme heat related to increased violence against women?)
"""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
# might need pip install geopandas for the maps
import geopandas as gpd



livwell_all = pd.read_csv("livwell.csv")
print(livwell_all.head())

# [5 rows x 409 columns]
# I extract the ones that are relevant for the primary RQ:

relevant_cols = [
    "DV_phys_12m_p",          # 'Women who experienced physical violence in the past 12 months (%)'
    "tmp_anom_p2sd_share12",  # 'Share of months in the past 12 months with a positive temperature anomaly exceeding 2 SD' --> LOCAL average! NOT global!
    "year",                   # time fixed effects
    "region_num_harmonized",  # region fixed effects
    "country_code",           # optional: maybe use for world-map later
]

# and to control for rainfall:
rainfall = "pre_anom_mean12"  # Mean deviation from long-run precipitation (12m)

livwell1 = pd.DataFrame({
    'dv': livwell_all[relevant_cols[0]],
    'heat': livwell_all[relevant_cols[1]],
    'year': livwell_all[relevant_cols[2]],
    'region': livwell_all[relevant_cols[3]],
    'country': livwell_all[relevant_cols[4]],
    'rain': livwell_all[rainfall]
})

print(livwell1.head(50))



"""
Excluding missing data:
"""
# NOTE: Issue! DV variable has many NAs! Exclude, but be CLEAR about
# what is missing.
total_rows = len(livwell1['dv'])
na_rows = len(livwell1['dv'][livwell1['dv'].isna()])
print(f'total rows: {total_rows}')
print(f'rows where dv is na: {na_rows}')
print(f'so, proportion excluded: {na_rows/total_rows}')

# now, how much am I excluding if I exlude ALL NAs for any rows?
any_missing = livwell1.isna().any(axis=1)
any_na_rows = len(livwell1[any_missing])
print(f'rows with any missing values: {any_na_rows}')
print(f'thus percentage total excluded: {any_na_rows/total_rows}')
# almost all exclusions are due to DV! But it is my main outcome, so this is necessary.

# check what I am excluding:
year_exclusion = livwell1.groupby("year")["dv"].apply(lambda x: x.isna().mean())
# . mean is turning the 1s and 0s from isna into a percentage
print(f'\nHow much am I excluding from which years?:\n {year_exclusion}')

country_exclusion = livwell1.groupby("country")["dv"].apply(lambda x: x.isna().mean())
print(f'\nHow much am I excluding from which countries?:\n {country_exclusion}')

# .agg lets me compute all four statistics for each column in each group at once
climate_comparison = livwell1.groupby(any_missing)[["heat", "rain"]].agg(
    ["count", "mean", "std", "median"]
).round(3)

print("\nClimate comparison (DV missing vs observed):\n")
print(climate_comparison)
# heat difference is 0.001; totally negligible
# but the retained data is from wetter regions: difference observed - missing
# is ≈ +0.1 -> remember this for later!
# DV availability is systematically related to rainfall conditions

# ~ means NOT. So without the rows where sth is missing:
clean_livwell1 = livwell1[~any_missing].copy()
print(clean_livwell1.head())

print('how do the regions work? In the paper, it seemed like there may be several countries with the same region.')
print('e.g. region 1 Nigeria is different from region 1 Germany')
print(clean_livwell1.groupby("region")["country"].nunique())
# TRUE! So make a unique identifier
clean_livwell1["region_id"] = clean_livwell1["country"].astype(str) + "_" + clean_livwell1["region"].astype(str)
# NOTE to self: use region_id for analyses!

# describe the final dataset
print(clean_livwell1.shape) # 758 observations
print(f'\nnumber of countries: {clean_livwell1["country"].nunique()}.\n')
print(f'number of regions: {clean_livwell1["region_id"].nunique()}.\n')
region_years = clean_livwell1.groupby("region_id")["year"].nunique() # how many years do I have in each region?
print(region_years.describe())
print((region_years >= 5).sum(), "regions with >=5 years")
# 758 observations
# 39 countries
# 349 regions, BUT ONLY 24 with more than 5 years??? oH OH


# Should I use countries as identifiers instead?

# how many years do I have in each country?
country_years = clean_livwell1.groupby("country")["year"].nunique()
print(f'\nYears available per country:\n{country_years.sort_values()}\n')

# list of all countries where I have more than 5 years available
countries_more_than_5_years = country_years[country_years > 5]
print(f'Countries with more than 5 years available:\n{countries_more_than_5_years}\n')

# map of countries to years available
country_to_years = clean_livwell1.groupby("country")["year"].unique()
print(f'Country -> years available:\n{country_to_years}\n')
# THIS IS ALSO A PROBLEM

# need a new strategy...


"""
DV_rct = beta * heat_rct + gamma * rain_rct + delta_country + tau_year + ε_rct

r: region, c = country, t = year
so outcome:
In region r, in country c, in year t, domestic violence was X%

the beta is: predicted contribution of heat to DV
If heat increases by 1 unit, how much does DV change?

gamma * rain_rc --> HOLDING RAIN CONSTANT
(based on Leonie's recommendation, and also because I systematically
excluded wet countries)

delta_c --> separate intercept for each country
(so country differences like culture are removed)

tau_year --> separate intercept for each year
--> so I'm checking for deviations from the global average in that year,
not comparing years
--> so if there is some GLOBAL catastrophe, its effect should be removed (e.g. covid)

so basically: subtract country averages and year averages from all variables,
run regression on what's left

so: “Among regions, removing country differences and global trends, are hotter places associated with more violence?”

This is good because:
- does not compare Armenia and Peru, which would get confounded by culture
    --> Remember Annika's paper!
- does not rely on region fixed effects, because I had way too little data
- does not throw away as much data as Country-year FE
- can keep countries even if they appear in only a few years -- but removes confounds

Question: the year control will remove climate change effects... is that a problem?

finally: CLUSTER standard errors (consider Leonie's hint: Conley standard error)
"""

# for this I need within country variation:
print('\n\n within country variation:')
within_country_heat_var = clean_livwell1.groupby("country")["heat"].std()
print(within_country_heat_var.sort_values())
# BGD, EGY, GAB, GHA, BEN will be excluded

print('\n\ncollinearity?')
print(clean_livwell1[["heat", "rain"]].corr())

# Drop countries with zero within-country heat variation (they don't matter because of the fixed effects)
# and will inflate my N
zero_var_countries = within_country_heat_var[within_country_heat_var == 0].index.tolist()
nan_var_countries = within_country_heat_var[within_country_heat_var.isna()].index.tolist()
excluded_countries = zero_var_countries + nan_var_countries
print(f'Excluding countries with no within-country heat variation: {excluded_countries}')

clean_livwell2 = clean_livwell1[~clean_livwell1["country"].isin(excluded_countries)].copy()

print(f'Observations before: {len(clean_livwell1)}, after: {len(clean_livwell2)}')
print(f'Countries before: {clean_livwell1["country"].nunique()}, after: {clean_livwell2["country"].nunique()}')
# Observations before: 758, after: 736
# Countries before: 39, after: 34
# This is fine.

# final descriptives before plotting:
print(clean_livwell2.shape)
print(f'Countries: {clean_livwell2["country"].nunique()}')
print(f'Regions: {clean_livwell2["region_id"].nunique()}')
print(f'Years: {clean_livwell2["year"].nunique()}, range: {clean_livwell2["year"].min()}-{clean_livwell2["year"].max()}')
print(clean_livwell2[["dv", "heat", "rain", "year"]].describe())



"""
Descriptive Plotting:
"""
# nice colours (can be modified):
orange = "#F8B195"
pink = "#F67280"
brighter_purple = "#C06C84"
darker_purple = "#6C5B7B"
blue = "#355C7D"
colors = [orange, pink, brighter_purple, darker_purple, blue]

from matplotlib.colors import LinearSegmentedColormap
hist_cmap = LinearSegmentedColormap.from_list("hist_gradient", [blue, orange, pink])

with PdfPages("Understanding_livWell.pdf") as pdf:
    # DV:
    n, bins, patches = plt.hist(clean_livwell2["dv"], bins=30)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(hist_cmap((left - bins[0]) / (bins[-1] - bins[0])))
    sns.kdeplot(clean_livwell2["dv"], color=orange)
    plt.title("DV Distribution")
    plt.xlabel('DV in % of surveyed women')
    pdf.savefig()
    plt.clf() # 'clear figure'

    # heat:
    n, bins, patches = plt.hist(clean_livwell2["heat"], bins=20)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(hist_cmap((left - bins[0]) / (bins[-1] - bins[0])))
    sns.kdeplot(clean_livwell2["heat"], color=orange)
    plt.title("Heat distribution")
    plt.xlabel('share of months with a positive temperature anomaly > 2 SD')
    pdf.savefig()
    plt.clf()

    # rainfall:
    n, bins, patches = plt.hist(clean_livwell2["rain"], bins=30)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(hist_cmap((left - bins[0]) / (bins[-1] - bins[0])))
    plt.title("Rainfall")
    plt.xlabel('Mean deviation from long-run precipitation over 12 months')
    pdf.savefig()
    plt.clf()

    # heat and DV?
    sns.scatterplot(x="heat", y="dv", data=clean_livwell2, alpha=0.3, color=colors[4])
    sns.regplot(x="heat", y="dv", data=clean_livwell2, scatter=False, color=colors[1])
    plt.title("Heat and DV association")
    pdf.savefig()
    plt.clf()

    # time trends (standardized bc heat is in sds and wasn't visible)
    # group and get the mean of dv and heat within each year
    time_trends = clean_livwell2.groupby("year")[["dv", "heat"]].mean()
    # standardize; divide deviation from mean by sd for both heat and dv
    time_trends_std = (time_trends - time_trends.mean()) / time_trends.std()
    time_trends_std.plot(color=[colors[4], colors[1]], marker="")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # This means "only show integer tick labels"
    plt.title("Time trends of heat and DV (standardized)")
    pdf.savefig()
    plt.clf()

    # visualise my exclusions!
    dv_by_year = livwell1.groupby("year")["dv"].apply(lambda x: x.notna().mean())
    plt.figure()
    sns.lineplot(x=dv_by_year.index, y=dv_by_year.values, color=colors[4])
    plt.title("DV Availability (plotting NAs)")
    plt.xlabel("Year")
    plt.ylabel("Share of Observations with DV Data by Year")
    plt.ylim(0, 1)
    pdf.savefig()
    plt.close()

    # also do some checks with maps (Annika's recommendation)
    # DV:
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
    # match the dataframe codes with the gpd countries
    country_data = clean_livwell2.groupby("country")[["dv", "heat"]].mean().reset_index()
    # codes in the dataset are same as some from the library; can be merged easily
    world = world.merge(country_data, how="left", left_on="SOV_A3", right_on="country")
    fig, ax = plt.subplots(figsize=(12, 6))
    # dv map
    world.plot(
        column="dv",
        cmap="OrRd",
        legend=True,
        missing_kwds={"color": "lightgrey"},
        ax=ax
    )
    ax.set_title("Average Domestic Violence (Country Level)")
    ax.set_axis_off()
    pdf.savefig(fig)
    plt.close(fig)

    # and heat map:
    fig, ax = plt.subplots(figsize=(12, 6))
    world.plot(
        column="heat",
        cmap="coolwarm",
        legend=True,
        missing_kwds={"color": "lightgrey"},
        ax=ax
    )
    ax.set_title("Average Heat Exposure (Country Level)")
    ax.set_axis_off()
    pdf.savefig(fig)
    plt.close(fig)

    # finally data coverage map
    # count how many years are available per country
    country_years = clean_livwell2.groupby("country")["year"].nunique().rename("n_years")
    country_data = clean_livwell2.groupby("country")[["dv", "heat"]].mean().reset_index()
    country_data = country_data.merge(country_years, on="country")
    world = world.merge(country_data, how="left", left_on="SOV_A3", right_on="country")
    fig, ax = plt.subplots(figsize=(12, 6))
    world.plot(
        column="n_years",
        cmap="RdYlGn",
        legend=True,
        missing_kwds={"color": "lightgrey"},
        vmin=0,
        vmax=7,
        ax=ax
    )
    ax.set_title("Number of Years Observed per Country")
    ax.set_axis_off()
    pdf.savefig(fig)
    plt.close(fig)

    # noticeable and also mentioned in the livWell description, but I
    # missed it:
    # countries are primarily low- and middle-income nations,
    # bc the dataset is built using Demographic and Health Survey (DHS) data,
    # which focuses on these regions.
    # Sub-Saharan Africa, South and Southeast Asia, and Latin America
    # Western high-income countries are missing

# NOTE for later checking: I checked the heat removal using my heat variable.
# But that is already coded in deviations from local means...
# so I MAY have removed warmer countries after all... definitely check!




"""
For the non-parametric idea:
"""

# to see whether I can just do a binary test or whether a Kenall's tau - ish
# approach is worth it; check how many observations i have per regions...
# - how many regions have more than two?
# - How many pairs would I have?

# 1: how many regions have more than two years?
grouped_livWell = clean_livwell2.groupby("region_id")
livwellyearsbyregion = grouped_livWell['year'].nunique()
uniquecounts = {}
for r in livwellyearsbyregion.index:
    if livwellyearsbyregion[r] in uniquecounts:
        uniquecounts[livwellyearsbyregion[r]] += 1
    else:
        uniquecounts[livwellyearsbyregion[r]] = 1
print(f'\nHlow often do you have each count of years available per country?\n')
for n_years, n_regions in sorted(uniquecounts.items()):
    print(f"{n_years} year(s): {n_regions} regions")


# I have many regions with more than 2 observations, so I need to think of
# something that's more than a yes / no.
# Idea:
# Run Model 2 as the main analysis: country + year fixed effects, rainfall control,
#     clustered standard errors.
# As a nonparametric verification, run a binomial test:
#
# 1. For each region, check if the hotter year is also the year with more violence
#     1a) If I have two years, this is just a 1 or 0
#     1b) If I have 3+, check all possible pairs and see if concordant pairs are more
#         common than discordant. If yes, region is a 1. If no, region is a 0.
#     1c) If equal, throw the region away (and report how many were dropped)
# 2. Count the number of 1s out of the total number of regions
# 3. Locate that in a Bernoulli distribution; that's the test
#
#
# After running the binomial test, check whether the hotter year was also always
# the later year. If yes, acknowledge this as a limitation,
# but note that survey years are NOT always the same two years. There are different
# years available per country.
# Also: Model 2 controls for year effects, which the binomial test cannot,
# so if both point in the same direction, that is reassuring.
# They account for each other's weaknesses.

# I would have to throw 142 regions away for this, because they only have
# a single year. But I'd still have
regions_with_several_years = 0
for count in uniquecounts:
    if count > 1:
        regions_with_several_years += uniquecounts[count]

print(f"\nRegions with more than one year: {regions_with_several_years}.\n")
# 185 regions! 594 observations. That's not too bad.

# make a dataset for this:
# find regions with more than 1 year
valid_regions = []

for r in livwellyearsbyregion.index:
    if livwellyearsbyregion[r] > 1:
        valid_regions.append(r)

# create new dataset with only those regions
non_parametric_livwell = clean_livwell2[clean_livwell2["region_id"].isin(valid_regions)].copy()

# check
print(f"Number of regions in new dataset: {non_parametric_livwell['region_id'].nunique()}")
print(f"Number of observations in new dataset: {len(non_parametric_livwell)}")



"""
Finally, save the cleaned dataset for future analyses!
"""
# save it here
clean_livwell2.to_csv("clean_livwell.csv", index=False)
# but also in the analysis folder (make sure it exists!)
clean_livwell2.to_csv("../Step2_Analysis/clean_livwell.csv", index=False)
# and in the step 3 folder!
clean_livwell2.to_csv("../Step3_PredictionTool/clean_livwell.csv", index=False)
