# Analysing LivWell
# Lioba Roggendorf
# May 2nd, 2026

"""
In this file, I conduct assumption checks and analysis using my models 2 and 3.

Reminder, model-2 is a fixed-effects regression, based on regional differences.
country + year fixed effects and rainfall control.
Compares regions within the same country
-- are hotter regions more violent?

DV_rct = β · Heat_rct + γ · Rain_rct + δ_c + τ_t + ε_rct
Heat is  main independent variable
Rain is a control variable or covariate
δ_c are country fixed effects; remove all time-invariant between-country differences
τ_t are year fixed effects; remove global shocks shared by all countries in a given year
β is the coefficient of interest; if heat increases by 1, DV changes by β (holding rain constant and subtracting year and country effects... etc. etc.)
γ is the coefficient on rain; should be reported, but not interpreted. just a control.

model 3 is a nonparametric binomial check, based on time differences.
For each region: is the hotter year also the more violent year? (1/0 per region)
Binomial test against p=0.5. Complements Model 2. If both point the same way, reassuring.
"""

import pandas as pd

# also again, use the same colours:
orange = "#F8B195"
pink = "#F67280"
brighter_purple = "#C06C84"
darker_purple = "#6C5B7B"
blue = "#355C7D"
colors = [orange, pink, brighter_purple, darker_purple, blue]

# Load the cleaned dataset from Step 1: understanding LivWell
livwell2 = pd.read_csv("clean_livwell.csv")


"""
Step 1: Model 2, assumption checks


1. Linearity
    - Plot residuals vs. fitted values
    - Plot heat vs. DV, eyeball the relationship
    - Use AIC to compare a squared to a linear model

2. Homoscedasticity
    - Plot residuals vs. fitted values

3. Normality of residuals
    - Q-Q plot
    - Histogram of residuals
    - Decided not to do Shapiro-Wilk or similar tests, because those become
    significant in almost all cases with large enough n, also
    - Less critical with large n (CLT), but let's check

4. Multicollinearity
    - Variance Inflation Factor (VIF) for heat and rainfall
    - VIF > 5 is concerning, > 10 is serious

5. Outliers / influential observations
    - Cook's distance (look at observations > 4/n,
    but generally don't remove without sufficient reason)
    - Standardised residuals (look at z > 3)
    - Note outliers, maybe rerun model without noted observations as robustness check

6. Skew of DV
    - DV histogram

A great package for this seems to be: statsmodels
https://www.statsmodels.org/stable/index.html
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_reset
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from math import comb

# Linearity and Homoscedasticity
# fit model

# notation for this package is like in R, you write DV ~ IV.
# the model just specifies heat and rain as continuous predictors.
# but country and year get their own intercepts, which is like subtracting their effects.
# https://www.statsmodels.org/stable/mixed_linear.html

# but problem, smf doesn't seem to have specific documentation for the
# type of fixed effects i want.
# BUT you can do dummy coding: https://www.statsmodels.org/dev/examples/notebooks/generated/contrasts.html,
# which is the same, mathematically ('do I add the intercept or not?').
# 'C' means 'categorical'. So this tells the programme to dummy-code the variable.
# model = smf.ols("dv ~ heat + rain + C(country) + C(year)", data=livwell2).fit()

# problem 2: statsmodels does not seem to have Conley SEs, and I couldn't find
# another package that does. It seems to be common in R.
# ask Leonie how important this is.
# solution for now:
# observations aren't independent.
# Regions within the same country share culture, institutions, climate patterns...
# errors are probably correlated.
# "there may be serial correlation in both climate variables and conflict variables
# that is not fully accounted for by detrending, and which has the potential to
# lead to incorrect inference if P values are not appropriately adjusted (Bertrand et al. 2004)"
# -- Burke, Hsiang, and Miguel (2015)
# so: https://stackoverflow.com/questions/54349525/clustered-standard-errors-in-statsmodels-with-categorical-variables-python
the_holy_grail = smf.ols("dv ~ heat + rain + C(country) + C(year)", data=livwell2).fit(
    cov_type="cluster", cov_kwds={"groups": livwell2["country"]}
)

# residuals vs. fitted
plt.scatter(the_holy_grail.fittedvalues, the_holy_grail.resid, alpha=0.3, color=brighter_purple)
plt.axhline(0, color=pink)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. fitted")
plt.savefig("residualsVsFitted.jpg")
plt.show()
plt.close()
# NOTE: Looks like heteroscedasticity!
# but clustered SEs handle that

# heat vs. DV
plt.scatter(livwell2["heat"], livwell2["dv"], alpha=0.3, color=darker_purple)
plt.xlabel("Heat")
plt.ylabel("DV")
plt.title("Heat vs. DV")
plt.savefig("heatVsDV.jpg")
plt.show()
plt.close()
# NOTE 2: without controlling for country etc., no obvious relationship emerges...
# but it could be weak or only emerge after controlling!

# AIC
the_holy_grail_two_degree_poly_version = smf.ols("dv ~ heat + I(heat**2) + rain + C(country) + C(year)", data=livwell2).fit(
    cov_type="cluster", cov_kwds={"groups": livwell2["country"]}
)
print(f'\n\nBase model AIC: {the_holy_grail.aic},\n Square model AIC: {the_holy_grail_two_degree_poly_version.aic}.\n\n')
# Base model AIC: 4934.44271738837,
# Square model AIC: 4935.098120706165.
# that's great! Or at least quadratic doesn't fit better.
# Conclusion: Some heteroscedasticity and non-obvious linearity, but relationship
# could be weak or appear only after controls.
# Furthermore, AIC indicates a linear relationship fits better than quadratic.


"""
3. Normality of residuals
    - Q-Q plot
    - Histogram of residuals
    - Less critical with large n (CLT), but let's check
"""

# how to qq: https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html

qqplot(the_holy_grail.resid, line='s', markerfacecolor=blue, markeredgecolor=blue, alpha=0.5)
plt.title("Q-Q plot of residuals")
plt.savefig("qq.jpg")
plt.show()
plt.close()

# plotting the residuals:
# googled 'residuals histogram in statsmodels' and was immediately shown this:
# https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.resid.html
residuals = the_holy_grail.resid

plt.hist(residuals, color=blue, edgecolor=brighter_purple)
plt.xlabel("Residuals")
plt.title("Histogram of residuals")
plt.savefig("residualHist.jpg")
plt.show()
plt.close()

# Conclusion: Bit of a heavy tail, but not extreme, mostly normal, and
# should be okay with large n.
# makes sense with right-skewed DV distribution, particularly in the upper tail.
# next up:

"""
4. Multicollinearity
    - Variance Inflation Factor (VIF) for heat and rainfall
    - VIF > 5 is concerning, > 10 is serious

https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
"""

heatAndRain = livwell2[["heat", "rain"]]
heatCol = 0 # which column is that of the variable i wanna check?
print(heatAndRain.columns)

vifHeat = variance_inflation_factor(heatAndRain.values, heatCol)
print(f"VIF for heat: {vifHeat}")



# NOTE: VIF for heat and rain correlation: 1.03
# great!

"""
5. Outliers / influential observations
    - Cook's distance (look at observations > 4/n,
    but generally don't remove without sufficient reason)
    - Standardised residuals (look at z > 3)
    - Note outliers, maybe rerun model without noted observations as robustness check

https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html
"""

influence = OLSInfluence(the_holy_grail)
# returns a tuple of two arrays: the Cook's distances themselves, and the p-values for each observation
cooks, _ = influence.cooks_distance
stdandardizedResiduals = influence.resid_studentized_internal

threshold = 4 / len(livwell2)
print(f"Observations with Cook's distance > {threshold:.4f}:")
print(livwell2[cooks > threshold])

print(f"\nObservations with |standardised residual| > 3:")
print(livwell2[abs(stdandardizedResiduals) > 3])

# quite some outliers!
# save them for possible re-analysis
cooking = livwell2.index[cooks > threshold].tolist()
farfarawaydatapoints= livwell2.index[abs(stdandardizedResiduals) > 3].tolist()
# use a set to keep indices from both cooks and sd, but not duplicate them
outliers = list(set(cooking + farfarawaydatapoints))
print(f"Outlier count: {len(outliers)}")
# Outlier count: 46

"""
Finally:
6. Skew of DV
    - Histogram
"""
plt.hist(livwell2["dv"], color=blue, edgecolor = brighter_purple)
plt.title("DV distribution")
plt.savefig("dvHist.jpg")
plt.show()
plt.close()

# NOTE: this IS skewed!
# BUT: OLS doesn't assume the variable
# itself is non-skewed.
# only that the residuals are normal, and that relaxes with large n.
# I already plotted the residuals, and they were okay.

# also just noticed... i think skew can be an issue if it leads to predictions
# below 0... because that's nonsensical for my data.
# BUT most of my values are between 5 and 30%, so I think I should be okay.


"""
Acknowledging the noted issues and asking Leonie about them... it seems overall
acceptable!
Let's look at the model!
"""
# dandandandandannnnnnnnnnnn (drumroll):
print('\nMODEL INCOMING:\n\n')
print(the_holy_grail.summary())
print('\n\n')

# PROBLEM: got a warning.
# ValueWarning: covariance of constraints does not have full rank. The number of constraints is 52, but rank is 17
#  warnings.warn('covariance of constraints does not have full '
#                            OLS Regression Results
# Have to find out what this means... ask a
# data scientist or Leonie. First hint:
#
# your fixed effects + clustering are overlapping in structure,
# so the software cannot fully separate all variance components,
# which makes the standard error calculations partially unstable.
#
# Key point:
# - Your coefficient estimates are still fine
# - but the p-values / standard errors may be less reliable than they look

# summary:
# R-squared:                       0.649
# Model:                            OLS   Adj. R-squared:                   0.623
# Method:                 Least Squares   F-statistic:                      321.1
# Prob (F-statistic):            1.11e-31

print('\SIGNIFICANCE INCOMING:\n\n')
print(f"Heat coefficient: {the_holy_grail.params['heat']} (SE: {the_holy_grail.bse['heat']}, p = {the_holy_grail.pvalues['heat']})")
print('\n\n')

# Heat coefficient: 8.240180571416907
# (SE: 3.6441872570635367, p = 0.023747820408485894)

# since both are measured in percent, a 1 increase doesn't make sense. BUT:
# A 10 percentage-point increase in extreme-heat-month count
# is associated with a roughly 0.8 percentage-point increase in DV prevalence
# (controlling for rainfall, country, and year fixed effects etc. etc.).
# This seems small, but it is meaningful if you consider one of my favourite papers
# ever:
# I forgot the name and couldn't find it, but the point is explained here:
# https://pmc.ncbi.nlm.nih.gov/articles/PMC2560511/
# the point being
# Even if the increase in risk for each person is tiny,
# multiplying that small increase across a large population
# can produce a large number of additional cases overall.
# imagine: 1,000,000 women experience DV in these countriees
# 0.8% increase could mean
# 8,000 additional cases... man
# NOTE: this would be great to mention and maybe visualise

"""
Now the fun part... coding the binary check!

For each region, check if the hotter year is also
the more violent year:

2 years → score 1 or 0 directly
3+ years → check all pairs; majority patternMatching = 1,
    NOT_patternMatching = 0,
    ties dropped
Count the 1s across all regions and locate that
in a Bernoulli(p = 0.5) distribution - that's the test.

"""

# Think about this and report it:
# this is the min heat difference necessary to be interpreted
# as meaningful:
min_heat_diff = 0.01

# I will need to check whether a hotter region has more
# dv many times, so I am making it a function:
def hotter_year_more_dv (list_with_two_dicts, min_heat_diff):
    # first of all, if it's a tie, I can save some computing:
    if abs(list_with_two_dicts[0]['heat'] - list_with_two_dicts[1]['heat']) < min_heat_diff:
        return 'tie'

    # if not:
    hotterindex = None
    if list_with_two_dicts[0]['heat'] > list_with_two_dicts[1]['heat']:
        hotterindex = 0
    else:
        hotterindex = 1
    # after identifying the hottest year...
    # does it also have more dv?

    # super cool trick for turning 0 into 1 and 1 into 0:
    otherindex = 1 - hotterindex
    if list_with_two_dicts[hotterindex]['dv'] > list_with_two_dicts[otherindex]['dv']:
        return True
    else:
        return False

# for every region, get the average dv and heat
# for every year
grouped_by_region = livwell2.groupby('region_id')
heat_dv_dicts = {}
# idea: make this dict structure:
# super-dict with entries being regions
# every region has entries that are lists
# dicts with years, heat, dv
for region_name, region in grouped_by_region:
    # for each region, group by years and get
    # average heat and dv per year
    region_grouped_yearly = region.groupby('year')
    looplist = []
    for year, info in region_grouped_yearly:
        dv_mean = info['dv'].mean()
        heat_mean = info['heat'].mean()

        #make a dict to append to the list
        loopdict = {
            'region' : region_name,
            'year' : year,
            'heat' : heat_mean,
            'dv' : dv_mean
        }
        looplist.append(loopdict)
    heat_dv_dicts[region_name] = looplist

# now use this for the algorithm I thought of!
yesnolist = [] # this will contain the ones and zeros for the bernoulli
tiecount = 0
only_one_year = 0

for region in heat_dv_dicts:
    current = heat_dv_dicts[region]
    # if it's only 2 years, just do 1 or 0 based on
    # hotter year having more dv or not
    if len(current) == 1:
        only_one_year += 1
    elif len(current) == 2:
        # int turns the true false into 1 0
        hotter = hotter_year_more_dv(current, min_heat_diff)
        if hotter == 'tie':
            tiecount += 1
            continue
        else:
            yesnolist.append(int(hotter))

    # now the harder part...
    # if it's not 2, check all pairs!
    # how to do this without checking doubles?
    # compare 1 to all others, then compare 2 to all others EXCEPT 1,
    # and 3 to all others except 1 and 2, etc. etc.
    elif len(current) > 2:
        patternMatching = 0
        NOT_patternMatching = 0
        for i in range(0,len(current)):
            # compare to all others EXCEPT the ones that were already checked!
            # i keeps track of this!
            for j in range((i+1), len(current)):
                selectedyears = [current[i], current[j]]
                result = hotter_year_more_dv(selectedyears, min_heat_diff)
                if result == 'tie':
                    continue
                if result:
                    patternMatching += 1
                else:
                    NOT_patternMatching += 1
        # now check if there were more patternMatching (1)
        # NOT_patternMatching (0),
        # or ties (drop):
        if patternMatching > NOT_patternMatching:
            yesnolist.append(1)
        elif NOT_patternMatching > patternMatching:
            yesnolist.append(0)
        else:
            tiecount += 1

# now the test!
# RQ is "is extreme heat related to increased violence"
# that is DIRECTIONAL!

# n is total number of regions
n = len(yesnolist)
# k is number of patternMatching regions.. think of a coin-flip!
k = sum(yesnolist)

# aks: "What is the probability of getting k or more patternMatching regions
# out of n, purely by chance (p=0.5)?"
# formula:
# P(X >= k) =
#    sum from i=k to n of: (n choose i) * p^i * (1-p)^(n-i)
#
# P(X >= k)      : probability of getting at least k patternMatching regions by chance
# sum i=k to n   : add up all outcomes at least as extreme as what was observed
# (n choose i)   : number of ways to get exactly i patternMatching regions out of n
# p^i            : probability that i regions ARE patternMatching (p=0.5, like a coin flip)
# (1-p)^(n-i)    : probability that the remaining (n-i) regions are NOT patternMatching

p = 0.5
p_value = sum(comb(n, i) * (p**i) * ((1-p)**(n-i)) for i in range(k, n+1))

print("\n\nBINOMIAL RESULTS INCOMING:\n")
print(f"Regions included: {n}")
print(f"Regions where hotter = more violent: {k}")
print(f"Regions dropped (ties): {tiecount}")
print(f"Regions with only 1 year (also dropped): {only_one_year}")
print(f"One-tailed p-value: {p_value}")


# NOTE: OH WHAT!!! THINK ABOUT THIS!
# Regions included: 109
# Regions where hotter = more violent: 51
# Regions dropped (ties): 76
# Regions with only 1 year (also dropped): 142
# One-tailed p-value: 0.7781652493988257
# it's basically 50/50

# may be because of coarseness of measure

# possible explanations:
# problem: need to interpret my results without too much biased, p-hacking adaptation.

#
# No real effect at the time scale I use / aggregation is too coarse
#
# Heat might only affect violence in the very short term (like during very hot days),
# or on very long time scales (how to test that?)
# but my data averages everything over a whole year.

# And I use the “share of extreme months” over a year.
# This turns detailed temperature changes into one simple number,
# which removes a lot of useful information.

#
# Measurement error
#
# My DV data comes from surveys, and people may not report violence accurately.
# My heat variable is also smoothed over many months.
# When both variables are noisy, real effects get blurred and look like zero.
# And the binomial test is not very powerful at all.

#
# Not enough change within regions
#
# Within each region, heat does not change that much from year to year.
# If nothing really changes, the model cannot detect any effect.
# Maybe average heat differences are just too small
# NOTE: try a model that only includes large differences!

#
# Cross-sectional bias in Model 2 (it's actually a 0 effect)
#
# Hotter regions are often also poorer or more rural.
# These factors could impact DV.
# So it may look like heat causes violence, but it’s actually these other differences.

# Also, different regions may have different norms about gender and violence.
# These differences don’t change much over time,
# but they can create strong differences between regions.

#
# Omitted variables
#
# There may be other things affecting violence (like economic stress or conflict)
# that I did not include in the model.
# If those are linked to heat, results can be wrong.


# Testing the "Not enough change within regions" explanation:
def run_binomial(heat_dv_dicts, min_heat_diff):
    """
    Identical algorithm to the original. Made into a function
    so I can reuse it.
    """
    yesnolist = []
    tiecount = 0
    only_one_year = 0

    for region in heat_dv_dicts:
        current = heat_dv_dicts[region]
        if len(current) == 1:
            only_one_year += 1
        elif len(current) == 2:
            hotter = hotter_year_more_dv(current, min_heat_diff)
            if hotter == 'tie':
                tiecount += 1
                continue
            else:
                yesnolist.append(int(hotter))
        elif len(current) > 2:
            patternMatching = 0
            NOT_patternMatching = 0
            for i in range(0, len(current)):
                for j in range((i + 1), len(current)):
                    selectedyears = [current[i], current[j]]
                    result = hotter_year_more_dv(selectedyears, min_heat_diff)
                    if result == 'tie':
                        continue
                    if result:
                        patternMatching += 1
                    else:
                        NOT_patternMatching += 1
            if patternMatching > NOT_patternMatching:
                yesnolist.append(1)
            elif NOT_patternMatching > patternMatching:
                yesnolist.append(0)
            else:
                tiecount += 1

    n = len(yesnolist)
    k = sum(yesnolist)
    p = 0.5
    p_value = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))

    return {
        "n": n, "k": k, "tiecount": tiecount,
        "only_one_year": only_one_year, "p_value": p_value
    }


# Build the region-year structure
grouped_by_region = livwell2.groupby('region_id')
heat_dv_dicts = {}
for region_name, region in grouped_by_region:
    region_grouped_yearly = region.groupby('year')
    looplist = []
    for year, info in region_grouped_yearly:
        dv_mean = info['dv'].mean()
        heat_mean = info['heat'].mean()
        loopdict = {'region': region_name, 'year': year,
                'heat': heat_mean, 'dv': dv_mean}
        looplist.append(loopdict)
    heat_dv_dicts[region_name] = looplist

# Now check for only large heat differences!

"""
Motivation: the real world is noise and a real effect
could look like 50/50 with too few observations, especially when heat differences
are tiny.
By restricting to regions where
heat actually changed, the test gets a better chance to work.

How: for each region, compute the range  of mean heat-s a year.
Keep only regions where that range goes over a threshold.
Try three thresholds to see whether results are sensitive to the choice.
"""

# First: just look at the distribution of within-region heat ranges.
region_heat_ranges = {}
for region_name, obs in heat_dv_dicts.items():
    heats = []
    for o in obs:
        heats.append(o['heat'])
    if len(heats) > 1:
        region_heat_ranges[region_name] = max(heats) - min(heats)

# seires so i can use .describe:
ranges_series = pd.Series(region_heat_ranges)
print("\n\nWITHIN-REGION HEAT RANGE DISTRIBUTION (mechanism 3 check):")
print(ranges_series.describe())

plt.hist(ranges_series, bins=30, color=blue, edgecolor=brighter_purple)
plt.xlabel("Within-region heat range (max − min annual mean heat)")
plt.ylabel("Number of regions")
plt.title("How much does heat actually vary within each region over time?")
plt.savefig("withinRegionHeatRange.jpg")
plt.show()
plt.close()
# If most regions cluster near 0, that supports this alternative explanation

# Now the robustness check itself.
# We try three thresholds. Adjust these based on the histogram above.
thresholds = [0.05, 0.10, 0.20]

print("\n\nROBUSTNESS CHECK: BINOMIAL WITH LARGE HEAT DIFFERENCES ONLY:")
print(f"{'Threshold'} | {'n'} | {'k'} | {'p-value'}")

for threshold in thresholds:
    # Filter to only regions where heat range exceeds threshold
    large_diff_regions = {}
    for region, obs in heat_dv_dicts.items():
        if region in ranges_series and ranges_series[region] >= threshold:
            large_diff_regions[region] = obs
    results = run_binomial(large_diff_regions, min_heat_diff)
    print(f"  >= {threshold} | {results['n']} | {results['k']} | {results['p_value']}")

print("""
  - If p shrinks as threshold rises, null result probably a power-problem.
  - If p stays ~0.5 regardless → the null result is probably robust.
  - Watch out for very small n at high thresholds (power drops!).
""")

# plot this:
thresh_results = []
for t in [0, 0.05, 0.10, 0.15, 0.20]:
    subset = {r: obs for r, obs in heat_dv_dicts.items() if r in ranges_series and ranges_series[r] >= t}
    res = run_binomial(subset, min_heat_diff)
    thresh_results.append({'threshold': t, 'p': res['p_value']})

df_thresh = pd.DataFrame(thresh_results)
plt.plot(df_thresh['threshold'], df_thresh['p'], marker='o', color=pink, linewidth=2)
plt.axhline(0.05, color=darker_purple, linestyle='--', label='Significance (0.05)')
plt.xlabel("Minimum Heat Difference Filter (Threshold)")
plt.ylabel("p-value")
plt.title("Does filtering for larger heat changes make the effect significant?")
plt.legend()
plt.savefig("heat_difference_binomial_p.jpg")
plt.show()
plt.close()

# Power analysis for the binomial test


# The binomial test is not very powerful with small n.
# Here I check: if the true effect were weak,
# would my test even detect it? Probably not.

# I need the actual n from the binomial I already ran.
# yesnolist already exists from the binomial code above:
n = len(yesnolist)

from scipy.stats import binom

# For a given true concordance rate, what is the probability
# that my test correctly rejects the null (= power)?
def compute_power(n, p_true, alpha=0.05):
    # Step 1: find the critical value.
    # That is: what is the minimum number of patternMatching regions
    # needed to reject H0 at alpha = 0.05, one-tailed?
    critical_value = binom.ppf(1 - alpha, n, 0.5)

    # Step 2: given the true rate, what is the probability
    # of reaching that critical value?
    power = 1 - binom.cdf(critical_value, n, p_true)
    return power

# Try a few plausible true effect sizes and see what power I have:
print(f"Power analysis (n = {n})")
print(f"'True concordance rate' 'Power'")

for p_true in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    power = compute_power(n, p_true)
    print(f"  {p_true:.0%}                      {power:.0%}")

# Power analysis (n = 109)
# 'True concordance rate' 'Power'
#   55%                      25%
#   60%                      65%
#   65%                      93%
#   70%                      100%

print(f"""
  Power = probability of detecting a true effect.
  A weak true effect (55%) would probably go undetected with n = {n}.
  So the null result is not strong evidence against an effect:
  the test just may not be sensitive enough with n = {n}.
""")

# plotting this:
# Concordance-Power Plot
concordances = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
powers = []
for concordance in concordances:
    powers.append(compute_power(n,concordance))
real_concordance = k / n

plt.fill_between(concordances, powers, color=blue, alpha=0.3)
plt.plot(concordances, powers, color=blue, linewidth=2)
plt.axvline(real_concordance, color=orange, linestyle='-', linewidth=3, label=f'Observed Data ({real_concordance:.1%})')
plt.xlabel("True Concordance Rate (If the effect exists)")
plt.ylabel("Probability of detecting it (Power)")
plt.title(f"Sensitivity Analysis: Could we even detect a weak effect with N={n}?")
plt.legend()
plt.savefig("binomial_sensitivity_analysis.jpg")
plt.show()
plt.close()
