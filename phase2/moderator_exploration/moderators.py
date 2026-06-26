import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

filename = "cleaned_moderator_livwell.csv"
save_plots_here = "/Users/lioba/Desktop/PIK/PikxDataProcessing/docs/phase2_plots"

livwell_all = pd.read_csv(filename)

"""
________________________________________________________________________________
STEP 1: Filter to ≥3 years per region (same as model 4)
"""

years_per_region = (
    livwell_all
    .groupby("region_id")["year"]
    .nunique()
)

min_years = 3
filterr = (years_per_region >= min_years)
valid_regions = years_per_region[filterr].index

livwell_3plus = livwell_all[
    livwell_all["region_id"].isin(valid_regions)
].copy()

# use the same colours:
orange = "#F8B195"
pink = "#F67280"
brighter_purple = "#C06C84"
darker_purple = "#6C5B7B"
blue = "#355C7D"
colors = [orange, pink, brighter_purple, darker_purple, blue]

"""
________________________________________________________________________________
STEP 2: BASE MODEL (model 4)
"""

"""
DV_rct = β · Heat_rct + γ · Rain_rct + α_r + τ_t + ε_rct
Region + year fixed effects, clustered SEs by region.
"""

the_holy_grail = smf.ols(
    "dv ~ heat + rain + C(region_id) + C(year)",
    data=livwell_3plus
).fit(
    cov_type="cluster",
    cov_kwds={"groups": livwell_3plus["region_id"]}
)

print("\nBASE MODEL (reference)")
print(f"Heat: {the_holy_grail.params['heat']:.4f} (SE: {the_holy_grail.bse['heat']:.4f}, p = {the_holy_grail.pvalues['heat']:.4f})")


"""
________________________________________________________________________________
STEP 3: What assumptions do i need to check?
"""
#  Within-region variance of each moderator --> is there any variance at all?

# VIF

# Correlation between heat and each moderator

"""
________________________________________________________________________________
STEP 4: Which variables are useful for the moderators?
"""

Urbanicity = ['DM_urban_p']
# this is a regional variable:
# "Women living in urban areas (%)"

Empowerment_Human_Capital = ['ED_educ_years_mean', 'ED_litt_p']
# women who are literate in that region. ALSO: education years are not percentages!
# pay attention to this.. this needs some form of scaling

Empowerment_Information_Access = ['EI_internet_week_p','EI_news_week_p', 'EI_radio_week_p', 'EI_tv_week_p']
# e.g. "Women using the internet at least one a week (%)"

Empowerment_Economic = ['DP_decide_money_p', 'DP_earn_more_equal_p', 'WK_working_paid_p', 'DP_owns_house_p', 'DP_owns_land_p']
# "Women currently married or in union who were paid cash for their work, who decide alone how their money earned is spent (%)"
# "Women currently married or in union who were paid cash for their work, earning more or equal than her partner (%)"
# "Women who women who worked in the past 12 months who are paid in their work (%)"
# --> maybe give one point for working, 2 for deciding how spent, and 3 for earning as much as their partners AND deciding how spent?
# "Women owning a house alone or jointly (%)"
# same for land

Empowerment_Autonomy = ['DP_decide_health_p', 'DP_decide_large_purchase_p', 'DP_decide_visits_p', 'DP_decide_contraception_p', 'DP_decide_no_contraception_p']

Partner_Violence_Proportion = ['DV_phys_partner_12m_p', 'DV_phys_12m_p']
# "Women who experienced physical violence by partner in the last 12 months(%)"
# "Women who experienced physical violence in the past 12 months(%)"

Wealth_Moderation = ['WL_wealth_median']
# "Median of the International Wealth Index"

Age_Gap = ['DM_age_diff_mean', 'DM_age_diff_10plus_p']
# "Average difference between women and husband/partner, For currently married women aged 15-24"

Cooling_Access = ['EI_women_elec_p']
# "Women living in an household with electricity (%)"

"""
________________________________________________________________________________
STEP 5: Make helper functions to
- make the variables
- check assumptions
"""

# urbanicity can stay the same
# is also already percent... no need to scale. I love livwell

# for education year mean, i'll take the max education years
# then make the variable a percentage so that it works for scoring
# so this is
# mean years of education a woman has completed in that region, as proportion
# of maximum years of education in the dataset
education_max = livwell_3plus['ED_educ_years_mean'].max()
# now for each row, make it a proportion
percentage_edu = []
for education in livwell_3plus['ED_educ_years_mean']:
    proportion = education/education_max
    percentage = proportion * 100
    percentage_edu.append(percentage)

# now make an education score column by just averaging my two variables
# FIX: renamed 'sum' to 'total' -- 'sum' is a python built-in function,
# using it as a variable name breaks score_maker later
education_score = []
for i in range(len(percentage_edu)):
    total = percentage_edu[i] + livwell_3plus['ED_litt_p'].iloc[i]
    av = total/2
    education_score.append(av)

# attach my score to livWell
livwell_3plus["education_empowerment_score"] = education_score


# NNNNNEEEEXT:... okay I can just make this a function, too
def score_maker(df, variables, non_percent_vars):

    list_of_lists = []
    # first, scale the non-percentage ones
    for col in variables:
        proportion_list = []
        # IF it's a non-percentage variable, scale it
        if col in non_percent_vars:
            max_val = df[col].max()

            # fix: avoid division errors if max is 0 or NaN
            if pd.isna(max_val) or max_val == 0:
                proportion_list = [0 for _ in df[col]]
            else:
                for row in df[col]:
                    proportion_list.append((row / max_val) * 100)

            list_of_lists.append(proportion_list)

        else:
            # convert to list to avoid pandas alignment issues
            list_of_lists.append(df[col].tolist())

    # now i have my list_of_lists where each list is a scaled variable
    # to make the score, I'll just loop through and calculate it
    # FIX: skip NaNs when averaging, so one missing sub-variable
    # doesn't wipe out the whole row's score
    score = []

    for i in range(len(list_of_lists[0])):
        available = []
        for listt in list_of_lists:
            value = listt[i]
            if not pd.isna(value):
                available.append(value)

        # only NaN if every single sub-variable is missing for this row
        if len(available) > 0:
            total = 0
            for value in available:
                total += value
            score.append(total / len(available))
        else:
            score.append(np.nan)

    return score


def check_moderation(moderator):

    # I didn't delete all NA rows before bc that would have removed ALL rows
    # so now, make a subset where I delete the NAs of that moderator
    subset = livwell_3plus.dropna(
        subset=[moderator]
    ).copy()

    print(f"\n\nMODERATOR: {moderator}")
    print(f"Observations: {len(subset)}")
    print(f"Regions: {subset['region_id'].nunique()}")

    # same model, just added the moderator
    moderation_model = smf.ols(
        f"dv ~ heat * {moderator} + rain + C(region_id) + C(year)",
        data=subset
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": subset["region_id"]}
    )

    interaction_term = f"heat:{moderator}"

    print("\nInteraction effect:")
    print(
        f"interaction is: heat:{moderator} :"
        f"{moderation_model.params[interaction_term]:.4f} "
        f"(SE: {moderation_model.bse[interaction_term]:.4f}, "
        f"p = {moderation_model.pvalues[interaction_term]:.4f})"
    )

    return


# okayyy, use my lil function to make all the scores.
# i already have urbanicity and education_score
livwell_3plus["info_empowerment_score"] = score_maker(
    livwell_3plus,
    Empowerment_Information_Access,
    []
)

livwell_3plus["economic_empowerment_score"] = score_maker(
    livwell_3plus,
    Empowerment_Economic,
    ['WK_working_paid_p']
)

livwell_3plus["autonomy_empowerment_score"] = score_maker(
    livwell_3plus,
    Empowerment_Autonomy,
    []
)

# partner violence needs a different strategy:
# is also not a moderator.. just interesting
proportion_partner_over_all_violence = []
for i in range(len(livwell_3plus['DV_phys_partner_12m_p'])):
    partnerviolence = livwell_3plus['DV_phys_partner_12m_p'].iloc[i]
    overallviolence = livwell_3plus['DV_phys_12m_p'].iloc[i]
    proportion_partner_over_all_violence.append(partnerviolence / overallviolence)
mean_proportion_partner_over_all_violence = np.mean(proportion_partner_over_all_violence)

print('\n\n\n######################')
print('mean proportion of violence that is partner violence in the whole dataset:')
print(f'{mean_proportion_partner_over_all_violence:.4f}')
print('\n\n')
# can do this by region later if interested

# mean proportion of violence that is partner violence in the whole dataset:
# 1.1658
# it's above 1!!! that is WEIRD! What happened???

# anyways..
# Wealth_Moderation can stay the same

# Age_Gap will also use only one of those variables, doesn't need a score
# and Cooling_Access can also stay


# and make the empowerment scores
# FIX: skip NaN sub-scores so one missing domain doesn't wipe the whole row
# also renamed 'sum' to 'total' -- same built-in shadowing issue as above
overall_empowerment = []
for i in range(len(livwell_3plus["autonomy_empowerment_score"])):

    available_scores = []

    autonomy  = livwell_3plus["autonomy_empowerment_score"].iloc[i]
    info      = livwell_3plus["info_empowerment_score"].iloc[i]
    economic  = livwell_3plus["economic_empowerment_score"].iloc[i]
    education = livwell_3plus["education_empowerment_score"].iloc[i]

    if not pd.isna(autonomy):
        available_scores.append(autonomy)
    if not pd.isna(info):
        available_scores.append(info)
    if not pd.isna(economic):
        available_scores.append(economic)
    if not pd.isna(education):
        available_scores.append(education)

    # only NaN if all four domains are missing for this row
    if len(available_scores) > 0:
        total = 0
        for s in available_scores:
            total += s
        overall_empowerment.append(total / len(available_scores))
    else:
        overall_empowerment.append(np.nan)

livwell_3plus['overall_empowerment_score'] = overall_empowerment
# talk about what i did here later: -> score by domain, then weigh each domain the same...
# but i could weigh them differently


# consideration for age-gap:
# DM_age_diff_mean could have negative values (woman older than partner)
# which could cancel out to make the mean 0?
# so let's make an absolute version

absolute_age_diff = []
for age_diff in livwell_3plus['DM_age_diff_mean']:
    absolute_age_diff.append(abs(age_diff))

livwell_3plus['DM_age_diff_abs'] = absolute_age_diff

"""
________________________________________________________________________________
STEP 6: ASSUMPTIOOOONS
"""

# let's go:
# list all the moderators I want to check
# FIX: 'age_gap_mean' was listed twice -- the second entry silently overwrites
# the first in a python dict, so DM_age_diff_mean was never actually checked.
# split into two separate keys.
all_moderators = {
    "empowerment": ["overall_empowerment_score"],
    "cooling": Cooling_Access,
    "age_gap_mean": ["DM_age_diff_mean"],
    "age_gap_abs": ["DM_age_diff_abs"],
    "age_gap_10plus": ["DM_age_diff_10plus_p"],
    "wealth": Wealth_Moderation,
    "urbanicity": Urbanicity,
}

# ______________________________________________________________________________
# Within-region variance

print("#" * 20)
print("CHECK 1: WITHIN-REGION VARIANCE")
print("#" * 20)
print("Mean std close to 0 = not enough within-region variation.")
print("That moderator is basically constant per region, so identification problem.")
print()

for group, moderator_list in all_moderators.items():
    for moderator in moderator_list:

        region_stds = []

        for region_id in livwell_3plus["region_id"].unique():
            # get all rows for this one region
            region_rows = livwell_3plus[livwell_3plus["region_id"] == region_id]
            moderator_values = region_rows[moderator].dropna()

            # need at least 2 data points to compute a std
            if len(moderator_values) >= 2:
                std = moderator_values.std()
                region_stds.append(std)

        if len(region_stds) == 0:
            print(f"{moderator}: no valid regions found")
        else:
            mean_std = np.mean(region_stds)
            print(f"{moderator:<35}  mean within-region std = {mean_std:.4f}")

print('\n')


# ______________________________________________________________________________
# Correlation between heat and each moderator

print("#" * 20)
print("CHECK 2: CORRELATION WITH HEAT")
print("#" * 20)
print('\n')

for group, moderator_list in all_moderators.items():
    for moderator in moderator_list:

        temp_df = livwell_3plus[["heat", moderator]].dropna()
        correlation = temp_df["heat"].corr(temp_df[moderator])
        print(f"{moderator:<35}  r = {correlation:.4f}")

print('\n')


# ______________________________________________________________________________
# VIF
# are heat, rain, and the moderator too correlated?
# VIF > 5 is problem, VIF > 10 is a problem problem

# Note: we only check heat, rain, and the moderator here,
# not the fixed-effect dummies (those would inflate VIF by design).

print("#" * 20)
print("CHECK 3: VIF")
print("#" * 20)
print("VIF > 5 problem.  VIF > 10 problem problem.")
print()

for group, moderator_list in all_moderators.items():
    for moderator in moderator_list:

        # drop any rows where these three are missing
        temp_df = livwell_3plus[["heat", "rain", moderator]].dropna().copy()

        # if nothing is left after dropping NAs, skip this moderator
        if len(temp_df) == 0:
            print(f"Moderator: {moderator}  --> skipped (no data after dropping NAs)")
            print()
            continue

        # vif needs a constant column added (why?? huh)
        temp_df["const"] = 1

        # put into a numpy matrix in a fixed order
        vif_fomrat = temp_df[["heat", "rain", moderator, "const"]].values

        vif_heat      = variance_inflation_factor(vif_fomrat, 0)
        vif_rain      = variance_inflation_factor(vif_fomrat, 1)
        vif_moderator = variance_inflation_factor(vif_fomrat, 2)

        print(f"Moderator: {moderator}")
        print(f"  heat VIF      = {vif_heat:.4f}")
        print(f"  rain VIF      = {vif_rain:.4f}")
        print(f"  moderator VIF = {vif_moderator:.4f}")
        print()


"""
________________________________________________________________________________
STEP 7: NOW DO THE ANALYSIS
"""

def run_selected_moderations(data, my_moderators):

    print("\n\n")
    print("#"*20)
    print("\n")
    print("MODERATION RESULTS INCOMIIING")
    print("#"*20)

    results = []

    for group, moderator_list in my_moderators.items():

        print(f"\n\n### {group.upper()} ###")
        print("-" * 40)

        for moderator in moderator_list:

            subset = data.dropna(subset=[moderator]).copy()

            # if nothing is left after dropping NAs, skip
            if len(subset) == 0:
                print(f"{moderator:<30} --> skipped (no rows left after dropping NAs)")
                continue

            # also need at least 2 years and 2 regions for the fixed effects to work
            if subset["year"].nunique() < 2 or subset["region_id"].nunique() < 2:
                print(f"{moderator:<30} --> skipped (not enough years or regions left)")
                continue

            n_obs = len(subset)
            n_clusters = subset["region_id"].nunique()

            model = smf.ols(
                f"dv ~ heat * {moderator} + rain + C(region_id) + C(year)",
                data=subset
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": subset["region_id"]}
            )

            term = f"heat:{moderator}"

            estimate = model.params[term]
            se = model.bse[term]
            p_val = model.pvalues[term]

            print(
                f"{moderator:<30} "
                f"β={estimate:>8.4f} "
                f"SE={se:>8.4f} "
                f"p={p_val:>8.4f} "
                f"(n={n_obs}, clusters={n_clusters})"
            )

            results.append({
                "group": group,
                "moderator": moderator,
                "beta": estimate,
                "se": se,
                "p": p_val,
                "n": n_obs,
                "clusters": n_clusters
            })

    return results

results = run_selected_moderations(livwell_3plus, all_moderators)

# none significant... think about why!

# after checking within region variation... I think that is it.
# it's tiny.
# and the thing i test for is complex.
# most effects are small and probably noise..... tell Leonie that






"""
________________________________________________________________________________
STEP 8: plotsssss

"""

moderator_labels = []
betas = []
ses = []

for result in results:
    moderator_labels.append(result["moderator"])
    betas.append(result["beta"])
    ses.append(result["se"])

ci_lower = []
ci_upper = []
for i in range(len(betas)):
    ci_lower.append(betas[i] - 1.96 * ses[i])
    ci_upper.append(betas[i] + 1.96 * ses[i])

fig, ax = plt.subplots(figsize=(9, 5))

for i in range(len(moderator_labels)):

    # CI bar
    ax.plot(
        [ci_lower[i], ci_upper[i]],
        [i, i],
        color=blue,
        linewidth=2,
        alpha=0.7
    )

    # point estimate
    ax.plot(
        betas[i], i,
        marker="o",
        color=pink,
        markersize=8,
        zorder=3
    )

# add a line so it's extra clear where there is no moderation at all
ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--", alpha=0.5, label="no effect")

ax.set_yticks(range(len(moderator_labels)))
ax.set_yticklabels(moderator_labels, fontsize=10)
ax.set_xlabel("Interaction coefficient β  (heat × moderator)  with 95% CI", fontsize=10)
ax.set_title("Moderation results: heat × moderator interaction", fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(save_plots_here + "/plot1_coefplot.png", dpi=150)
plt.show()
print("saved: plot1_coefplot.png")
