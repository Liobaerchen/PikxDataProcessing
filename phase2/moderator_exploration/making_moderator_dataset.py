# Understanding LivWell

# Lioba Roggendorf

# April 23rd, 2026

"""
Here, I create a cleaned dataset for the LivWell moderator analyses.

Main variables:

Domestic violence
Heat exposure
Rainfall control

Moderators:
  Urbanicity
  Women's empowerment
  Wealth
  Age gaps
  Cooling (closest proxy: electricity access)

 It's the same as the Understanding_livWell cleaning, just re-done to include the
 new variables i need.
    """

import pandas as pd

complete_livwell = pd.read_csv("livwell.csv")
print(complete_livwell.head())

# Main variables:

cols_i_need = [
"DV_phys_12m_p",          # Women who experienced physical violence in past 12 months (%)
"tmp_anom_p2sd_share12",  # Share of months with positive temperature anomaly > 2 SD
"year",
"region_num_harmonized",
"country_code",
]

# rainfall control:

rainfall = "pre_anom_mean12"

"""
Moderator variables:
"""

Urbanicity = ['DM_urban_p']

Empowerment_Human_Capital = [
'ED_educ_years_mean',
'ED_litt_p'
]

Empowerment_Information_Access = [
'EI_internet_week_p',
'EI_news_week_p',
'EI_radio_week_p',
'EI_tv_week_p'
]

Empowerment_Economic = [
'DP_decide_money_p',
'DP_earn_more_equal_p',
'WK_working_paid_p',
'DP_owns_house_p',
'DP_owns_land_p'
]

Empowerment_Autonomy = [
'DP_decide_health_p',
'DP_decide_large_purchase_p',
'DP_decide_visits_p',
'DP_decide_contraception_p',
'DP_decide_no_contraception_p'
]

Partner_Violence_Proportion = [
'DV_phys_partner_12m_p',
'DV_phys_12m_p'
]

Wealth_Moderation = [
'WL_wealth_median'
]

Age_Gap = [
'DM_age_diff_mean',
'DM_age_diff_10plus_p'
]

Cooling_Access = [
'EI_women_elec_p'
]

# combine all moderator variables

all_moderators = (
Urbanicity
+ Empowerment_Human_Capital
+ Empowerment_Information_Access
+ Empowerment_Economic
+ Empowerment_Autonomy
+ Partner_Violence_Proportion
+ Wealth_Moderation
+ Age_Gap
+ Cooling_Access
)

"""
Create dataframe:
"""

livwell1 = pd.DataFrame({
'dv': complete_livwell[cols_i_need[0]],
'heat': complete_livwell[cols_i_need[1]],
'year': complete_livwell[cols_i_need[2]],
'region': complete_livwell[cols_i_need[3]],
'country': complete_livwell[cols_i_need[4]],
'rain': complete_livwell[rainfall]
})

# add moderator variables

for variable in all_moderators:
    livwell1[variable] = complete_livwell[variable]

print(livwell1.head())


"""
Excluding missing data:
"""

# only exclude rows missing the original core analysis variables

core_variables = [
'dv',
'heat',
'rain',
'year',
'region',
'country'
]

rowsWith_Core_NA = livwell1[core_variables].isna().any(axis=1)

any_rowsWithDV_NA = len(livwell1[rowsWith_Core_NA])

# ~ means NOT

cleaned_moderator_livwell = livwell1[~rowsWith_Core_NA].copy()

print(cleaned_moderator_livwell.head())

"""
Create unique region identifier:
"""

cleaned_moderator_livwell["region_id"] = (
cleaned_moderator_livwell["country"].astype(str)
+ "_"
+ cleaned_moderator_livwell["region"].astype(str)
)

print(cleaned_moderator_livwell.groupby("region")["country"].nunique())

"""
Remove countries with no within-country heat variation:
"""

within_country_heat_var = (
cleaned_moderator_livwell
.groupby("country")["heat"]
.std()
)

countries_withNOvar = (
within_country_heat_var[
within_country_heat_var == 0
].index.tolist()
)

countries_with_NA_var = (
within_country_heat_var[
within_country_heat_var.isna()
].index.tolist()
)

excluded_countries = countries_withNOvar + countries_with_NA_var

print(f'Excluding countries with no within-country heat variation: {excluded_countries}')

cleaned_moderator_livwell = cleaned_moderator_livwell[
~cleaned_moderator_livwell["country"].isin(excluded_countries)
].copy()

"""
Final descriptives:
"""

print(cleaned_moderator_livwell.shape)

print(f'\nnumber of countries: {cleaned_moderator_livwell["country"].nunique()}.\n')

print(f'number of regions: {cleaned_moderator_livwell["region_id"].nunique()}.\n')

print(f'Years: {cleaned_moderator_livwell["year"].nunique()}')

print(cleaned_moderator_livwell.describe())


"""
Finally, save the cleaned dataset for future analyses!
"""

cleaned_moderator_livwell.to_csv(
"cleaned_moderator_livwell.csv",
index=False
)
