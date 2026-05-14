# Understanding LivWell
# Lioba Roggendorf
# May 6th, 2026

"""
Here I check:
    - how much DV data there really was; also how many observations
        per country, not only years per country.
    - Which variables had the best temporal coverage.
"""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

livwell_all = pd.read_csv("livwell.csv")
print(livwell_all.head())

# NOTE to self: If there are many observations per region, is the limited year
# availability in issue?
# and is my model looking at region averages? No, right? It is looking at
# individual observations... I think? (Update, see end of document, it IS an issue)

# Question 1: how much DV data is there really per country? In Observations, nOT years!
# plan: group data by country, and just count nr of rows that are not na

# I will re-use this, so make it a funtion:
def print_data_availability_per(grouping_col, data_kind, how_many, what):
    grouped_by_country = livwell_all.groupby(grouping_col)
    # 'count' counts only non-Nan! So only available data.
    non_na_counts = grouped_by_country[data_kind].count()
    # found a pandas function that turns the Series into a df:
    # the 'name' argument names the column with the counts
    count_df = non_na_counts.reset_index(name="valid_observations")

    # Rank them
    non_na_ranked = count_df.sort_values(by="valid_observations", ascending=False)

    print(f"Number of total DV observations per {what}:")
    print(non_na_ranked.head(how_many))

print_data_availability_per("country_name", "DV_phys_12m_p",500, 'country')

# Question 2: how much DV data is there really per region?
# same, but with region:
print_data_availability_per("region_name_harmonized", "DV_phys_12m_p",30, 'region')


# Step 2: Temporal Coverage Analysis
cols = list(livwell_all.columns)
def check_avg_years_per_country(df, cols):
    coverage_results = []

    for col in cols:
        # 1. Filter rows where this column is not NA
        mask = df[col].notna()
        temp_df = df[mask]

        # NEW STEP: group by country to see how many unique years each country has
        # This helps determine the average "depth" of the panel for each variable
        grouped_by_country = temp_df.groupby('country_name')
        years_per_country = grouped_by_country['year'].nunique()

        # 2. count the average unique years per country for this variable
        if not years_per_country.empty:
            avg_years = years_per_country.mean()
        else:
            avg_years = 0

        coverage_results.append({
            'variable': col,
            'avg_years_per_country': avg_years
        })

    # Create DataFrame and rank by the best coverage
    coverage_df = pd.DataFrame(coverage_results)
    coverage_df = coverage_df.sort_values(by='avg_years_per_country', ascending=False)

    return coverage_df

# Mapping the LivWell dataset indicator ranges for analysis
livwell_categories = {
    'Basic Demographics': range(11, 45),        # Age, Urban/Rural, Marital Status
    'Household Infrastructure': range(45, 61),   # Water, Sanitation, Flooring
    'Household Composition': range(61, 63),     # Household Size, Number of Children
    'Employment (WK)': range(63, 71),           # Working status, Paid work, Agriculture
    'Education (ED)': range(71, 89),            # Years of schooling, Literacy, Attainment
    'Decision Power (DP)': range(109, 131),     # Financial & Personal Autonomy
    'Domestic Violence (DV)': range(131, 151),  # Physical and Sexual Violence indicators
    'Reproductive Health (RH)': range(163, 185) # Fertility, Contraception, Sexual History
}

# now, I can apply my function to the categories:
for category_name, index_range in livwell_categories.items():
    #".items" returns the keys (mapped to cat name) and values (mapped to index_range)
    print(f"\n--- CATEGORY: {category_name} ---")

    current_cols = []
    for i in index_range:
        # map the index numbers to the actual column names from our list
        # use 'cols' (the list of all column names) to pick the ones in the range
        current_cols.append(cols[i])

    # Apply my temporal coverage function (now checking avg years per country)
    results = check_avg_years_per_country(livwell_all, current_cols)

    # Print the top 25 (or as many as exist in that category)
    print(results.head(25))

# NOTE: Also realised that the interviews were done in different months.
# need to re-categorise the years! So if it's done in January, it probably relates
# more to the weather of the previous year...
# this is a to-do!

#print(cols[231:365])
# climate seems to be:
range_start = cols.index('pre_anom_mean12')
# to
range_end = cols.index('drought_spei03_n2_share60')
current_cols = cols[range_start : range_end + 1]

# this should show me the max coverage!
# because they should have good climate data...
climate_results = check_avg_years_per_country(livwell_all, current_cols)

# Print the top 25 (or as many as exist in that category)
print(climate_results.head(100))
# so they have only ~ 4 distinct years from every country??
# but that makes sense, because they researched every 4 years and covered only ~20 years!

"""
"The majority of these are based on 199 Demographic and Health Surveys (DHS)
for the period !!!1990–2019!!!"

"As DHS sur-veys are collected in every country on average only every five years,
the original version of LivWell entails gaps between the years of data collection."
"""

## plotting:

# My colors
orange = "#F8B195"
pink = "#F67280"
brighter_purple = "#C06C84"
darker_purple = "#6C5B7B"
blue = "#355C7D"

# 1. How many years do I have (ALL variables)?

all_coverage = check_avg_years_per_country(livwell_all, cols)

plt.figure(figsize=(8, 5))
plt.hist(all_coverage['avg_years_per_country'],
         bins=30,
         color=blue,
         edgecolor=brighter_purple)

plt.title('How many years do the variables actually have?')
plt.xlabel('Average years per country (per variable)')
plt.ylabel('Number of variables')

plt.axvline(all_coverage['avg_years_per_country'].mean(),
            color=pink, linestyle='--', label='Mean')

plt.legend()
plt.tight_layout()
plt.savefig('plot1_coverage_distribution.png')
plt.show()


# 2. DV vs Climate coverage


dv_cols = [cols[i] for i in livwell_categories['Domestic Violence (DV)']]
climate_cols = current_cols

dv_cov = check_avg_years_per_country(livwell_all, dv_cols)
climate_cov = check_avg_years_per_country(livwell_all, climate_cols)

plt.figure(figsize=(8, 5))

plt.hist(dv_cov['avg_years_per_country'],
         bins=15,
         alpha=0.6,
         label='DV',
         color=pink)

plt.hist(climate_cov['avg_years_per_country'],
         bins=15,
         alpha=0.6,
         label='Climate',
         color=blue)

plt.title('DV and climate temporal coverage?')
plt.xlabel('Average years per country')
plt.ylabel('Number of variables')

plt.legend()
plt.tight_layout()
plt.savefig('plot2_dv_vs_climate.png')
plt.show()


# 3. DV observations per region (data imbalance)

region_counts = livwell_all.groupby("region_name_harmonized")["DV_phys_12m_p"].count()
region_counts = region_counts.sort_values(ascending=False)

plt.figure(figsize=(10, 5))

plt.bar(range(len(region_counts)),
        region_counts.values,
        color=brighter_purple)

plt.title('How much DV data exists per region?')
plt.xlabel('Regions (sorted)')
plt.ylabel('Number of observations')

plt.tight_layout()
plt.savefig('plot3_dv_per_region.png')
plt.show()


# 4: Availability per category (distribution)

category_detailed = []

for cat, indices in livwell_categories.items():
    cat_cols = [cols[i] for i in indices]
    res = check_avg_years_per_country(livwell_all, cat_cols)

    for val in res['avg_years_per_country']:
        category_detailed.append({
            'Category': cat,
            'Years': val
        })

df_cat = pd.DataFrame(category_detailed)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df_cat,
    x='Years',
    y='Category',
    palette=[orange, pink, brighter_purple, darker_purple, blue]
)

plt.title('How much data is available for each category?')
plt.xlabel('Average years per country')
plt.ylabel('')

plt.tight_layout()
plt.savefig('category_availability_boxplot.png')
plt.show()

"""
So in Conclusion:
Within a region and year, all people experience the same heat level.
I DO have many individuals, but that doesn't help...
They don't compare “more heat vs less heat”.
They only help measure DV more precisely at that one heat level. But all
individuals SHARE the heat in their region. So I need regions across many years.
Many individuals is NOT enough.
So, individual-level comparisons don't say much about the effect of heat itself.
To study heat effects, I need variation in heat across time
(the same region in different years).
So the problem is that I don’t have enough repeated,
comparable time changes per region,
so I can’t reliably see whether changes in heat are followed by changes in DV within the same place.

Mathematically, I predict using beta*heat, and heat does NOT vary within a region.
It's constant for everyone.
"""
