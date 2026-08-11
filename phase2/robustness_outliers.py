# Robustness check: Model 4 excluding flagged outliers
# Lioba Roggendorf

"""
Reruns Model 4 (dv ~ heat + rain + C(region_id) + C(year)) after dropping
observations flagged as outliers by Cook's distance (> 4/n, Hair et al., 1998)
or standardised residuals (|resid| > 3), to check whether the heat effect
is being driven by a handful of influential region-years.
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

# --- load and filter, same as analysis2.py ---

livwell_all = pd.read_csv("clean_livwell.csv")

years_per_region = livwell_all.groupby("region_id")["year"].nunique()
min_years = 3
valid_regions = years_per_region[years_per_region >= min_years].index
livwell_3plus = livwell_all[livwell_all["region_id"].isin(valid_regions)].copy()

# --- main model ---

main_model = smf.ols(
    "dv ~ heat + rain + C(region_id) + C(year)", data=livwell_3plus
).fit(cov_type="cluster", cov_kwds={"groups": livwell_3plus["region_id"]})

print(f"=== MAIN MODEL (n={len(livwell_3plus)}) ===")
print(f"heat: {main_model.params['heat']:.4f}, "
      f"SE: {main_model.bse['heat']:.4f}, "
      f"p: {main_model.pvalues['heat']:.4f}")
print(f"R2: {main_model.rsquared:.4f}\n")

# --- flag outliers ---

influence = OLSInfluence(main_model)
cooks, _ = influence.cooks_distance
studentized = influence.resid_studentized_internal

threshold = 4 / len(livwell_3plus)
cooking = livwell_3plus.index[cooks > threshold].tolist()
farfaraway = livwell_3plus.index[abs(studentized) > 3].tolist()
outliers = sorted(set(cooking + farfaraway))

print(f"Flagged outliers: {len(outliers)} observations "
      f"(Cook's distance > {threshold:.4f}, Hair et al., 1998; "
      f"|standardised residual| > 3)")

# --- rerun without them ---

livwell_no_outliers = livwell_3plus.drop(index=outliers).copy()

robust_model = smf.ols(
    "dv ~ heat + rain + C(region_id) + C(year)", data=livwell_no_outliers
).fit(cov_type="cluster", cov_kwds={"groups": livwell_no_outliers["region_id"]})

ci = robust_model.conf_int().loc["heat"]

print(f"\n=== MODEL EXCLUDING FLAGGED OUTLIERS (n={len(livwell_no_outliers)}) ===")
print(f"heat: {robust_model.params['heat']:.4f}, "
      f"SE: {robust_model.bse['heat']:.4f}, "
      f"p: {robust_model.pvalues['heat']:.4f}")
print(f"R2: {robust_model.rsquared:.4f}")
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# RESULT (for reference):
# Main model (n=464):              heat = 10.31, R2 = 0.836
# Excluding outliers (n=400):      heat = 9.60, SE = 2.18, p < 0.001,
#                                   R2 = 0.937, 95% CI [5.32, 13.87]
# -> Effect holds after excluding the 64 flagged observations; coefficient
#    shrinks slightly but remains highly significant. Consistent with the
#    bootstrap and 2%-trim robustness checks.
