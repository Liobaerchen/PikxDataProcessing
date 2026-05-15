# Analysing LivWell
# Lioba Roggendorf
# May 12th, 2026

# NOTE Simon's recommendation: bokeh!

"""
First step: I have a model to predict increase in DV from increase in Heatwaves,
but I have no prediction for increase in Heatwaves! For this, I use the
IPCC-WGI AR6 Interactive Atlas Dataset
by
Instituto de Fisica de Cantabria (IFCA, CSIC-UC, Spain)
Date of data collection:
2022-09-30
Here is the readme: https://digital.csic.es/bitstream/10261/332744/110/README.txt

I will use: the "tx35ba" variable
"Bias adjusted (ISIMIP3 trend preserving method) monthly count of days with
maximum near-surface (usually, 2 meters) temperature above 35 degC"
--> Closest to my heatwave model
"""

import pandas as pd
# if there is an error, might need to:
# pip install xarray netcdf4 h5netcdf pandas matplotlib
import xarray as xr

# also again, use the same colours:
orange = "#F8B195"
pink = "#F67280"
brighter_purple = "#C06C84"
darker_purple = "#6C5B7B"
blue = "#355C7D"
colors = [orange, pink, brighter_purple, darker_purple, blue]

# Load the cleaned dataset from Step 1: understanding LivWell
livwell2 = pd.read_csv("clean_livwell.csv")






# IMPORTANT NOTE!!!!
"""
The .nc files were too large to upload to github. So, this code will not run!
They can be downloaded here: https://ipcc-browser.ipcc-data.org/browser/dataset/6171
"""



# Load the .nc prediction data from IPCC
historical = xr.open_dataset("IPCC-WGI_Dataset/tx35ba_CMIP6_historical_mon_195001-201412.nc")

# understanding ssp:
"""
'the AR6
WGI report to fill certain gaps identified in the RCPs. The first
number in the label is the particular set of socioeconomic
assumptions driving the emissions and other climate forcing
inputs taken up by climate models and the second number is the
radiative forcing level reached in 2100.'
https://www.ipcc.ch/report/ar6/wg2/downloads/outreach/IPCC_AR6_WGII_IntroductionWGII.pdf

and radiative forcing can be converted to global warming like this:
"commonly accepted value of climate sensitivity parameter λ is 0.8"
https://en.wikipedia.org/wiki/Radiative_forcing#Basic_estimates

"Radiative forcing and climate feedbacks can be used together to estimate a subsequent change in steady-state (often denoted ‘equilibrium’) surface temperature (ΔTs) via the equation:
ΔTs = λ̃ ΔF
where λ̃ commonly denotes the climate sensitivity parameter, usually with units K/(W/m²),
and ΔF is the radiative forcing in W/m²."
https://en.wikipedia.org/wiki/Radiative_forcing#Basic_estimates

So lambda means:
"How many degrees of warming you get per unit of radiative forcing?"
Which is what I need!
"""

ssp370 = xr.open_dataset("IPCC-WGI_Dataset/tx35ba_CMIP6_ssp370_mon_201501-210012.nc")
ssp585 = xr.open_dataset("IPCC-WGI_Dataset/tx35ba_CMIP6_ssp585_mon_201501-210012.nc")
ssp126 = xr.open_dataset("IPCC-WGI_Dataset/tx35ba_CMIP6_ssp126_mon_201501-210012.nc")


# Inspecting the data
# print(ssp370)
# print(list(ssp370.data_vars))
# print(ssp370["tx35ba"].attrs)

# my datasets are: ssp370, ssp585, ssp126
# the first number is "the particular set of socioeconomic
# assumptions driving the emissions and other climate forcing
# inputs taken up by climate models" -- not relevant!
# but the second number is:
"""
"The second number is the radiative forcing level reached in 2100"
https://www.ipcc.ch/report/ar6/wg2/downloads/outreach/IPCC_AR6_WGII_IntroductionWGII.pdf
"""

# I can convert this to degrees celcius of warming based on my research!
def forcing_to_warming(forcing, lambdaa=0.8):
    """
    Convert radiative forcing INCREASE to global mean warming INCREASE (°C).

    Parameters:
    - forcing: the global forcing; how much it increased relative to pre-industrial
    - lambdaa: climate sensitivity parameter, common default is = 0.8

    Returns:
    - the estimated GLOBAL mean temperature change in °C - relative to pre-industrial
    """
    return lambdaa * forcing

radiative_forcing_2100 = [2.6, 7.0, 8.5]
warmingsby2100 = []
for forcing in radiative_forcing_2100:
    warmingsby2100.append(forcing_to_warming(forcing))

print(f"global warming in degrees that the IPCC scenarios are based on:\n{warmingsby2100}")

# for interpretation, radioactive forcing means:
# How much extra (or less) energy is the Earth system receiving compared to a reference state?”
# and the reference state for IPCC is: pre-industrial climate (≈1750)
# so after conversion, my variables are:
# ESTIMATE of how global mean temperature would respond to different IPCC
# radiative forcing scenarios,
# giving a rough sense of low-, medium-, and high-emission worlds in terms of warming magnitude.
# This is only a simplified illustrative conversion of IPCC scenario inputs.
# MENTION THIS!

# [2.1, 5.6, 6.8]
# So, rename:
data_2_1 = ssp126["tx35ba"]
data_5_6 = ssp370["tx35ba"]
data_6_8 = ssp585["tx35ba"]

# NOTE: spatial structure is still intact here (lat/lon preserved)

def make_table(data, label):

    # 1. binary heatwave occurrence per grid cell
    # this converts heat to "at least one hot day"
    data = (data > 0)

    # 2. fraction of Earth affected (average across earth)
    data = data.mean(dim=["lat", "lon"])

    # 3. average across model members (these are different simulations
    # with the same presets, I think)
    data = data.mean(dim="member")

    # 4. turn from months to years
    data = data.resample(time="1YE").mean()

    df = data.to_dataframe().reset_index()
    df = df.rename(columns={"tx35ba": "share_earth_in_heatwave"})
    df["scenario"] = label
    df["year"] = df["time"].dt.year

    return df


df_2_1 = make_table(data_2_1, "2.1")
df_5_6 = make_table(data_5_6, "5.6")
df_6_8 = make_table(data_6_8, "6.8")

# finally, save them:
df_2_1.to_csv("heatwaves_2_1.csv", index=False)
df_5_6.to_csv("heatwaves_5_6.csv", index=False)
df_6_8.to_csv("heatwaves_6_8.csv", index=False)

# this now means:
"""
In this year, on average, X% of the Earth’s grid cells experienced at least some heatwave conditions each month.
"fraction of grid cells with at least one heatwave day in that month"
"""
# NOTE: merge these into one file later, for readability and nice data structure
