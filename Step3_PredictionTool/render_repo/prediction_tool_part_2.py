# Analysing LivWell
# Lioba Roggendorf
# May 12th, 2026

# NOTE Simon's recommendation: bokeh!

"""
This is part two of my visualisation of my predictions.
Converting the IPCC data to what I needed takes a long time, because it is huge.
So, I saved them to csvs and load them again here, for the visualisation:
"""

import pandas as pd
import numpy as np
from bokeh.plotting import figure, curdoc
# might need pip install bokeh!
import pandas as pd
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Div, Slider, LinearColorMapper, WMTSTileSource, LabelSet
from bokeh.layouts import column, row
from bokeh.palettes import RdYlGn

df_2_1 = pd.read_csv("heatwaves_2_1.csv")
df_5_6 = pd.read_csv("heatwaves_5_6.csv")
df_6_8 = pd.read_csv("heatwaves_6_8.csv")

print(df_2_1.head())


# okay, great!
# now, for the visualisation, I will treat the global increase in
# heatwave-affected global earth-area as a an approximation to the increase in
# extreme-heat months experienced locally.
# NOTE: this is SUPER vague! It was coded in months above 35 degrees, not 2 sds,
# and obviously, these two measures are not the same thing. BUT it is for
# illustrative purposes, not for perfect scientific modelling

# Step 1: remember the model output! My effect size was:
BETA = 8.24 # the β coefficient

# I have data from 2015, so the model will have to start at 2015

# Step 2: Get average DV per country from livwell
"""
Loads LivWell CSV and returns a dictionary:
country_code -> average DV
"""
# robust read (handles comma or tab separation)
df = pd.read_csv("clean_livwell.csv")

# compute mean DV per country
country_means = df.groupby("country")["dv"].mean()

# convert to dict
country_dv = country_means.to_dict()

print(country_dv)


# approximate female populations
# get them by just dividing total population by two (percentage of females is
# between like 49 and 51 percent everywhere)
# Source: https://data.worldbank.org/indicator/SP.POP.TOTL

# World Bank file needs skipping first rows until header (got an error before)
df = pd.read_csv("worldbank_population.csv", skiprows=4)

# keep only needed columns
df = df[["Country Code", "2024"]].copy()

# turn to int or float
df["2024"] = pd.to_numeric(df["2024"], errors="coerce")
# get rid of na
df = df.dropna(subset=["2024"])

# make the population dictionary
country_pop = dict(zip(df["Country Code"], df["2024"]))

# keep only the countries that are in dv:
common_countries = set(country_dv.keys()) & set(country_pop.keys())
pop_df = {}
for c in common_countries:
    pop_df[c] = country_pop[c]

print(pop_df)

# __________________________________________________

# PLAN:
#     bokeh.sampledata.world_countries has country borders built in — no shapefile needed
# Precompute all predicted_dvections once upfront (nested dict: scenario → year → country → DV), so callbacks are instant
# Map scenario: fixed to medium (5.6°C) — simplest, add selector later if needed
# Two callbacks only: country_select fillPlotsWithDatas line chart + text; year_slider fillPlotsWithDatas map
# Layout: country selector on top, line chart and map side by side, text div below

# Before all, map the country abbreviations to actual names for readability
# NOTE: This Dictionary is AI generated, I was too lazy to type all of this out
abbrToCountry = {
    'ARM': 'Armenia', 'BFA': 'Burkina Faso', 'CIV': 'Côte d’Ivoire', 'CMR': 'Cameroon',
    'COD': 'DR Congo', 'COL': 'Colombia', 'ETH': 'Ethiopia', 'GTM': 'Guatemala',
    'HND': 'Honduras', 'HTI': 'Haiti', 'IND': 'India', 'JOR': 'Jordan',
    'KEN': 'Kenya', 'KHM': 'Cambodia', 'LBR': 'Liberia', 'MLI': 'Mali',
    'MOZ': 'Mozambique', 'MWI': 'Malawi', 'NAM': 'Namibia', 'NGA': 'Nigeria',
    'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'RWA': 'Rwanda',
    'SEN': 'Senegal', 'SLE': 'Sierra Leone', 'TGO': 'Togo', 'TJK': 'Tajikistan',
    'TLS': 'Timor-Leste', 'TZA': 'Tanzania', 'UGA': 'Uganda', 'ZAF': 'South Africa',
    'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
}
# also make the reverse dictionary for later use:
countryToAbbr = {}
for abbr, coun in abbrToCountry.items():
    countryToAbbr[coun] = abbr

# ###### 2. World map # ###### # ###### # ######
# I have only country names, but bokeh needs longitudes and latitudes!
# I need to convert these somehow!
# very closely modelled after this Stackoverflow discussion:
#   https://stackoverflow.com/questions/57178783/how-to-plot-latitude-and-longitude-in-bokeh
"""
Note: This code is modelled after online tutorials and stackoverflow discussions.
I do not fully understand it myself.
What it does, though, is:

takes: lat = latitude (north/south position on Earth)
lon = longitude (east/west position)
Outputs x, y coordinates on a flat map predicted_dvection
"""
def to_merc(lat, lon):
    r = 6378137.0 # r is the radius of the Earth in meters
    x = r * np.radians(lon)
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat*(np.pi/180.0)/2.0)) * (x/lon)
    return x, y

# The way Bokeh makes map coordinates is with something called
# EPSG:3857 (Web Mercator)
    # Defined as:
    # X axis: predicted_dvected longitude
    # Y axis: predicted_dvected latitude (nonlinear)
    # Units:
    # meters on a hypothetical cylinder wrapped around the Earth
# Which I don't really understand. This function converst lats and longs into these units

# These are based on: https://gist.github.com/metal3d/5b925077e66194551df949de64e910f6
centers = {
    "ARM": [40.0, 45.0],
    "BFA": [13.0, -2.0],
    "CIV": [8.0, -5.0],
    "CMR": [6.0, 12.0],
    "COD": [0.0, 25.0],
    "COL": [4.0, -72.0],
    "ETH": [8.0, 38.0],
    "GTM": [15.5, -90.25],
    "HND": [15.0, -86.5],
    "HTI": [19.0, -72.4167],
    "IND": [20.0, 77.0],
    "JOR": [31.0, 36.0],
    "KEN": [1.0, 38.0],
    "KHM": [13.0, 105.0],
    "LBR": [6.5, -9.5],
    "MLI": [17.0, -4.0],
    "MOZ": [-18.25, 35.0],
    "MWI": [-13.5, 34.0],
    "NAM": [-22.0, 17.0],
    "NGA": [10.0, 8.0],
    "PAK": [30.0, 70.0],
    "PER": [-10.0, -76.0],
    "PHL": [13.0, 122.0],
    "RWA": [-2.0, 30.0],
    "SEN": [14.0, -14.0],
    "SLE": [8.5, -11.5],
    "TGO": [8.0, 1.1667],
    "TJK": [39.0, 71.0],
    "TLS": [-8.55, 125.5167],
    "TZA": [-6.0, 35.0],
    "UGA": [1.0, 32.0],
    "ZAF": [-29.0, 24.0],
    "ZMB": [-15.0, 30.0],
    "ZWE": [-20.0, 30.0]
}


# Also: Initial heat for plotting purposes, based on this:
# https://en.wikipedia.org/wiki/List_of_countries_by_average_yearly_temperature
country_temp = {
    "BFA": 30.40,  # Burkina Faso
    "SEN": 28.90,  # Senegal
    "NGA": 27.30,  # Nigeria
    "TGO": 27.33,  # Togo
    "SLE": 26.54,  # Sierra Leone
    "LBR": 25.45,  # Liberia
    "GHA": 27.66,  # Ghana (NOT in your centers, so excluded)
    "CIV": 26.80,  # Ivory Coast / Côte d’Ivoire
    "MLI": 29.21,  # Mali
    "NER": 28.04,  # Niger (NOT in your centers, so excluded)
    "CMR": 24.80,  # Cameroon
    "COD": 24.35,  # Democratic Republic of the Congo
    "RWA": 20.03,  # Rwanda
    "UGA": 23.25,  # Uganda
    "TZA": 22.92,  # Tanzania
    "KEN": 25.08,  # Kenya
    "ETH": 23.36,  # Ethiopia
    "ZMB": 22.23,  # Zambia
    "ZWE": 21.90,  # Zimbabwe
    "MOZ": 24.41,  # Mozambique
    "MWI": 22.66,  # Malawi
    "NAM": 20.45,  # Namibia
    "ZAF": 18.23,  # South Africa
    "IND": 24.94,  # India
    "PAK": 21.38,  # Pakistan
    "BGD": 25.71,  # Bangladesh (NOT in your centers, so excluded)
    "KHM": 27.41,  # Cambodia
    "PHL": 27.10,  # Philippines
    "LKA": 27.25,  # Sri Lanka (NOT in your centers, so excluded)
    "IDN": 25.96,  # Indonesia (NOT in your centers, so excluded)
    "PER": 20.07,  # Peru
    "COL": 25.00,  # Colombia
    "HTI": 24.95,  # Haiti
    "HND": 24.72,  # Honduras
    "GTM": 23.65,  # Guatemala
    "NGA": 27.30,  # Nigeria (already included; duplicate kept clean logically)
    "PAK": 21.38,  # Pakistan (duplicate resolved above logically)
    "IND": 24.94   # India (duplicate resolved above logically)
}

# and normalise it for later plotting:
temp_values = np.array(list(country_temp.values()))
t_min = temp_values.min()
t_max = temp_values.max()

country_temp_norm = {
    c: (t - t_min) / (t_max - t_min)
    for c, t in country_temp.items()
}

# Now I make lists that have data for bokeh
map_x = [] # x-coordinates (Mercator)
map_y = [] # y-coordinates (Mercator)
map_codes = [] # ISO country codes (like "NGA")
map_names = []

# loop over the geographic data I have for my countries
for c, coords in centers.items():
    # and for every country I have in my violence dict
    if c in country_dv:
        # use the function that turns these coordinates into the bokeh format
        mx, my = to_merc(coords[0], coords[1])
        # append
        map_x.append(mx)
        map_y.append(my)
        map_codes.append(c)
        map_names.append(abbrToCountry.get(c, c))
        # now the lists have the x,y coords of the country, + the code and the name,
        # all with the same index

BETA = 8.24 # FROM MY REGRESSION! IMPORTANT!
# these are the years that are covered in the IPCC scenarios
years = df_2_1.year.tolist()
# Store them in a dict for easier use -> so I remember what each one is
scens = {"Low": df_2_1, "Med": df_5_6, "High": df_6_8}

predicted_dv = {}
# loop over each scenario in the dict
for scenario_name, scenario_df in scens.items():
    predicted_dv[scenario_name] = {}
    # get the first percentage of heat-wave countries (2015 I think)
    baseline = scenario_df.iloc[0].share_earth_in_heatwave
    # go over that scenario and get the different years
    for _, r in scenario_df.iterrows():
        # store the year
        year = r.year
        # get how many people are experiencing heatwwaves right now (extract from the csv I made)
        current_heatwave = r.share_earth_in_heatwave
        # now, look at the expected increase in heat and save the expected increase in DV in this dict
        predicted_dv[scenario_name][year] = {}
        for country in country_dv:
        # the formula works like this:
        # for each year, for every country, take the dv level, check how different
        # it is from the baseline, and then multiply it by beta.
        # Because beta shows how much betwa will increase from baseline dv with
        # x change in dv.
            dv_prediction = (
                country_dv[country]
                + (current_heatwave - baseline) * BETA
            )

            # this dict will store: for each IPSS scenario, year, and country, what is the
            # expected dv level?
            # E.g. "Nigeria’s projected DV in 2050 under the High scenario is 13.2."
            predicted_dv[scenario_name][year][country] = dv_prediction

# ##### 3. Making Dataframe-like-things for bokeh
# Tutorial on the column data source thingy:
# https://docs.bokeh.org/en/2.3.3/docs/user_guide/data.html
mapWithCoordinatesEtc = ColumnDataSource(data={'x': map_x, 'y': map_y, 'code': map_codes, 'names': map_names, 'color_val': [0]*len(map_codes), 'size': [20]*len(map_codes)})
# added this bc I want a red circle for the selected country and this will allow that
selectedCountry = ColumnDataSource(data={'x': [], 'y': [], 'size': []})
# this is for the line chart
line_src = ColumnDataSource(data={'year': years,
    # this just makes an empty chart that is then later filled with data when the user clicks a country
    'lowWarming': [0]*len(years),
    'mediumWarming': [0]*len(years),
    'highWarming': [0]*len(years)})
# this makes a list full of zeros that has the same length as the years list.

# MAP
# basically a copy of this: https://docs.bokeh.org/en/latest/docs/user_guide/topics/geo.html
p_map = figure(x_axis_type="mercator", x_range=(-12000000, 10000000), y_range=(-5000000, 7000000),
               title="DV RISK MAP", tools="tap") # tools tap lets me make the map interactive; see later
# this somehow automatically loads an image where I can input x,y, and z coordinates
p_map.add_tile(WMTSTileSource(url="https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"))
p_map.sizing_mode = "stretch_both" # make it fill the element

# this maps my values to colours for the dv
# red, yellow, green.
# adapted from here: https://docs.bokeh.org/en/latest/docs/reference/models/mappers.html
# 11 is the stepsize
mapper = LinearColorMapper(palette=RdYlGn[11], low=0.2, high=2)
# draw my countries as bubbles
# the fill colour will be determined by 'colorval', which is later going to be
# determined using the heat
# transform is really cool; you can dynamically adjust your plot
# by computing stuff on your computer: https://docs.bokeh.org/en/latest/docs/reference/transform.html
p_map.scatter('x', 'y', size='size', source=mapWithCoordinatesEtc, fill_color={'field': 'color_val', 'transform': mapper}, fill_alpha=0.8, line_color="black")
# this just draws a red outline around the selected country
p_map.scatter('x', 'y', size='size', source=selectedCountry, fill_alpha=0, line_color="#FF0000", line_width=5)

# add the country names to make the plot more understandable
# by choosing 'names' from 'mapWithCoordinatesEtc', I tell it to write the country names
# rest is adapter from here: https://docs.bokeh.org/en/3.2.0/docs/examples/basic/annotations/label.html
labels = LabelSet(x='x', y='y', text='names', level='glyph', x_offset=5, y_offset=5, source=mapWithCoordinatesEtc,
                  text_font_size="10pt", text_font_style="bold")
p_map.add_layout(labels)

# making the line chart
p_line = figure(width=400, title="Predicted DV prevalence in percent of women", sizing_mode="stretch_height")
p_line.line('year', 'lowWarming', source=line_src, color="#355C7D", line_width=3, legend_label="Low")
p_line.line('year', 'mediumWarming', source=line_src, color="#F8B195", line_width=3, legend_label="Med")
p_line.line('year', 'highWarming', source=line_src, color="#F67280", line_width=3, legend_label="High")
p_line.legend.location = "top_left"
# fixed x and y axes for country comparability
p_line.y_range.start = 5
p_line.y_range.end = 40
p_line.x_range.start = 2015
p_line.x_range.end = 2100


# clickable stuff
# this is the dropdown menu, with the values from my country dict
# was too small, so made it bigger
sel = Select(title="CHOOSE COUNTRY", value="Nigeria", options=sorted(abbrToCountry.values()),
             sizing_mode="stretch_width", styles={'font-size': '20px', 'font-weight': 'bold'})
# this is the year slider, with the years available in the IPCC simulation
sld = Slider(start=2015, end=2100, value=2015, title="pick a year")
# This is the textbox where I will later show the percentage of women affected
txt = Div(sizing_mode="stretch_width", styles={'font-size': '20px', 'padding': '15px', 'background': '#f9f9f9', 'border-left': '8px solid #d9534f'})

# NOW THE VERY HARD PART:
# fillPlotsWithData MY VISUALS BASED ON MY DATA

# NOTE: THE LINES ARE NOT RE-CALCULATED FOR EACH COUNTRY....

def fillPlotsWithData():
    country_name = sel.value # get the current country name (from sel,
    # which is the dropdown where you can choose the country)
    country_abb = countryToAbbr[country_name] # convert to abbreviation to access the data
    y = sld.value # update the year based on the slider where the users can select a year

    # 1. fill Line Chart
    low_vals = []
    med_vals = []
    high_vals = []

    # this accesses the predicted dv values from the computations
    # i made earlier, for each scenario, year, and country
    # and then get a list for those
    for yr in years:
        low_vals.append(predicted_dv["Low"][yr][country_abb])
        med_vals.append(predicted_dv["Med"][yr][country_abb])
        high_vals.append(predicted_dv["High"][yr][country_abb])

    line_src.data = {
        "year": years,
        "lowWarming": low_vals,
        "mediumWarming": med_vals,
        "highWarming": high_vals
    }

    # 2. fill Map Visuals
    new_colors = []
    new_sizes = []
    # this will be used to make the countries redder as time passes -> to visualise heating
    # (so the larger the difference to 2015, the more added heating)
    heating_visual = (y - 2015) / (2100 - 2015) #2100 is my max year
    # these store the coordinates of the countries I have selected right now
    sel_x, sel_y, sel_size = 0, 0, 0

    # this loops over all countries on the map
    # so I get the index and the abbreviation
    for i, code in enumerate(map_codes):
        # COLOR based on dv:
        # then I go to that country and get the heatwave share for that year from your scenario dataframe
        # from my csv
        base_temp = country_temp_norm.get(code, 0.5) #0.5 is just gonna be the safety middle filler value
        # then I raise it to a power to amplify the differences
        gamma = 1.6
        color_value = base_temp ** gamma
        color_value += heating_visual
        new_colors.append(color_value)

        # but BUBBLE size is based on heat!
        # I get the acutal dv for that country
        initial_dv = country_dv[code]
        # get the prediction (based on the HIGH scenario)
        # so the map is based on the HIGH scenario. DRAMA
        predicted_dv_val = predicted_dv["High"][y].get(code, initial_dv)
        # Calculate size based strictly on the DV change
        # the multiplications adjust the size change so it becomes more visible
        bubble_size = (initial_dv * 1.2) + ((predicted_dv_val - initial_dv) * 85)
        new_sizes.append(bubble_size)
        # now update the coordinates and code for the country!
        # this saves the selected country info for later use (drawing the outline, calculating things, etc.)
        if code == country_abb:
            sel_x, sel_y, sel_size = map_x[i], map_y[i], bubble_size

    # Now I change the map!
    # I changed the colours to new colours, sizes to new sizes, and now I update
    # and then the library uses this dataset instead for the drawing! I think...
    mapWithCoordinatesEtc.data = {
    'x': map_x,
    'y': map_y,
    'code': map_codes,
    'names': map_names,
    'color_val': new_colors,
    'size': new_sizes
    }
    selectedCountry.data = {'x': [sel_x], 'y': [sel_y], 'size': [sel_size + 4]}

    # 3. How many women experience DV?
    # this variable gets the predicted dv for the present year and country
    # and then gets the difference between that and the starting value

    diff_in_dv = predicted_dv["High"][y][country_abb] - country_dv[country_abb]
    diff_in_dv_low = predicted_dv["Low"][y][country_abb] - country_dv[country_abb]

    # then it gets the total population
    total_pop = country_pop.get(country_abb, 0)

    # and divides it by 2, to get only the female population
    female_pop = total_pop / 2

    # then I multiply the predicted difference in dv by the total female population
    # (btw, this assumes population stays constant, so real numbers are probbaly worse...
    # populations grow)
    additional_women_country = int((diff_in_dv / 100) * female_pop)
    additional_women_country_low = int((diff_in_dv_low / 100) * female_pop)

    # total affected women in that country (NOT just increase)
    total_women_country = int((predicted_dv["High"][y][country_abb] / 100) * female_pop)
    total_women_country_low = int((predicted_dv["Low"][y][country_abb] / 100) * female_pop)

    # also get the difference, for impact etc.
    difference_total_women = total_women_country - total_women_country_low

    txt.text = (
        f"<b>{country_name} Year: ({y}):</b><br>"
        f"Increase in DV compared to baseline (High): +{diff_in_dv:.2f} percentage points.<br>"
        f"Increase in DV compared to baseline (Low): +{diff_in_dv_low:.2f} percentage points.<br><br>"

        f"Total affected (High): <b>{total_women_country:,}</b> women.<br>"
        f"Total affected (Low): <b>{total_women_country_low:,}</b> women.<br><br>"

        f"Additional affected (High): <b>{additional_women_country:,}</b> women.<br>"
        f"Additional affected (Low): <b>{additional_women_country_low:,}</b> women.<br>"

        f"So, between the low and the high temperature increase scenario, <b>~{difference_total_women}</b> more women "
        f"experience domestic violence in {y} in {country_name} than in 2015."
    )


# This defines what happens when a user clicks a country
# from here:
# https://discourse.bokeh.org/t/live-plot-with-interactivity/10890/6
# and here: https://docs.bokeh.org/en/3.0.0/docs/user_guide/interaction/python_callbacks.html
# this confused me, but I think it just updates 'sel', the selected country,
# to the name of that country when tapped
def on_tap(attr, old, new):
    if new:
        index = new[0]
        sel.value = mapWithCoordinatesEtc.data['names'][index]

mapWithCoordinatesEtc.selected.on_change('indices', on_tap)
# this changes the selected country --> from dropdown selector
sel.on_change('value', lambda a,o,n: fillPlotsWithData())
# this the selected year --> from slider (sld)
sld.on_change('value', lambda a,o,n: fillPlotsWithData())

# this calls the function and does the whole calculating updating etc etc.
fillPlotsWithData()

# pretty layout
# basically copied from here: https://docs.bokeh.org/en/latest/docs/reference/layouts.html
layout = column(sel, row(p_line, p_map, sizing_mode="stretch_both"), sld, txt, sizing_mode="stretch_both")
curdoc().add_root(layout)
curdoc().title = "Climate Risk Dashboard"
