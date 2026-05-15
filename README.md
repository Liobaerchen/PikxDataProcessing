# Does Extreme Heat Increase Violence Against Women?

This project investigates whether extreme heat events are associated with higher rates of domestic violence (DV), using survey data from 34 low- and middle-income countries. It combines statistical analysis with an interactive climate projection tool.

Research question: Are extreme heat months associated with higher rates of physical violence against women?

---

# Structure

# Step1_Understanding_LivWell
This contains data cleaning and exploration

# Step2_Analysis
This contains my assumption checks and results of statistical analyses for three models
+ some interpretations

# Step3_PredictionTool
This contains an interactive visualisation using bokeh that I made to illustrate
what my analysis might mean.
This also needed some work with a different data format than the previous csvs (.nc),
and some additional calculations. Importantly, this is just a visualisation!
It makes many assumptions that are not viable, and is therefore not scientific
prediction.

# docs (AND PROCESS BOOK!)
This is a website where I visualised and documented all my steps.
To understand this project, I would highly recommend opening it!
I also host it at: https://liobaerchen.github.io/PikxDataProcessing/index.html
It also contains my !!!process book!!!

# literature
This contains the articles I based my analysis
and interpretations on.

---

## The Three Steps

# Step 1 - Understanding the Data (`understanding_livWell.py`)
Loads the LivWell dataset (https://www.pik-potsdam.de/en/output/projects/all/579), selects the key variables (DV prevalence, heat anomalies, rainfall, country, year, region), and cleans it for analysis.
Here, I documents what data is missing and why. If this ever became something like a paper, I'd need
to refer back to this, to talk about the effects of exclusion etc.
This also makes descriptive plots and saves `clean_livwell.csv`, which is used in all later steps (important!).


# Step 2 — Analysis (`analysis.py`)
Runs two models on the cleaned data.

Main Result: A 10 percentage-point increase in extreme-heat months is associated with roughly a 0.8 percentage-point increase in DV prevalence (p ≈ 0.024).

NOTE: I SKIPPED THE EMPOWERMENT ANALYSIS -> With the limited number of years
I had available, I had barely any power as is, and not enough room for things
like moderation analysis. I added complexity with my visualisation, though, and
the data-cleaning and analysis were quite tedious, too.

# Step 3 — Prediction Tool (`prediction_tool_part_2.py`)
An interactive visualisation built with Bokeh. I used IPCC climate projections (the SSP scenarios) and the regression coefficient from Step 2 to project how DV prevalence might change until 2100 under low, medium, and high warming scenarios.

To run it locally:
pip install bokeh pandas numpy
bokeh serve --show prediction_tool_part_2.py

Also! To run the first part of the prediction_tool script, you need files that
were too big to upload to github. If you want to run the analysis: the script
includes links to the data.

render_repo is a folder you can ignore. I have it to run the simulation without
having to run bokeh serve in the terminal. But it just contains copies of
scripts, partly outdated.

# Data Sources

- LivWell — Wellbeing and livelihoods data for women in low- and middle-income countries.

- IPCC AR6 / CMIP6 — Climate projections under SSP1-2.6, SSP3-7.0, and SSP5-8.5 scenarios (`tx35ba` variable: months exceeding 35°C).
- World Bank populations — Global populations from 2024 for estimating numbers of affected women.

---

# Author
Me :)
Lioba Roggendorf, 2026.
