
import numpy as np
from scipy import stats
import parse_data
from matplotlib import pyplot as plt
import pandas as pd

df = parse_data.parse()


# Hypothesis test1: Do people rent more bikes on holidays than on non-holidays?
# 2 sample t-test
def analysis1():
    # data subsets:
    holiday = df[df["Holiday"] == 1]["Bike Count"]
    nonholiday = df[df["Holiday"] == 0]["Bike Count"]

    t1, p1 = stats.ttest_ind(holiday, nonholiday)

    #plot
    mean_non = nonholiday.mean()
    mean_hol = holiday.mean()
    plt.figure(figsize=(7,5))
    plt.scatter([0] * len(nonholiday), nonholiday, alpha=0.6, s=40)
    plt.scatter([1] * len(holiday), holiday, alpha=0.6, s=40)
    mean_non = nonholiday.mean()
    mean_hol = holiday.mean()
    plt.plot([0, 1],[mean_non, mean_hol],'-o',color='tab:red',linewidth=2,markersize=8,label="Group Means")
    plt.xticks([0, 1], ["Non-Holiday", "Holiday"])
    plt.ylabel("Bike Rentals")
    plt.title("Holiday vs Non-Holiday: Scatterplot with Mean Connection Line")
    plt.grid(axis='y', alpha=0.25)
    plt.legend()

    plt.show()
    return t1, p1

# Hypothesis test2: Do temperature effects change by time of day?
def analysis2():

    #time of day groups
    df["TimeGroup"] = pd.cut(
    df["Hour"],
    bins=[-1, 5, 10, 16, 23],
    labels=["LateNight", "Morning", "Day", "Evening"])
    
    colours = {"LateNight": "gray", "Morning": "yellow", "Day": "green", "Evening": "blue"}

    plt.figure(figsize=(9,6))
    for group in df["TimeGroup"].unique():
        sub = df[df["TimeGroup"] == group]

        # Scatter
        plt.scatter(
            sub["Temperature"],
            sub["Bike Count"],
            color=colours[group],
            alpha=0.25,
            label=group
        )

    plt.title("Temperature vs Bike Rentals (Grouped by Time of Day)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Bike Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


    # Focus on warm hours = top 30% of temperature
    temp_thresh = df["Temperature"].quantile(0.7)
    warm_df = df[df["Temperature"] >= temp_thresh]

    # Linear regression
    resultslr = {}
    for group in warm_df["TimeGroup"].unique():
        sub = warm_df[warm_df["TimeGroup"] == group]
        rlr = stats.linregress(sub["Temperature"], sub["Bike Count"])
        resultslr[group] = (rlr.pvalue, rlr.slope)
    
    # ANOVA
    groupe1 = warm_df[warm_df["TimeGroup"] == "LateNight"]["Bike Count"]
    groupe2 = warm_df[warm_df["TimeGroup"] == "Morning"]["Bike Count"]
    groupe3 = warm_df[warm_df["TimeGroup"] == "Day"]["Bike Count"]
    groupe4 = warm_df[warm_df["TimeGroup"] == "Evening"]["Bike Count"]
    resultsANOVA = stats.f_oneway(groupe1, groupe2, groupe3, groupe4)

    #plot
    colours = {"LateNight": "gray", "Morning": "yellow", "Day": "green", "Evening": "blue"}
    for group in warm_df["TimeGroup"].unique():
        sub = warm_df[warm_df["TimeGroup"] == group]
        lr = stats.linregress(sub["Temperature"], sub["Bike Count"])
        plt.scatter(
            sub["Temperature"],
            sub["Bike Count"],
            color=colours[group],
            alpha=0.25,
            label=f"{group}"
        )
        xs = np.linspace(sub["Temperature"].min(), sub["Temperature"].max(), 100)
        ys = lr.slope * xs + lr.intercept
        plt.plot(xs, ys, color=colours[group], linewidth=2)

    plt.title("Temperature vs Bike Rentals on Warm Hours (by Time of Day)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Bike Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return resultslr, resultsANOVA