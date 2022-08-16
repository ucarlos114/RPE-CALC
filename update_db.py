import csv
import pandas as pd

def add_to_data(id, time, reps, hip_av, knee_av, bar_speed, rpe):
    file = 'RPE-calc-data.csv'
    df = pd.read_csv(file)

    if (id not in df['Lift-ID'].unique()):  # vid not already catalogued
        df.loc[len(df.index)] = [id, time, reps, hip_av, knee_av, bar_speed, rpe]   # new row

    else:   # update row with same id
        df.loc[df['Lift-ID'] == id] = [id, time, reps, hip_av, knee_av, bar_speed, rpe]
        print(df.loc[df['Lift-ID'] == id])
    df.to_csv(file, index=False)