import pandas as pd
import matplotlib.pyplot as plt

GAME = "sonic"
LEVEL = "GreenHillZone/Act2"
WORKER_NUM = 4
ALG = "bestTrajectory_512"


plotDF = pd.DataFrame()
filepath = "./testing/" + GAME + "/" + LEVEL + "/scratch/action_repeat_4/80/" + ALG +"/" + ALG + "_score.csv"
df = pd.read_csv(filepath)
df.fillna(0, inplace=True)
print(df)
