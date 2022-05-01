from pf import ParticleFilter
import pandas as pd

a = -2
b = -1

n_particle = 100
alpha_2 = 1 #10**b
sigma_2 = 0.5



df= pd.read_csv("data/ftx_book_snapshot_25_2020-01-01_BTC-PERP.csv.gz")
df['simple_mid'] = 0.5*(df['bids[0].price']+df['asks[0].price'])
df['ret'] = df['simple_mid'].pct_change()
cleaned_df = df[df.ret!=0].reset_index()


pf=ParticleFilter(cleaned_df.simple_mid.values[1000:1500], n_particle,sigma_2, alpha_2)


pf.simulate(roll_window=100)
pf.draw_graph()