"""Export a transition-level dataset and per-episode logs used by the study."""
import numpy as np, pandas as pd, gymnasium as gym
from fql_improved import FuzzyQLearningImproved, train

# Train an improved agent, then roll out to log labelled transitions.
agent = FuzzyQLearningImproved(seed=0)
train(agent, episodes=300, seed=0)

env = gym.make("CartPole-v1")
rows=[]
for ep in range(20):
    s,_=env.reset(seed=1000+ep); done=False; t=0
    while not done:
        a,phi,g=agent.act(s,greedy=True)
        ns,r,term,trunc,_=env.step(a); done=term or trunc
        rows.append({"episode":ep,"t":t,"cart_pos":s[0],"cart_vel":s[1],
                     "pole_angle":s[2],"pole_ang_vel":s[3],"action":a,
                     "reward":r,"next_cart_pos":ns[0],"next_cart_vel":ns[1],
                     "next_pole_angle":ns[2],"next_pole_ang_vel":ns[3],
                     "terminated":int(term)})
        s=ns; t+=1
env.close()
df=pd.DataFrame(rows)
df.to_csv("../Data/cartpole_transitions.csv",index=False)
print("transitions:",len(df),"episodes:",df.episode.nunique(),
      "mean_len:",df.groupby('episode').size().mean().round(1))
