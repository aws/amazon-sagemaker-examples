import platform

from eplus.envs import DataCenterEnv

os_type = platform.system()
if os_type == "Linux":
    eplus_path = "/usr/local/EnergyPlus-8-8-0/"
elif os_type == "nt":  # windows
    eplus_path = "C:\EnergyPlus-8-8-0\\"
else:  # mac
    eplus_path = "/Applications/EnergyPlus-8-8-0/"

config = {
    "eplus_path": eplus_path,
    "weather_file": "weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
}

env = DataCenterEnv(config)
env.reset()
done = False
print("Started simulation, taking first action.")
outputs = []
while not done:
    action = [0.5, 0.5]
    obs, reward, done, info = env.step(action)
    print("reward: %.2f, action: %.2f, %.2f" % (reward, action[0], action[1]))
    print("zone temp: %.2f, drybulb: %.2f, humidity: %.2f" % tuple(obs))
    outputs.append(obs)
print("Completed simulation.")
