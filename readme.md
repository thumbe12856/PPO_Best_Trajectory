

## Memory Replay with Trajectory for Side-Scrolling Video Games ##
Based on previous related works,to enhance the curiosity-driven exploration andutilize prior experience more effectively, we develop a new memory replay mechanism, whichconsists of two modules: Trajectory Replay Module (TRM) to record the agent moving trajectory information with much less space, and the Trajectory Optimization Module (TOM) to formulate the state information as reward.

Supported platforms:
- Linux (ubuntu 16.04 desktop)

Supported Pythons:
- 3.5

### 1. Prerequisites 
```
sudo apt-get update
sudo apt-get upgrade
```


### 2. Build virtual env
```
python3 -m venv tom
source tom/bin/activate
cd tom
```
### 3. Build Gym Retro, see its readme to set up
```
git clone --recursive https://github.com/openai/retro.git gym-retro
```

### Steamcmd (optional)
If you don't have the ROM file, you can buy it legally on Steam.
```
sudo apt-get install steamcmd
python retro/scripts/import_sega_classics.py #(you need to enter your steam username and password)
```

### 4. Build Gym Baselines, see its readme to set up
```
git clone https://github.com/openai/baselines.git 
```
### 5. Build Retro-Contest
```
git clone https://github.com/openai/retro-contest
cd retro-contest/setup
./setup.sh
```

### 6. Some packages
```
pip3 install pandas
```
### 7. Training agent
```
python3 agent.py
```
### 8. Testing agent
```
python3 test.py
```
