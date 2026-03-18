# Legged Robots

This repository is based on the course [Legged Robots](https://edu.epfl.ch/coursebook/en/legged-robots-MICRO-507). This repository will contain all labs & miniprojects done in the course.

This repository covers from basic topics about legged robots as in kinematics & dynamics, Jacobian, going all the way to complex topics as deep RL.

These labs are mostly in the form of [jupyter notebooks](https://jupyter.org/), with a few helper functions in `.py` files.

To setup the environment: 

```
conda create -n LR python=3.10 numpy scipy matplotlib sympy ffmpeg ipykernel
conda activate LR
pip install -r requirements.txt
```
> [!NOTE]
> `Pybullet` in [`requirements.txt`](requirements.txt) would need you installing [Visual Studio](https://visualstudio.microsoft.com/), along with Desktop C++.

And then in your VS Code window, choose the `LR` environment to run notebook, or if it is a python file, then activate the virtual environment, then run the `.py` script.

## Code Structure

[`env`](env/) for the leg environment files, please see the gym simulation environment [`leg_gym_env.py`](env/leg_gym_env.py), the robot specific functionalities in [`leg.py`](env/leg.py), and config variables in [`configs_leg.py`](env/configs_leg.py). Review [`leg.py`](env/leg.py) carefully for accessing robot states.

## Code Resources

- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation.

## Tips

- If your simulation is very slow, remove the calls to time.sleep() and disable the camera resets in [`leg_gym_env.py`](env/leg_gym_env.py).
- The camera viewer can be modified in `_render_step_helper()` in [`leg_gym_env.py`](env/leg_gym_env.py) to track the hopper.