# Curiosity

## Scope
The scope of the project is to train an agent to learn new things about an environment by using curiosity. After the agent learn how to use the tools available in the enivornment, he will be able to solve specific problems.

## Similar projects
- https://blogs.unity3d.com/2018/06/26/solving-sparse-reward-tasks-with-curiosity/#780000872
- https://pathak22.github.io/noreward-rl/
- https://www.youtube.com/watch?v=9YomzVTv3Ho


## Start training
- create a `venv` environment using Python 3
```bash
./install_libraries.sh
mlagents-learn config-ml-agents/trainer_config.yaml --run-id=run-01 --train
```
- start the `Pyaramids` scene from Unity
- save the generated model (`*.nn` file) in `UnityProject/Assets/ML-Agents/Applications/Pyramids/TFModels`
- then configure the  agent to use the new model
