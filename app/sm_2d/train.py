import datetime

import os
import numpy
import time
import collections

import torch
import torch_ac

import gym
import gym_minigrid

from app.sm_2d.models import ACModel

from app.sm_2d.utils import get_obss_preprocessor, get_model_dir, make_env, get_status, get_status_path

from app.sm_2d.env_registers import *

env_name = 'MiniGridNoLimit-Empty-6x6-v0'
seed = 1
mem = False
text = False

frames = 10**7
frames_per_proc = None
discount = 0.99
lr = 0.001
gae_lambda = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm=0.5
recurrence = 1
optim_eps=1e-08
clip_eps=0.2
epochs=4
batch_size=256
log_interval=1
save_interval = 10
number_of_envs = 300

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{env_name}_PPO_seed{seed}_{date}"

model_name = 'PPO'
model_dir = get_model_dir(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

envs = []
for i in range(number_of_envs):
    envs.append(make_env(env_name, seed + 10000 * i))

env = envs[0]


obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)

try:
    status = get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}


acmodel = ACModel(obs_space, env.action_space, mem, text)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])

acmodel.to(device)


algo = torch_ac.PPOAlgo(
    envs,
    acmodel,
    device,
    frames_per_proc,
    discount,
    lr,
    gae_lambda,
    entropy_coef,
    value_loss_coef,
    max_grad_norm,
    recurrence,
    optim_eps,
    clip_eps,
    epochs,
    batch_size,
    preprocess_obss,
)


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


print("Starting the training..")
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()
while num_frames < frames:
    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    if update % log_interval == 0:
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = synthesize(logs["return_per_episode"])
        rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        print(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} |"
            " H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(*data),
        )

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

    # Save status

    if save_interval > 0 and update % save_interval == 0:
        status = {
            "num_frames": num_frames,
            "update": update,
            "model_state": acmodel.state_dict(),
            "optimizer_state": algo.optimizer.state_dict(),
        }
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        save_status(status, model_dir)
        print("Status saved")

print("Training has ended.")

if __name__ == '__main__':
    pass
