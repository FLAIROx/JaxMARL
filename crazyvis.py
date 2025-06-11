import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxmarl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=128, help='Batch size')
    parser.add_argument('--timesteps', type=int, default=500, help='Rollout length')
    parser.add_argument('--model_path', type=str, default='actor_model.tflite')
    args = parser.parse_args()

    # 1) Load TFLite model and resize input for batch inference
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    interpreter.resize_tensor_input(inp_det['index'], [args.num_envs, inp_det['shape'][-1]])
    interpreter.allocate_tensors()

    # 2) Create the environment (use jaxmarl.make for correct wrappers)
    env = jaxmarl.make(
      "multiquad_ix4",
      policy_freq=500,
      sim_steps_per_action=1,
      episode_length=args.timesteps,
      reward_coeffs=None,
      obs_noise=0.3,
      act_noise=0.1,
      max_thrust_range=0.3,
      num_quads=2,
      cable_length=0.4,
    )
