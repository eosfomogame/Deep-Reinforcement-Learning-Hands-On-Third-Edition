#!/usr/bin/env python3
import ptan
import pathlib
import argparse
from gymnasium import wrappers
import numpy as np
import time
import os

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from lib import environ, data, models, common, validation

SAVES_DIR = pathlib.Path("saves")
STOCKS = "data/YNDX_160101_161231.csv"
VAL_STOCKS = "data/YNDX_150101_151231.csv"

BATCH_SIZE = 32
BARS_COUNT = 10

EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000
CHECKPOINT_EVERY = 50000  # Salva un checkpoint ogni 50k iterazioni


def save_checkpoint(engine, path, net, tgt_net, selector, buffer, optimizer, eps_tracker):
    """Salva un checkpoint completo dell'addestramento"""
    checkpoint = {
        'iteration': engine.state.iteration,
        'net_state_dict': net.state_dict(),
        'tgt_net_state_dict': tgt_net.target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': selector.epsilon,
        'best_mean_val': getattr(engine.state, "best_mean_val", None),
        'best_val_reward': getattr(engine.state, "best_val_reward", None),
        'buffer': buffer.buffer if hasattr(buffer, 'buffer') else None,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato in {path}")


def load_checkpoint(path, net, tgt_net, selector, buffer, optimizer):
    """Carica un checkpoint salvato"""
    if not os.path.exists(path):
        print(f"Il checkpoint {path} non esiste.")
        return None
    
    checkpoint = torch.load(path)
    
    # Carica i parametri del modello
    net.load_state_dict(checkpoint['net_state_dict'])
    tgt_net.target_model.load_state_dict(checkpoint['tgt_net_state_dict'])
    
    # Carica lo stato dell'ottimizzatore
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Imposta epsilon
    selector.epsilon = checkpoint['epsilon']
    
    # Carica il buffer di replay se disponibile
    if checkpoint['buffer'] is not None and hasattr(buffer, 'buffer'):
        buffer.buffer = checkpoint['buffer']
    
    print(f"Checkpoint caricato da {path}, iterazione {checkpoint['iteration']}")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", help="Training device name", default="cpu")
    parser.add_argument("--data", default=STOCKS, help=f"Stocks file or dir, default={STOCKS}")
    parser.add_argument("--year", type=int, help="Year to train on, overrides --data")
    parser.add_argument("--val", default=VAL_STOCKS, help="Validation data, default=" + VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    parser.add_argument("--checkpoint", help="Path to checkpoint file to riprendere l'addestramento")
    args = parser.parse_args()
    device = torch.device(args.dev)

    saves_path = SAVES_DIR / f"simple-{args.run}"
    saves_path.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(args.data)
    val_path = pathlib.Path(args.val)

    if args.year is not None or data_path.is_file():
        if args.year is not None:
            stock_data = data.load_year_data(args.year)
        else:
            stock_data = {"YNDX": data.load_relative(data_path)}
        env = environ.StocksEnv(
            stock_data, bars_count=BARS_COUNT)
        env_tst = environ.StocksEnv(
            stock_data, bars_count=BARS_COUNT)
    elif data_path.is_dir():
        env = environ.StocksEnv.from_dir(
            data_path, bars_count=BARS_COUNT)
        env_tst = environ.StocksEnv.from_dir(
            data_path, bars_count=BARS_COUNT)
    else:
        raise RuntimeError("No data to train on")

    env = wrappers.TimeLimit(env, max_episode_steps=1000)
    val_data = {"YNDX": data.load_relative(val_path)}
    env_val = environ.StocksEnv(val_data, bars_count=BARS_COUNT)

    net = models.SimpleFFDQN(env.observation_space.shape[0],
                             env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)
    eps_tracker = ptan.actions.EpsilonTracker(
        selector, EPS_START, EPS_FINAL, EPS_STEPS)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Definizione della nostra funzione di processo batch
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss(
            batch, net, tgt_net.target_model,
            gamma=GAMMA ** REWARD_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        eps_tracker.frame(engine.state.iteration)

        if getattr(engine.state, "eval_states", None) is None:
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.asarray(transition.state)
                           for transition in eval_states]
            engine.state.eval_states = np.asarray(eval_states)

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    tb = common.setup_ignite(engine, exp_source, f"simple-{args.run}",
                             extra_metrics=('values_mean',))

    # Se è specificato un checkpoint, caricalo
    start_iteration = 0
    if args.checkpoint:
        checkpoint_path = args.checkpoint if os.path.exists(args.checkpoint) else saves_path / args.checkpoint
        checkpoint = load_checkpoint(checkpoint_path, net, tgt_net, selector, buffer, optimizer)
        if checkpoint:
            start_iteration = checkpoint['iteration']
            engine.state.iteration = start_iteration
            if 'best_mean_val' in checkpoint:
                engine.state.best_mean_val = checkpoint['best_mean_val']
            if 'best_val_reward' in checkpoint:
                engine.state.best_val_reward = checkpoint['best_val_reward']

    @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def sync_eval(engine: Engine):
        tgt_net.sync()

        mean_val = common.calc_values_of_states(
            engine.state.eval_states, net, device=device)
        engine.state.metrics["values_mean"] = mean_val
        if getattr(engine.state, "best_mean_val", None) is None:
            engine.state.best_mean_val = mean_val
        if engine.state.best_mean_val < mean_val:
            print("%d: Best mean value updated %.3f -> %.3f" % (
                engine.state.iteration, engine.state.best_mean_val,
                mean_val))
            path = saves_path / ("mean_value-%.3f.data" % mean_val)
            torch.save(net.state_dict(), path)
            engine.state.best_mean_val = mean_val

    @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def validate(engine: Engine):
        res = validation.validation_run(env_tst, net, device=device)
        print("%d: tst: %s" % (engine.state.iteration, res))
        for key, val in res.items():
            engine.state.metrics[key + "_tst"] = val
        res = validation.validation_run(env_val, net, device=device)
        print("%d: val: %s" % (engine.state.iteration, res))
        for key, val in res.items():
            engine.state.metrics[key + "_val"] = val
        val_reward = res['episode_reward']
        if getattr(engine.state, "best_val_reward", None) is None:
            engine.state.best_val_reward = val_reward
        if engine.state.best_val_reward < val_reward:
            print("Best validation reward updated: %.3f -> %.3f, model saved" % (
                engine.state.best_val_reward, val_reward
            ))
            engine.state.best_val_reward = val_reward
            path = saves_path / ("val_reward-%.3f.data" % val_reward)
            torch.save(net.state_dict(), path)

    # Aggiungiamo un evento periodico per salvare un checkpoint completo
    @engine.on(ptan.ignite.PeriodEvents.ITERATION_COMPLETED)
    def maybe_save_checkpoint(engine: Engine):
        iteration = engine.state.iteration
        if iteration % CHECKPOINT_EVERY == 0:
            checkpoint_path = saves_path / f"checkpoint-{iteration}.pth"
            save_checkpoint(engine, checkpoint_path, net, tgt_net, selector, buffer, optimizer, eps_tracker)

    event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED
    tst_metrics = [m + "_tst" for m in validation.METRICS]
    tst_handler = tb_logger.OutputHandler(
        tag="test", metric_names=tst_metrics)
    tb.attach(engine, log_handler=tst_handler, event_name=event)

    val_metrics = [m + "_val" for m in validation.METRICS]
    val_handler = tb_logger.OutputHandler(
        tag="validation", metric_names=val_metrics)
    tb.attach(engine, log_handler=val_handler, event_name=event)

    # Se stiamo riprendendo da un checkpoint e il buffer non era nel checkpoint
    # assicuriamoci di riempirlo prima di iniziare l'addestramento
    if args.checkpoint and start_iteration > 0 and not buffer.buffer:
        print("Riempimento del buffer di replay...")
        for _ in range(REPLAY_INITIAL):
            buffer.populate(1)

    # Avvio dell'addestramento
    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))
