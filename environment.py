import numpy as np
import gym
from gym import spaces
from gym.spaces import Box, Discrete
from credit_settlement import CreditSettlementEngine
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import myParams
import random
from scipy.interpolate import splprep, splev
from collections import deque
import multiprocessing
from typing import Optional, Dict, Any
import llm_credit_selector as llm_credit_selector_py37_patched
import time

class ETCSelectorWrapper(object):


    def __init__(self, llm_selector, llm_ratio=0.4, min_interval_sec=0.5, debug=False):
        self.llm = llm_selector            
        self.llm_ratio = float(llm_ratio)
        self.min_interval_sec = float(min_interval_sec)
        self.debug = bool(debug)

        self._last_llm_t = 0.0
        self._sticky_mode = None          

    def _strong_gate(self, ctx, env):

        
        overlap_avg = float(getattr(ctx, "overlap_avg", 0.0) or 0.0)
        overlap_peak = float(getattr(ctx, "overlap_peak", 0.0) or 0.0)
        wait_steps = int(getattr(ctx, "wait_steps", 0) or 0)

       
        remaining_ratio = 1.0
        try:
            num_tasks = int(getattr(env, "num_tasks", 0) or 0)
            if num_tasks > 0 and getattr(env, "completed_tasks", None) is not None:
                done = sum(1 for x in env.completed_tasks if x)
                remaining_ratio = float(max(num_tasks - done, 0)) / float(num_tasks)
        except Exception:
            remaining_ratio = 1.0

      
        if overlap_peak >= 3.0 or overlap_avg >= 1.2:
            return "crowd_only_penalty_v1"

       
        if wait_steps >= 300 and overlap_avg <= 0.3:
            return "wait_only_penalty_v1"

        
        if remaining_ratio <= 0.2:
           
            if hasattr(env, "credit_engine") and "late_stage_completion_bonus_v1" in getattr(env.credit_engine, "_rules", {}):
                return "late_stage_completion_bonus_v1"

        return None

    def __call__(self, ctx, env):
       
        gate_mode = self._strong_gate(ctx, env)
        if gate_mode is not None:
            self._sticky_mode = gate_mode
            return gate_mode

       
        ep = int(getattr(env, "episode_idx", 0) or 0)
        max_ep = int(getattr(env, "max_episodes", 0) or 0)
        use_llm = False
        if max_ep > 0:
            use_llm = (ep < int(self.llm_ratio * max_ep))
        else:
           
            use_llm = False

        if not use_llm:
            return self._sticky_mode or str(getattr(env, "credit_mode", "crowd_wait_penalty_v1"))

        now = time.time()
        if now - self._last_llm_t < self.min_interval_sec:
            return self._sticky_mode or str(getattr(env, "credit_mode", "crowd_wait_penalty_v1"))

        try:
            mode = str(self.llm(ctx, env))
        except Exception:
            mode = self._sticky_mode or str(getattr(env, "credit_mode", "crowd_wait_penalty_v1"))
        finally:
            self._last_llm_t = time.time()


        self._sticky_mode = mode
        return mode


class Trajectory:
    def __init__(self, num_agents, sampling_factor=10):
        self.num_agents = num_agents
        self.sampling_factor = sampling_factor  
        self.trajectories = {i: [] for i in range(num_agents)}
        self.milestones = {i: [] for i in range(num_agents)}
        self.smoothed_trajectories = {i: [] for i in range(num_agents)}
        self.deque_maxlen = 100
        self.cur_queue = [deque(maxlen=self.deque_maxlen) for _ in range(num_agents)]
        self.cur_store = [[] for _ in range(num_agents)]
        

    def add(self, agent_id, position):
        if agent_id in self.trajectories and all(p >= 0 for p in position):
            if not self.trajectories[agent_id] or self.trajectories[agent_id][-1] != position:
                self.trajectories[agent_id].append(position)

    def milestone(self, agent_id):
        if agent_id in self.trajectories and self.trajectories[agent_id]:
            last_milestone = self.milestones[agent_id][-1] if self.milestones[agent_id] else 0
            current_milestone = len(self.trajectories[agent_id]) - 1
            self.milestones[agent_id].append(current_milestone)
            
            if False:
                self._smooth_and_store_curvature(agent_id, last_milestone, current_milestone)

    def render(self, *agent_ids):
        plt.figure(figsize=(10, 6))
        for agent_id in agent_ids:
            if agent_id in self.trajectories and self.trajectories[agent_id]:
                trajectory = self.trajectories[agent_id]
                if trajectory:
                    x, y = zip(*trajectory)
                    if self.smoothed_trajectories[agent_id]:
                        smoothed_x, smoothed_y = zip(*self.smoothed_trajectories[agent_id])
                        plt.plot(smoothed_x, smoothed_y, linestyle='--', label=f'Agent {agent_id} - Smoothed')
                    for milestone_index in self.milestones[agent_id]:
                        if milestone_index < len(trajectory):
                            plt.scatter(*trajectory[milestone_index], marker='x', color='red', s=100, zorder=5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.75)
        plt.title('Agent Trajectories')
        plt.show()
    
    def render_in_process(renderer, *agent_ids):
        process = multiprocessing.Process(target=renderer.render, args=agent_ids)
        process.start()
        
        return process
    
    def _insert_points(self, p1, p2, n):
        return [(p1[0] + (p2[0] - p1[0]) * i / n, p1[1] + (p2[1] - p1[1]) * i / n) for i in range(1, n)]

    def _process_segments(self, segments):
        processed_segment = []
        skip_next = False
        processed_segment.append(segments[0])
        for k in range(1, len(segments) - 1):
            if skip_next:
                skip_next = False
                continue

            p0, p1, p2 = segments[k - 1], segments[k], segments[k + 1]

            if (p1[0] == p0[0] and p1[1] != p0[1] and p1[1] == p2[1]) or (p1[1] == p0[1] and p1[0] != p0[0] and p1[0] == p2[0]):
                processed_segment.append([((p0[0] + p2[0]) / 2 + p1[0]) / 2, ((p0[1] + p2[1]) / 2 + p1[1]) / 2])
            elif (p0[0] == p2[0]) and (p0[1] == p2[1]):
                if p0[1] == p1[1]:
                    if p0[0] < p1[0]:
                        processed_segment.append([p1[0], p1[1] - 0.1])
                        processed_segment.append([p1[0] + 0.1, p1[1]])
                        processed_segment.append([p1[0], p1[1] + 0.1])
                    elif p0[0] > p1[0]:
                        processed_segment.append([p1[0], p1[1] + 0.1])
                        processed_segment.append([p1[0] - 0.1, p1[1]])
                        processed_segment.append([p1[0], p1[1] - 0.1])
                    else:
                        processed_segment.append(p1)
                elif p0[0] == p1[0]:
                    if p0[1] < p1[1]:
                        processed_segment.append([p1[0] + 0.1, p1[1]])
                        processed_segment.append([p1[0], p1[1] + 0.1])
                        processed_segment.append([p1[0] - 0.1, p1[1]])
                    elif p0[1] > p1[1]:
                        processed_segment.append([p1[0] - 0.1, p1[1]])
                        processed_segment.append([p1[0], p1[1] - 0.1])
                        processed_segment.append([p1[0] + 0.1, p1[1]])
                    else:
                        processed_segment.append(p1)
                else:
                    processed_segment.append(p1)
            else:
                processed_segment.append(p1)

        processed_segment.append(segments[-1])
        return processed_segment

    def _smooth_and_store_curvature(self, agent_id, start, end):
        trajectory = np.array(self.trajectories[agent_id][start:end + 1])
        if len(trajectory) < 3:
            return
        
        new_segment = []

        for j in range(len(trajectory) - 1):
            new_segment.append(tuple(trajectory[j]))
            new_segment.extend(self._insert_points(trajectory[j], trajectory[j + 1], 2))

        new_segment.append(tuple(trajectory[-1]))

        processed_segment = self._process_segments(new_segment)

        if len(processed_segment) < 3:
            self.cur_queue[agent_id].append(0.0)
            return

        tck, u = splprep([np.array(processed_segment)[:, 0], np.array(processed_segment)[:, 1]], s=0, k=2)
        u_fine = np.linspace(0, 1, len(processed_segment) * self.sampling_factor)
        x_fine, y_fine = splev(u_fine, tck)

        self.smoothed_trajectories[agent_id].extend(zip(x_fine, y_fine))

        dx = np.gradient(x_fine)
        dy = np.gradient(y_fine)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        total_curvature = np.sum(curvature * ds)

        self.cur_queue[agent_id].append(total_curvature)
        self.cur_store[agent_id].append(total_curvature)
        
    def delete(self, agent_id):
        if agent_id in self.trajectories and self.milestones[agent_id]:
            last_milestone = self.milestones[agent_id][-1]
            
            self.trajectories[agent_id] = self.trajectories[agent_id][:last_milestone + 1]
            
            if self.smoothed_trajectories[agent_id]:
                num_points_to_remove = len(self.smoothed_trajectories[agent_id]) - last_milestone - 1
                self.smoothed_trajectories[agent_id] = self.smoothed_trajectories[agent_id][:-num_points_to_remove]

            for _ in range(len(self.cur_queue[agent_id]) - last_milestone):
                self.cur_queue[agent_id].pop()
            
            self.milestones[agent_id].pop()
    
    def clear(self, agent_id):
        self.trajectories[agent_id].clear()
        self.milestones[agent_id].clear()
        self.smoothed_trajectories[agent_id].clear()
        self.cur_queue[agent_id].clear()   
        self.cur_store[agent_id].clear()   

    def clear_all(self):
        for agent_id in range(self.num_agents):
            self.clear(agent_id)
            
    def get_curvature(self, agent_id):
        return self.cur_queue[agent_id]
    
    def get_distance(self):
        
        real_distances = []
        for agent_id in range(self.num_agents):
            real_traj = self.trajectories[agent_id]
            real_len = 0.0
            if len(real_traj) > 1:
                for i in range(1, len(real_traj)):
                    p1, p2 = np.array(real_traj[i - 1]), np.array(real_traj[i])
                    real_len += np.linalg.norm(p2 - p1)
            real_distances.append(real_len)


        return real_distances

    def start_episode(self):
        self.episode_start_idx = [len(self.trajectories[i]) for i in range(self.num_agents)]

    def get_distance_since_episode(self):
        distances = []
        for i in range(self.num_agents):
            traj = self.trajectories[i]
            s = self.episode_start_idx[i]
            d = 0.0
            if len(traj) - s > 1:
                for k in range(s + 1, len(traj)):
                    p1, p2 = np.array(traj[k - 1]), np.array(traj[k])
                    d += np.linalg.norm(p2 - p1)
            distances.append(d)
        return distances


class MultiAgentTaskEnv(gym.Env):
    def __init__(self,para):
        self.para=para
        self.num_agents=para.num_agents
        self.num_tasks=para.num_tasks
        self.map_size=para.map_size
        self.action_space = spaces.Discrete(para.action_size)
        max_x = self.map_size[0] - 1
        max_y = self.map_size[1] - 1
        high = np.array([max_x, max_y, 1, max(self.num_tasks, self.num_agents)])
        self.observation_space = Box(low=0, high=max_x, shape=(self.num_agents + self.num_tasks, 4), dtype=np.int64)

        

        self.agent_positions = np.random.randint(0, self.map_size[0], size=(self.num_agents, 2))
        self.task_positions = np.random.randint(0, self.map_size[0], size=(self.num_tasks, 2))
        self.completed_tasks = np.zeros(self.num_tasks, dtype=bool)
        self.agent_occupied = np.zeros(self.num_agents, dtype=bool)
        self.agent_occupied_taskid = np.zeros(self.num_agents, dtype=int)
        self.synch_num = np.zeros(self.num_tasks, dtype=int)
        
        self.done=False
        self.total_distance = 0
        plt.ion()

        self.traj = Trajectory(self.num_agents)
        self.batch_size = 5
        self.seq_length = 300
        self.traj_record =[np.zeros((self.seq_length,self.batch_size,2)) for _ in range(self.num_agents)]
        self.traj_mask =[np.ones((self.batch_size,self.seq_length),dtype=bool) for _ in range(self.num_agents)]
        self.traj_obs = [np.zeros((self.batch_size, self.num_agents+self.num_tasks, 4)) for _ in range(self.num_agents)]
        self.traj_batch_pointer = np.zeros(self.num_agents,dtype=int)
        self.traj_position_pointer = np.zeros(self.num_agents,dtype=int)
        self.traj_output = np.zeros((self.num_agents), dtype=bool)
        self.idle_streak = np.zeros(self.num_agents, dtype=np.int32)

        self.idle_streak_coeff = 0.02     
        self.idle_streak_cap = 50         


        self.ep_step_count = 0
        self.ep_reward_sum = 0.0

        self.ep_task_reward_sum = 0.0      
        self.ep_done_bonus_sum = 0.0       

        self.ep_move_penalty_sum = 0.0    
        self.ep_crowd_penalty_sum = 0.0   
        self.ep_idle_penalty_sum = 0.0    

        self.ep_tasks_completed = 0      
        self.ep_idle_count = 0            
        self.ep_move_count = 0             
        self.ep_total_manhattan = 0.0     
        self.ep_progress_reward_sum = 0.0     
        self.ep_progress_steps = 0           

        self.task_overlap_sum = np.zeros(self.num_tasks, dtype=np.float32)   
        self.task_overlap_peak = np.zeros(self.num_tasks, dtype=np.float32)   
        self.task_active_steps = np.zeros(self.num_tasks, dtype=np.int32)    
        self.task_first_claim_step = -np.ones(self.num_tasks, dtype=np.int32) 

        self.ep_task_credit_sum = 0.0
        self.ep_task_credit_count = 0

        self.episode_idx = 0 
        self.max_episodes = self.para.episode_max


        variant = str(getattr(para, "credit_variant", "ours")).lower()

        self.credit_config = {
            "overlap_coeff": float(getattr(para, "credit_overlap_coeff", 0.2)),
            "wait_coeff": float(getattr(para, "credit_wait_coeff", 0.001)),
            "overlap_peak_weight": float(getattr(para, "credit_overlap_peak_weight", 0.5)),
        }

        self.credit_mode_selector = None

        if variant == "ours":
            self.credit_mode = str(getattr(para, "credit_mode", "crowd_wait_penalty_v1"))

            llm_sel = llm_credit_selector_py37_patched.LLMCreditModeSelector(
                model=str(getattr(para, "credit_llm_model", "qwen3.5-flash")),
                debug=False,
                min_interval_sec=float(getattr(para, "credit_llm_min_interval", 0.5)),
                cache_ttl_sec=float(getattr(para, "credit_llm_cache_ttl", 5.0)),
                per_episode_budget=int(getattr(para, "credit_per_episode_budget", self.num_tasks)),
            )

            self.credit_mode_selector = ETCSelectorWrapper(
                llm_selector=llm_sel,
                llm_ratio=float(getattr(para, "credit_llm_ratio", 0.3)),
                min_interval_sec=float(getattr(para, "credit_gate_min_interval", 1.0)),
                debug=False,
            )

        elif variant == "team":
            self.credit_mode = "team_avg_v1"
            self.credit_mode_selector = None

        elif variant == "no_llm":
            self.credit_mode = str(getattr(para, "credit_mode", "crowd_wait_penalty_v1"))
            self.credit_mode_selector = None

        elif variant == "no_hier":
            self.credit_mode = "none"
            self.credit_mode_selector = None

        else:
            self.credit_mode = str(getattr(para, "credit_mode", "crowd_wait_penalty_v1"))
            self.credit_mode_selector = None

        self.credit_engine = CreditSettlementEngine(
            mode=self.credit_mode,
            config=self.credit_config,
            selector=lambda ctx, env:
                env.credit_mode_selector(ctx, env)
                if callable(getattr(env, "credit_mode_selector", None))
                else env.credit_mode,
        )


        self.credit_event_log = []  
        self.agent_task_steps = np.zeros((self.num_agents, self.num_tasks), dtype=np.int32)


        self.ep_credit_corr = 0.0
        self.ep_peak_synch = 0

    def random_selector(ctx, env):
        return random.choice(list(env.credit_engine._rules.keys()))

    def set_credit_mode(self, mode: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.credit_mode = str(mode)
        if config:
            self.credit_config.update(config)
        # Keep engine in sync
        if hasattr(self, "credit_engine") and self.credit_engine is not None:
            self.credit_engine.mode = self.credit_mode
            self.credit_engine.config = self.credit_config

    def reset(self):
        self.agent_positions = np.random.randint(0, self.map_size[0], size=(self.num_agents, 2))
        self.task_positions = np.random.randint(0, self.map_size[0], size=(self.num_tasks, 2))
        
        self.completed_tasks = np.zeros(self.num_tasks, dtype=bool)
        self.agent_occupied = np.zeros(self.num_agents, dtype=bool)
        self.agent_occupied_taskid = np.zeros(self.num_agents, dtype=int)
        self.synch_num = np.zeros(self.num_tasks, dtype=int)
        self.idle_penalty_coeff=0.05

        self.total_distance = 0
        self.done=False
        self.traj.clear_all()
        self.traj.start_episode()
        self.traj_record = [np.zeros((self.seq_length, self.batch_size, 2)) for _ in range(self.num_agents)]
        self.traj_mask   = [np.ones((self.batch_size, self.seq_length), dtype=bool) for _ in range(self.num_agents)]
        self.traj_obs    = [np.zeros((self.batch_size, self.num_agents + self.num_tasks, 4)) for _ in range(self.num_agents)]
        self.traj_batch_pointer[:]    = 0
        self.traj_position_pointer[:] = 0
        self.traj_output[:]           = False   
        self.idle_streak[:] = 0
        self.idle_punish_max = 1
        self.ep_step_count = 0
        self.ep_reward_sum = 0.0

        self.ep_task_reward_sum = 0.0
        self.ep_done_bonus_sum = 0.0

        self.ep_move_penalty_sum = 0.0
        self.ep_crowd_penalty_sum = 0.0
        self.ep_idle_penalty_sum = 0.0

        self.ep_tasks_completed = 0
        self.ep_idle_count = 0
        self.ep_move_count = 0
        self.ep_total_manhattan = 0.0
        self.ep_progress_reward_sum = 0.0      
        self.ep_progress_steps = 0            

        self.task_overlap_sum = np.zeros(self.num_tasks, dtype=np.float32)   
        self.task_overlap_peak = np.zeros(self.num_tasks, dtype=np.float32)   
        self.task_active_steps = np.zeros(self.num_tasks, dtype=np.int32)     
        self.task_first_claim_step = -np.ones(self.num_tasks, dtype=np.int32) 

        self.ep_task_credit_sum = 0.0
        self.ep_task_credit_count = 0

        self.episode_idx += 1

        self.credit_engine.mode = self.credit_mode
        self.credit_event_log.clear()
        self.agent_task_steps.fill(0)
        self.ep_credit_corr = 0.0
        self.ep_peak_synch = 0

        self.credit_engine.config = dict(self.credit_config)
        if hasattr(self, "credit_mode_selector") and hasattr(self.credit_mode_selector, "_sticky_mode"):
            self.credit_mode_selector._sticky_mode = None
            self.credit_mode_selector._last_llm_t = 0.0
        

        return self.get_state()
      
    
    
    def orietation(self, postion_id, agent_id):
        if postion_id == self.num_tasks:
            return int(0)
        if self.task_positions[postion_id][0] > self.agent_positions[agent_id][0]:
            return int(2)
        elif self.task_positions[postion_id][0] < self.agent_positions[agent_id][0]:
            return int(1)
        else:
            if self.task_positions[postion_id][1] > self.agent_positions[agent_id][1]:
                return int(4)
            elif self.task_positions[postion_id][1] < self.agent_positions[agent_id][1]:
                return int(3)
            else:
                return int(0)  
    
    def is_idle(self, task_id):
        return task_id == self.num_tasks
    
    def print_task_position_stats(self):
        if len(self.task_positions) == 0:
            print("No tasks available.")
            return

        xs = [pos[0] for pos in self.task_positions]
        ys = [pos[1] for pos in self.task_positions]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        print("Task position statistics:")
        print(f"  x_min = {x_min}, x_max = {x_max}, span = {x_max - x_min}")
        print(f"  y_min = {y_min}, y_max = {y_max}, span = {y_max - y_min}")
        print(f"  approx covered area = {(x_max - x_min + 1) * (y_max - y_min + 1)}")
        print(f"  num_tasks = {len(self.task_positions)}")



    def step(self, actions_set):
        rewards = []
        actions = []
        actions_id = []
        milestone = np.zeros(self.num_agents, dtype=bool)
        step_team_credit = np.zeros(self.num_agents, dtype=np.float32)
        for i ,actionset in enumerate(actions_set):
            if self.agent_occupied[i]==True and self.completed_tasks[self.agent_occupied_taskid[i]]==False:
                task_id = self.agent_occupied_taskid[i]
                actions.append(self.orietation(task_id, i))
                actions_id.append(task_id)
            elif self.agent_occupied[i]==True and self.completed_tasks[self.agent_occupied_taskid[i]]==True:
                old_tid = int(self.agent_occupied_taskid[i])
                if 0 <= old_tid < self.num_tasks:
                    self.synch_num[old_tid] = max(0, self.synch_num[old_tid] - 1)

                self.agent_occupied[i] = False
                self.agent_occupied_taskid[i] = 0 
                picked = False
                for j ,action in enumerate(actionset):
                    if action !=self.num_tasks and self.completed_tasks[action]:
                        continue
                    else:
                        task_id = action
                        if task_id != self.num_tasks:
                            self.agent_occupied_taskid[i]=task_id
                            self.agent_occupied[i]=True  
                            self.synch_num[task_id]+=1
                        actions.append(self.orietation(task_id, i))
                        actions_id.append(task_id)
                        picked = True                 
                        break
                if not picked:
                    actions.append(0)
                    actions_id.append(self.num_tasks) 
            else:
                picked = False
                for j ,action in enumerate(actionset):
                    if action !=self.num_tasks and self.completed_tasks[action]:
                        continue
                    else:
                        task_id = action
                        if task_id != self.num_tasks:
                            self.agent_occupied_taskid[i]=task_id
                            self.agent_occupied[i]=True  
                            self.synch_num[task_id]+=1
                        actions.append(self.orietation(task_id, i))
                        actions_id.append(task_id)
                        picked = True              
                        break
                if not picked:
                    actions.append(0)
                    actions_id.append(self.num_tasks) 

        task_reward = 10
        move_penalty_coeff = 0.02
        crowd_penalty_coeff = 0.06
        idle_penalty_coeff = self.compute_idle_penalty()
        progress_coeff = 0.1

        synch_snapshot = self.synch_num.copy()
        self.ep_peak_synch = max(self.ep_peak_synch, int(np.max(synch_snapshot)))

        rewards = []
        self.ep_step_count += 1
        for t in range(self.num_tasks):
            if self.completed_tasks[t]:
                continue
            s = float(synch_snapshot[t]) 
            if s > 0:
                if self.task_first_claim_step[t] < 0:
                    self.task_first_claim_step[t] = int(self.ep_step_count)

                self.task_active_steps[t] += 1

                overlap = max(0.0, s - 1.0)
                self.task_overlap_sum[t] += overlap
                if overlap > self.task_overlap_peak[t]:
                    self.task_overlap_peak[t] = overlap





        for i, action in enumerate(actions):
            old_position = self.agent_positions[i].copy()
            new_position = self.agent_positions[i] + [(0,0),(-1,0),(1,0),(0,-1),(0,1)][action]
            new_position = np.clip(new_position, 0, self.map_size[0]-1)

            distance = np.sum(np.abs(new_position - self.agent_positions[i]))
            self.total_distance += distance
            self.agent_positions[i] = new_position
            self.traj.add(i, (new_position.tolist()))
            if self.traj_position_pointer[i] < self.seq_length:
                self.traj_record[i][self.traj_position_pointer[i]][self.traj_batch_pointer[i]][0]=new_position[0]
                self.traj_record[i][self.traj_position_pointer[i]][self.traj_batch_pointer[i]][1]=new_position[1]
                self.traj_mask[i][self.traj_batch_pointer[i]][self.traj_position_pointer[i]]=False
            self.traj_position_pointer[i] += 1 
            reward_i = 0.0

            if distance != 0:
                penalty = move_penalty_coeff * distance
                reward_i -= penalty
                self.ep_move_penalty_sum += penalty
                self.ep_move_count += 1
                self.ep_total_manhattan += distance
            else:
                penalty = move_penalty_coeff  * 2
                reward_i -= penalty
                self.ep_move_penalty_sum += penalty
                self.ep_total_manhattan += 0

            if self.agent_occupied[i]:
                t_id = self.agent_occupied_taskid[i]
                overlap = synch_snapshot[t_id] - 1
                if overlap > 0:
                    penalty = crowd_penalty_coeff * overlap
                    reward_i -= penalty
                    self.ep_crowd_penalty_sum += penalty

            if actions_id[i] == self.num_tasks:
                self.idle_streak[i] = min(self.idle_streak[i] + 1, self.idle_streak_cap)
                base = idle_penalty_coeff
                extra = self.idle_streak_coeff * float(self.idle_streak[i])
                penalty = min(base + extra, self.idle_punish_max)
                reward_i -= penalty
                self.ep_idle_penalty_sum += penalty
                self.ep_idle_count += 1
            else:
                self.idle_streak[i] = 0

            if self.agent_occupied[i]:
                t_id = int(self.agent_occupied_taskid[i])

                if 0 <= t_id < self.num_tasks and (not self.completed_tasks[t_id]):
                    self.agent_task_steps[i, t_id] += 1

                    old_d = np.sum(np.abs(old_position - self.task_positions[t_id]))
                    new_d = np.sum(np.abs(new_position - self.task_positions[t_id]))
                    delta_d = old_d - new_d

                    crowd = float(synch_snapshot[t_id])

                    crowd_pen = 0 * crowd          
                    effective_progress = delta_d - crowd_pen

                    pr = progress_coeff * effective_progress
                    reward_i += pr

                    self.ep_progress_reward_sum += pr
                    self.ep_progress_steps += 1
          


            for j, task_pos in enumerate(self.task_positions):
                if np.all(new_position == task_pos) and not self.completed_tasks[j]:

                    reward_i += task_reward
                   
                    if str(getattr(self, "credit_mode", "")) == "team_avg_v1":
                        credit_vec, credit_meta = self.credit_engine.settle(env=self, task_id=j, finisher_id=i)
                        step_team_credit += credit_vec
                        event_credit_total  = float(np.sum(credit_vec))   
                    else:
                        task_credit, credit_meta = self.credit_engine.settle_finisher_scalar(env=self, task_id=j, finisher_id=i)
                        reward_i += float(task_credit)
                        event_credit_total  = float(task_credit)
                    task_active = int(self.task_active_steps[j]) if hasattr(self, "task_active_steps") else 0
                    finisher_active = int(self.agent_task_steps[i, j])

                    self.credit_event_log.append({
                        "task_id": int(j),
                        "finisher_id": int(i),
                        "task_credit_total": float(event_credit_total),         
                        "task_active_steps": int(task_active),  
                        "mode": str(credit_meta.get("mode", getattr(self, "credit_mode", ""))),
                    })

                    self.ep_task_credit_sum += float(credit_meta.get('task_credit_total', 0.0))
                    self.ep_task_credit_count += 1

                    self.ep_task_reward_sum += task_reward
                    self.ep_tasks_completed += 1
                    self.completed_tasks[j] = True
                    self.traj.milestone(i)
                    milestone[i] = True
                    rewards_tag = 1 
                    self.traj_obs[i][self.traj_batch_pointer[i]] = self.get_state()
                    self.traj_batch_pointer[i] += 1
                    self.traj_position_pointer[i] = 0
                    if self.traj_batch_pointer[i] == self.batch_size:
                        self.traj_batch_pointer[i] = 0
                        self.traj_output[i] = True

                    self.completed_tasks[j] = True
                    self.agent_occupied[i] = False

                    old_tid = int(self.agent_occupied_taskid[i])
                    self.agent_occupied_taskid[i] = 0

                    if 0 <= old_tid < self.num_tasks:
                        self.synch_num[old_tid] = max(0, self.synch_num[old_tid] - 1)

                    break


            rewards.append(reward_i)
            self.ep_reward_sum += reward_i

        if str(getattr(self, "credit_mode", "")) == "team_avg_v1":
            for k in range(self.num_agents):
                rewards[k] += float(step_team_credit[k])
                self.ep_reward_sum += float(step_team_credit[k])

        if np.all(self.completed_tasks):
            for i in range(len(rewards)):
                rewards[i]+=0
            self.done=True
            self._finalize_credit_metrics()
        assert len(rewards)==self.num_agents
        info = {
                    "ep_credit_corr": float(getattr(self, "ep_credit_corr", 0.0)),
                    "ep_peak_synch": int(getattr(self, "ep_peak_synch", 0)),
                    "ep_credit_events": int(len(getattr(self, "credit_event_log", []))),
                }
        return self.get_state(), rewards, [self.done] * self.num_agents, info, actions_id
    

    def _finalize_credit_metrics(self):
        events = self.credit_event_log
        if not events:
            self.ep_credit_corr = 0.0
            return



        xs = np.array([float(e.get("task_credit_total", 0.0)) for e in events])
        ys = np.array([float(e.get("task_active_steps", 0.0)) for e in events], dtype=np.float32)

        if len(xs) >= 2 and float(np.std(xs)) > 1e-8 and float(np.std(ys)) > 1e-8:
            self.ep_credit_corr = float(np.corrcoef(xs, ys)[0, 1])
        else:
            self.ep_credit_corr = 0.0

    def compute_idle_penalty(self):
        remaining_tasks = np.sum(~self.completed_tasks)
        remaining_ratio = remaining_tasks / self.num_tasks  

        return self.idle_penalty_coeff * remaining_ratio



    def render(self, mode='human', close=False):
        self.grid_size = self.map_size[0] 
        grid = np.zeros((self.grid_size, self.grid_size))
        grid_width = self.grid_size
        grid_height = self.grid_size
        extent = (0, grid_width, 0, grid_height)
        
        for j, task_pos in enumerate(self.task_positions):
            x, y = task_pos
            if self.completed_tasks[j]==False:
                grid[x, y] = 0.85  
        for agent_pos in self.agent_positions:
            grid[x, y] = 1 
                   
        plt.imshow(grid, cmap='CMRmap', interpolation='nearest', extent=extent) 
        for x in range(1, grid_width):
            plt.plot([x, x], [0, grid_height], color='r', linewidth=0.5)
        for y in range(1, grid_height):
            plt.plot([0, grid_width], [y, y], color='r', linewidth=0.5)
            
        plt.pause(0.01)  
        plt.clf()  
        
    def get_state(self):
        self.agent_positions_obs = np.hstack((self.agent_positions, self.agent_occupied.reshape(-1, 1), self.agent_occupied_taskid.reshape(-1, 1)))
        self.task_positions_obs = np.hstack((self.task_positions, self.completed_tasks.reshape(-1, 1), self.synch_num.reshape(-1, 1)))
        return np.vstack((self.agent_positions_obs , self.task_positions_obs))
    
    def env_save(self,current_time,path):
        np.save(os.path.join(path,f"agent_positions_{current_time}.npy"), self.agent_positions)
        np.save(os.path.join(path,f"agent_occupied_{current_time}.npy"), self.agent_occupied)
        np.save(os.path.join(path,f"agent_occupied_taskid_{current_time}.npy"), self.agent_occupied_taskid)
        np.save(os.path.join(path,f"task_positions_{current_time}.npy"), self.task_positions)
        np.save(os.path.join(path,f"completed_tasks_{current_time}.npy"), self.completed_tasks)
        np.save(os.path.join(path,f"synch_num_{current_time}.npy"), self.synch_num)
        np.save(os.path.join(path,f"total_distance_{current_time}.npy"), self.total_distance)
        np.save(os.path.join(path,f"done_{current_time}.npy"), self.done)


    def env_load(self,current_time, path):
        self.agent_positions = np.load(os.path.join(path,f"agent_positions_{current_time}.npy"))
        self.agent_occupied = np.load(os.path.join(path,f"agent_occupied_{current_time}.npy"))
        self.agent_occupied_taskid = np.load(os.path.join(path,f"agent_occupied_taskid_{current_time}.npy"))
        self.task_positions = np.load(os.path.join(path,f"task_positions_{current_time}.npy"))
        self.completed_tasks = np.load(os.path.join(path,f"completed_tasks_{current_time}.npy"))
        self.synch_num = np.load(os.path.join(path,f"synch_num_{current_time}.npy"))
        self.total_distance = np.load(os.path.join(path,f"total_distance_{current_time}.npy"))
        self.done = np.load(os.path.join(path,f"done_{current_time}.npy"))
        return self.get_state()

    def env_save_traj(self,current_time,path):
        np.save(os.path.join(path,f"cur_store_{current_time}.npy"), self.traj.cur_store)
        np.save(os.path.join(path,f"trajectories_{current_time}.npy"), self.traj.trajectories)
        np.save(os.path.join(path,f"milestones_{current_time}.npy"), self.traj.milestones)

    def get_traj_distance(self):
        return self.traj.get_distance_since_episode()



