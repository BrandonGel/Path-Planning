from math import fabs
import random
from Simulation.CBS.cbs import Environment
from collections import defaultdict
import numpy as np
import torch
from collections import deque
import time
import math


class NeuralATTF(object):
    def __init__(self, agents, dimesions, obstacles, non_task_endpoints, simulation, a_star_max_iter=4000, encoder=None):
        self.agents = agents
        self.dimensions = dimesions
        self.obstacles = set(obstacles)
        self.non_task_endpoints = non_task_endpoints
        self.assigned_tasks = set()
        self.total_astar_iter = 0
        self.max_iters = 0
        self.encoder = encoder
        print(len(non_task_endpoints))
        if len(agents) > len(non_task_endpoints):
            print('There are more agents than non task endpoints, instance is not well-formed.')
            exit(1)
        self.token = {}
        self.simulation = simulation
        self.a_star_max_iter = a_star_max_iter
        self.init_token()

    def init_token(self):
        self.token['agents'] = {}
        self.token['tasks'] = {}
        self.token['start_tasks_times'] = {}
        self.token['completed_tasks_times'] = {}
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        self.token['agents_to_tasks'] = {}
        self.token['completed_tasks'] = 0
        self.token['n_replans'] = 0
        self.token['path_ends'] = set()
        self.token['occupied_non_task_endpoints'] = set()
        self.token['delayed_agents'] = []
        self.token['delayed_agents_to_reach_task_start'] = []
        for a in self.agents:
            self.token['agents'][a['name']] = [a['start']]
            self.token['path_ends'].add(tuple(a['start']))
            if a['start'] in self.non_task_endpoints:
                self.token['occupied_non_task_endpoints'].add(tuple(a['start']))
        self.token['deadlock_count_per_agent'] = defaultdict(lambda: 0)

    def get_idle_agents(self):
        agents = {}
        for name, path in self.token['agents'].items():
            if len(path) == 1:
                agents[name] = path
            if name in self.token['agents_to_tasks'] and self.token['agents_to_tasks'][name]['task_name'] == 'safe_idle':
                agents[name] = path
        return agents

    def admissible_heuristic(self, task_pos, agent_pos):
        manhattan_distance = fabs(task_pos[0] - agent_pos[0]) + fabs(task_pos[1] - agent_pos[1])
        euclidean_distance = ((task_pos[0] - agent_pos[0]) ** 2 + (task_pos[1] - agent_pos[1]) ** 2) ** 0.5
        return manhattan_distance + 0.001*euclidean_distance

    def find_closest_agent(self, available_tasks, idle_agents, token):

        pairs = []

        for agent in idle_agents.keys():
            
            if agent in token['agents_to_tasks'] and token['agents_to_tasks'][agent]['task_name'] != 'safe_idle':
                task_name = token['agents_to_tasks'][agent]['task_name']
                if agent in token['delayed_agents_to_reach_task_start']:
                    task = [token['agents'][agent][0], token['agents_to_tasks'][agent]['goal']]
                else:
                    task = [token['agents_to_tasks'][agent]['start'], token['agents_to_tasks'][agent]['goal']]
                pairs.append((agent, task_name, task, -1))
        
            elif len(available_tasks) != 0:
                agent_position = idle_agents[agent][0]
                
                for task_name, task_positions in available_tasks.items():
                    task_start = task_positions[0]
                    d = self.admissible_heuristic(task_start, agent_position)
                    pairs.append((agent, task_name, task_positions, d))
        
            cost = len(token['agents'][agent]) + 2*(self.dimensions[0] + self.dimensions[1])
            pairs.append((agent, None, None, cost))

        pairs = sorted(pairs, key=lambda x: x[3])
        assigned_pairs = deque()
        assigned_tasks = set()
        assigned_agents = set()
        valid_pairs = []
        for pair in pairs:
            agent = pair[0]
            task_name = pair[1]
            task = pair[2]
            if task_name: 
                if task_name not in assigned_tasks and agent not in assigned_agents:
                    assigned_pairs.append(pair)
                    assigned_tasks.add(task_name)
                    assigned_agents.add(agent)
                    valid_pairs.append(pair)
            else:
                if agent not in assigned_agents:
                    assigned_pairs.append(pair)
                    assigned_agents.add(agent)

        return assigned_pairs, valid_pairs
 
    def get_moving_obstacles_agents(self, agents, time_start):
        obstacles = {}
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    obstacles[(path[i][0], path[i][1], k)] = name
                    if i == len(path) - 1:
                        obstacles[(path[i][0], path[i][1], k+1)] = name
        return obstacles

    def get_idle_obstacles_agents(self, agents_paths, delayed_agents, time_start):
        obstacles = set()
        for path in agents_paths:
            if len(path) == 1:# and path[0] in self.non_task_endpoints:
                obstacles.add((path[0][0], path[0][1]))
            elif path[-1] in self.non_task_endpoints:
                obstacles.add((path[-1][0], path[-1][1]))
        for agent_name in delayed_agents:
            obstacles.add(tuple(self.token['agents'][agent_name][0]))
        return obstacles

    def check_safe_idle(self, agent_pos, agent_name):
        for task_name, task in self.token['tasks'].items():
            if tuple(task[0]) == tuple(agent_pos) or tuple(task[1]) == tuple(agent_pos):
                return False
        if len(self.token['agents'][agent_name]) != 1:
            return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        return True

    def get_closest_non_task_endpoint(self, agent_pos):
        dist = -1
        res = -1
        for endpoint in self.non_task_endpoints:
            if endpoint not in self.token['occupied_non_task_endpoints']:
                if dist == -1:
                    dist = self.admissible_heuristic(endpoint, agent_pos)
                    res = endpoint
                else:
                    tmp = self.admissible_heuristic(endpoint, agent_pos)
                    if tmp < dist:
                        dist = tmp
                        res = endpoint
        if res == -1:
            print('Error in finding non-task endpoint, is instance well-formed?')
            breakpoint()
            exit(1)
        return res

    def update_ends(self, agent_pos, agent_name=None):
        if not isinstance(agent_pos, tuple):
            agent_pos = tuple(agent_pos)
        if agent_pos not in [tuple(path[-1]) for agent,path in self.token['agents'].items() if agent != agent_name]:
            if agent_pos in self.token['path_ends']:
                self.token['path_ends'].remove(tuple(agent_pos))
            if agent_pos in self.token['occupied_non_task_endpoints']:
                self.token['occupied_non_task_endpoints'].remove(tuple(agent_pos))

    def get_agents_to_tasks_goals(self):
        goals = set()
        for el in self.token['agents_to_tasks'].values():
            goals.add(tuple(el['goal']))
        return goals

    def get_agents_to_tasks_starts_goals(self):
        starts_goals = set()
        for el in self.token['agents_to_tasks'].values():
            starts_goals.add(tuple(el['goal']))
            starts_goals.add(tuple(el['start']))
        return starts_goals

    def get_completed_tasks(self):
        return self.token['completed_tasks']

    def get_completed_tasks_times(self):
        return self.token['completed_tasks_times']

    def get_n_replans(self):
        return self.token['n_replans']

    def get_token(self):
        return self.token
    
    def plan(self, agent_name, start, goal, all_idle_agents, all_delayed_agents, cost_map, cost):
        moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], cost)
        idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, cost)
        agent = {'name': agent_name, 'start': start, 'goal': goal}
        env = Environment(self.dimensions, agent, self.obstacles | idle_obstacles_agents,
                            moving_obstacles_agents, a_star_max_iter=self.a_star_max_iter, cost_map=cost_map)
        path_to_task_goal, iters = env.search()
        self.total_astar_iter += iters
        if iters != self.a_star_max_iter:
            self.max_iters = max(self.max_iters, iters)
        return path_to_task_goal

    def go_to_closest_non_task_endpoint(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, cost_map):

        if tuple(self.token['agents'][agent_name][-1]) in self.non_task_endpoints:
            print('Agent', agent_name, 'already going to non-task endpoint.')
            return
        closest_non_task_endpoint = self.get_closest_non_task_endpoint(agent_pos)
        print('Closest non-task endpoint for agent', agent_name, 'is', closest_non_task_endpoint)
        path_to_non_task_endpoint = self.plan(agent_name, agent_pos, closest_non_task_endpoint, all_idle_agents, all_delayed_agents, None, 0)
        if not path_to_non_task_endpoint:
            print("Solution to non-task endpoint not found for agent", agent_name, " instance is not well-formed.")
            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, 4)
        else:
            print('No available task for agent', agent_name, ' moving to safe idling position...')
            self.update_ends(agent_pos)
            self.token['occupied_non_task_endpoints'].add(tuple(closest_non_task_endpoint))
            self.token['agents_to_tasks'][agent_name] = {'task_name': 'safe_idle', 'start': agent_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['agents'][agent_name] = []
            for el in path_to_non_task_endpoint[agent_name]:
                self.token['agents'][agent_name].append([el['x'], el['y']])

    def get_random_close_cell(self, agent_pos, r):
        while True:
            cell = (agent_pos[0] + random.choice(range(-r - 1, r + 1)), agent_pos[1] + random.choice(range(-r - 1, r + 1)))
            if cell not in self.obstacles and cell not in self.token['path_ends'] and \
                    cell not in self.token['occupied_non_task_endpoints'] \
                    and cell not in self.get_agents_to_tasks_goals() \
                    and 0 <= cell[0] < self.dimensions[0] and 0 <= cell[1] < self.dimensions[1]:
                return cell

    def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, r):
        self.token['deadlock_count_per_agent'][agent_name] += 1
        if self.token['deadlock_count_per_agent'][agent_name] >= 2:
            self.token['deadlock_count_per_agent'][agent_name] = 0
            random_close_cell = self.get_random_close_cell(agent_pos, r)
            path_to_non_task_endpoint = self.plan(agent_name, agent_pos, random_close_cell, all_idle_agents, all_delayed_agents, None, 0)
            if not path_to_non_task_endpoint:
                print("No solution to deadlock recovery for agent", agent_name, " retrying later.")
            else:
                print('Agent', agent_name, 'causing deadlock, moving to safer position...')
                self.update_ends(agent_pos)
                self.token['agents'][agent_name] = []
                for el in path_to_non_task_endpoint[agent_name]:
                    self.token['agents'][agent_name].append([el['x'], el['y']])

    def time_forward(self):

        # Update completed tasks
        for agent_name in self.token['agents']:
            pos = self.simulation.actual_paths[agent_name][-1]
            if agent_name in self.token['agents_to_tasks'] and \
                (pos['x'], pos['y']) == tuple(self.token['agents_to_tasks'][agent_name]['goal']) and \
                    len(self.token['agents'][agent_name]) == 1:
                if self.token['agents_to_tasks'][agent_name]['task_name'] != 'safe_idle':
                    self.token['completed_tasks'] = self.token['completed_tasks'] + 1
                    self.token['completed_tasks_times'][self.token['agents_to_tasks'][agent_name]['task_name']] = self.simulation.get_time()
                    self.token['agents_to_tasks'].pop(agent_name)
                else:
                    self.token['agents_to_tasks'].pop(agent_name)

        # Check delayed agents and agents affected by delays
        self.token['delayed_agents'] = self.simulation.get_delayed_agents()
        for name in self.token['delayed_agents']:
            print('Agent', name, 'delayed or affected by delay!')
            path = self.token['agents'][name]
            self.token['n_replans'] = self.token['n_replans'] + 1
            self.update_ends(path[-1])
            if path[0] in self.non_task_endpoints:
                self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
            else:
                self.token['path_ends'].add(tuple(path[0]))
            if name in self.token['agents_to_tasks']:
                if self.token['agents_to_tasks'][name]['start'] not in path:
                    self.token['delayed_agents_to_reach_task_start'].append(name)
            self.token['agents'][name] = [path[0]]

        # Collect new tasks and assign them, if possible
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()


        idle_agents = self.get_idle_agents()
        available_tasks = {}
        for task_name, task in self.token['tasks'].items():
            if task_name not in self.assigned_tasks:
                available_tasks[task_name] = task
        assigned_pairs, valid_pairs = self.find_closest_agent(available_tasks, idle_agents, self.token)

        if self.encoder:
            enc_size = math.ceil(max(self.dimensions[0], self.dimensions[1])/16) * 16
            maze = torch.tensor(np.ones((enc_size, enc_size)), device='cuda').unsqueeze(0).float()
            for o in self.obstacles | self.get_idle_obstacles_agents(self.token['agents'].values(), self.token['delayed_agents'], 0):
                maze[0, o[0], o[1]] = 0

            if valid_pairs:
                start_goals = torch.zeros((2*len(valid_pairs), 1, enc_size, enc_size), device='cuda')

                for i, (agent_name, _, closest_task, _ )in enumerate(valid_pairs):
                    start, goal = self.token['agents'][agent_name][0], closest_task[0]
                    start_goals[2*i, 0, start[0], start[1]] = 1
                    start_goals[2*i, 0, goal[0], goal[1]] = 1
                    start, goal = closest_task
                    start_goals[2*i+1, 0, start[0], start[1]] = 1
                    start_goals[2*i+1, 0, goal[0], goal[1]] = 1

                batch = torch.cat([maze.expand(2*len(valid_pairs), -1, -1, -1), start_goals], dim=1)
                start_time = time.time()
                with torch.no_grad():
                    cost_maps = self.encoder(batch).squeeze(1).detach().cpu().numpy()
                print('Encoder time:', time.time() - start_time)
            else:
                cost_maps = []
        else:
            cost_maps = None

        cost_map_idx = 0
        while len(idle_agents) > 0:            
            
            agent_name, closest_task_name, closest_task, _ = assigned_pairs.popleft()
            if cost_maps is not None and cost_map_idx < len(cost_maps):
                cost_map_1 = cost_maps[cost_map_idx]
                cost_map_2 = cost_maps[cost_map_idx + 1]
                cost_map_idx += 2
            else:
                cost_map_1 = None
                cost_map_2 = None
            
            all_idle_agents = self.token['agents'].copy()
            all_idle_agents.pop(agent_name)
            all_delayed_agents = self.token['delayed_agents'].copy()
            if agent_name in all_delayed_agents:
                all_delayed_agents.remove(agent_name)
            agent_pos = idle_agents.pop(agent_name)[0]

            if closest_task:
                path_to_task_start = self.plan(agent_name, agent_pos, closest_task[0], all_idle_agents, all_delayed_agents, cost_map_1, 0)
                if not path_to_task_start:
                    print("Solution not found to", closest_task_name, "start for", agent_name, "idling at current position...")
                    if len(self.token['delayed_agents']) == 0:
                        self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, 4)
                else:
                    print("Solution found to", closest_task_name, "start for", agent_name, "searching solution to task goal...")
                    cost1 = sum([len(path) for path in path_to_task_start.values()])
                    path_to_task_goal = self.plan(agent_name, closest_task[0], closest_task[1], all_idle_agents, all_delayed_agents, cost_map_2, cost1-1)
                    if not path_to_task_goal:
                        print("Solution not found to", closest_task_name,  "goal for", agent_name, "idling at current position...")
                        if len(self.token['delayed_agents']) == 0:
                            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, 4)
                    else:
                        print("Solution found to", closest_task_name, "goal for agent", agent_name, "doing task...")
                        cost2 = sum([len(path) for path in path_to_task_goal.values()])
                        self.assigned_tasks.add(closest_task_name)
                        if agent_name not in self.token['agents_to_tasks']:
                            self.token['tasks'].pop(closest_task_name)
                            task = available_tasks.pop(closest_task_name)
                        else:
                            task = closest_task
                        if agent_name in self.token['delayed_agents_to_reach_task_start']:
                            self.token['delayed_agents_to_reach_task_start'].remove(agent_name)
                        last_step = path_to_task_goal[agent_name][-1]
                        self.update_ends(agent_pos,agent_name)
                        if len(self.token['agents'][agent_name]) > 1:
                            self.update_ends(self.token['agents'][agent_name][-1], agent_name)
                        self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                        self.token['agents_to_tasks'][agent_name] = {'task_name': closest_task_name, 'start': task[0],
                                                                     'goal': task[1], 'predicted_cost': cost1 + cost2}
                        self.token['agents'][agent_name] = []
                        for el in path_to_task_start[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
                        self.token['agents'][agent_name] = self.token['agents'][agent_name][:-1]
                        for el in path_to_task_goal[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
            
            elif self.check_safe_idle(agent_pos, agent_name):
                if agent_name in self.token['delayed_agents']:
                    if agent_name in self.token['agents_to_tasks'] and self.token['agents_to_tasks'][agent_name]['task_name'] == 'safe_idle':
                        if tuple(self.token['agents_to_tasks'][agent_name]['goal']) in self.token['occupied_non_task_endpoints']: 
                            self.token['occupied_non_task_endpoints'].remove(tuple(self.token['agents_to_tasks'][agent_name]['goal']))
                        
            else:
                self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents, all_delayed_agents, cost_map=np.ones((self.dimensions[0], self.dimensions[1])))

                    
        print('Number of completed tasks:', self.token['completed_tasks'])
