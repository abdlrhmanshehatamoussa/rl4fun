import random
import numpy as np
import json

class MazeSettings:
    def __init__(self, rewards: [[]], max_steps: int):
        if(type(rewards) == type('')):
            rewards = json.loads(rewards)
        self.RewardsMatrix: [[]] = rewards
        self.MaximumSteps: int = max_steps

class Maze2D:
    def __init__(self, settings: MazeSettings):
        rewards = settings.RewardsMatrix
        max_steps: int = settings.MaximumSteps

        # Number of layers
        self.Width = len(rewards[0])
        self.Height = len(rewards)
        # Number of nodes
        self.N = self.Width * self.Height
        self.Si = 1
        self.Sc = self.Si
        self.TakenSteps = 0
        self.MaxSteps = max_steps
        self.Rewards = np.concatenate(rewards, axis=0)

    def reset(self):
        self.Sc = self.Si
        self.TakenSteps = 0
        return self.Si

    def get_reward(self, s):
        return self.Rewards[s - 1]

    def flat2mat(self, s):
        if s > self.N:
            return None
        c = s % self.Width
        r = s // self.Height + 1
        if c == 0:
            c = self.Width
            r = r - 1
        return r, c

    def mat2flat(self, r, c):
        if 1 <= r <= self.Height and 1 <= c <= self.Width:
            res = (r - 1) * self.Width + c
            return res
        return 0

    def get_transitions(self, s):
        r, c = self.flat2mat(s)
        actions = self.actions_space()
        transitions = {
            actions[0]: self.mat2flat(r - 1, c),
            actions[1]: self.mat2flat(r, c + 1),
            actions[2]: self.mat2flat(r + 1, c),
            actions[3]: self.mat2flat(r, c - 1),
            actions[4]: self.mat2flat(r - 1, c + 1),
            actions[5]: self.mat2flat(r + 1, c + 1),
            actions[6]: self.mat2flat(r + 1, c - 1),
            actions[7]: self.mat2flat(r - 1, c - 1),
            actions[8]: s
        }
        return transitions

    def get_next_state(self, a):
        transitions = self.get_transitions(self.Sc)
        s2 = transitions[a]
        return s2

    def states_space(self):
        space = list(range(self.Si, self.N + 1))
        return space

    def actions_space(self):
        return ["n", "e", "s", "w", "ne", "se", "sw", "nw", "~"]

    def episode_ended(self):
        return self.TakenSteps >= self.MaxSteps

    # Define step function
    def step(self, a):
        s2 = self.get_next_state(a)
        r = self.get_reward(s2)
        if a == "~":
            r = 0
        self.Sc = s2
        self.TakenSteps += 1
        d = self.episode_ended()
        return r, s2, d

    def possible_actions(self, s):
        transitions = self.get_transitions(s)
        actions = []
        for (k, v) in transitions.items():
            if self.Si <= v <= self.N:
                if self.Rewards[v - 1] is not None:
                    actions.append(k)
        return actions

    def random_action(self):
        possible_actions = self.possible_actions(self.Sc)
        rand = random.randint(0, len(possible_actions) - 1)
        a = possible_actions[rand]
        return a


class MazeSolverSettings:
    def __init__(self, gamma: float, termination_threshold: float, history: bool):
        self.Gamma: float = gamma if gamma is not None else 0.95
        self.TerminationThreshold: float = termination_threshold if termination_threshold is not None else 0.0001
        self.SaveHistory: bool = history if history is not None else False


class Maze2DSolver:
    def __init__(self, env: Maze2D, settings: MazeSolverSettings):
        self.GAMMA: float = settings.Gamma
        self.TERMINATION_THRESHOLD: float = settings.TerminationThreshold
        self.SaveTrainingHistory: bool = settings.SaveHistory
        self.env: Maze2D = env
        self.states = self.env.states_space()
        self.actions = self.env.actions_space()

    # Train the model
    # Asynchronous value iteration for MDPs
    def train(self):
        q = np.random.rand(len(self.actions), len(self.states))
        qs = []
        prev_q = None
        count = 0
        while True:
            done = False
            current_state = self.env.reset()
            while not done:
                # Suitable Action
                action = self.env.random_action()
                action_index = self.actions.index(action)

                # Take a step
                reward, new_state, done = self.env.step(action)
                if not done:
                    # Update Q Table
                    new_state_index = new_state - 1
                    current_state_index = current_state - 1
                    max_q = np.max(q[:, new_state_index])
                    q[action_index, current_state_index] = reward + self.GAMMA * max_q
                    current_state = new_state

            # For visualization purposes
            if self.SaveTrainingHistory:
                qs.append(q.copy())

            # Check termination
            if prev_q is not None:
                diff = np.abs(np.subtract(prev_q, q))
                diff = np.less_equal(diff, self.TERMINATION_THRESHOLD)
                if np.all(diff):
                    if count > 20:
                        break
                    else:
                        count += 1
                else:
                    count = 0
            prev_q = q.copy()

        return q, qs

    def run_bulk(self, qs):
        trials = []
        for q in qs:
            trial = {}
            sln, score = self.run(q)
            trial['solution'] = [int(a) for a in sln]
            trial['score'] = int(score)
            trials.append(trial)
        return trials

    # Test the model
    def run(self, q):
        current_state = self.env.reset()
        actions = self.env.actions_space()
        final_solution = [current_state]
        score = 0
        while True:
            current_state_index = current_state - 1
            q_values = list(q[:, current_state_index])
            possible_actions = self.env.possible_actions(current_state)

            # Choose the best action
            max_q = min(q_values)
            action_ = actions[q_values.index(max_q)]
            for i, q_value in enumerate(q_values):
                action = actions[i]
                if action in possible_actions and q_value > max_q:
                    max_q = q_value
                    action_ = action

            # Perform the action
            reward, new_state, done = self.env.step(action_)
            final_solution.append(new_state)
            current_state = new_state
            score += reward
            if done:
                break
        return final_solution, score

    def solve(self):
        response = {}
        trained_q, trial_qs = self.train()

        if self.SaveTrainingHistory:
            trials = self.run_bulk(trial_qs)
            response["history"] = trials

        answer, score = self.run(trained_q)
        response["score"] = int(score)
        response["answer"] = [int(a) for a in answer]
        return response


if __name__ == "__main__":
    solver_settings = MazeSolverSettings(0.95, 0.0001, False)
    maze_settings = MazeSettings([
        [-2, -2, -2],
        [-2, -2, -2],
        [7, -2, 10]
    ], 20)
    maze = Maze2D(maze_settings)
    solver = Maze2DSolver(maze, solver_settings)
    solution = solver.solve()
    print(solution)
