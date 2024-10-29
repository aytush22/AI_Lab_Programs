class GoalBasedAgent:
    def __init__(self, goal):
        self.position = [0, 0]  # Start position
        self.goal = goal

    def perceive(self):
        return self.position

    def decide_action(self):
        if self.position[0] < self.goal[0]:
            return 'down'
        elif self.position[0] > self.goal[0]:
            return 'up'
        elif self.position[1] < self.goal[1]:
            return 'right'
        elif self.position[1] > self.goal[1]:
            return 'left'
        else:
            return 'goal reached'

    def act(self, action):
        if action == 'up':
            self.position[0] -= 1
        elif action == 'down':
            self.position[0] += 1
        elif action == 'left':
            self.position[1] -= 1
        elif action == 'right':
            self.position[1] += 1

    def run(self):
        while True:
            percept = self.perceive()
            action = self.decide_action()
            if action == 'goal reached':
                print(f"Goal reached at position {self.position}")
                break
            self.act(action)
            print(f"Action: {action}, New Position: {self.position}")

# Example usage
goal_position = [2, 3] 
agent = GoalBasedAgent(goal_position)
agent.run()