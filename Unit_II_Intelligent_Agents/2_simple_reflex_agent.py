class SimpleReflexAgent:
    def __init__(self):
        self.rules = {
            'dirty': self.clean,
            'clean': self.no_op
        }
        self.location = None

    def perceive(self, environment):
        self.location, status = environment
        return status

    def act(self, status):
        action = self.rules.get(status, self.no_op)
        return action()

    def clean(self):
        print(f"Agent cleaned the room at location {self.location}.")
        return "clean"

    def no_op(self):
        print(f"Agent did nothing in the room at location {self.location}.")
        return "no_op"

# Example usage
if __name__ == "__main__":
    environment_state = ("A", "dirty") 
    agent = SimpleReflexAgent()
    agent.location = "A" 

    while True:
        percept = agent.perceive(environment_state)
        action_result = agent.act(percept)

        if action_result == "clean":
            environment_state = (agent.location, "clean")  
        else:
            break

    print("Final state of the environment:", environment_state)
