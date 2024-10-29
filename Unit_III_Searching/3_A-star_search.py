class AStarSearch:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.open_list = [start]
        self.closed_list = set()
        self.g_scores = {start: 0}
        self.f_scores = {start: self.heuristic(start)}
        self.came_from = {}

    def heuristic(self, node):
        dx = abs(node[0] - self.goal[0])
        dy = abs(node[1] - self.goal[1])
        return dx + dy

    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]

    def get_neighbors(self, node):
        x, y = node
        neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]) and self.grid[nx][ny] == 0]

    def search(self):
        while self.open_list:
            current = min(self.open_list, key=lambda node: self.f_scores.get(node, float('inf')))
            if current == self.goal:
                return self.reconstruct_path(current)

            self.open_list.remove(current)
            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue
                
                tentative_g_score = self.g_scores[current] + 1
                if neighbor not in self.open_list:
                    self.open_list.append(neighbor)
                elif tentative_g_score >= self.g_scores.get(neighbor, float('inf')):
                    continue

                self.came_from[neighbor] = current
                self.g_scores[neighbor] = tentative_g_score
                self.f_scores[neighbor] = tentative_g_score + self.heuristic(neighbor)

        return None 


# Example usage
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (4, 4)   

    a_star = AStarSearch(grid, start, goal)
    path = a_star.search()

    if path:
        print("Path found:", path)
    else:
        print("No path found.")
