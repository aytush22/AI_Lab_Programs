graph = {
   '1' : ['3','7'],
  '3' : ['5', '4'],
  '7' : ['9'],
  '5' : [],
  '4' : ['9'],
  '9' : []
}

visited = [] 
queue = []    

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:        
    m = queue.pop(0) 
    print (m, end = " ") 

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)
start_node = '1'
print("Breadth-First Search: ")
bfs(visited, graph, start_node)  