import heapq
from database import Database

class Navigator:
    def __init__(self, database):
        self.db = database
        self.current_path = []
        self.current_step = 0
    
    def plan_route(self, location_name, current_node_id, destination_node_id=None):
        """
        Plan route using A* algorithm
        If destination_node_id is None, route to end of location
        """
        nodes, edges = self.db.get_navigation_graph(location_name)
        
        if not nodes:
            return None
        
        # If no destination specified, go to last node
        if destination_node_id is None:
            destination_node_id = max(nodes.keys(), 
                                     key=lambda k: nodes[k]['position'])
        
        # Build adjacency list
        graph = {node_id: [] for node_id in nodes}
        for from_node, to_node, distance in edges:
            graph[from_node].append((to_node, distance))
        
        # A* algorithm
        path = self._astar(graph, nodes, current_node_id, destination_node_id)
        
        if path:
            self.current_path = path
            self.current_step = 0
            return path
        return None
    
    def _astar(self, graph, nodes, start, goal):
        """A* pathfinding algorithm"""
        # Priority queue: (f_score, node_id)
        open_set = [(0, start)]
        came_from = {}
        g_score = {node: float('inf') for node in graph}
        g_score[start] = 0
        
        f_score = {node: float('inf') for node in graph}
        f_score[start] = self._heuristic(nodes, start, goal)
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for neighbor, distance in graph[current]:
                tentative_g = g_score[current] + distance
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(nodes, neighbor, goal)
                    
                    if neighbor not in [node for _, node in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _heuristic(self, nodes, node1, node2):
        """Heuristic function: position difference (Manhattan distance)"""
        pos1 = nodes[node1]['position']
        pos2 = nodes[node2]['position']
        return abs(pos2 - pos1) * 0.7  # 0.7m per position unit
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from A* came_from dict"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def get_next_instruction(self, nodes):
        """Get next navigation instruction"""
        if not self.current_path or self.current_step >= len(self.current_path):
            return "Destination reached"
        
        current_node_id = self.current_path[self.current_step]
        
        if self.current_step == len(self.current_path) - 1:
            return "Destination reached"
        
        next_node_id = self.current_path[self.current_step + 1]
        
        # Calculate distance to next waypoint
        # Each position unit is approximately 0.7 meters
        distance = 0.7
        
        return f"Continue forward {distance:.1f} meters"
    
    def advance_step(self):
        """Move to next step in path"""
        if self.current_step < len(self.current_path) - 1:
            self.current_step += 1
            return True
        return False
