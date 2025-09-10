import time
import psutil
import os
from collections import defaultdict

class SimpleMonitor:
    def __init__(self):
        self.stats = {
            'encoder': {'time': 0, 'memory': 0, 'calls': 0},
            'random_walk': {'time': 0, 'memory': 0, 'calls': 0},
            'embeddings': {'time': 0, 'memory': 0, 'calls': 0},
            'loss': {'time': 0, 'memory': 0, 'calls': 0},
            'kmeans': {'time': 0, 'memory': 0, 'calls': 0},
            'preprocessing': {'time': 0, 'memory': 0, 'calls': 0}
        }
        self.process = psutil.Process(os.getpid())
    
    def get_memory_mb(self):
        return self.process.memory_info().rss / 1024 / 1024
    
    def start_timer(self, component):
        self.start_time = time.time()
        self.start_memory = self.get_memory_mb()
        self.current_component = component
    
    def end_timer(self):
        if hasattr(self, 'current_component'):
            elapsed = time.time() - self.start_time
            memory_diff = self.get_memory_mb() - self.start_memory
            
            self.stats[self.current_component]['time'] += elapsed
            self.stats[self.current_component]['memory'] += memory_diff
            self.stats[self.current_component]['calls'] += 1
    
    def print_stats(self):
        print("\n" + "="*60)
        print("COMPONENT PERFORMANCE STATISTICS")
        print("="*60)
        
        for component, data in self.stats.items():
            if data['calls'] > 0:
                avg_time = data['time'] / data['calls']
                avg_memory = data['memory'] / data['calls']
                print(f"{component.upper()}:")
                print(f"  Total Time: {data['time']:.3f}s")
                print(f"  Avg Time: {avg_time:.3f}s")
                print(f"  Total Memory: {data['memory']:.1f}MB")
                print(f"  Avg Memory: {avg_memory:.1f}MB")
                print(f"  Calls: {data['calls']}")
                print()

# Global monitor
monitor = SimpleMonitor()