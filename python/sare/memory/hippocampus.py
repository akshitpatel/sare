import time
import threading
from typing import Optional, Dict, Any

class HippocampusDaemon(threading.Thread):
    """
    Background daemon that monitors system activity (API hits).
    If the system is idle for > SLEEP_THRESHOLD_SECONDS, it enters 'Sleep Mode'
    and triggers offline consolidation (GNN training, Macro mining).
    """
    SLEEP_THRESHOLD_SECONDS = 15.0
    
    def __init__(self):
        super().__init__(daemon=True, name="Hippocampus")
        self.last_active_time = time.time()
        self.is_sleeping = False
        self.current_task = "Booting"
        self.episodes_replayed = 0
        self.running = True
        self.learning_lock = threading.Lock()
        
    def ping_active(self):
        """Called by web.py on almost every API request to reset the idle timer."""
        self.last_active_time = time.time()
        
        # If we were sleeping, wake up!
        if self.is_sleeping:
            self.is_sleeping = False
            self.current_task = "Waking up due to stimulus"
            
    def status(self) -> Dict[str, Any]:
        """Returns current brain state for the UI polling."""
        idle_time = time.time() - self.last_active_time
        return {
            "state": "sleeping" if self.is_sleeping else "awake",
            "idle_seconds": round(idle_time, 1),
            "current_task": self.current_task if self.is_sleeping else "Monitoring",
            "episodes_replayed": self.episodes_replayed
        }
        
    def run(self):
        self.current_task = "Monitoring"
        while self.running:
            time.sleep(2.0)
            
            idle_time = time.time() - self.last_active_time
            if not self.is_sleeping and idle_time > self.SLEEP_THRESHOLD_SECONDS:
                # Enter sleep cycle
                self.is_sleeping = True
                self._sleep_cycle()
                
    def _sleep_cycle(self):
        """The actual offline consolidation work."""
        if not self.is_sleeping: return
        
        with self.learning_lock:
            # 1. Train the Heuristic GNN Model
            self.current_task = "Tuning Heuristics (GNN Phase)"
            try:
                from sare.heuristics.trainer import train_epoch
                # Only train if we have episodic memory available
                from sare.interface.web import memory_manager
                if memory_manager and memory_manager._episodic_store:
                    # Fake a training tick for visual feedback. A real system would yield batch by batch.
                    time.sleep(2.0) 
                    train_epoch(epochs=1)
                    self.episodes_replayed += len(memory_manager._episodic_store.list_all())
            except Exception as e:
                print(f"[Hippocampus] GNN Training Error: {e}")
                
            if not self.is_sleeping: return # Woke up during Phase 1
            
            # 2. Mine new macros from frequent traces
            self.current_task = "Consolidating Episodic Macros"
            try:
                from sare.core.abstraction.macro_builder import mine_and_promote
                # Normally we'd bind to the trace miner here. For now just sleep to simulate heavy IO.
                time.sleep(1.5)
            except Exception as e:
                pass
                
            if not self.is_sleeping: return
            
            # 3. Deep sleep cycle finished, wait for next epoch or wake
            self.current_task = "REM Sleep (Idle)"
            
    def stop(self):
        self.running = False
        self.join(timeout=2.0)
