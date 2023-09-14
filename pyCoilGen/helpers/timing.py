import logging
import time

log = logging.getLogger(__name__)

class Timing:
    def __init__(self):
        self.start_times = []
    
    def start(self):
        self.start_times.append(time.time())
    
    def stop(self):
        if self.start_times:
            elapsed_time = time.time() - self.start_times.pop()
            if len(self.start_times) == 0:
                log.info(f"Total elapsed time: {elapsed_time:.6f} seconds")
            else:
                log.debug(f"Elapsed time: {elapsed_time:.6f} seconds")
        else:
            log.warning("No active timer to stop.")

# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    timer = Timing()

    timer.start()  # Start total timer
    timer.start()  # Start task 1 timer
    time.sleep(1)   # Simulate task 1
    timer.stop()   # Measure time of task 1 and write time to log.debug

    timer.start()  # Start task 2 timer
    time.sleep(2)   # Simulate task 2
    timer.stop()   # Measure time of task 2 and write time to log.debug

    timer.stop()   # Measure and write total time to log.debug
