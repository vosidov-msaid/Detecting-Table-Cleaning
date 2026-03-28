import pandas as pd
import numpy as np
from datetime import datetime
import config

STATE_EMPTY = config.STATE_EMPTY
STATE_OCCUPIED = config.STATE_OCCUPIED
STATE_APPROACH = config.STATE_APPROACH

class TableTracker:
    def __init__(self, roi: tuple[int, int, int, int], fps: float):
        self.roi = roi
        self.fps = fps

        self.state = STATE_EMPTY
        self.empty_counter = 0

        self.events = pd.DataFrame(columns=["timestamp", "wall_time", "event", "frame_no"])

    def update(self, frame_no: int, person_in_roi: bool):
        ts = frame_no / self.fps
        wall = str(datetime.now().strftime("%H:%M:%S.%f")[:-3])

        if person_in_roi:
            self.empty_counter = 0
            if self.state == STATE_EMPTY:
                self.log_events(ts, wall, STATE_APPROACH, frame_no)
                self.state = STATE_APPROACH
            elif self.state == STATE_APPROACH:
                self.log_events(ts, wall, STATE_OCCUPIED, frame_no)
                self.state = STATE_OCCUPIED
        else:
            self.empty_counter += 1
            if (self.state in (STATE_OCCUPIED, STATE_APPROACH) and 
                    self.empty_counter >= config.EMPTY_FRAMES_NEEDED):
                self.log_events(ts, wall, STATE_EMPTY, frame_no)
                self.state = STATE_EMPTY
                self.empty_counter = 0

    def log_events(self, ts, wall, event, frame_no):
        row = pd.DataFrame([{
            "timestamp": round(ts, 2),
            "wall_time": wall,
            "event": event,
            "frame_no": frame_no,
        }])
        self.events = pd.concat([self.events, row], ignore_index=True)
        print(f"[EVENT] {event:10s} t={ts:7.2f}s frame={frame_no}")

    def stat_guests(self) -> dict:
        df = self.events

        none_response = {"avg_response_sec": None, "n_cycles": 0}

        if df.empty:
            return none_response
        
        delays = []
        empties = df[df["event"] == STATE_EMPTY]["timestamp"].tolist()
        approaches = df[df["event"] == STATE_APPROACH]["timestamp"].tolist()

        for emps in empties:
            nexts = [a for a in approaches if a > emps]
            if nexts:
                delays.append(nexts[0] - emps)

        if delays:
            return {
                "avg_response_sec": round(float(np.mean(delays)), 2),
                "min_response_sec": round(float(np.min(delays)), 2),
                "max_response_sec": round(float(np.max(delays)), 2),
                "n_cycles": len(delays),
            }

        return none_response