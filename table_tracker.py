import pandas as pd

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

    def log_events(self, ts, wall, event, frame_no):
        row = pd.DataFrame([{
            "timestamp": round(ts, 2),
            "wall_time": wall,
            "event": event,
            "frame_no": frame_no,
        }])
        self.events = pd.concat([self.events, row], ignore_index=True)
        print(f"[EVENT] {event:10s} t={ts:7.2f}s frame={frame_no}")