# dataset_collection/collect_traj_data.py
#
# Virtual trajectory dataset collection for Network I.
# Drives Unity via TCP to grow the soft robot through randomised bending
# configurations and saves all tip trajectories to coordinates_new.json.
#
# Pair with: SoftRobotDataCollector.cs
#
# Usage:
#   1. Open Unity, attach SoftRobotDataCollector.cs, press Play.
#   2. python dataset_collection/collect_traj_data.py

import json, math, os, platform, random, select, signal, socket, sys, time
from datetime import datetime
import numpy as np

_OS = platform.system()


def get_save_dir() -> str:
    user = os.getlogin()
    if _OS == "Windows":
        if user == "ayaka": return r"C:\Users\ayaka\PycharmProjects\pythonProject1\diffusion"
        if user == "B610":  return r"C:\Users\B610\PycharmProjects\PythonProject\diffusion"
    if _OS == "Linux":  return "/media/dockang/D/haoran/pythonProject1/diffusion"
    return "/Users/ayaka/Desktop/pythonProject1/diffusion"


class UnityClient:
    """TCP client for SoftRobotDataCollector.cs."""

    def __init__(self, host="localhost", port=5555):
        self.host, self.port = host, port
        self._sock = None; self._buf = ""; self._seq = 0; self._last_state = None

    def connect(self, timeout=30.0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        if self._recv_line(timeout) != "READY":
            raise RuntimeError("Handshake failed")
        print(f"[Client] Connected and handshake OK")

    def disconnect(self):
        if self._sock:
            try: self._sock.close()
            except: pass
            self._sock = None

    def reset(self):
        self._last_state = None
        self._send("START", "START_DONE", 15.0)
        return self._last_state

    def step(self, angle: float):
        self._send(f"{angle:.6f}", "SEGMENT_DONE", 10.0)
        return self._last_state

    def end_episode(self):
        self._send("END", "END_DONE", 20.0)

    @property
    def last_state(self): return self._last_state

    def _send(self, payload, ack, timeout):
        self._seq += 1
        self._sock.sendall(f"SEQ:{self._seq}:{payload}\n".encode())
        deadline = time.monotonic() + timeout
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0: raise TimeoutError(f"Waiting for {ack}")
            line = self._recv_line(remain)
            if line.startswith("STATE:"): self._parse_state(line)
            elif line.startswith(f"ACK:{self._seq}:"):
                if ack in line: return
                raise RuntimeError(f"Unexpected: {line}")

    def _recv_line(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                return line.strip()
            remain = deadline - time.monotonic()
            if remain <= 0: raise TimeoutError("recv_line timeout")
            rd, _, _ = select.select([self._sock], [], [], min(remain, 0.5))
            if rd:
                data = self._sock.recv(4096)
                if not data: raise ConnectionError("Unity disconnected")
                self._buf += data.decode()

    def _parse_state(self, line):
        try:
            x, z, rot = map(float, line[6:].split(","))
            self._last_state = (x, z, rot)
        except: pass


def sample_config(sim_idx):
    steps = random.randint(38, 50)
    if sim_idx < 100:
        t1 = random.uniform(8, 18);  t2 = t3 = 0.0
        p1 = random.randint(9, 15);  p2, p3 = random.randint(0, 1), random.randint(2, 3)
    elif sim_idx < 200:
        t1 = random.uniform(-25, -15); t2 = t3 = 0.0
        p1 = random.randint(19, 25);   p2, p3 = random.randint(0, 1), random.randint(2, 3)
    elif sim_idx < 300:
        t1 = random.uniform(8, 18); t2 = random.uniform(-27, -17); t3 = 0.0
        p1 = random.randint(8, 14); p2 = random.randint(19, 25);   p3 = random.randint(2, 3)
    else:
        t1 = random.uniform(-20, 20); t2 = random.uniform(-20, 20); t3 = random.uniform(-20, 20)
        p1 = random.randint(5, 12);   p2 = random.randint(13, 25);  p3 = random.randint(26, 35)
    return dict(theta1=t1, theta2=t2, theta3=t3, pos1=p1, pos2=p2, pos3=p3, steps=steps)


def config_to_angles(cfg):
    angles = [0.0] * cfg["steps"]
    for pos, theta in [(cfg["pos1"], cfg["theta1"]), (cfg["pos2"], cfg["theta2"]),
                       (cfg["pos3"], cfg["theta3"])]:
        if 0 <= pos < cfg["steps"]: angles[pos] = theta
    return angles


TARGET_POINTS = [(1.599, 4.317), (-1.38, 4.313), (-0.631, 4.302), (-0.893, 4.263), (-1.078, 4.270)]
SUCCESS_RADIUS = 0.2


def reached_target(x, z):
    return any(math.hypot(tx - x, tz - z) < SUCCESS_RADIUS for tx, tz in TARGET_POINTS)


def run_episode(client, sim_idx):
    cfg = sample_config(sim_idx); angles = config_to_angles(cfg)
    print(f"\n[Episode {sim_idx+1}]  steps={cfg['steps']}  "
          f"θ1={cfg['theta1']:.1f}°@{cfg['pos1']}  θ2={cfg['theta2']:.1f}°@{cfg['pos2']}")
    client.reset(); time.sleep(0.1)
    reached = False
    for idx, angle in enumerate(angles):
        state = client.step(angle); time.sleep(0.05)
        if state is None: continue
        x, z, _ = state
        if idx % 5 == 0: print(f"  step {idx:3d}  angle={angle:+6.1f}°  pos=({x:.3f},{z:.3f})")
        if idx > 0 and reached_target(x, z):
            reached = True; break
    client.end_episode(); time.sleep(0.5)
    return {"reached": reached, "config": cfg}


def main():
    random.seed(42); np.random.seed(42)
    NUM_EPISODES = 600; save_dir = get_save_dir()
    client = UnityClient(); log = []

    def on_exit(sig, frame):
        _save_log(log, save_dir); client.disconnect(); sys.exit(0)
    signal.signal(signal.SIGINT, on_exit); signal.signal(signal.SIGTERM, on_exit)

    print(f"[Main] Connecting to Unity localhost:5555 ..."); client.connect(timeout=60.0)
    t0, success = time.monotonic(), 0
    try:
        for idx in range(NUM_EPISODES):
            result = run_episode(client, idx); log.append(result)
            if result["reached"]: success += 1
            print(f"  {idx+1}/{NUM_EPISODES}  reach={success/(idx+1)*100:.1f}%  "
                  f"elapsed={time.monotonic()-t0:.0f}s")
            if (idx + 1) % 50 == 0: _save_log(log, save_dir)
            time.sleep(1.0)
    finally:
        _save_log(log, save_dir); client.disconnect()
        print(f"\n[Main] Done  episodes={len(log)}  "
              f"reach={success/max(len(log),1)*100:.1f}%  time={time.monotonic()-t0:.0f}s")


def _save_log(log, directory):
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(directory, f"collection_log_{ts}.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)


if __name__ == "__main__":
    main()
