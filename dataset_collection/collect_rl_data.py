# dataset_collection/collect_rl_data.py
#
# Offline RL dataset collection for Network II.
# Drives Unity via TCP through random bending + tendon rotation episodes and
# records 6-D state vectors for both the original and comparison (noisy) scenes.
#
# Pair with: SoftRobotRLDataCollector.cs
#
# Usage:
#   1. Open Unity, attach SoftRobotRLDataCollector.cs, press Play.
#   2. python dataset_collection/collect_rl_data.py

import json, os, platform, random, signal, socket, sys, time, threading
from datetime import datetime
from queue import Queue

import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

_OS = platform.system()
TCP_HOST, TCP_PORT = "localhost", 5555

# Current 6-D state (thread-shared)
_state_lock = threading.Lock()
_current_state = [0.0] * 6   # [length, ex, ez, angle, ex_comp, ez_comp]
_seq = 0


def get_save_dir() -> str:
    user = os.getlogin()
    if _OS == "Windows":
        if user == "ayaka": return r"C:\Users\ayaka\PycharmProjects\pythonProject1\diffusion"
        if user == "B610":  return r"C:\Users\B610\PycharmProjects\PythonProject\diffusion"
    if _OS == "Linux":  return "/media/dockang/D/haoran/pythonProject1/diffusion"
    return "/Users/ayaka/Desktop/pythonProject1/diffusion"


# ── TCP client ────────────────────────────────────────────────────────────────

class TCPClient:
    def __init__(self, host=TCP_HOST, port=TCP_PORT):
        self.host, self.port = host, port
        self.socket = None; self.connected = False
        self._queue = Queue(); self._stop = threading.Event()

    def connect(self, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5); self.socket.connect((self.host, self.port))
                self.socket.settimeout(None); self.connected = True
                self._stop.clear()
                threading.Thread(target=self._recv_loop, daemon=True).start()
                print(f"[Client] Connected to {self.host}:{self.port}")
                return True
            except: time.sleep(1)
        return False

    def disconnect(self):
        self.connected = False; self._stop.set()
        try: self.socket.close()
        except: pass

    def send(self, msg):
        if not self.connected: return False
        try: self.socket.sendall((msg + "\n").encode()); return True
        except: self.connected = False; return False

    def wait(self, prefix=None, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self._queue.empty():
                msg = self._queue.get_nowait()
                if prefix is None or msg.startswith(prefix): return msg
                self._dispatch(msg)
            time.sleep(0.01)
        return None

    def _recv_loop(self):
        buf = ""
        while not self._stop.is_set() and self.connected:
            try:
                data = self.socket.recv(4096)
                if not data: break
                buf += data.decode()
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.strip(): self._queue.put(line.strip())
            except: break

    def _dispatch(self, msg):
        global _current_state
        if msg.startswith("STATE_VECTOR:"):
            parts = msg[13:].split(",")
            if len(parts) >= 6:
                try:
                    with _state_lock:
                        _current_state = [float(p) for p in parts[:6]]
                except: pass


def send_cmd(client, command, timeout=10):
    global _seq
    _seq += 1
    seq = _seq
    client.send(f"SEQ:{seq}:{command}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = client.wait(f"ACK:{seq}:", timeout=0.1)
        if msg:
            if "ERROR" in msg or "INSUFFICIENT" in msg:
                print(f"[WARN] {msg}"); return False
            return True
    print(f"[WARN] Timeout waiting for ACK#{seq}: {command}")
    return False


# ── Action generation ─────────────────────────────────────────────────────────

def generate_actions(sim_idx):
    """
    Generate random bending parameters.

    sim_idx < 2000: constrained range matching the physical prototype scenarios.
    sim_idx ≥ 2000: wider random range for generalisation.
    """
    if sim_idx < 2000:
        t1 = random.uniform(14, 18); t2 = random.uniform(12, 16); t3 = 0.0
        p1 = random.randint(25, 29); p2 = 34; p3 = random.randint(2, 3)
    else:
        t1 = random.uniform(-15, 15); t2 = random.uniform(-15, 15); t3 = random.uniform(-15, 15)
        p1 = random.randint(5, 10);   p2 = random.randint(15, 20); p3 = random.randint(25, 30)
    return [t1, t2, t3, p1, p2, p3]


def generate_rotation_sequence(seq_len):
    """Sparse rotation sequence: non-zero only at steps 50 and 60."""
    rots = np.zeros(seq_len)
    for pos in [50, 60]:
        if pos < seq_len:
            u = np.random.rand()
            rots[pos] = (u - 0.5) * 60   # U(−30, 30)
    return rots


# ── Single episode ────────────────────────────────────────────────────────────

def run_episode(client, seq_len, action_params, rotation_seq):
    """
    Execute one growth + rotation episode.
    Returns (actions_sent, rotation_params_list).
    """
    global _current_state

    actions_sent = []
    rot_log = []

    # START
    actions_sent.append("START")
    if not send_cmd(client, "START", timeout=10): return actions_sent, rot_log
    time.sleep(0.2)

    t1, t2, t3 = action_params[0], action_params[1], action_params[2]
    p1, p2, p3 = int(action_params[3]), int(action_params[4]), int(action_params[5])

    for idx in range(seq_len):
        # Choose bending angle for this step
        if   idx == p1: angle = t1
        elif idx == p2: angle = t2
        elif idx == p3: angle = t3
        else:           angle = 0.0

        actions_sent.append(angle)
        if not send_cmd(client, str(angle), timeout=8): break
        time.sleep(0.15)

        # Rotation (sparse)
        if idx < len(rotation_seq) and abs(rotation_seq[idx]) > 3:
            pivot = idx - 10
            rot_cmd = f"ROTATE:{rotation_seq[idx]:.4f}:{pivot}"
            actions_sent.append(rot_cmd)
            success = send_cmd(client, rot_cmd, timeout=12)
            rot_log.append({"step": idx, "angle": round(rotation_seq[idx], 2),
                             "pivot": pivot, "success": success})
            time.sleep(0.6)

        if idx % 10 == 0:
            with _state_lock: s = list(_current_state)
            print(f"  step {idx:3d}  angle={angle:+5.1f}°  "
                  f"end=({s[1]:.3f},{s[2]:.3f})  comp=({s[4]:.3f},{s[5]:.3f})")

    actions_sent.append("END")
    send_cmd(client, "END", timeout=15)
    time.sleep(1.0)
    return actions_sent, rot_log


def save_episode(file_path, episode_num, actions, rotations):
    entry = {"Episode": episode_num, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "Actions": {"sequence": actions, "rotations": rotations}}
    data = []
    if os.path.exists(file_path):
        with open(file_path) as f: data = json.load(f)
    data.append(entry)
    with open(file_path, "w") as f: json.dump(data, f, indent=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42); np.random.seed(42)
    NUM_EPISODES = 2000; save_dir = get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"rl_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    client = TCPClient()

    def on_exit(sig, frame):
        client.disconnect(); sys.exit(0)
    signal.signal(signal.SIGINT, on_exit); signal.signal(signal.SIGTERM, on_exit)

    print(f"[Main] Connecting to Unity {TCP_HOST}:{TCP_PORT} ...")
    if not client.connect(30):
        print("[Main] Cannot connect to Unity"); return

    # Wait for READY
    msg = client.wait("READY", timeout=30)
    if msg != "READY": print("[Main] Unity not ready"); client.disconnect(); return
    print("[Main] Unity ready")

    seq_len = random.randint(62, 70)
    print(f"[Main] Sequence length: {seq_len}")
    t0 = time.monotonic()

    try:
        for sim_idx in range(NUM_EPISODES):
            print(f"\n{'='*55}\nEpisode {sim_idx+1}/{NUM_EPISODES}")
            actions  = generate_actions(sim_idx)
            rot_seq  = generate_rotation_sequence(seq_len)
            print(f"  actions=[{actions[0]:.1f},{actions[1]:.1f},{actions[2]:.1f}]  "
                  f"pos=[{int(actions[3])},{int(actions[4])},{int(actions[5])}]  "
                  f"rotations={np.count_nonzero(rot_seq)}")

            acts, rots = run_episode(client, seq_len, actions, rot_seq)
            save_episode(file_path, sim_idx + 1, acts, rots)
            print(f"  Saved. Rotations executed: {len(rots)}")
            time.sleep(1.0)
    finally:
        client.disconnect()
        print(f"\n[Main] Done  elapsed={time.monotonic()-t0:.0f}s")


if __name__ == "__main__":
    main()
