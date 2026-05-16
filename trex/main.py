import time
import threading
import cv2
import mss
import numpy as np
import pyautogui

DEBUG_WINDOW = True

MIN_HEIGHT = 10
MIN_AREA   = 60

CACTUS_SMALL_MAX_H  = 35
CACTUS_MEDIUM_MAX_H = 55
GROUP_WIDE_THRESH   = 70

JUMP_PROFILES = {
    "cactus_small":      (0.95, 0.10),
    "cactus_medium":     (1.00, 0.15),
    "cactus_large":      (1.05, 0.22),
    "cactus_group_wide": (0.80, 0.27),
    "bird_lo_jump":      (1.00, 0.13),
}

BASE_JUMP_DIST   = 90
SPEED_JUMP_COEFF = 3.5
SPEED_JUMP_QUAD  = 0.18
MIN_JUMP_DIST    = 80
MAX_JUMP_DIST    = 500

SPEED_INIT = 10.0
MERGE_GAP  = 80

BIRD_HI_MIN_MULT  = 1.3
BIRD_MD_MIN_MULT  = 0.55
BIRD_MAX_GAP_MULT = 2.5

UI_FILTER_MIN_H = 35
UI_FILTER_RATIO = 0.80

GROUND_GAP_THRESH   = 8
JUMP_AIRBORNE_EXTRA = 0.40

BIRD_DUCK_DIST     = 400
DUCK_HOLD_MIN      = 0.20
DUCK_HOLD_MAX      = 1.10
DUCK_HOLD_COEFF    = 1.5
AFTER_DUCK_LOCKOUT = 0.08


class SpeedEstimator:
    SPEED_START = 6.0
    SPEED_RATE  = 0.008

    def __init__(self):
        self._start_time = None

    def reset(self):
        self._start_time = None

    def update(self, gray=None, dino_right_x=None):
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elapsed = time.perf_counter() - self._start_time
        return self.speed_at(elapsed)

    @staticmethod
    def speed_at(elapsed_sec: float) -> float:
        s = SpeedEstimator.SPEED_START + SpeedEstimator.SPEED_RATE * elapsed_sec * 60
        return float(min(s, 35.0))

    @property
    def speed(self) -> float:
        return self.update()


def find_dino_bbox(gray_strip):
    h, w = gray_strip.shape
    left_half = gray_strip[:, :w // 2].copy()
    _, bw = cv2.threshold(left_half, 100, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for cnt in cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if abs((y + ch) - h) <= 20 and 20 <= ch <= 120 and 20 <= cw <= 150:
            if cw * ch > best_score:
                best_score = cw * ch
                best = (x, y, cw, ch)
    return best


def auto_detect_game_zone(sct):
    mon  = sct.monitors[1]
    gray = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2GRAY)
    h_full, w_full = gray.shape

    ground_y, best_dark = None, 0
    for y in range(h_full // 3, h_full - 20):
        dark_count = int(np.sum(gray[y, w_full // 4:3 * w_full // 4] < 100))
        if dark_count > best_dark and dark_count > w_full // 8:
            best_dark, ground_y = dark_count, y

    if not ground_y:
        return None

    STRIP_H   = 130
    strip_top = max(0, ground_y - STRIP_H)
    strip     = gray[strip_top: ground_y + 5, :]
    bbox      = find_dino_bbox(strip)
    if bbox is None:
        return None

    dx, dy, dw, dh = bbox
    return {
        "region": {
            "left": 0, "top": strip_top,
            "width": mon["width"], "height": STRIP_H + 10, "mon": 1,
        },
        "dino_right_x": dx + dw,
        "ground_y":     ground_y - strip_top,
        "dino_bbox":    (dx, dy, dw, dh),
        "dino_h":       dh,
    }


def wait_for_game(sct):
    print("[AUTO] Ищу игру... Откройте браузер.")
    t0 = time.time()
    while time.time() - t0 < 30:
        r = auto_detect_game_zone(sct)
        if r:
            print(f"[AUTO] dino_right_x={r['dino_right_x']}  "
                  f"ground_y={r['ground_y']}  dino_h={r['dino_h']}")
            return r
        time.sleep(0.5)
    raise RuntimeError("Игра не найдена!")


def classify_cactus(w, h):
    if w >= GROUP_WIDE_THRESH:   return "cactus_group_wide"
    if h <= CACTUS_SMALL_MAX_H:  return "cactus_small"
    if h <= CACTUS_MEDIUM_MAX_H: return "cactus_medium"
    return "cactus_large"


def classify_bird(gap: int, dino_h: int) -> str:
    if gap >= dino_h * BIRD_HI_MIN_MULT:
        return "bird_hi"
    if gap >= dino_h * BIRD_MD_MIN_MULT:
        return "bird_lo_duck"
    return "bird_lo_jump"


def is_ui(bw: int, bh: int) -> bool:
    if bh <= UI_FILTER_MIN_H:
        return False
    return (bw / max(bh, 1)) < UI_FILTER_RATIO


def detect_obstacles(gray, dino_x, ground_y, thr, dino_h):
    frame_w = gray.shape[1]

    _, bin_img = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    bin_img[ground_y:, :]               = 0
    bin_img[:max(0, ground_y - 115), :] = 0
    bin_img[:, :dino_x + 20]            = 0
    bin_img[:, int(frame_w * 0.88):]    = 0

    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ground_boxes = []
    air_boxes    = []
    bird_max_gap = int(dino_h * BIRD_MAX_GAP_MULT)

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < MIN_HEIGHT:    continue
        if bw * bh < MIN_AREA: continue

        gap = ground_y - (y + bh)

        if gap <= GROUND_GAP_THRESH:
            if bh <= 110:
                ground_boxes.append((x, y, bw, bh))
        else:
            if gap > bird_max_gap: continue
            if is_ui(bw, bh):     continue
            air_boxes.append((x, y, bw, bh, gap))

    res = []

    if ground_boxes:
        ground_boxes.sort(key=lambda b: b[0])
        merged = [list(ground_boxes[0])]
        for bx, by, bw, bh in ground_boxes[1:]:
            px, py, pw, ph = merged[-1]
            if bx <= px + pw + MERGE_GAP:
                merged[-1] = [
                    px, min(py, by),
                    max(px + pw, bx + bw) - px,
                    max(py + ph, by + bh) - min(py, by),
                ]
            else:
                merged.append([bx, by, bw, bh])

        for x, y, cw, ch in merged:
            dist = x - dino_x
            if dist <= 0: continue
            gap = ground_y - (y + ch)
            res.append((dist, classify_cactus(cw, ch), cw, ch, gap))

    for x, y, bw, bh, gap in air_boxes:
        dist = x - dino_x
        if dist <= 0: continue
        kind = classify_bird(gap, dino_h)
        res.append((dist, kind, bw, bh, gap))

    return sorted(res, key=lambda r: r[0])


def calc_duck_hold(dist: int, speed: float) -> float:
    px_per_sec = max(speed * 100.0, 200.0)
    return float(np.clip(
        dist / px_per_sec * DUCK_HOLD_COEFF,
        DUCK_HOLD_MIN, DUCK_HOLD_MAX
    ))


def is_game_over(gray, ground_y):
    y1 = max(0, ground_y - 80)
    y2 = max(1, ground_y - 25)
    if y2 <= y1:
        return False
    strip = gray[y1:y2, gray.shape[1] // 3: 2 * gray.shape[1] // 3]
    if strip.size == 0:
        return False
    _, bw = cv2.threshold(strip, 140, 255, cv2.THRESH_BINARY_INV)
    return (np.sum(bw) / (bw.size * 255 + 1e-8)) > 0.12


class SharedState:
    def __init__(self):
        self._lock  = threading.Lock()
        self.frame  = None
        self.status = ""
        self.quit   = False


def bot_thread(shared: SharedState):
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0

    with mss.mss() as sct:
        zone   = wait_for_game(sct)
        reg    = zone["region"]
        dino_x = zone["dino_right_x"]
        gr_y   = zone["ground_y"]
        dino_h = zone["dino_h"]

        print(f"[CAL] dino_h={dino_h}  "
              f"BIRD_HI_MULT={BIRD_HI_MIN_MULT}*{dino_h}={int(BIRD_HI_MIN_MULT*dino_h)}px  "
              f"BIRD_MD_MULT={BIRD_MD_MIN_MULT}*{dino_h}={int(BIRD_MD_MIN_MULT*dino_h)}px")
        print("[OK] Старт через 3 сек...")
        time.sleep(3)
        pyautogui.press("space")
        time.sleep(1)

        gray0 = cv2.cvtColor(np.array(sct.grab(reg)), cv2.COLOR_BGRA2GRAY)
        hb    = cv2.calcHist([gray0], [0], None, [256], [0, 256]).flatten()
        hb[:150] = 0
        thr = max(int(np.argmax(hb)) - 40, 100)
        print(f"[CAL] thr={thr}")

        speed_est = SpeedEstimator()

        space_release_time = 0.0
        landing_time       = 0.0
        down_release_time  = 0.0
        next_action_time   = 0.0
        is_space_down      = False
        is_down_down       = False
        game_over_cd       = 0.0
        game_over_streak   = 0
        GAME_OVER_CONFIRM  = 3
        action_str         = "—"

        while not shared.quit:
            t0  = time.perf_counter()
            now = t0

            gray  = cv2.cvtColor(np.array(sct.grab(reg)), cv2.COLOR_BGRA2GRAY)
            speed = speed_est.update(gray, dino_x)

            if is_space_down and now >= space_release_time:
                pyautogui.keyUp("space")
                is_space_down = False

            if is_down_down and now >= down_release_time:
                pyautogui.keyUp("down")
                is_down_down = False
                next_action_time = max(next_action_time, now + AFTER_DUCK_LOCKOUT)

            if now > game_over_cd:
                if is_game_over(gray, gr_y):
                    game_over_streak += 1
                else:
                    game_over_streak = 0

                if game_over_streak >= GAME_OVER_CONFIRM:
                    game_over_streak = 0
                    if is_space_down:
                        pyautogui.keyUp("space"); is_space_down = False
                    if is_down_down:
                        pyautogui.keyUp("down");  is_down_down  = False
                    print("[BOT] GAME OVER — перезапуск")
                    time.sleep(0.5)
                    pyautogui.press("space")
                    time.sleep(1.0)
                    game_over_cd     = now + 3.0
                    next_action_time = now + 3.0
                    landing_time     = 0.0
                    speed_est.reset()
                    action_str       = "RESTART"
                    continue
            else:
                game_over_streak = 0

            obs = detect_obstacles(gray, dino_x, gr_y, thr, dino_h)
            base_dist = (BASE_JUMP_DIST
                         + SPEED_JUMP_COEFF * speed
                         + SPEED_JUMP_QUAD  * speed ** 2)
            base_dist = max(base_dist, MIN_JUMP_DIST)

            is_airborne = now < landing_time

            duck_target = None
            if not is_airborne:
                for item in obs:
                    d, k, ow, oh, gap = item
                    if d > BIRD_DUCK_DIST: break
                    if k == "bird_lo_duck":
                        duck_target = item
                        break

            if duck_target and not is_space_down:
                d, k, ow, oh, gap = duck_target
                hold_duck = calc_duck_hold(d, speed)
                if is_down_down:
                    new_rel = now + hold_duck
                    if new_rel > down_release_time:
                        down_release_time = new_rel
                        next_action_time  = down_release_time + AFTER_DUCK_LOCKOUT
                else:
                    pyautogui.keyDown("down")
                    is_down_down      = True
                    down_release_time = now + hold_duck
                    next_action_time  = down_release_time + AFTER_DUCK_LOCKOUT
                    action_str = f"DUCK d={d} hold={hold_duck:.2f}s spd={speed:.1f}"
                    print(f"[BOT] {action_str}")

            elif obs and now >= next_action_time and not is_down_down:
                dist, kind, ow, oh, gap = obs[0]

                if kind != "bird_hi" and kind in JUMP_PROFILES:
                    mult, hold   = JUMP_PROFILES[kind]
                    trigger_dist = int(np.clip(base_dist * mult, MIN_JUMP_DIST, MAX_JUMP_DIST))
                    if dist <= trigger_dist and not is_space_down:
                        pyautogui.keyDown("space")
                        is_space_down      = True
                        space_release_time = now + hold
                        landing_time       = now + hold + JUMP_AIRBORNE_EXTRA
                        next_action_time   = landing_time
                        action_str = (f"JUMP {kind} d={dist} trig={trigger_dist} "
                                      f"spd={speed:.1f}")
                        print(f"[BOT] {action_str}")

            if DEBUG_WINDOW:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                h, w = vis.shape[:2]

                cv2.line(vis, (0, gr_y), (w, gr_y), (0, 220, 0), 1)
                cv2.line(vis, (dino_x, 0), (dino_x, h), (255, 80, 0), 2)

                bx = int(dino_x + base_dist)
                if bx < w:
                    cv2.line(vis, (bx, 0), (bx, h), (180, 180, 180), 1)

                dlx = dino_x + BIRD_DUCK_DIST
                if dlx < w:
                    cv2.line(vis, (dlx, 0), (dlx, h), (0, 165, 255), 1)

                zones = [
                    (int(dino_h * BIRD_HI_MIN_MULT), (0, 220, 220),
                     f"bird_hi≥{int(dino_h*BIRD_HI_MIN_MULT)}"),
                    (int(dino_h * BIRD_MD_MIN_MULT), (0, 140, 255),
                     f"duck≥{int(dino_h*BIRD_MD_MIN_MULT)}"),
                ]
                for gap_val, color, label in zones:
                    ly = gr_y - gap_val
                    if 0 <= ly < h:
                        cv2.line(vis, (0, ly), (w, ly), color, 1)
                        cv2.putText(vis, label, (4, max(ly - 2, 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

                KIND_COLOR = {
                    "bird_hi":           (0,   220, 220),
                    "bird_lo_duck":      (0,   140, 255),
                    "bird_lo_jump":      (50,  50,  255),
                    "cactus_small":      (0,   80,  255),
                    "cactus_medium":     (0,   80,  255),
                    "cactus_large":      (0,   80,  255),
                    "cactus_group_wide": (0,   40,  200),
                }
                for item in obs[:6]:
                    d, k, ow, oh, gap = item
                    ox    = dino_x + d
                    top_y = gr_y - oh
                    color = KIND_COLOR.get(k, (0, 80, 255))
                    cv2.rectangle(vis, (ox, top_y), (ox + ow, gr_y), color, 2)
                    cv2.putText(vis, f"{k} d={d}", (ox, max(top_y - 3, 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

                state_str = f"SPD:{speed:.1f} base={int(base_dist)} | "
                if is_space_down:
                    state_str += f"[JUMP {max(space_release_time-now,0):.2f}s] "
                elif is_airborne:
                    state_str += f"[AIR {max(landing_time-now,0):.2f}s] "
                elif is_down_down:
                    state_str += f"[DUCK {max(down_release_time-now,0):.2f}s] "
                elif now < next_action_time:
                    state_str += f"[WAIT {next_action_time-now:.2f}s] "
                else:
                    state_str += "[READY] "
                state_str += action_str

                cv2.putText(vis, state_str, (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

                with shared._lock:
                    shared.frame  = vis
                    shared.status = state_str

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.001, 0.010 - elapsed))

    if is_space_down: pyautogui.keyUp("space")
    if is_down_down:  pyautogui.keyUp("down")


def main():
    shared = SharedState()
    worker = threading.Thread(target=bot_thread, args=(shared,), daemon=True)
    worker.start()

    if DEBUG_WINDOW:
        print("[GUI] Q — выход")
        while worker.is_alive():
            with shared._lock:
                frame = shared.frame
            if frame is not None:
                cv2.imshow("Dino", frame)
            if cv2.waitKey(8) & 0xFF == ord('q'):
                shared.quit = True
                break
        cv2.destroyAllWindows()
    else:
        worker.join()


if __name__ == "__main__":
    main()