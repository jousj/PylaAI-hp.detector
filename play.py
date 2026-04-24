import math
import random
import time
import threading

import cv2
import numpy as np
from state_finder import get_state
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info
from hpdetector import HPdetectBYjousj

brawl_stars_width, brawl_stars_height = 1920, 1080
debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
visual_debug = load_toml_as_dict("cfg/general_config.toml").get('visual_debug', 'no') == "yes"

def vlog(*args):
    if visual_debug:
        print("[DBG]", *args)
super_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['super']
gadget_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['gadget']
hypercharge_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['hypercharge']

class Movement:

    def __init__(self, window_controller):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config["unstuck_movement_delay"],
            "duration": bot_config["unstuck_movement_hold_time"],
            "toggled": False,
            "started_at": time.time(),
            "fixed": ""
        }
        self.game_mode = bot_config["gamemode_type"]
        gadget_value = bot_config["bot_uses_gadgets"]
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.super_treshold = time_config["super"]
        self.gadget_treshold = time_config["gadget"]
        self.hypercharge_treshold = time_config["hypercharge"]
        self.walls_treshold = time_config["wall_detection"]
        self.keep_walls_in_memory = self.walls_treshold <= 1
        self.last_walls_data = []
        self.keys_hold = []
        self.time_since_different_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.window_controller = window_controller
        self.TILE_SIZE = 60
        # Separate unstuck state for analog (angle-based) movement, so the
        # existing string-based unstuck for other game modes is untouched.
        self.fix_angle_state = {
            "delay_to_trigger": bot_config["unstuck_movement_delay"],
            "duration": bot_config["unstuck_movement_hold_time"],
            "toggled": False,
            "started_at": time.time(),
            "last_angle": None,
            "last_angle_change": time.time(),
            "fixed_angle": None,
        }
        
    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    @staticmethod
    def get_horizontal_move_key(direction_x, opposite=False):
        if opposite:
            return "A" if direction_x > 0 else "D"
        return "D" if direction_x > 0 else "A"

    @staticmethod
    def get_vertical_move_key(direction_y, opposite=False):
        if opposite:
            return "W" if direction_y > 0 else "S"
        return "S" if direction_y > 0 else "W"

    def attack(self, touch_up=True, touch_down=True):
        self.window_controller.press_key("M", touch_up=touch_up, touch_down=touch_down)

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press_key("H")

    def use_gadget(self):
        print("Using gadget")
        self.window_controller.press_key("G")

    def use_super(self):
        print("Using super")
        self.window_controller.press_key("E")

    @staticmethod
    def get_random_attack_key():
        random_movement = random.choice(["A", "W", "S", "D"])
        random_movement += random.choice(["A", "W", "S", "D"])
        return random_movement

    @staticmethod
    def angle_from_direction(dx: float, dy: float) -> float:
        """Return joystick angle in degrees from a direction vector.

        Uses screen coordinates: 0° = right, 90° = down, 180° = left, 270° = up.
        """
        return math.degrees(math.atan2(dy, dx)) % 360

    @staticmethod
    def angle_opposite(angle_degrees: float) -> float:
        """Return the opposite direction angle (retreat)."""
        return (angle_degrees + 180) % 360

    @staticmethod
    def reverse_movement(movement):
        # Create a translation table
        movement = movement.lower()
        translation_table = str.maketrans("wasd", "sdwa")
        return movement.translate(translation_table)

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()
        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False
                vlog("unstuck: finished")
            else:
                vlog(f"unstuck: active → {self.fix_movement_keys['fixed']}")

            return self.fix_movement_keys['fixed']

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            self.time_since_different_movement = current_time

        # print(f"Last change: {self.time_since_different_movement}", f" self.hold: {self.keys_hold}",f" c movement: {movement}")
        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            reversed_movement = self.reverse_movement(movement)

            if reversed_movement == "s":
                reversed_movement = random.choice(['aw', 'dw'])
            elif reversed_movement == "w":
                reversed_movement = random.choice(['as', 'ds'])

            """
            If reverse movement is either "w" or "s" it means the bot is stuck
            going forward or backward. This happens when it doesn't detect a wall in front
            so to go around it it could either go to the left diagonal or right
            """

            self.fix_movement_keys['fixed'] = reversed_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            vlog(f"unstuck triggered: {movement} → {reversed_movement}")
            return reversed_movement

        return movement

    def unstuck_angle_if_needed(self, angle: float, current_time: float = None) -> float:
        """Unstuck routine for analog joystick movement.

        Works by tracking how long the requested angle has stayed roughly the
        same (±20°). If it has been locked for longer than delay_to_trigger,
        we assume the bot is pressed against something the wall detector
        missed and we force a ~135° deflection for `duration` seconds to
        wiggle free.
        """
        if current_time is None:
            current_time = time.time()

        state = self.fix_angle_state

        # If an unstuck deflection is already active, keep using it until it expires
        if state['toggled']:
            if current_time - state['started_at'] > state['duration']:
                state['toggled'] = False
                state['last_angle'] = angle
                state['last_angle_change'] = current_time
                vlog("unstuck(angle): finished")
                return angle
            vlog(f"unstuck(angle): active → {state['fixed_angle']:.1f}°")
            return state['fixed_angle']

        # Track how long the commanded angle has been (roughly) unchanged
        if state['last_angle'] is None:
            state['last_angle'] = angle
            state['last_angle_change'] = current_time
            return angle

        diff = abs((angle - state['last_angle'] + 180) % 360 - 180)
        if diff > 20:
            state['last_angle'] = angle
            state['last_angle_change'] = current_time
            return angle

        # Angle has stayed similar — check if it has been long enough to trigger
        if current_time - state['last_angle_change'] > state['delay_to_trigger']:
            # Deflect by ±135° (random side) to wiggle around the obstacle
            offset = random.choice((135.0, -135.0))
            deflected = (angle + offset) % 360
            state['fixed_angle'] = deflected
            state['toggled'] = True
            state['started_at'] = current_time
            vlog(f"unstuck(angle) triggered: {angle:.1f}° → {deflected:.1f}°")
            return deflected

        return angle


class Play(Movement):

    def __init__(self, main_info_model, tile_detector_model, window_controller):
        super().__init__(window_controller)

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main_info = Detect(main_info_model, classes=['enemy', 'teammate', 'player'])
        self.tile_detector_model_classes = bot_config["wall_model_classes"]
        self.Detect_tile_detector = Detect(
            tile_detector_model,
            classes=self.tile_detector_model_classes
        )

        # ── HP-детектор byjousj (фоновый поток) ─────────────────
        _hp_model_path = "./models/byjousj.onnx"
        try:
            self.HP_detector    = HPdetectBYjousj(
                _hp_model_path,
                preferred_device=load_toml_as_dict("cfg/general_config.toml")['cpu_or_gpu']
            )
            self._hp_ok         = True
        except FileNotFoundError:
            self.HP_detector    = None
            self._hp_ok         = False

        self._hp_last_results   = []          # читается из main() без блокировки
        self._hp_frame_queue    = None        # последний кадр для обработки
        self._hp_lock           = threading.Lock()
        self._onnx_lock         = threading.Lock()  # глобальный лок для всех ONNX-вызовов

        if self._hp_ok:
            t = threading.Thread(target=self._hp_worker, daemon=True)
            t.start()
        # ────────────────────────────────────────────────────────

        self.time_since_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.time_since_hypercharge_checked = time.time()
        self.time_since_super_checked = time.time()
        self.time_since_walls_checked = 0
        self.time_since_movement_change = time.time()
        self.time_since_player_last_found = time.time()
        self.current_brawler = None
        self.is_hypercharge_ready = False
        self.is_gadget_ready = False
        self.is_super_ready = False
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement = None
        self.last_movement_time = time.time()
        self.locked_teammate = None
        self.locked_teammate_distance = float('inf')
        self.teammate_hysteresis = 0.20  # Switch only if another teammate is 20% closer
        self.wall_history = []
        self.wall_history_length = 3  # Number of frames to keep walls
        self.scene_data = []
        self.should_detect_walls = bot_config["gamemode"] in ["brawlball", "brawl_ball", "brawll ball", "showdown"]
        self.is_showdown = bot_config["gamemode"] == "showdown"
        self.minimum_movement_delay = bot_config["minimum_movement_delay"]
        self.no_detection_proceed_delay = time_config["no_detection_proceed"]
        self.gadget_pixels_minimum = bot_config["gadget_pixels_minimum"]
        self.hypercharge_pixels_minimum = bot_config["hypercharge_pixels_minimum"]
        self.super_pixels_minimum = bot_config["super_pixels_minimum"]
        self.wall_detection_confidence = bot_config["wall_detection_confidence"]
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self.time_since_holding_attack = None
        self.seconds_to_hold_attack_after_reaching_max = load_toml_as_dict("cfg/bot_config.toml")["seconds_to_hold_attack_after_reaching_max"]
        self.current_frame = None
        # Fog color (poison gas in showdown) — sampled from images/fog_sample.png.
        # Narrow range because the fog fully overlays whatever is under it.
        self.fog_hsv_low = (50, 95, 215)
        self.fog_hsv_high = (60, 125, 245)

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info['attack_range']
            safe_range = info['safe_range']
            super_range = info['super_range']
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info[brawler]['ignore_walls_for_attacks']
        elif skill_type == "super":
            return brawlers_info[brawler]['ignore_walls_for_supers']
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        return brawlers_info[brawler]['hold_attack'] > 0

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls):
        if not walls:
            return False

        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x, max_x = min(p1_t[0], p2_t[0]), max(p1_t[0], p2_t[0])
        min_y, max_y = min(p1_t[1], p2_t[1]), max(p1_t[1], p2_t[1])
        for wall in walls:
            x1, y1, x2, y2 = wall

            if max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2:
                continue

            rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                return True
        return False

    def no_enemy_movement(self, player_data, walls):
        player_position = self.get_player_pos(player_data)
        preferred_movement = 'W' if self.game_mode == 3 else 'D'  # Adjust based on game mode

        if not self.is_path_blocked(player_position, preferred_movement, walls):
            return preferred_movement
        else:
            # Try alternative movements
            alternative_moves = ['W', 'A', 'S', 'D']
            alternative_moves.remove(preferred_movement)
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_position, move, walls):
                    return move
            print("no movement possible ?")
            # If no movement is possible, return empty string
            return preferred_movement

    def detect_fog_direction(self, frame, player_position):
        """Find the dominant fog direction relative to the player.

        Builds an HSV mask of the fog color and computes the centroid of fog
        pixels. Returns the angle FROM the player TO the fog centroid (degrees),
        or None if there is not enough fog in the frame.
        """
        if frame is None:
            return None

        import numpy as np
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        low = np.array(self.fog_hsv_low, dtype=np.uint8)
        high = np.array(self.fog_hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

        fog_pixel_count = int(cv2.countNonZero(mask))
        # Need at least a meaningful chunk of fog on screen to trust the result
        if fog_pixel_count < 500:
            return None

        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] == 0:
            return None
        fog_cx = moments["m10"] / moments["m00"]
        fog_cy = moments["m01"] / moments["m00"]

        dx = fog_cx - player_position[0]
        dy = fog_cy - player_position[1]
        if math.hypot(dx, dy) < 1:
            return None

        vlog(f"fog centroid=({int(fog_cx)},{int(fog_cy)}) pixels={fog_pixel_count}")
        return self.angle_from_direction(dx, dy)

    def showdown_roam(self, player_data, walls):
        """Roam movement for showdown: move away from the poison fog.

        If fog is visible, head away from its centroid. Otherwise spin in
        place (rotate the joystick angle each call) to look around without
        committing to a direction.
        """
        player_position = self.get_player_pos(player_data)

        fog_angle = self.detect_fog_direction(self.current_frame, player_position)
        if fog_angle is not None:
            desired_angle = self.angle_opposite(fog_angle)
            vlog(f"roam: fog detected → flee angle={desired_angle:.1f}° (fog at {fog_angle:.1f}°)")
            return self.find_best_angle(player_position, desired_angle, walls)

        # Nothing to chase, nothing to flee — spin in place
        self._roam_spin_angle = (getattr(self, "_roam_spin_angle", 0.0) + 30.0) % 360
        vlog(f"roam: idle spin → angle={self._roam_spin_angle:.1f}°")
        return self._roam_spin_angle

    def showdown_follow_teammate(self, player_data, teammate_data, walls):
        """Move towards the closest visible teammate, avoiding walls."""
        player_pos = self.get_player_pos(player_data)

        # Find the closest detected teammate this frame
        closest_teammate = None
        closest_distance = float('inf')
        for tm in teammate_data:
            tm_pos = self.get_enemy_pos(tm)
            dist = self.get_distance(tm_pos, player_pos)
            if dist < closest_distance:
                closest_distance = dist
                closest_teammate = tm_pos

        if closest_teammate is None:
            self.locked_teammate = None
            self.locked_teammate_distance = float('inf')
            return self.showdown_roam(player_data, walls)

        # Hysteresis only applies when there are multiple teammates to choose from.
        # If we already have a locked target, check whether to switch to a closer one.
        # Either way, always update the locked target's position to this frame's value.
        if self.locked_teammate is not None:
            locked_dist = self.get_distance(self.locked_teammate, player_pos)
            if closest_distance < locked_dist * (1 - self.teammate_hysteresis):
                vlog(f"follow teammate: switched target ({int(locked_dist)}px → {int(closest_distance)}px)")
                self.locked_teammate = closest_teammate
                self.locked_teammate_distance = closest_distance
            else:
                # Same target (or similar) — update its position to the current frame
                self.locked_teammate = closest_teammate
                self.locked_teammate_distance = closest_distance
        else:
            self.locked_teammate = closest_teammate
            self.locked_teammate_distance = closest_distance

        direction_x = self.locked_teammate[0] - player_pos[0]
        direction_y = self.locked_teammate[1] - player_pos[1]

        angle = self.angle_from_direction(direction_x, direction_y)
        best = self.find_best_angle(player_pos, angle, walls)
        vlog(f"follow teammate → angle={best:.1f}° (desired={angle:.1f}°, dist={int(closest_distance)}px, "
             f"player={int(player_pos[0])},{int(player_pos[1])} tm={int(self.locked_teammate[0])},{int(self.locked_teammate[1])})")
        return best

    def get_showdown_movement(self, player_data, enemy_data, teammate_data, walls, brawler):
        """Showdown movement using analog joystick angles.

        Always returns a float angle in degrees (0–360).
        0° = right, 90° = down, 180° = left, 270° = up.
        """
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")

        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and \
                time.time() - self.time_since_holding_attack >= brawler_info['hold_attack'] + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)

        # --- No enemy in sight: follow teammate or roam ---
        if not self.is_there_enemy(enemy_data):
            if teammate_data:
                vlog(f"no enemy → follow teammate ({len(teammate_data)} visible)")
                return self.showdown_follow_teammate(player_data, teammate_data, walls)
            vlog("no enemy, no teammate → roam")
            return self.showdown_roam(player_data, walls)

        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None:
            if teammate_data:
                vlog("enemy detected but unreachable → follow teammate")
                return self.showdown_follow_teammate(player_data, teammate_data, walls)
            vlog("enemy detected but unreachable, no teammate → roam")
            return self.showdown_roam(player_data, walls)

        # --- Compute exact angle toward/away from enemy, then wall-avoid ---
        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]
        toward_angle = self.angle_from_direction(direction_x, direction_y)

        if enemy_distance > safe_range:
            desired = toward_angle
            vlog(f"enemy detected → approach desired={desired:.1f}° (dist={int(enemy_distance)}px, safe={safe_range}px)")
        else:
            desired = self.angle_opposite(toward_angle)
            vlog(f"enemy too close → retreat desired={desired:.1f}° (dist={int(enemy_distance)}px, safe={safe_range}px)")

        angle = self.find_best_angle(player_pos, desired, walls)
        vlog(f"showdown: movement angle={angle:.1f}° (desired={desired:.1f}°)")

        # --- Skills ---
        if self.is_super_ready and self.time_since_holding_attack is None:
            super_type = brawler_info['super_type']
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")
            if (enemy_hittable and
                    (enemy_distance <= super_range
                     or super_type in ["spawnable", "other"]
                     or (brawler in ["stu", "surge"] and super_type == "charge" and enemy_distance <= super_range + attack_range)
                    )):
                if self.is_hypercharge_ready:
                    self.use_hypercharge()
                    self.time_since_hypercharge_checked = time.time()
                    self.is_hypercharge_ready = False
                self.use_super()
                self.time_since_super_checked = time.time()
                self.is_super_ready = False

        vlog(f"showdown movement → angle={angle:.1f}°")

        if enemy_distance <= attack_range:
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
            vlog(f"enemy in attack range (dist={int(enemy_distance)}px, range={attack_range}px), hittable={enemy_hittable}")
            if enemy_hittable:
                if self.should_use_gadget and self.is_gadget_ready and self.time_since_holding_attack is None:
                    self.use_gadget()
                    self.time_since_gadget_checked = time.time()
                    self.is_gadget_ready = False

                if not must_brawler_hold_attack:
                    self.attack()
                else:
                    if self.time_since_holding_attack is None:
                        self.time_since_holding_attack = time.time()
                        self.attack(touch_up=False, touch_down=True)
                    elif time.time() - self.time_since_holding_attack >= self.brawlers_info[brawler]['hold_attack']:
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None
        else:
            vlog(f"enemy out of attack range (dist={int(enemy_distance)}px, range={attack_range}px)")

        return angle

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        if self.walls_block_line_of_sight(player_pos, enemy_pos, walls):
            return False
        return True

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            return closest_hittable
        elif closest_unhittable:
            return closest_unhittable

        return None, None

    def _hp_worker(self):
        """
        Фоновый поток — непрерывно обрабатывает последний доступный кадр.
        Главный цикл не блокируется: он только кладёт кадр в очередь
        и читает последний кэшированный результат.
        """
        while True:
            with self._hp_lock:
                frame = self._hp_frame_queue

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                with self._onnx_lock:
                    results = self.HP_detector.detect(frame)
                with self._hp_lock:
                    self._hp_last_results = results
                    self._hp_frame_queue  = None   # сигнал: кадр обработан
            except Exception:
                with self._hp_lock:
                    self._hp_frame_queue = None

    def get_main_data(self, frame):
        with self._onnx_lock:
            data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        return data

    def is_path_blocked(self, player_pos, move_direction, walls, distance=None):  # Increased distance
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= distance
        if 's' in move_direction.lower():
            dy += distance
        if 'a' in move_direction.lower():
            dx -= distance
        if 'd' in move_direction.lower():
            dx += distance
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        return self.walls_block_line_of_sight(player_pos, new_pos, walls)

    def is_path_blocked_angle(self, player_pos, angle_degrees, walls, distance=None):
        """Check if the path in the given angle direction is blocked by walls.

        Uses two probe distances (half-tile and full-tile) so that walls that
        start very close to the player are also detected.
        """
        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Check both close (half-tile) and far (full-tile) points
        for d in (distance * 0.5, distance):
            new_pos = (player_pos[0] + cos_a * d, player_pos[1] + sin_a * d)
            if self.walls_block_line_of_sight(player_pos, new_pos, walls):
                return True
        return False

    def find_best_angle(self, player_pos, desired_angle, walls, sweep_range=160, step=10):
        """Find the closest unblocked angle to desired_angle within ±sweep_range degrees.

        Sweeps outward from the desired angle in alternating left/right steps so
        the first hit is always the least deviation from the goal direction.
        Returns desired_angle unchanged if no walls (or no clear path found).
        """
        if not self.is_path_blocked_angle(player_pos, desired_angle, walls):
            return desired_angle

        for offset in range(step, sweep_range + 1, step):
            for sign in (1, -1):
                candidate = (desired_angle + sign * offset) % 360
                if not self.is_path_blocked_angle(player_pos, candidate, walls):
                    return candidate

        # Nothing clear found — return desired anyway (better than stopping)
        return desired_angle

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if "player" not in data.keys():
            incomplete = True  # This is required so track_no_detections can also keep track if enemy is missing

        if "enemy" not in data.keys():
            data['enemy'] = None

        if "teammate" not in data.keys():
            data['teammate'] = None

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        return False if incomplete else data

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None,
                "teammate": None,
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    def do_movement(self, movement):
        if isinstance(movement, float):
            # Analog joystick path: movement is an angle in degrees
            self.window_controller.move_joystick_angle(movement)
            self.keys_hold = []
        else:
            # Legacy WASD path
            movement = movement.lower()
            keys_to_keyDown = []
            keys_to_keyUp = []
            for key in ['w', 'a', 's', 'd']:
                if key in movement:
                    keys_to_keyDown.append(key)
                else:
                    keys_to_keyUp.append(key)

            if keys_to_keyDown:
                self.window_controller.keys_down(keys_to_keyDown)

            self.window_controller.keys_up(keys_to_keyUp)

            self.keys_hold = keys_to_keyDown

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    def _debounce_angle(self, angle: float, threshold_deg: float = 10.0) -> float:
        """Suppress small angle changes to avoid jitter.

        Only adopts the new angle if it differs by more than threshold_deg
        from the last committed angle, OR if no angle was committed yet.
        """
        if self.last_movement is None or not isinstance(self.last_movement, float):
            self.last_movement = angle
            self.last_movement_time = time.time()
            return angle

        diff = abs((angle - self.last_movement + 180) % 360 - 180)
        if diff > threshold_deg:
            self.last_movement = angle
            self.last_movement_time = time.time()

        return self.last_movement

    def loop(self, brawler, data, current_time):
        if self.is_showdown:
            movement = self.get_showdown_movement(
                player_data=data['player'][0],
                enemy_data=data['enemy'],
                teammate_data=data['teammate'],
                walls=data['wall'],
                brawler=brawler,
            )
            # Debounce small angle jitter before sending to joystick
            movement = self._debounce_angle(movement)
        else:
            movement = self.get_movement(player_data=data['player'][0], enemy_data=data['enemy'], walls=data['wall'], brawler=brawler)

        current_time = time.time()
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            if isinstance(movement, float):
                movement = self.unstuck_angle_if_needed(movement, current_time)
            else:
                movement = self.unstuck_movement_if_needed(movement, current_time)
            self.do_movement(movement)
            self.time_since_movement = time.time()
        return movement

    def check_if_hypercharge_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(hypercharge_crop_area[0] * wr), int(hypercharge_crop_area[1] * hr)
        x2, y2 = int(hypercharge_crop_area[2] * wr), int(hypercharge_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        purple_pixels = count_hsv_pixels(screenshot, (137, 158, 159), (179, 255, 255))
        if debug:
            print("hypercharge purple pixels:", purple_pixels, "(if > ", self.hypercharge_pixels_minimum, " then hypercharge is ready)")
            cv2.imwrite(f"debug_frames/hypercharge_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if purple_pixels > self.hypercharge_pixels_minimum:
            return True
        return False

    def check_if_gadget_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(gadget_crop_area[0] * wr), int(gadget_crop_area[1] * hr)
        x2, y2 = int(gadget_crop_area[2] * wr), int(gadget_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        green_pixels = count_hsv_pixels(screenshot, (57, 219, 165), (62, 255, 255))
        if debug:
            print("gadget green pixels:", green_pixels, "(if > ", self.gadget_pixels_minimum, " then gadget is ready)")
            cv2.imwrite(f"debug_frames/gadget_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if green_pixels > self.gadget_pixels_minimum:
            return True
        return False

    def check_if_super_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(super_crop_area[0] * wr), int(super_crop_area[1] * hr)
        x2, y2 = int(super_crop_area[2] * wr), int(super_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        yellow_pixels = count_hsv_pixels(screenshot, (17, 170, 200), (27, 255, 255))
        if debug:
            print("super yellow pixels:", yellow_pixels, "(if > ", self.super_pixels_minimum, " then super is ready)")
            cv2.imwrite(f"debug_frames/super_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        if yellow_pixels > self.super_pixels_minimum:
            return True
        return False

    def get_tile_data(self, frame):
        with self._onnx_lock:
            tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    def process_tile_data(self, tile_data):
        walls = []
        for class_name, boxes in tile_data.items():
            if class_name != 'bush':
                walls.extend(boxes)

        # Add walls to history
        self.wall_history.append(walls)
        if len(self.wall_history) > self.wall_history_length:
            self.wall_history.pop(0)
        # Combine walls from history
        combined_walls = self.combine_walls_from_history()

        return combined_walls

    def combine_walls_from_history(self):
        unique_walls = {tuple(wall) for walls in self.wall_history for wall in walls}
        return list(unique_walls)

    def get_movement(self, player_data, enemy_data, walls, brawler):
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")
        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        # if a brawler has been holding an attack for its max duration + the bot setting, then we release
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and time.time() - self.time_since_holding_attack >= brawler_info['hold_attack'] + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)
        if debug: print("found player pos:", player_pos)
        if not self.is_there_enemy(enemy_data):
            return self.no_enemy_movement(player_data, walls)
        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None:
            return self.no_enemy_movement(player_data, walls)
        if debug: print("found enemy pos:", enemy_coords)
        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        # Determine initial movement direction
        if enemy_distance > safe_range:  # Move towards the enemy
            move_horizontal = self.get_horizontal_move_key(direction_x)
            move_vertical = self.get_vertical_move_key(direction_y)
        else:  # Move away from the enemy
            move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
            move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

        movement_options = [move_horizontal + move_vertical]
        if self.game_mode == 3:
            movement_options += [move_vertical, move_horizontal]
        elif self.game_mode == 5:
            movement_options += [move_horizontal, move_vertical]
        else:
            raise ValueError("Gamemode type is invalid")

        # Check for walls and adjust movement
        for move in movement_options:
            if not self.is_path_blocked(player_pos, move, walls):
                movement = move
                break
        else:
            print("default paths are blocked")
            # If all preferred directions are blocked, try other directions
            alternative_moves = ['W', 'A', 'S', 'D']
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break
            else:
                # if no movement is available, we still try to go in the best direction
                # because it's better than doing nothing
                movement = move_horizontal + move_vertical

        current_time = time.time()
        if movement != self.last_movement:
            if current_time - self.last_movement_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_time = current_time
            else:
                movement = self.last_movement  # Continue previous movement
        else:
            self.last_movement_time = current_time  # Reset timer if movement didn't change

        if self.is_super_ready and self.time_since_holding_attack is None:
            super_type = brawler_info['super_type']
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")

            if (enemy_hittable and
                    (enemy_distance <= super_range
                     or super_type in ["spawnable", "other"]
                     or (brawler in ["stu", "surge"] and super_type == "charge" and enemy_distance <= super_range + attack_range)
                    )):
                if self.is_hypercharge_ready:
                    self.use_hypercharge()
                    self.time_since_hypercharge_checked = time.time()
                    self.is_hypercharge_ready = False
                self.use_super()
                self.time_since_super_checked = time.time()
                self.is_super_ready = False

        # Attack if enemy is within attack range and hittable
        if enemy_distance <= attack_range:
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
            if enemy_hittable:
                if self.should_use_gadget == True and self.is_gadget_ready and self.time_since_holding_attack is None:
                    self.use_gadget()
                    self.time_since_gadget_checked = time.time()
                    self.is_gadget_ready = False

                if not must_brawler_hold_attack:
                    self.attack()
                else:
                    if self.time_since_holding_attack is None:
                        self.time_since_holding_attack = time.time()
                        self.attack(touch_up=False, touch_down=True)
                    elif time.time() - self.time_since_holding_attack >= self.brawlers_info[brawler]['hold_attack']:
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None


        return movement

    def main(self, frame, brawler, main):
        current_time = time.time()
        data = self.get_main_data(frame)
        if self.should_detect_walls and current_time - self.time_since_walls_checked > self.walls_treshold:

            tile_data = self.get_tile_data(frame)

            walls = self.process_tile_data(tile_data)

            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            data['wall'] = walls
        elif self.keep_walls_in_memory:
            data['wall'] = self.last_walls_data


        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            self.time_since_player_last_found = time.time()
            if main.state != "match":
                main.state = get_state(frame)
                if main.state != "match":
                    data = None
        if not data:
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.keys_up(list("wasd"))
            self.time_since_different_movement = time.time()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame)
                if current_state != "match":
                    self.time_since_last_proceeding = current_time
                else:
                    print("haven't detected the player in a while proceeding")
                    self.window_controller.press_key("Q")
                    self.time_since_last_proceeding = time.time()
            return
        self.time_since_last_proceeding = time.time()
        self.is_hypercharge_ready = False
        if current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
        self.is_gadget_ready = False
        if current_time - self.time_since_gadget_checked > self.gadget_treshold:
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
        self.is_super_ready = False
        if current_time - self.time_since_super_checked > self.super_treshold:
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time

        self.current_frame = frame
        movement = self.loop(brawler, data, current_time)

        # ── HP-детектор: кладём кадр в очередь (не блокируемся) ─
        if self._hp_ok:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) \
                if frame.shape[2] == 3 else frame
            with self._hp_lock:
                self._hp_frame_queue = frame_bgr   # воркер возьмёт при следующей итерации
        # ────────────────────────────────────────────────────────

        if visual_debug:
            with self._hp_lock:
                hp_snap = list(self._hp_last_results)
            self.show_visual_debug(frame, data, hp_snap)

        # if data:
        #     # Record scene data
        #     self.scene_data.append({
        #         'frame_number': len(self.scene_data),
        #         'player': data.get('player', []),
        #         'enemy': data.get('enemy', []),
        #         'wall': data.get('wall', []),
        #         'movement': movement,
        #     })

    def show_visual_debug(self, frame, data, hp_results=None):
        import numpy as np
        img = frame.copy() if isinstance(frame, np.ndarray) else np.array(frame)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # → BGR для всех cv2 операций

        COLORS = {
            "player":   (60,  220, 60),    # зелёный
            "enemy":    (50,  50,  220),   # красный
            "teammate": (220, 140, 50),    # синий
            "wall":     (100, 100, 100),   # серый
        }
        HP_COLORS = {      # BGR цвета для HP-бара на экране
            "player":   (60,  220, 60),
            "enemy":    (50,  50,  220),
            "team":     (220, 140, 50),
        }
        FILL_GOOD   = (60,  210, 60)    # >60% — зелёный
        FILL_MID    = (30,  175, 255)   # >30% — оранжевый
        FILL_LOW    = (45,  45,  215)   # ≤30% — красный
        FILL_EMPTY  = (40,  40,  40)    # фон пустой части

        overlay = img.copy()

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low  = np.array(self.fog_hsv_low,  dtype=np.uint8)
        high = np.array(self.fog_hsv_high, dtype=np.uint8)
        fog_mask = cv2.inRange(hsv_img, low, high)
        if cv2.countNonZero(fog_mask) > 0:
            tint = np.zeros_like(img)
            tint[:, :] = (180, 0, 180)
            img = np.where(fog_mask[..., None] > 0,
                           cv2.addWeighted(img, 0.55, tint, 0.45, 0), img)
            moments = cv2.moments(fog_mask, binaryImage=True)
            if moments["m00"] > 0 and data.get("player"):
                fog_cx = int(moments["m10"] / moments["m00"])
                fog_cy = int(moments["m01"] / moments["m00"])
                cv2.circle(img, (fog_cx, fog_cy), 10, (180, 0, 180), -1)
                cv2.putText(img, "FOG", (fog_cx + 12, fog_cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 80, 220), 2, cv2.LINE_AA)
                px, py = self.get_player_pos(data["player"][0])
                cv2.arrowedLine(img, (int(px), int(py)), (fog_cx, fog_cy),
                                (180, 0, 180), 2, tipLength=0.12)

        wall_boxes = data.get("wall") or []
        for box in wall_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS["wall"], -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        for box in wall_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["wall"], 1)

        def _draw_entity_box(bimg, x1, y1, x2, y2, label, color, hp_pct=None):
            ov = bimg.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            bimg[:] = cv2.addWeighted(ov, 0.15, bimg, 0.85, 0)

            corner = min(12, (x2 - x1) // 3, (y2 - y1) // 3)
            t = 2
            for (cx, cy, dx, dy) in [
                (x1, y1,  1,  1),
                (x2, y1, -1,  1),
                (x1, y2,  1, -1),
                (x2, y2, -1, -1),
            ]:
                cv2.line(bimg, (cx, cy), (cx + dx * corner, cy), color, t + 1, cv2.LINE_AA)
                cv2.line(bimg, (cx, cy), (cx, cy + dy * corner), color, t + 1, cv2.LINE_AA)

            cv2.rectangle(bimg, (x1, y1), (x2, y2), color, 1)

            if hp_pct is not None:
                bar_h    = 6
                bar_gap  = 3
                bar_y1   = max(0, y1 - bar_gap - bar_h)
                bar_y2   = bar_y1 + bar_h
                bar_x1   = x1
                bar_x2   = x2
                bar_w    = bar_x2 - bar_x1

                cv2.rectangle(bimg, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 30, 30), -1)
                cv2.rectangle(bimg, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 70), 1)

                filled_w = int(bar_w * hp_pct / 100)
                if filled_w > 0:
                    if hp_pct > 60:
                        fc = FILL_GOOD
                    elif hp_pct > 30:
                        fc = FILL_MID
                    else:
                        fc = FILL_LOW
                    cv2.rectangle(bimg, (bar_x1, bar_y1),
                                  (bar_x1 + filled_w, bar_y2), fc, -1)

                pct_str = f"{hp_pct:.0f}%"
                (tw, th), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                tx = bar_x1 + (bar_w - tw) // 2
                ty = bar_y2 - 1
                cv2.putText(bimg, pct_str, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

            lbl_h = 16
            lbl_y1 = y2
            lbl_y2 = y2 + lbl_h
            cv2.rectangle(img, (x1, lbl_y1), (x2, lbl_y2), color, -1)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(bimg, label,
                        (x1 + (x2 - x1 - lw) // 2, lbl_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10, 10, 10), 1, cv2.LINE_AA)

        hp_map = {}   # (x1,y1,x2,y2) → hp_pct
        if hp_results:
            for hbar in hp_results:
                hx1, hy1, hx2, hy2 = hbar.box_img
                hcx = (hx1 + hx2) / 2
                hcy = (hy1 + hy2) / 2
                hp_map[(hcx, hcy, hbar.class_name)] = hbar.hp_pct

        def _find_hp(cx, cy, cls_name, search_r=80):
            best_dist = search_r
            best_hp   = None
            for (hcx, hcy, hcls), pct in hp_map.items():
                if hcls[:4] != cls_name[:4]:   # player/enemy/team prefix match
                    continue
                d = ((cx - hcx) ** 2 + (cy - hcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_hp   = pct
            return best_hp

        entity_map = {
            "player":   ("player", COLORS["player"]),
            "enemy":    ("enemy",  COLORS["enemy"]),
            "teammate": ("team",   COLORS["teammate"]),
        }
        for key, (hp_cls, color) in entity_map.items():
            boxes = data.get(key)
            if not boxes:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                hp_pct = _find_hp(cx, cy, hp_cls)
                _draw_entity_box(img, x1, y1, x2, y2, key, color, hp_pct)

        if hp_results:
            for hbar in hp_results:
                hx1, hy1, hx2, hy2 = hbar.box_img
                color = HP_COLORS.get(hbar.class_name, (180, 180, 180))
                if hbar.hp_pct is not None:
                    bar_cy = (hy1 + hy2) // 2
                    bar_cx = (hx1 + hx2) // 2
                    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), color, 1)
                    bar_h   = hy2 - hy1
                    fill_w  = int((hx2 - hx1) * hbar.hp_pct / 100)
                    if fill_w > 0:
                        if hbar.hp_pct > 60:   fc = FILL_GOOD
                        elif hbar.hp_pct > 30: fc = FILL_MID
                        else:                   fc = FILL_LOW
                        alpha_ov = img.copy()
                        cv2.rectangle(alpha_ov, (hx1, hy1),
                                      (hx1 + fill_w, hy2), fc, -1)
                        img[:] = cv2.addWeighted(alpha_ov, 0.45, img, 0.55, 0)

        if hp_results:
            _draw_hp_hud(img, hp_results)

        cv2.imshow("Pyla Visual Debug", img)
        cv2.waitKey(1)

    @staticmethod
    def movement_to_direction(movement):
        mapping = {
            'w': 'up',
            'a': 'left',
            's': 'down',
            'd': 'right',
            'wa': 'up-left',
            'aw': 'up-left',
            'wd': 'up-right',
            'dw': 'up-right',
            'sa': 'down-left',
            'as': 'down-left',
            'sd': 'down-right',
            'ds': 'down-right',
        }
        movement = movement.lower()
        movement = ''.join(sorted(movement))
        return mapping.get(movement, 'idle' if movement == '' else movement)

def _draw_hp_hud(img, hp_results):
    import numpy as np

    if not hp_results:
        return

    H, W = img.shape[:2]

    PAD    = 8
    ROW_H  = 24
    BAR_W  = 88
    BAR_H  = 10
    ICON_W = 18
    TXT_W  = 52
    ROW_W  = PAD + ICON_W + 6 + BAR_W + 6 + TXT_W + PAD

    ORDER    = {"player": 0, "team": 1, "enemy": 2}
    sorted_r = sorted(hp_results, key=lambda r: (ORDER.get(r.class_name, 9), -r.confidence))

    HEADER_H = 22
    hud_h    = HEADER_H + PAD + len(sorted_r) * ROW_H + PAD
    hud_w    = ROW_W
    ox       = W - hud_w - 10
    oy       = 10

    ox = max(0, min(ox, W - hud_w))
    oy = max(0, min(oy, H - hud_h))

    panel = img[oy:oy + hud_h, ox:ox + hud_w].copy()
    dark  = np.full_like(panel, 18)
    panel = cv2.addWeighted(panel, 0.20, dark, 0.80, 0)

    cv2.rectangle(panel, (0, 0), (hud_w - 1, hud_h - 1), (180, 170, 220), 1)

    cv2.rectangle(panel, (1, 1), (hud_w - 2, HEADER_H - 2), (45, 35, 70), -1)
    cv2.putText(panel, "HP MONITOR", (PAD, HEADER_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 190, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (1, HEADER_H - 1), (hud_w - 2, HEADER_H - 1), (100, 90, 140), 1)

    CLS_COLOR = {"player": (60, 220, 60), "enemy": (50, 50, 220), "team": (220, 140, 50)}
    CLS_LETTER = {"player": "P", "enemy": "E", "team": "T"}

    for i, bar in enumerate(sorted_r):
        row_y  = HEADER_H + PAD + i * ROW_H
        cx     = PAD
        color  = CLS_COLOR.get(bar.class_name, (160, 160, 160))
        pct    = bar.hp_pct if bar.hp_pct is not None else 0.0

        ico_cx = cx + ICON_W // 2
        ico_cy = row_y + ROW_H // 2
        cv2.circle(panel, (ico_cx, ico_cy), ICON_W // 2 - 1, color, -1)
        cv2.circle(panel, (ico_cx, ico_cy), ICON_W // 2 - 1, tuple(min(255, c + 60) for c in color), 1)
        letter = CLS_LETTER.get(bar.class_name, "?")
        (lw, lh), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(panel, letter, (ico_cx - lw // 2, ico_cy + lh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 1, cv2.LINE_AA)
        cx += ICON_W + 6

        bx1 = cx
        by1 = row_y + (ROW_H - BAR_H) // 2
        bx2 = bx1 + BAR_W
        by2 = by1 + BAR_H

        cv2.rectangle(panel, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.rectangle(panel, (bx1, by1), (bx2, by2), (75, 75, 75), 1)

        fw = int(BAR_W * pct / 100)
        if fw > 0:
            if pct > 60:   fc = (55, 205, 55)
            elif pct > 30: fc = (30, 170, 255)
            else:           fc = (45, 45, 210)
            cv2.rectangle(panel, (bx1, by1), (bx1 + fw, by2), fc, -1)
            # блик
            cv2.rectangle(panel, (bx1, by1), (bx1 + fw, by1 + BAR_H // 3 + 1),
                          tuple(min(255, c + 55) for c in fc), -1)
        cx += BAR_W + 6

        pct_str = f"{pct:.0f}%" if bar.hp_pct is not None else "N/A"
        label   = f"{bar.class_name[:3].upper()} {pct_str}"
        if pct > 60:   tc = (75,  225, 75)
        elif pct > 30: tc = (60,  185, 255)
        else:           tc = (70,  70,  225)
        if bar.hp_pct is None:
            tc = (120, 120, 120)
        cv2.putText(panel, label, (cx, by2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, tc, 1, cv2.LINE_AA)

    img[oy:oy + hud_h, ox:ox + hud_w] = panel