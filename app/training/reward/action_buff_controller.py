from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Sequence
from app.config import (
    RoundConfig,
    ActionBuffConfig,
    get_buff_group
)

DEFAULT_BUFF_FILENAME = "action_buff_state.json"

@dataclass(frozen=True)
class GroupBuffState:
    ema_ratio: float
    buff: float
    prev_error: float = 0.0
    
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class EMABuffController:
    def __init__(self, groups: Sequence[(str, str)], namespace: str):
        self.groups = tuple(groups)
        self.namespace = namespace
        self.states: Dict[str, Dict[str, GroupBuffState]] = {}
        
    def init(self, round_config: RoundConfig):
        for zone_key, action_key in self.groups:
            print(f"Init buff for {zone_key}:{action_key}")
            if zone_key not in self.states:
                self.states[zone_key] = {}
                
            self.states[zone_key][action_key] = get_buff_group(round_config, zone_key, action_key)
            
    def on_step_end(self, round_config: RoundConfig, zone_key: str, counts: Dict[str, int], total: int) -> None:
        if total <= 0:
            return
        alpha = round_config.alpha
        kp = round_config.kp
        kd = round_config.kd
        step_max = round_config.step_max
        
        for z_key, action_key in self.groups:
            if z_key != zone_key:
                continue
            action_buff_cfg: ActionBuffConfig = round_config.zone_buffs[zone_key].action_buffs[action_key]
            
            state = self.states[zone_key][action_key]
            count = counts.get(action_key, 0)
            ratio = count / total
            
            # EMA update
            ema_ratio = alpha * ratio + (1 - alpha) * state.ema_ratio
            
            # Error and derivative
            error = action_buff_cfg.target_ratio - ema_ratio
            d_error = error - state.prev_error
            
            # Buff update
            buff_delta = kp * error + kd * d_error
            buff_delta = _clip(buff_delta, -step_max, step_max)
            new_buff = _clip(
                state.buff + buff_delta, 
                action_buff_cfg.buff_min, 
                action_buff_cfg.buff_max
            )
            
            # Update state
            self.states[zone_key][action_key] = GroupBuffState(
                ema_ratio=ema_ratio,
                buff=new_buff,
                prev_error=error
            )
    
    def get_buff(self, zone_key: str, action_key: str) -> float:
        state = self.states.get(zone_key).get(action_key)
        if state is None:
            return 0.0
        return state.buff
    
    def snapshot(self) -> Dict[str, Dict[str, GroupBuffState]]:
        return {
            zone_key: {
                action_key: {
                    "ema_ratio": state.ema_ratio, 
                    "buff": state.buff, 
                    "prev_error": state.prev_error
                }
                for action_key, state in actions.items()
            }
            for zone_key, actions in self.states.items()
        }
        
    def state_dict(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return self.snapshot()
    
    def load_state_dict(self, data: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        for zone_key, zone_value in data.items():
            if zone_key not in self.states:
                self.states[zone_key] = {}
            for action_key, d in zone_value.items():
                self.states[zone_key][action_key] = GroupBuffState(
                    ema_ratio=float(d["ema_ratio"]),
                    buff=float(d["buff"]),
                    prev_error=float(d.get("prev_error", 0.0)),
                )
    
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False), encoding="utf-8")
        
    def load(self, path: str) -> bool:
        """Trả True nếu load thành công. Caller PHẢI gọi
        seed_from_round_config() khi trả về False — KHÔNG được để states
        rỗng (get_buff sẽ âm thầm trả 0.0 cho group thiếu)."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.load_state_dict(data)
            return True
        except Exception:
            return False
        
    @classmethod
    def load_or_init(cls, round_config: RoundConfig, resume_checkpoint: str = None) -> EMABuffController:
        groups = []
        for zone_key, zone_buff in round_config.zone_buffs.items():
            for action_key, _ in zone_buff.action_buffs.items():
                groups.append((zone_key, action_key))
        
        buff_controller = EMABuffController(groups=groups, namespace="action")

        import os
        buff_path = os.path.join(resume_checkpoint, DEFAULT_BUFF_FILENAME) if resume_checkpoint else None
        if buff_path and Path(buff_path).exists():
            print(f"Load action buff state from {buff_path}")
            buff_controller.load(buff_path)
        else:
            buff_controller.init(round_config)   # round MỚI hoặc load thất bại -> seed lại từ config

        return buff_controller