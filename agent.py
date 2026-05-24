"""
agent.py
LLM agent that navigates the ITER splat world and moves breeder blanket panels
using Claude as the reasoning engine.
"""

import json
import numpy as np
import anthropic
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time

# ── WORLD CONFIG ──────────────────────────────────────────────────────────────
PANEL_WEIGHT_KG  = 4000   # breeder blanket panel ~4 tonnes (abstracted)
ARM_REACH_M      = 2.0    # robot arm reach in metres
MOVE_STEP        = 1      # grid cells per move action
MAX_STEPS        = 500    # max steps per episode
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = [
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "pick_up_panel",
    "drop_off_panel",
    "describe_surroundings",
    "done",
]

SYSTEM_PROMPT = """You are an autonomous robot operating inside the ITER Assembly Building — a nuclear fusion facility.
Your robot has a wheeled base for navigation and a heavy-duty robotic arm capable of handling breeder blanket panels (each ~4 tonnes).

Your goal is to move breeder blanket panels from their storage positions to the reactor port dropoff zones.

Rules:
- You can only carry ONE panel at a time
- You must be within 2 grid cells of a panel to pick it up
- You must be at a dropoff zone to drop off a panel
- You cannot move through walls or obstacles
- Choose the most efficient path

Always respond with ONLY the action name from the list provided. No explanation, no punctuation, just the action.
"""


@dataclass
class Panel:
    id: str
    x: int
    z: int
    picked_up: bool = False
    delivered: bool = False


@dataclass  
class DropoffZone:
    id: str
    x: int
    z: int


@dataclass
class AgentState:
    x: int
    z: int
    carrying: Optional[str] = None
    steps: int = 0
    log: list = field(default_factory=list)


class ITERWorld:
    def __init__(self, map_dir: str):
        map_dir = Path(map_dir)
        self.occupancy = np.load(map_dir / "occupancy.npy")
        with open(map_dir / "map_meta.json") as f:
            self.meta = json.load(f)

        self.nz, self.nx = self.occupancy.shape

        # Place panels and dropoff zones in free space
        self.panels = self._place_panels()
        self.dropoffs = self._place_dropoffs()
        self.agent = self._place_agent()

        print(f"World loaded: {self.nx}x{self.nz} grid")
        print(f"Panels: {[p.id for p in self.panels]}")
        print(f"Dropoffs: {[d.id for d in self.dropoffs]}")
        print(f"Agent start: ({self.agent.x}, {self.agent.z})")

    def _find_free_cells(self, n: int, region=None):
        """Find n free cells, optionally within a region (x_min, x_max, z_min, z_max)."""
        free = []
        if region:
            x_min, x_max, z_min, z_max = region
        else:
            x_min, x_max, z_min, z_max = 0, self.nx, 0, self.nz

        for z in range(z_min, z_max):
            for x in range(x_min, x_max):
                if self.occupancy[z, x] == 0:
                    free.append((x, z))

        if len(free) < n:
            # Fall back to all free cells
            free = [(x, z) for z in range(self.nz) for x in range(self.nx)
                    if self.occupancy[z, x] == 0]

        indices = np.random.choice(len(free), n, replace=False)
        return [free[i] for i in indices]

    def _place_panels(self):
        cells = self._find_free_cells(3, region=(0, self.nx//3, 0, self.nz))
        return [Panel(id=f"panel_{i+1}", x=c[0], z=c[1]) for i, c in enumerate(cells)]

    def _place_dropoffs(self):
        cells = self._find_free_cells(2, region=(2*self.nx//3, self.nx, 0, self.nz))
        return [DropoffZone(id=f"reactor_port_{i+1}", x=c[0], z=c[1]) for i, c in enumerate(cells)]

    def _place_agent(self):
        cells = self._find_free_cells(1, region=(self.nx//3, 2*self.nx//3, 0, self.nz))
        return AgentState(x=cells[0][0], z=cells[0][1])

    def is_free(self, x, z):
        if x < 0 or x >= self.nx or z < 0 or z >= self.nz:
            return False
        return self.occupancy[z, x] == 0

    def get_nearby_objects(self, radius=5):
        nearby = []
        ax, az = self.agent.x, self.agent.z

        for panel in self.panels:
            if not panel.delivered:
                dist = abs(panel.x - ax) + abs(panel.z - az)
                if dist <= radius:
                    status = "carried by you" if panel.picked_up else f"{dist} cells away"
                    nearby.append(f"{panel.id} ({status})")

        for dropoff in self.dropoffs:
            dist = abs(dropoff.x - ax) + abs(dropoff.z - az)
            if dist <= radius:
                nearby.append(f"{dropoff.id} (dropoff zone, {dist} cells away)")

        return nearby if nearby else ["nothing nearby"]

    def get_observation(self):
        ax, az = self.agent.x, self.agent.z
        nearby = self.get_nearby_objects()
        carrying = self.agent.carrying if self.agent.carrying else "nothing"
        remaining = [p.id for p in self.panels if not p.delivered]

        # Direction hints
        hints = []
        for panel in self.panels:
            if not panel.delivered and not panel.picked_up:
                dx = panel.x - ax
                dz = panel.z - az
                dir_x = "east" if dx > 0 else "west"
                dir_z = "north" if dz > 0 else "south"
                hints.append(f"{panel.id} is {abs(dx)} cells {dir_x} and {abs(dz)} cells {dir_z}")

        for dropoff in self.dropoffs:
            dx = dropoff.x - ax
            dz = dropoff.z - az
            dir_x = "east" if dx > 0 else "west"
            dir_z = "north" if dz > 0 else "south"
            hints.append(f"{dropoff.id} is {abs(dx)} cells {dir_x} and {abs(dz)} cells {dir_z}")

        obs = f"""=== ITER Assembly Building — Robot Status ===
Position: ({ax}, {az})
Carrying: {carrying}
Panels remaining: {remaining}
Nearby: {', '.join(nearby)}
Navigation hints: {'; '.join(hints)}
Steps taken: {self.agent.steps}/{MAX_STEPS}
Available actions: {', '.join(ACTIONS)}
"""
        return obs

    def execute_action(self, action: str) -> str:
        ax, az = self.agent.x, self.agent.z
        self.agent.steps += 1
        result = ""

        if action == "move_north":
            if self.is_free(ax, az + MOVE_STEP):
                self.agent.z += MOVE_STEP
                result = f"Moved north to ({self.agent.x}, {self.agent.z})"
            else:
                result = "Cannot move north — obstacle or boundary"

        elif action == "move_south":
            if self.is_free(ax, az - MOVE_STEP):
                self.agent.z -= MOVE_STEP
                result = f"Moved south to ({self.agent.x}, {self.agent.z})"
            else:
                result = "Cannot move south — obstacle or boundary"

        elif action == "move_east":
            if self.is_free(ax + MOVE_STEP, az):
                self.agent.x += MOVE_STEP
                result = f"Moved east to ({self.agent.x}, {self.agent.z})"
            else:
                result = "Cannot move east — obstacle or boundary"

        elif action == "move_west":
            if self.is_free(ax - MOVE_STEP, az):
                self.agent.x -= MOVE_STEP
                result = f"Moved west to ({self.agent.x}, {self.agent.z})"
            else:
                result = "Cannot move west — obstacle or boundary"

        elif action == "pick_up_panel":
            if self.agent.carrying:
                result = f"Already carrying {self.agent.carrying}"
            else:
                picked = False
                for panel in self.panels:
                    if not panel.picked_up and not panel.delivered:
                        dist = abs(panel.x - ax) + abs(panel.z - az)
                        if dist <= 2:
                            panel.picked_up = True
                            self.agent.carrying = panel.id
                            result = f"Picked up {panel.id} — arm engaged, panel secured"
                            picked = True
                            break
                if not picked:
                    result = "No panel within reach (must be within 2 cells)"

        elif action == "drop_off_panel":
            if not self.agent.carrying:
                result = "Not carrying any panel"
            else:
                dropped = False
                for dropoff in self.dropoffs:
                    dist = abs(dropoff.x - ax) + abs(dropoff.z - az)
                    if dist <= 2:
                        for panel in self.panels:
                            if panel.id == self.agent.carrying:
                                panel.delivered = True
                                panel.picked_up = False
                        result = f"Delivered {self.agent.carrying} to {dropoff.id} — panel installed"
                        self.agent.carrying = None
                        dropped = True
                        break
                if not dropped:
                    result = "Not at a dropoff zone (must be within 2 cells of reactor port)"

        elif action == "describe_surroundings":
            nearby = self.get_nearby_objects(radius=10)
            result = f"Surroundings: {', '.join(nearby)}"

        elif action == "done":
            result = "Agent signalled task complete"

        else:
            result = f"Unknown action: {action}"

        self.agent.log.append({"step": self.agent.steps, "action": action, "result": result})
        return result

    def is_complete(self):
        return all(p.delivered for p in self.panels)

    def render(self, save_path=None):
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(self.occupancy, cmap="gray_r", origin="lower", alpha=0.6)

        # Draw panels
        for panel in self.panels:
            colour = "green" if panel.delivered else ("yellow" if panel.picked_up else "blue")
            ax.plot(panel.x, panel.z, "s", color=colour, markersize=10)
            ax.annotate(panel.id, (panel.x, panel.z), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=colour)

        # Draw dropoffs
        for dropoff in self.dropoffs:
            ax.plot(dropoff.x, dropoff.z, "^", color="red", markersize=12)
            ax.annotate(dropoff.id, (dropoff.x, dropoff.z), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="red")

        # Draw agent
        ax.plot(self.agent.x, self.agent.z, "o", color="orange", markersize=14)
        ax.annotate("ROBOT", (self.agent.x, self.agent.z), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color="orange", fontweight="bold")

        # Legend
        legend = [
            mpatches.Patch(color="blue", label="Panel (storage)"),
            mpatches.Patch(color="yellow", label="Panel (carried)"),
            mpatches.Patch(color="green", label="Panel (delivered)"),
            mpatches.Patch(color="red", label="Reactor port (dropoff)"),
            mpatches.Patch(color="orange", label="Robot"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8)
        ax.set_title(f"ITER Assembly Building — Step {self.agent.steps}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class ClaudeAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.history = []

    def get_action(self, observation: str) -> str:
        self.history.append({"role": "user", "content": observation})

        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=50,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )

        action = response.content[0].text.strip().lower().replace(" ", "_")
        self.history.append({"role": "assistant", "content": action})

        # Validate action
        if action not in ACTIONS:
            # Try to find closest match
            for a in ACTIONS:
                if a in action:
                    action = a
                    break
            else:
                action = "describe_surroundings"

        return action


def run_episode(map_dir: str, output_dir: str, max_steps: int = MAX_STEPS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Starting ITER Assembly Building Agent ===\n")
    world = ITERWorld(map_dir)
    agent = ClaudeAgent()

    # Save initial state
    world.render(save_path=output_dir / "step_000.png")

    step = 0
    while step < max_steps and not world.is_complete():
        obs = world.get_observation()
        print(f"\n--- Step {step + 1} ---")
        print(obs)

        action = agent.get_action(obs)
        print(f"Claude chose: {action}")

        result = world.execute_action(action)
        print(f"Result: {result}")

        if (step + 1) % 5 == 0:
            world.render(save_path=output_dir / f"step_{step+1:03d}.png")

        if action == "done":
            break

        step += 1
        time.sleep(0.5)  # rate limit

    # Final render
    world.render(save_path=output_dir / f"step_{step:03d}_final.png")

    # Save log
    delivered = sum(1 for p in world.panels if p.delivered)
    total = len(world.panels)
    success = world.is_complete()

    summary = {
        "steps": world.agent.steps,
        "panels_delivered": delivered,
        "panels_total": total,
        "success": success,
        "log": world.agent.log,
    }

    with open(output_dir / "episode_log.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Episode Complete ===")
    print(f"Steps: {world.agent.steps}")
    print(f"Panels delivered: {delivered}/{total}")
    print(f"Success: {success}")
    print(f"Output saved to {output_dir}")

    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python agent.py <map_dir> <output_dir>")
        sys.exit(1)
    run_episode(sys.argv[1], sys.argv[2])
