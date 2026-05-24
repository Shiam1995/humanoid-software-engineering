# LLM Agent in the ITER Assembly Building
### Humanoid Internship Challenge — Software Engineering

An LLM-powered robot agent that navigates a 2D world derived from a real Gaussian Splat reconstruction of the ITER nuclear fusion facility, reasoning about moving breeder blanket panels from storage to reactor ports.

## What it does

A robot operates inside a 2D occupancy map automatically extracted from the ITER Gaussian Splat point cloud. The circular map shape is the actual geometry of the ITER tokamak assembly hall — never drawn manually, it emerged from the real building captured on video.

Every step, the world state is described as text and sent to Claude via the Anthropic API. Claude returns a single action. The world executes it, checks physics, and feeds the result back.

**Task:** move 3 breeder blanket panels from storage (left) to reactor ports (right). Every episode is randomised — Claude must reason fresh each time.

## Setup

```bash
conda create -n humanoid_agent python=3.11 -y
conda activate humanoid_agent
pip install anthropic numpy matplotlib plyfile opencv-python-headless scipy
```

## Running

### Step 1 — Extract occupancy map
```bash
python extract_map.py /path/to/point_cloud.ply ./map_output
```

### Step 2 — Run the agent
```bash
export ANTHROPIC_API_KEY=your_key_here
python agent.py ./map_output ./episode_output
```

### Step 3 — View results
```bash
eog ./episode_output/
```

## Action space
move_north, move_south, move_east, move_west, pick_up_panel, drop_off_panel, describe_surroundings, done

## Design choices

- Text observations — LLMs reason natively in text, explicit coordinates give Claude everything it needs
- Randomised placement — prevents memorisation, forces genuine reasoning each episode
- ITER world — map comes directly from Project 1 Gaussian Splat, task is grounded in real fusion engineering
- Connected to Project 2 — DTT arm policy would execute physical grasps at panel positions

## Future work
- A* pathfinding for efficient navigation
- Raw sensor observations to force genuine spatial reasoning
- Direct DTT arm integration for physical manipulation
- Multi-agent coordination
- Real-time Unreal Engine 5 visualisation

## Connection to Projects 1 and 2
- Project 1 built the world (Gaussian Splat → occupancy map)
- Project 2 built the arm (DTT RL policy for fusion reactor manipulation)
- Project 3 (this) built the brain (Claude agent navigating and reasoning)

## References
- Kerbl et al. 2023 — 3D Gaussian Splatting (SIGGRAPH 2023)
- Anthropic — Claude API
- Zoppoli et al. 2024 — DTT kinematic simulation
