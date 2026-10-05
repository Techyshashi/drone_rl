"""
NIDAR AirMouse arena generator (MuJoCo MJCF).

Writes:
  nidar_arena.xml          - worldbody-only MJCF (same name/format as the original script)
  nidar_arena_layout.json  - ground truth: grid labels, rooms, survivors, gates, wall rectangles

Arena is built from a modular grid defined by LAYOUT (ASCII map, north at the top):
  '#'  solid / unused cell (no space inside)
  '.'  corridor cell (1 m clear)
  A-F  a 2 x 2-cell room (each letter must cover exactly a 2 x 2 block)
Walls are placed on cell edges automatically:
  - between open and solid cells
  - between a room and a corridor / another room, UNLESS listed in DOORS (open doorway)
  - around the outer boundary, UNLESS a gate is defined in GATES (entry / exit opening)
"""
import json
import xml.etree.ElementTree as ET
from collections import deque

FT = 0.3048

# ------------------------------------------------------------------ mission-brief constants
CELL_CLEAR = 1.0          # uniform clear corridor width (brief: >= 1 m)
WALL_T = 0.05             # wall thickness; pitch = CELL_CLEAR + WALL_T keeps clear width exactly 1 m
WALL_H = 8 * FT           # 8 ft = 2.4384 m clearance
MAX_ARENA = 15.0          # brief: arena <= 15 m x 15 m
LAUNCH = 2 * FT           # 2 ft x 2 ft launch area
LAUNCH_GAP = 0.30         # distance from outer face of the gate wall to the launch pad
FLOOR_MARGIN = 2.0        # extra floor around the arena

CEILING_NET = True        # brief: arena top is covered with a net
CEILING_COLLIDES = True   # True = drone cannot fly over the walls (important for RL)
DOOR_MARKERS = True       # visual-only floor strips at doorways
SURVIVOR_ROOMS = "ABCDEF" # one dummy survivor at the centre of each of these rooms

# ------------------------------------------------------------------ layout (north = top row)
LAYOUT = [
    "AA#BB##CC##DD#",
    "AA#BB##CC##DD#",
    "..............",
    "#.###.###.###.",
    "EE#...#FF...#.",
    "EE#.###FF##.#.",
    "#...#.....#...",
    "#.###.###.###.",
    "#.....#.....#.",
    "#######.###.#.",
    "........#...#.",
    ".########.###.",
    ".########.###.",
    "..............",
]

# open doorways: pairs of adjacent cells (row, col), row 0 = south row
DOORS = [
    ((12, 1), (11, 1)),   # room A -> corridor
    ((12, 4), (11, 4)),   # room B -> corridor
    ((12, 7), (11, 7)),   # room C -> corridor
    ((12, 12), (11, 12)), # room D -> corridor
    ((9, 1), (10, 1)),    # room E north door
    ((8, 1), (7, 1)),     # room E south door (pass-through room)
    ((9, 8), (9, 9)),     # room F east door
    ((8, 8), (7, 8)),     # room F south door (pass-through room)
]

# gates: name -> (side 'S'/'N'/'W'/'E', index of the cell along that side)
# brief: SAME entry and exit point -> both names point to the same opening
GATES = {"entry": ("S", 6), "exit": ("S", 6)}

# ------------------------------------------------------------------ derived geometry
P = CELL_CLEAR + WALL_T
N = len(LAYOUT)
GRID = LAYOUT[::-1]                # GRID[r][c], r = 0 is the south row
W = N * P + WALL_T                 # total arena side length


def cx(c):
    return c * P + WALL_T + CELL_CLEAR / 2


def cy(r):
    return r * P + WALL_T + CELL_CLEAR / 2


def label(r, c):
    """Grid box label, e.g. column G, row 4 -> 'G4'."""
    return f"{chr(ord('A') + c)}{r + 1}"


def in_grid(r, c):
    return 0 <= r < N and 0 <= c < N


# ------------------------------------------------------------------ validation of inputs
assert all(len(row) == N for row in LAYOUT), "LAYOUT must be square"
assert W <= MAX_ARENA, f"arena {W:.2f} m exceeds {MAX_ARENA} m"

rooms = {}
for r in range(N):
    for c in range(N):
        ch = GRID[r][c]
        if ch.isalpha():
            rooms.setdefault(ch, []).append((r, c))
for k, cells in rooms.items():
    rs = sorted({r for r, _ in cells})
    cs = sorted({c for _, c in cells})
    assert len(cells) == 4 and len(rs) == 2 and len(cs) == 2 \
        and rs[1] - rs[0] == 1 and cs[1] - cs[0] == 1, f"room {k} is not a 2x2 block"


def gate_cells(side, i):
    """Return (inner cell, outer cell) of a boundary gate."""
    if side == "S":
        return (0, i), (-1, i)
    if side == "N":
        return (N - 1, i), (N, i)
    if side == "W":
        return (i, 0), (i, -1)
    if side == "E":
        return (i, N - 1), (i, N)
    raise ValueError(side)


GATE_EDGES = set()
for _name, (_side, _i) in GATES.items():
    _inner, _outer = gate_cells(_side, _i)
    assert GRID[_inner[0]][_inner[1]] != "#", f"gate {_name} opens into a solid cell"
    GATE_EDGES.add(frozenset((_inner, _outer)))

DOOR_EDGES = set()
for _a, _b in DOORS:
    assert in_grid(*_a) and in_grid(*_b), f"door {_a}-{_b} outside grid"
    assert abs(_a[0] - _b[0]) + abs(_a[1] - _b[1]) == 1, f"door {_a}-{_b} not adjacent"
    _ca, _cb = GRID[_a[0]][_a[1]], GRID[_b[0]][_b[1]]
    assert _ca != "#" and _cb != "#" and _ca != _cb, f"door {_a}-{_b} must join two different open spaces"
    DOOR_EDGES.add(frozenset((_a, _b)))


def needs_wall(a, b):
    """True if a wall panel is required on the edge between cells a and b."""
    e = frozenset((a, b))
    if e in GATE_EDGES:
        return False
    if not (in_grid(*a) and in_grid(*b)):
        return True                       # outer boundary
    ca, cb = GRID[a[0]][a[1]], GRID[b[0]][b[1]]
    if ca == "#" and cb == "#":
        return False
    if (ca == "#") != (cb == "#"):
        return True
    if ca == cb:
        return False                      # same room, or two corridor cells
    return e not in DOOR_EDGES            # room<->corridor / room<->room needs a door


def runs(flags):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i))
            start = None
    return out


# ------------------------------------------------------------------ wall rectangles (merged)
wall_rects = []   # (xmin, ymin, xmax, ymax)
for k in range(N + 1):
    # horizontal wall line at y index k (between row k-1 and row k)
    for s, e in runs([needs_wall((k - 1, c), (k, c)) for c in range(N)]):
        wall_rects.append((s * P, k * P, e * P + WALL_T, k * P + WALL_T))
    # vertical wall line at x index k (between column k-1 and column k)
    for s, e in runs([needs_wall((r, k - 1), (r, k)) for r in range(N)]):
        wall_rects.append((k * P, s * P, k * P + WALL_T, e * P + WALL_T))

# ------------------------------------------------------------------ self-checks on the layout
def passable(a, b):
    return in_grid(*a) and in_grid(*b) and GRID[a[0]][a[1]] != "#" \
        and GRID[b[0]][b[1]] != "#" and not needs_wall(a, b)


def flood_from(start):
    seen, dq = {start}, deque([start])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n not in seen and passable((r, c), n):
                seen.add(n)
                dq.append(n)
    return seen


entry_inner = gate_cells(*GATES["entry"])[0]
reach = flood_from(entry_inner)
open_cells = {(r, c) for r in range(N) for c in range(N) if GRID[r][c] != "#"}
unreachable = open_cells - reach
assert not unreachable, f"unreachable open cells: {sorted(unreachable)}"
exit_inner = gate_cells(*GATES["exit"])[0]
assert exit_inner in reach, "exit not reachable from entry"

wide = [(r, c) for r in range(N - 1) for c in range(N - 1)
        if all(GRID[r + dr][c + dc] == "." for dr in (0, 1) for dc in (0, 1))]
if wide:
    print("WARNING: 2x2 open corridor blocks (corridor wider than 1 cell) at (row, col):", wide)

# ------------------------------------------------------------------ build MJCF
def add_box(parent, name, x, y, z, hx, hy, hz, rgba, collide=True):
    attrs = dict(type="box", pos=f"{x:.4f} {y:.4f} {z:.4f}",
                 size=f"{hx:.4f} {hy:.4f} {hz:.4f}", rgba=rgba)
    if name:
        attrs["name"] = name
    if not collide:
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
    return ET.SubElement(parent, "geom", **attrs)


def create_maze():
    mujoco = ET.Element("mujoco", model="nidar_airmouse_arena")
    worldbody = ET.SubElement(mujoco, "worldbody")

    # lighting (no shadows, so the ceiling net does not darken the arena) and floor
    ET.SubElement(worldbody, "light", pos=f"{W/2:.3f} {W/2:.3f} 10", dir="0 0 -1",
                  castshadow="false")
    fh = W / 2 + FLOOR_MARGIN
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size=f"{fh:.3f} {fh:.3f} 0.1",
                  rgba="0.8 0.8 0.8 1", pos=f"{W/2:.4f} {W/2:.4f} 0")

    # walls (8 ft high: centre z = H/2, half-height = H/2)
    hz = WALL_H / 2
    for i, (x0, y0, x1, y1) in enumerate(wall_rects):
        add_box(worldbody, f"wall_{i}", (x0 + x1) / 2, (y0 + y1) / 2, hz,
                (x1 - x0) / 2, (y1 - y0) / 2, hz, "0.4 0.4 0.4 1")

    # ceiling net (transparent). Collidable by default so the drone cannot leave over the walls.
    if CEILING_NET:
        add_box(worldbody, "ceiling_net", W / 2, W / 2, WALL_H + 0.005,
                W / 2, W / 2, 0.005, "0.7 0.7 0.7 0.12", collide=CEILING_COLLIDES)

    # gates: floor tile + visual posts on both jambs; launch pad outside the ENTRY gate
    outward = {"S": (0, -1), "N": (0, 1), "W": (-1, 0), "E": (1, 0)}
    seen_positions = {}
    for name, pos in GATES.items():
        seen_positions.setdefault(pos, []).append(name)
    gate_info = {}
    for (side, i), names in seen_positions.items():
        inner, _ = gate_cells(side, i)
        gx, gy = cx(inner[1]), cy(inner[0])
        ox, oy = outward[side]
        rgba = ("1 0.8 0 1" if len(names) > 1 else
                "0.1 0.8 0.2 1" if names[0] == "entry" else "0.9 0.1 0.1 1")
        tag = "_".join(names)
        add_box(worldbody, f"gate_{tag}_tile", gx, gy, 0.002, 0.4, 0.4, 0.002, rgba, collide=False)
        for s in (-1, 1):
            px = gx + (-oy) * s * (CELL_CLEAR / 2 + 0.03) + ox * (CELL_CLEAR / 2 + WALL_T / 2)
            py = gy + (ox) * s * (CELL_CLEAR / 2 + 0.03) + oy * (CELL_CLEAR / 2 + WALL_T / 2)
            hx_ = 0.03 if ox == 0 else WALL_T / 2 + 0.01
            hy_ = 0.03 if oy == 0 else WALL_T / 2 + 0.01
            add_box(worldbody, f"gate_{tag}_post_{'p' if s > 0 else 'n'}", px, py, hz,
                    hx_, hy_, hz, rgba, collide=False)
        gate_info[tag] = {"side": side, "cell": label(*inner), "xy": [round(gx, 4), round(gy, 4)]}

    es, ei = GATES["entry"]
    inner, _ = gate_cells(es, ei)
    ox, oy = outward[es]
    d = CELL_CLEAR / 2 + WALL_T + LAUNCH_GAP + LAUNCH / 2
    lx, ly = cx(inner[1]) + ox * d, cy(inner[0]) + oy * d
    add_box(worldbody, "launch_pad", lx, ly, 0.002, LAUNCH / 2, LAUNCH / 2, 0.002,
            "0.8 0.2 0.2 1", collide=False)

    # doorway markers (visual only)
    if DOOR_MARKERS:
        for j, (a, b) in enumerate(DOORS):
            mx, my = (cx(a[1]) + cx(b[1])) / 2, (cy(a[0]) + cy(b[0])) / 2
            hx_, hy_ = (0.04, 0.4) if a[0] == b[0] else (0.4, 0.04)
            add_box(worldbody, f"door_marker_{j}", mx, my, 0.002, hx_, hy_, 0.002,
                    "0.2 0.5 0.9 1", collide=False)

    # dummy survivors at room centres (0.4 m cubes, as in the original script)
    survivors = []
    for n, k in enumerate(SURVIVOR_ROOMS, start=1):
        cells = rooms[k]
        door_cells = [x for d in DOORS for x in d if x in cells]

        def clearance(cell):
            return min((abs(cell[0] - d[0]) + abs(cell[1] - d[1]) for d in door_cells), default=0)

        sr, sc = max(sorted(cells), key=clearance)   # cell farthest from the doors
        sx, sy = cx(sc), cy(sr)
        add_box(worldbody, f"survivor_{n}", sx, sy, 0.2, 0.2, 0.2, 0.2, "0 0.8 0 1")
        survivors.append({"name": f"survivor_{n}", "room": k, "xy": [round(sx, 4), round(sy, 4)],
                          "grid_box": label(sr, sc)})

    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    tree.write("nidar_arena.xml", encoding="utf-8", xml_declaration=True)

    layout = {
        "arena_size_m": round(W, 3), "cell_pitch_m": P, "wall_thickness_m": WALL_T,
        "wall_height_m": round(WALL_H, 4), "grid_n": N,
        "grid_box_label": "column letter (A=west) + row number (1=south)",
        "gates": gate_info,
        "launch_pad": {"xy": [round(lx, 4), round(ly, 4)], "size_m": round(LAUNCH, 4)},
        "rooms": {k: {"cells": [label(r, c) for r, c in v],
                      "center_xy": [round(sum(cx(c) for _, c in v) / 4, 4),
                                    round(sum(cy(r) for r, _ in v) / 4, 4)]}
                  for k, v in rooms.items()},
        "doors": [[label(*a), label(*b)] for a, b in DOORS],
        "survivors": survivors,
        "walls_xyxy": [[round(v, 4) for v in w] for w in wall_rects],
    }
    with open("nidar_arena_layout.json", "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    print("Successfully generated nidar_arena.xml and nidar_arena_layout.json")
    print(f"  arena {W:.2f} m x {W:.2f} m | grid {N}x{N} | pitch {P} m | wall boxes {len(wall_rects)}")
    print(f"  rooms {sorted(rooms)} | doors {len(DOORS)} | gates {list(gate_info)}")
    print(f"  launch pad centre ({lx:.2f}, {ly:.2f}) m | all {len(open_cells)} open cells reachable from entry")


if __name__ == "__main__":
    create_maze()
