import xml.etree.ElementTree as ET

def create_maze():
    mujoco = ET.Element("mujoco")
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Add basic lighting and 15x15m floor
    ET.SubElement(worldbody, "light", pos="7.5 7.5 10", dir="0 0 -1")
    # Floor half-size is 7.5 for a 15m x 15m arena
    ET.SubElement(worldbody, "geom", type="plane", size="7.5 7.5 0.1", rgba="0.8 0.8 0.8 1", pos="7.5 7.5 0")

    # 1. Launch Area: 2ft x 2ft (0.61m x 0.61m)
    # MuJoCo size is half-length, so 0.305m x 0.305m
    ET.SubElement(worldbody, "geom", name="launch_pad", type="box", 
                  pos="1.0 1.0 0.01", size="0.305 0.305 0.01", rgba="0.8 0.2 0.2 1")

    # 2. Wall Height: 8 feet (2.4384 meters)
    # MuJoCo size is half-length, so z = 1.2192
    z_height = 1.2192
    wall_thickness = 0.1 # 10cm walls

    # Define walls as (x_center, y_center, x_half_width, y_half_length)
    # This representative layout enforces 1m corridors and 2x2m internal rooms
    walls = [
        # Outer boundary (15x15m)
        (7.5, 0, 7.5, wall_thickness),   (7.5, 15, 7.5, wall_thickness),
        (0, 7.5, wall_thickness, 7.5),   (15, 7.5, wall_thickness, 7.5),

        # Room 1 (Top-Left) - 2x2m internal space
        (2.1, 11.9, 2.1, wall_thickness),  (4.1, 13.4, wall_thickness, 1.5),
        
        # Room 2 (Top-Middle) - 2x2m internal space
        (7.5, 11.9, 2.0, wall_thickness),  
        (5.5, 13.4, wall_thickness, 1.5), 
        (9.5, 13.4, wall_thickness, 1.5),
        
        # Room 3 (Top-Right) - 2x2m internal space
        (12.9, 11.9, 2.1, wall_thickness), (10.9, 13.4, wall_thickness, 1.5),

        # Room 4 (Middle-Left) - 2x2m internal space
        (2.1, 8.5, 2.1, wall_thickness),   (2.1, 5.5, 2.1, wall_thickness),    
        (4.1, 7.0, wall_thickness, 1.5),

        # Room 5 (Middle-Right) - 2x2m internal space
        (12.9, 8.5, 2.1, wall_thickness),  (12.9, 5.5, 2.1, wall_thickness),   
        (10.9, 7.0, wall_thickness, 1.5),

        # Bottom-Center T-Wall / Room 6 structures enforcing 1m corridors
        (6.5, 4.0, 1.5, wall_thickness),   (6.0, 2.5, wall_thickness, 1.5),
        (12.5, 4.0, 2.5, wall_thickness),  (10.0, 2.5, wall_thickness, 1.5)
    ]

    for i, (x, y, hx, hy) in enumerate(walls):
        ET.SubElement(worldbody, "geom", name=f"wall_{i}", type="box", 
                      pos=f"{x} {y} {z_height}", size=f"{hx} {hy} {z_height}", rgba="0.4 0.4 0.4 1")

    # Add reference dummy survivors (0.2m cubes) in rooms for testing
    survivors = [(1.5, 13.5), (7.5, 13.5), (13.5, 13.5), (1.5, 7.0), (13.5, 7.0), (12.5, 2.0)]
    for i, (sx, sy) in enumerate(survivors):
        ET.SubElement(worldbody, "geom", name=f"survivor_{i+1}", type="box",
                      pos=f"{sx} {sy} 0.2", size="0.2 0.2 0.2", rgba="0 0.8 0 1")

    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    tree.write("nidar_arena.xml", encoding="utf-8", xml_declaration=True)
    print("Successfully generated nidar_arena.xml")

if __name__ == "__main__":
    create_maze()