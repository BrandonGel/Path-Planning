import python_motion_planning as pmp
import numpy as np

def convert_to_pixel(x,y):
    pixel_x = int(np.round(x))
    pixel_y = int(np.round(-1 - y))
    return pixel_x, pixel_y

def convert_grid_to_yaml(env: pmp.Grid, filename: str = None):
    """
    Convert a grid to a YAML-like text file.
    Args:
        env: The environment to convert.
        filename: The filename to save the YAML-like text file.
    Returns:
        None
    """
    xlen, ylen = env.x_range, env.y_range
    obstacle_map = np.zeros((ylen, xlen))
    for i in range(xlen):
        for j in range(ylen):
            if (i, j) in env.obstacles:
                pixel_x, pixel_y = convert_to_pixel(i, j)
                obstacle_map[pixel_y, pixel_x] = 1

    # Convert the obstacle_map to . and T
    symbol_map = np.where(obstacle_map == 1, 'T', '.')
    
    # Write to YAML-like text file
    if filename is None:
        filename = 'obstacle_map.yaml'
    elif not filename.endswith('.yaml'):
        filename = filename + '.yaml'
    with open(filename, 'w') as f:
        f.write('dimensions:\n ' + str(xlen) + ' ' + str(ylen) + '\n')
        f.write('obstacle:\n ')
        for row in symbol_map:
            f.write(''.join(row) + '\n ')

def convert_map_to_yaml(env: pmp.Map, filename: str = None):
    """
    Convert a map to a YAML-like text file.
    Args:
        env: The environment to convert.
        filename: The filename to save the YAML-like text file.
    Returns:
        None
    """
    xlen, ylen = env.x_range, env.y_range
    
    # Write to YAML-like text file
    if filename is None:
        filename = 'obstacle_map.yaml'
    elif not filename.endswith('.yaml'):
        filename = filename + '.yaml'
    with open(filename, 'w') as f:
        f.write('dimensions:\n ' + str(xlen) + ' ' + str(ylen) + '\n')
        f.write('obstacle_rect:\n')
        if env.obs_rect:
            for rect in env.obs_rect:
                # rect format: [x, y, width, height]
                f.write('  - [' + ', '.join(str(x) for x in rect) + ']\n')
        f.write('obstacle_circ:\n')
        if env.obs_circ:
            for circ in env.obs_circ:
                # circ format: [x, y, radius]
                f.write('  - [' + ', '.join(str(x) for x in circ) + ']\n')

def convert_from_pixel(pixel_x, pixel_y, ylen):
    """
    Convert pixel coordinates back to grid coordinates.
    Reverse of convert_to_pixel.
    Args:
        pixel_x: pixel x coordinate
        pixel_y: pixel y coordinate
    Returns:
        x, y: grid coordinates
    """
    x = pixel_x
    y = ylen - 1 - pixel_y
    return x, y

def read_grid_from_yaml(filename: str):
    """
    Read a YAML file and recreate a Grid environment.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        env: A Grid object with obstacles loaded from the YAML file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse dimensions
    dimensions_line = None
    obstacle_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('dimensions:'):
            dimensions_line = lines[i + 1].strip()
            xlen, ylen = map(int, dimensions_line.split())
            break
    
    # Find obstacle section
    for i, line in enumerate(lines):
        if line.strip().startswith('obstacle:'):
            obstacle_start_idx = i + 1
            break
    
    if obstacle_start_idx is None:
        raise ValueError("Could not find 'obstacle:' section in YAML file")
    
    # Parse obstacle rows
    obstacle_rows = []
    for i in range(obstacle_start_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('obstacle'):
            obstacle_rows.append(line)
        elif line.startswith('obstacle'):
            break
    
    # Convert obstacle map back to obstacle coordinates
    obstacles = set()
    for pixel_y, row in enumerate(obstacle_rows):
        for pixel_x, char in enumerate(row):
            if char == 'T':  # Obstacle
                x, y = convert_from_pixel(pixel_x, pixel_y,ylen)
                # Ensure coordinates are within bounds
                if 0 <= x < xlen and 0 <= y < ylen:
                    obstacles.add((x, y))
    
    # Create Grid object
    env = pmp.Grid(xlen, ylen)
    env.update(obstacles)
    
    return env

def read_map_from_yaml(filename: str):
    """
    Read a YAML file and recreate a Map environment.
    Args:
        filename: The filename of the YAML file to read.
    Returns:
        env: A Map object with obstacles loaded from the YAML file.
    """
    import re
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Parse dimensions
    dim_match = re.search(r'dimensions:\s*\n\s*(\d+)\s+(\d+)', content)
    if not dim_match:
        raise ValueError("Could not find 'dimensions:' in YAML file")
    xlen, ylen = int(dim_match.group(1)), int(dim_match.group(2))
    
    # Parse obstacle_rect
    obs_rect = []
    rect_match = re.search(r'obstacle_rect:\s*\n(.*?)(?=obstacle_circ:|$)', content, re.DOTALL)
    if rect_match:
        rect_section = rect_match.group(1)
        # Find all list patterns like [x, y, width, height]
        rect_pattern = r'\[([\d\s,]+)\]'
        for match in re.finditer(rect_pattern, rect_section):
            rect_str = match.group(1)
            rect = [int(x.strip()) for x in rect_str.split(',')]
            if len(rect) == 4:  # [x, y, width, height]
                obs_rect.append(rect)
    
    # Parse obstacle_circ
    obs_circ = []
    circ_match = re.search(r'obstacle_circ:\s*\n(.*?)$', content, re.DOTALL)
    if circ_match:
        circ_section = circ_match.group(1)
        # Find all list patterns like [x, y, radius]
        circ_pattern = r'\[([\d\s,]+)\]'
        for match in re.finditer(circ_pattern, circ_section):
            circ_str = match.group(1)
            circ = [int(x.strip()) for x in circ_str.split(',')]
            if len(circ) == 3:  # [x, y, radius]
                obs_circ.append(circ)
    
    # Create Map object
    env = pmp.Map(xlen, ylen)
    env.update(obs_rect=obs_rect if obs_rect else None, 
               obs_circ=obs_circ if obs_circ else None)
    
    return env