import json
import os
import numpy as np
from collections import defaultdict, Counter
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Dict, Any, Tuple, Set
from itertools import product

# --- Configuration ---
# PUZZLE_JSON_PATH = "DATA1/training/d4469b4b.json"
PUZZLE_JSON_PATH = "DATA1/training/007bbfb7.json"
# PUZZLE_JSON_PATH = "DATA1/evaluation/00576224.json"
OUTPUT_DIR = "FEATURES"  # Save in the current directory for now

# --- Basic Analysis Functions ---

def analyze_grid_dimensions(grid: List[List[int]]) -> Tuple[int, int]:
    """Analyzes the dimensions (rows, columns) of a grid."""
    if not grid:
        return 0, 0
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return rows, cols

def find_unique_colors(grid: List[List[int]]) -> Set[int]:
    """Finds all unique integer values (colors) present in a grid."""
    colors = set()
    for row in grid:
        for cell in row:
            colors.add(cell)
    return colors

# --- Enhanced Analysis Functions ---

def identify_objects(grid: List[List[int]], background_color=0) -> List[Dict]:
    """
    Identifies distinct objects in the grid based on connected components.
    Returns a list of object properties (color, position, size, bounding box).
    """
    grid_array = np.array(grid)
    objects = []
    
    # Process each color separately
    unique_colors = set(np.unique(grid_array)) - {background_color}
    
    for color in unique_colors:
        # Create binary mask for this color
        mask = (grid_array == color)
        
        # Label connected components
        labeled_array, num_features = label(mask)
        
        for i in range(1, num_features + 1):
            # Get object mask
            obj_mask = (labeled_array == i)
            obj_pixels = np.argwhere(obj_mask)
            
            if len(obj_pixels) == 0:
                continue
                
            # Calculate properties
            min_y, min_x = obj_pixels.min(axis=0)
            max_y, max_x = obj_pixels.max(axis=0)
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            
            # Calculate center
            center_y = (min_y + max_y) / 2
            center_x = (min_x + max_x) / 2
            
            # Create object dictionary
            obj = {
                'color': int(color),
                'position': (int(min_y), int(min_x)),
                'size': (int(height), int(width)),
                'area': int(np.sum(obj_mask)),
                'center': (float(center_y), float(center_x)),
                'bounding_box': ((int(min_y), int(min_x)), (int(max_y), int(max_x))),
                'pixels': obj_pixels.tolist(),
                'shape_complexity': calculate_shape_complexity(obj_mask)
            }
            
            objects.append(obj)
    
    return objects

def calculate_shape_complexity(mask):
    """Calculate a simple complexity score for a shape based on perimeter/area ratio."""
    # Convert to binary image if not already
    binary = mask.astype(np.uint8)
    
    # Find perimeter pixels (pixels that have at least one non-object neighbor)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    interior = binary.copy()
    for i in range(1, binary.shape[0]-1):
        for j in range(1, binary.shape[1]-1):
            if binary[i, j] == 1:
                neighborhood = binary[i-1:i+2, j-1:j+2]
                if np.sum(neighborhood * kernel) == 8:  # All neighbors are 1
                    interior[i, j] = 2  # Mark as interior
    
    perimeter_pixels = np.sum(binary) - np.sum(interior == 2)
    area = np.sum(binary)
    
    if area == 0:
        return 0
        
    # Simple complexity measure: perimeter / sqrt(area)
    # This adjusts for the fact that perimeter scales with sqrt(area) for similar shapes
    return perimeter_pixels / np.sqrt(area)

def detect_grid_structure(grid: List[List[int]]) -> Dict:
    """
    Detects if the puzzle includes a regular grid structure.
    Returns information about cell size, grid dimensions, etc.
    """
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    result = {'has_grid_structure': False}
    
    # Look for horizontal and vertical lines of the same color
    for color in set(np.unique(grid_array)):
        color_mask = (grid_array == color)
        
        # Check for horizontal lines
        h_lines = []
        for r in range(rows):
            if np.all(color_mask[r, :]):
                h_lines.append(r)
        
        # Check for vertical lines
        v_lines = []
        for c in range(cols):
            if np.all(color_mask[:, c]):
                v_lines.append(c)
        
        # If we have both horizontal and vertical lines of the same color
        if len(h_lines) > 1 and len(v_lines) > 1:
            # Calculate cell sizes
            h_spacing = [h_lines[i+1] - h_lines[i] for i in range(len(h_lines)-1)]
            v_spacing = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines)-1)]
            
            # Check if spacings are regular
            if len(set(h_spacing)) == 1 and len(set(v_spacing)) == 1:
                result['has_grid_structure'] = True
                result['grid_color'] = int(color)
                result['h_lines'] = h_lines
                result['v_lines'] = v_lines
                result['cell_height'] = h_spacing[0] - 1  # -1 because line itself takes space
                result['cell_width'] = v_spacing[0] - 1
                result['grid_cells_y'] = len(h_lines) - 1
                result['grid_cells_x'] = len(v_lines) - 1
                break
    
    return result

def detect_symmetry(grid: List[List[int]]) -> Dict:
    """
    Detects various forms of symmetry in the grid.
    Returns information about horizontal, vertical, diagonal symmetry.
    """
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    result = {
        'horizontal_symmetry': False,
        'vertical_symmetry': False,
        'diagonal_symmetry_main': False,
        'diagonal_symmetry_anti': False,
        'rotational_symmetry_180': False,
        'rotational_symmetry_90': False
    }
    
    # Check horizontal symmetry (top-bottom)
    horizontal_diffs = 0
    for r in range(rows // 2):
        if not np.array_equal(grid_array[r, :], grid_array[rows-1-r, :]):
            horizontal_diffs += 1
    result['horizontal_symmetry'] = (horizontal_diffs == 0)
    result['horizontal_symmetry_score'] = 1.0 - (horizontal_diffs / (rows // 2)) if rows > 1 else 0
    
    # Check vertical symmetry (left-right)
    vertical_diffs = 0
    for c in range(cols // 2):
        if not np.array_equal(grid_array[:, c], grid_array[:, cols-1-c]):
            vertical_diffs += 1
    result['vertical_symmetry'] = (vertical_diffs == 0)
    result['vertical_symmetry_score'] = 1.0 - (vertical_diffs / (cols // 2)) if cols > 1 else 0
    
    # Check diagonal symmetry (main diagonal: top-left to bottom-right)
    if rows == cols:  # Only check diagonal symmetry for square grids
        diag_main_diffs = 0
        for r in range(rows):
            for c in range(r):
                if grid_array[r, c] != grid_array[c, r]:
                    diag_main_diffs += 1
        result['diagonal_symmetry_main'] = (diag_main_diffs == 0)
        max_diag_diffs = (rows * (rows - 1)) // 2
        result['diagonal_symmetry_main_score'] = 1.0 - (diag_main_diffs / max_diag_diffs) if max_diag_diffs > 0 else 0
        
        # Check anti-diagonal symmetry (top-right to bottom-left)
        diag_anti_diffs = 0
        for r in range(rows):
            for c in range(cols):
                if r + c < rows - 1:
                    if grid_array[r, c] != grid_array[rows-1-c, cols-1-r]:
                        diag_anti_diffs += 1
        result['diagonal_symmetry_anti'] = (diag_anti_diffs == 0)
        result['diagonal_symmetry_anti_score'] = 1.0 - (diag_anti_diffs / max_diag_diffs) if max_diag_diffs > 0 else 0
    
    # Check 180-degree rotational symmetry
    rot_180_diffs = 0
    for r in range(rows):
        for c in range(cols):
            if grid_array[r, c] != grid_array[rows-1-r, cols-1-c]:
                rot_180_diffs += 1
    total_cells = rows * cols
    result['rotational_symmetry_180'] = (rot_180_diffs == 0)
    result['rotational_symmetry_180_score'] = 1.0 - (rot_180_diffs / (total_cells // 2)) if total_cells > 1 else 0
    
    # Check 90-degree rotational symmetry (only for square grids)
    if rows == cols:
        rot_90_diffs = 0
        for r in range(rows):
            for c in range(cols):
                if grid_array[r, c] != grid_array[c, rows-1-r]:
                    rot_90_diffs += 1
        result['rotational_symmetry_90'] = (rot_90_diffs == 0)
        result['rotational_symmetry_90_score'] = 1.0 - (rot_90_diffs / total_cells) if total_cells > 0 else 0
    
    return result

def analyze_pattern_frequency(grid: List[List[int]], pattern_size=(2, 2)) -> Dict:
    """
    Analyzes the frequency of different patterns in the grid.
    Returns the most common pattern and its frequency.
    """
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    pattern_h, pattern_w = pattern_size
    
    if rows < pattern_h or cols < pattern_w:
        return {
            'status': 'grid_too_small',
            'patterns': []
        }
    
    patterns = []
    pattern_count = Counter()
    
    # Extract all patterns of the given size
    for r in range(rows - pattern_h + 1):
        for c in range(cols - pattern_w + 1):
            pattern = grid_array[r:r+pattern_h, c:c+pattern_w].tolist()
            pattern_tuple = tuple(map(tuple, pattern))
            pattern_count[pattern_tuple] += 1
            if pattern_tuple not in patterns:
                patterns.append(pattern_tuple)
    
    # Find most common patterns
    most_common = pattern_count.most_common(5)
    
    return {
        'status': 'success',
        'pattern_size': pattern_size,
        'unique_patterns_count': len(patterns),
        'most_common_patterns': [
            {
                'pattern': [list(row) for row in pattern],
                'count': count,
                'frequency': count / (rows - pattern_h + 1) / (cols - pattern_w + 1)
            }
            for pattern, count in most_common
        ]
    }

def analyze_color_distribution(grid: List[List[int]]) -> Dict:
    """
    Analyzes the distribution of colors in the grid.
    Returns color counts, percentages, and distribution properties.
    """
    grid_array = np.array(grid)
    flat_grid = grid_array.flatten()
    unique, counts = np.unique(flat_grid, return_counts=True)
    total_cells = len(flat_grid)
    
    color_info = []
    for color, count in zip(unique, counts):
        color_info.append({
            'color': int(color),
            'count': int(count),
            'percentage': float(count / total_cells * 100)
        })
    
    # Sort by count (most frequent first)
    color_info.sort(key=lambda x: x['count'], reverse=True)
    
    # Calculate entropy as a measure of color diversity
    entropy = 0
    for info in color_info:
        p = info['percentage'] / 100
        if p > 0:
            entropy -= p * np.log2(p)
    
    return {
        'color_counts': color_info,
        'most_common_color': int(color_info[0]['color']) if color_info else None,
        'unique_colors_count': len(color_info),
        'color_entropy': float(entropy),
        'is_binary': len(color_info) <= 2,
        'is_grayscale': all(0 <= c['color'] <= 9 for c in color_info)
    }

def compare_input_output(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict:
    """
    Compares input and output grids to identify transformations.
    Returns information about color mappings, size changes, etc.
    """
    input_array = np.array(input_grid)
    output_array = np.array(output_grid)
    input_rows, input_cols = input_array.shape
    output_rows, output_cols = output_array.shape
    
    result = {
        'size_relation': {
            'rows_ratio': output_rows / input_rows if input_rows > 0 else None,
            'cols_ratio': output_cols / input_cols if input_cols > 0 else None,
            'area_ratio': (output_rows * output_cols) / (input_rows * input_cols) if (input_rows * input_cols) > 0 else None
        },
        'size_transformation': None,
        'color_transformation': None,
        'preserved_colors': [],
    }
    
    # Determine size transformation
    if input_rows == output_rows and input_cols == output_cols:
        result['size_transformation'] = 'preserved'
    elif output_rows == input_rows * 2 and output_cols == input_cols * 2:
        result['size_transformation'] = 'doubled'
    elif output_rows == input_rows // 2 and output_cols == input_cols // 2:
        result['size_transformation'] = 'halved'
    elif output_rows > input_rows or output_cols > input_cols:
        result['size_transformation'] = 'expanded'
    elif output_rows < input_rows or output_cols < input_cols:
        result['size_transformation'] = 'contracted'
    
    # Analyze color mappings if sizes allow direct comparison
    if input_rows == output_rows and input_cols == output_cols:
        # Count how many cells have the same color in both grids
        matching_cells = np.sum(input_array == output_array)
        result['matching_cells_percentage'] = matching_cells / (input_rows * input_cols) * 100
        
        # Check for specific color transformations
        input_colors = set(np.unique(input_array))
        output_colors = set(np.unique(output_array))
        
        # Colors that appear in both input and output
        preserved_colors = input_colors.intersection(output_colors)
        result['preserved_colors'] = sorted(list(preserved_colors))
        
        # New colors that appear only in output
        new_colors = output_colors - input_colors
        result['new_colors'] = sorted(list(new_colors))
        
        # Colors that disappeared from input to output
        disappeared_colors = input_colors - output_colors
        result['disappeared_colors'] = sorted(list(disappeared_colors))
        
        # Check for color inversion (particularly in binary grids)
        if len(input_colors) == 2 and len(output_colors) == 2:
            input_color_list = sorted(list(input_colors))
            output_color_list = sorted(list(output_colors))
            
            # Check if colors are preserved but swapped positions
            if set(input_color_list) == set(output_color_list):
                swapped_grid = np.zeros_like(input_array)
                for r in range(input_rows):
                    for c in range(input_cols):
                        if input_array[r, c] == input_color_list[0]:
                            swapped_grid[r, c] = input_color_list[1]
                        else:
                            swapped_grid[r, c] = input_color_list[0]
                
                if np.array_equal(swapped_grid, output_array):
                    result['color_transformation'] = 'inverted'
    
    # Check if output is a shifted version of input
    if input_rows == output_rows and input_cols == output_cols:
        for row_shift in range(-input_rows+1, input_rows):
            for col_shift in range(-input_cols+1, input_cols):
                shifted = np.roll(input_array, (row_shift, col_shift), axis=(0, 1))
                if np.array_equal(shifted, output_array):
                    result['transformation'] = 'shift'
                    result['shift_amount'] = (row_shift, col_shift)
                    return result
    
    # Check if output is a rotated version of input
    if input_rows == input_cols and output_rows == output_cols and input_rows == output_rows:
        # Check 90, 180, 270 degree rotations
        for k in range(1, 4):
            rotated = np.rot90(input_array, k=k)
            if np.array_equal(rotated, output_array):
                result['transformation'] = f'rotation_{k*90}'
                return result
    
    # Check if output is a flipped version of input
    if input_rows == output_rows and input_cols == output_cols:
        # Check horizontal flip
        flipped_h = np.fliplr(input_array)
        if np.array_equal(flipped_h, output_array):
            result['transformation'] = 'horizontal_flip'
            return result
            
        # Check vertical flip
        flipped_v = np.flipud(input_array)
        if np.array_equal(flipped_v, output_array):
            result['transformation'] = 'vertical_flip'
            return result
            
        # Check both flips (equivalent to 180-degree rotation for square grids)
        flipped_both = np.flipud(np.fliplr(input_array))
        if np.array_equal(flipped_both, output_array):
            result['transformation'] = 'both_flip'
            return result
    
    # No simple transformation found
    result['transformation'] = 'complex'
    return result

def detect_spatial_patterns(grid: List[List[int]]) -> Dict:
    """
    Detect common spatial patterns in the grid.
    Returns information about lines, corners, enclosed regions, etc.
    """
    grid_array = np.array(grid)
    result = {
        'lines': {'horizontal': 0, 'vertical': 0, 'diagonal': 0},
        'corners': 0,
        'enclosed_regions': 0,
        'isolated_pixels': 0,
        'border_pixels': 0
    }
    
    rows, cols = grid_array.shape
    
    # Check for horizontal, vertical, and diagonal lines
    for color in set(np.unique(grid_array)):
        color_mask = (grid_array == color)
        
        # Horizontal lines
        for r in range(rows):
            line_length = 0
            for c in range(cols):
                if color_mask[r, c]:
                    line_length += 1
                else:
                    if line_length >= 3:  # Consider 3+ consecutive pixels as a line
                        result['lines']['horizontal'] += 1
                    line_length = 0
            # Check the last line if it extends to the edge
            if line_length >= 3:
                result['lines']['horizontal'] += 1
        
        # Vertical lines
        for c in range(cols):
            line_length = 0
            for r in range(rows):
                if color_mask[r, c]:
                    line_length += 1
                else:
                    if line_length >= 3:
                        result['lines']['vertical'] += 1
                    line_length = 0
            if line_length >= 3:
                result['lines']['vertical'] += 1
        
        # Diagonal lines (main diagonal: top-left to bottom-right)
        for offset in range(-(rows-1), cols):
            diag = np.diagonal(color_mask, offset)
            line_length = 0
            for i in range(len(diag)):
                if diag[i]:
                    line_length += 1
                else:
                    if line_length >= 3:
                        result['lines']['diagonal'] += 1
                    line_length = 0
            if line_length >= 3:
                result['lines']['diagonal'] += 1
        
        # Diagonal lines (anti-diagonal: top-right to bottom-left)
        flipped = np.fliplr(color_mask)
        for offset in range(-(rows-1), cols):
            diag = np.diagonal(flipped, offset)
            line_length = 0
            for i in range(len(diag)):
                if diag[i]:
                    line_length += 1
                else:
                    if line_length >= 3:
                        result['lines']['diagonal'] += 1
                    line_length = 0
            if line_length >= 3:
                result['lines']['diagonal'] += 1
    
    # Detect corners (simple version)
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            color = grid_array[r, c]
            if color == 0:  # Skip background
                continue
            
            # Check for L-shape patterns
            patterns = [
                # Top-left corner
                [[0, 0, 0],
                 [0, 1, 1],
                 [0, 1, 0]],
                # Top-right corner
                [[0, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0]],
                # Bottom-left corner
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]],
                # Bottom-right corner
                [[0, 1, 0],
                 [1, 1, 0],
                 [0, 0, 0]]
            ]
            
            for pattern in patterns:
                match = True
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if 0 <= r+dr < rows and 0 <= c+dc < cols:
                            if pattern[dr+1][dc+1] == 1:
                                if grid_array[r+dr, c+dc] != color:
                                    match = False
                                    break
                            else:
                                if grid_array[r+dr, c+dc] == color:
                                    match = False
                                    break
                    if not match:
                        break
                
                if match:
                    result['corners'] += 1
    
    # Detect isolated pixels
    for r in range(rows):
        for c in range(cols):
            color = grid_array[r, c]
            if color == 0:  # Skip background
                continue
            
            isolated = True
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid_array[nr, nc] == color:
                        isolated = False
                        break
                if not isolated:
                    break
            
            if isolated:
                result['isolated_pixels'] += 1
    
    # Detect border pixels
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:  # If on the border
                color = grid_array[r, c]
                if color != 0:  # If not background
                    result['border_pixels'] += 1
    
    # Detect enclosed regions (simplistic approach using binary fill)
    for color in set(np.unique(grid_array)) - {0}:  # Skip background
        binary = (grid_array == color).astype(np.uint8)
        
        # Create a padded version with background around the edges
        padded = np.zeros((rows+2, cols+2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary
        
        # Flood fill from the outside
        filled = padded.copy()
        nodes = [(0, 0)]
        while nodes:
            r, c = nodes.pop()
            if 0 <= r < rows+2 and 0 <= c < cols+2 and filled[r, c] == 0:
                filled[r, c] = 2  # Mark as visited
                nodes.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        
        # Count enclosed regions (areas of 0s that weren't reached by the flood fill)
        enclosed = (filled[1:-1, 1:-1] == 0).astype(np.uint8)
        if np.any(enclosed):
            labeled, num = label(enclosed)
            result['enclosed_regions'] += num
    
    return result

def visualize_grid(grid: List[List[int]], filename: str, title: str = None):
    """
    Creates a visualization of the grid and saves it to a file.
    """
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    
    # Create a color map that's distinct and readable
    unique_colors = sorted(list(set(np.unique(grid_array))))
    n_colors = len(unique_colors)
    
    # Generate a distinct colormap
    if n_colors <= 10:
        cmap = plt.cm.get_cmap('tab10', n_colors)
    else:
        cmap = plt.cm.get_cmap('viridis', n_colors)
    
    color_map = {color: mcolors.to_hex(cmap(i)) for i, color in enumerate(unique_colors)}
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(max(6, cols/2), max(6, rows/2)))
    
    # Plot each cell with its color
    for r in range(rows):
        for c in range(cols):
            color_val = grid_array[r, c]
            rect = patches.Rectangle((c, rows-1-r), 1, 1, linewidth=1, 
                                   edgecolor='black', facecolor=color_map[color_val])
            ax.add_patch(rect)
            
            # Add text label for the color value (optional)
            ax.text(c + 0.5, rows-1-r + 0.5, str(color_val), 
                   ha='center', va='center', fontsize=max(8, min(24, 72/max(rows, cols))))
    
    # Set limits and aspect
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Add grid lines
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    
    # Adjust axis ticks
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    
    return filename

def identify_task_type(analysis_results):
    """
    Try to identify the likely ARC task type based on the analysis results.
    Returns a dictionary with task type probabilities and explanations.
    """
    task_types = {
        'object_preservation': 0,
        'object_counting': 0,
        'pattern_completion': 0,
        'symmetry': 0,
        'color_mapping': 0,
        'spatial_transformation': 0,
        'grid_manipulation': 0,
        'background_transformation': 0,
        'object_matching': 0,
        'sorting': 0
    }
    
    explanations = {}
    
    # Check for object preservation/counting patterns
    for idx, example in enumerate(analysis_results['train_examples']):
        input_objs = len(example['input']['objects'])
        output_objs = len(example['output']['objects'])
        
        if input_objs == output_objs:
            task_types['object_preservation'] += 1
            explanations.setdefault('object_preservation', []).append(
                f"Example {idx+1}: Same number of objects ({input_objs})"
            )
        else:
            task_types['object_counting'] += 1
            explanations.setdefault('object_counting', []).append(
                f"Example {idx+1}: Objects changed from {input_objs} to {output_objs}"
            )
    
    # Check for symmetry tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_sym = example['input']['symmetry']
        output_sym = example['output']['symmetry']
        
        # If output has symmetry that input doesn't
        if (not input_sym['horizontal_symmetry'] and output_sym['horizontal_symmetry']) or \
           (not input_sym['vertical_symmetry'] and output_sym['vertical_symmetry']):
            task_types['symmetry'] += 2
            explanations.setdefault('symmetry', []).append(
                f"Example {idx+1}: Output has symmetry not present in input"
            )
        
        # If both have similar symmetry properties
        if input_sym['horizontal_symmetry'] == output_sym['horizontal_symmetry'] and \
           input_sym['vertical_symmetry'] == output_sym['vertical_symmetry']:
            task_types['symmetry'] += 1
    
    # Check for color mapping tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        rel = example['input_output_relation']
        
        if 'color_transformation' in rel and rel['color_transformation']:
            task_types['color_mapping'] += 2
            explanations.setdefault('color_mapping', []).append(
                f"Example {idx+1}: Clear color transformation ({rel['color_transformation']})"
            )
        
        # If there's a large set of new colors
        if 'new_colors' in rel and len(rel['new_colors']) > 0:
            task_types['color_mapping'] += len(rel['new_colors'])
            explanations.setdefault('color_mapping', []).append(
                f"Example {idx+1}: {len(rel['new_colors'])} new colors in output"
            )
    
    # Check for spatial transformation tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        rel = example['input_output_relation']
        
        if 'transformation' in rel and rel['transformation'] != 'complex':
            task_types['spatial_transformation'] += 2
            explanations.setdefault('spatial_transformation', []).append(
                f"Example {idx+1}: {rel['transformation']} transformation detected"
            )
    
    # Check for grid manipulation tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_grid = example['input']['grid_structure']
        output_grid = example['output']['grid_structure']
        
        if input_grid['has_grid_structure'] and output_grid['has_grid_structure']:
            task_types['grid_manipulation'] += 2
            explanations.setdefault('grid_manipulation', []).append(
                f"Example {idx+1}: Both input and output have grid structures"
            )
    
    # Check for pattern completion tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_pattern = example['input']['pattern_frequency_2x2']
        output_pattern = example['output']['pattern_frequency_2x2']
        
        if input_pattern['status'] == 'success' and output_pattern['status'] == 'success':
            if input_pattern['unique_patterns_count'] < output_pattern['unique_patterns_count']:
                task_types['pattern_completion'] += 1
                explanations.setdefault('pattern_completion', []).append(
                    f"Example {idx+1}: Output has more unique patterns than input"
                )
    
    # Check for background transformation tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_color = example['input']['color_distribution']
        output_color = example['output']['color_distribution']
        
        # If the most common color is different
        if input_color['most_common_color'] != output_color['most_common_color']:
            task_types['background_transformation'] += 1
            explanations.setdefault('background_transformation', []).append(
                f"Example {idx+1}: Most common color changed from {input_color['most_common_color']} to {output_color['most_common_color']}"
            )
    
    # Check for object matching tasks
    if len(analysis_results['train_examples']) > 1:
        objs_match = True
        for i in range(1, len(analysis_results['train_examples'])):
            if len(analysis_results['train_examples'][i]['input']['objects']) != \
               len(analysis_results['train_examples'][0]['input']['objects']):
                objs_match = False
                break
        
        if objs_match:
            task_types['object_matching'] += 2
            explanations.setdefault('object_matching', []).append(
                "All training examples have the same number of input objects"
            )
    
    # Check for sorting tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_objs = example['input']['objects']
        output_objs = example['output']['objects']
        
        if len(input_objs) == len(output_objs) and len(input_objs) > 1:
            # Check if output objects are more ordered by position
            input_positions = [(obj['position'][0], obj['position'][1]) for obj in input_objs]
            output_positions = [(obj['position'][0], obj['position'][1]) for obj in output_objs]
            
            # Check if positions are aligned horizontally or vertically in output
            output_row_aligned = len(set(pos[0] for pos in output_positions)) <= len(input_objs) // 2
            output_col_aligned = len(set(pos[1] for pos in output_positions)) <= len(input_objs) // 2
            
            if output_row_aligned or output_col_aligned:
                task_types['sorting'] += 1
                explanations.setdefault('sorting', []).append(
                    f"Example {idx+1}: Objects appear more aligned in output"
                )
            
            # Check if objects are ordered by color
            input_colors = [obj['color'] for obj in input_objs]
            output_colors = [obj['color'] for obj in output_objs]
            
            input_sorted = sorted(input_colors)
            output_sorted = sorted(output_colors)
            
            if output_colors == output_sorted and input_colors != input_sorted:
                task_types['sorting'] += 2
                explanations.setdefault('sorting', []).append(
                    f"Example {idx+1}: Objects appear sorted by color in output"
                )
    
    # Normalize scores to get probabilities
    total_score = sum(task_types.values())
    if total_score > 0:
        probabilities = {task: score / total_score for task, score in task_types.items()}
    else:
        probabilities = {task: 0 for task in task_types}
    
    # Get top 3 most likely task types
    top_tasks = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    result = {
        'probabilities': probabilities,
        'top_tasks': top_tasks,
        'explanations': explanations
    }
    
    return result

########################

def identify_task_type(analysis_results):
    """
    Try to identify the likely ARC task type based on the analysis results.
    Returns a dictionary with task type probabilities and explanations.
    """
    task_types = {
        'object_preservation': 0,
        'object_counting': 0,
        'pattern_completion': 0,
        'symmetry': 0,
        'color_mapping': 0,
        'spatial_transformation': 0,
        'grid_manipulation': 0,
        'background_transformation': 0,
        'object_matching': 0,
        'sorting': 0
    }
    
    explanations = {}
    
    # Check for object preservation/counting patterns
    for idx, example in enumerate(analysis_results['train_examples']):
        input_objs = len(example['input']['objects'])
        output_objs = len(example['output']['objects'])
        
        if input_objs == output_objs:
            task_types['object_preservation'] += 1
            explanations.setdefault('object_preservation', []).append(
                f"Example {idx+1}: Same number of objects ({input_objs})"
            )
        else:
            task_types['object_counting'] += 1
            explanations.setdefault('object_counting', []).append(
                f"Example {idx+1}: Objects changed from {input_objs} to {output_objs}"
            )
    
    # Check for symmetry tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_sym = example['input']['symmetry']
        output_sym = example['output']['symmetry']
        
        # If output has symmetry that input doesn't
        if (not input_sym['horizontal_symmetry'] and output_sym['horizontal_symmetry']) or \
           (not input_sym['vertical_symmetry'] and output_sym['vertical_symmetry']):
            task_types['symmetry'] += 2
            explanations.setdefault('symmetry', []).append(
                f"Example {idx+1}: Output has symmetry not present in input"
            )
        
        # If both have similar symmetry properties
        if input_sym['horizontal_symmetry'] == output_sym['horizontal_symmetry'] and \
           input_sym['vertical_symmetry'] == output_sym['vertical_symmetry']:
            task_types['symmetry'] += 1
    
    # Check for color mapping tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        rel = example['input_output_relation']
        
        if 'color_transformation' in rel and rel['color_transformation']:
            task_types['color_mapping'] += 2
            explanations.setdefault('color_mapping', []).append(
                f"Example {idx+1}: Clear color transformation ({rel['color_transformation']})"
            )
        
        # If there's a large set of new colors
        if 'new_colors' in rel and len(rel['new_colors']) > 0:
            task_types['color_mapping'] += len(rel['new_colors'])
            explanations.setdefault('color_mapping', []).append(
                f"Example {idx+1}: {len(rel['new_colors'])} new colors in output"
            )
    
    # Check for spatial transformation tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        rel = example['input_output_relation']
        
        if 'transformation' in rel and rel['transformation'] != 'complex':
            task_types['spatial_transformation'] += 2
            explanations.setdefault('spatial_transformation', []).append(
                f"Example {idx+1}: {rel['transformation']} transformation detected"
            )
    
    # Check for grid manipulation tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_grid = example['input']['grid_structure']
        output_grid = example['output']['grid_structure']
        
        if input_grid['has_grid_structure'] and output_grid['has_grid_structure']:
            task_types['grid_manipulation'] += 2
            explanations.setdefault('grid_manipulation', []).append(
                f"Example {idx+1}: Both input and output have grid structures"
            )
    
    # Check for pattern completion tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_pattern = example['input']['pattern_frequency_2x2']
        output_pattern = example['output']['pattern_frequency_2x2']
        
        if input_pattern['status'] == 'success' and output_pattern['status'] == 'success':
            if input_pattern['unique_patterns_count'] < output_pattern['unique_patterns_count']:
                task_types['pattern_completion'] += 1
                explanations.setdefault('pattern_completion', []).append(
                    f"Example {idx+1}: Output has more unique patterns than input"
                )
    
    # Check for background transformation tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_color = example['input']['color_distribution']
        output_color = example['output']['color_distribution']
        
        # If the most common color is different
        if input_color['most_common_color'] != output_color['most_common_color']:
            task_types['background_transformation'] += 1
            explanations.setdefault('background_transformation', []).append(
                f"Example {idx+1}: Most common color changed from {input_color['most_common_color']} to {output_color['most_common_color']}"
            )
    
    # Check for object matching tasks
    if len(analysis_results['train_examples']) > 1:
        objs_match = True
        for i in range(1, len(analysis_results['train_examples'])):
            if len(analysis_results['train_examples'][i]['input']['objects']) != \
               len(analysis_results['train_examples'][0]['input']['objects']):
                objs_match = False
                break
        
        if objs_match:
            task_types['object_matching'] += 2
            explanations.setdefault('object_matching', []).append(
                "All training examples have the same number of input objects"
            )
    
    # Check for sorting tasks
    for idx, example in enumerate(analysis_results['train_examples']):
        input_objs = example['input']['objects']
        output_objs = example['output']['objects']
        
        if len(input_objs) == len(output_objs) and len(input_objs) > 1:
            # Check if output objects are more ordered by position
            input_positions = [(obj['position'][0], obj['position'][1]) for obj in input_objs]
            output_positions = [(obj['position'][0], obj['position'][1]) for obj in output_objs]
            
            # Check if positions are aligned horizontally or vertically in output
            output_row_aligned = len(set(pos[0] for pos in output_positions)) <= len(input_objs) // 2
            output_col_aligned = len(set(pos[1] for pos in output_positions)) <= len(input_objs) // 2
            
            if output_row_aligned or output_col_aligned:
                task_types['sorting'] += 1
                explanations.setdefault('sorting', []).append(
                    f"Example {idx+1}: Objects appear more aligned in output"
                )
            
            # Check if objects are ordered by color
            input_colors = [obj['color'] for obj in input_objs]
            output_colors = [obj['color'] for obj in output_objs]
            
            input_sorted = sorted(input_colors)
            output_sorted = sorted(output_colors)
            
            if output_colors == output_sorted and input_colors != input_sorted:
                task_types['sorting'] += 2
                explanations.setdefault('sorting', []).append(
                    f"Example {idx+1}: Objects appear sorted by color in output"
                )
    
    # Normalize scores to get probabilities
    total_score = sum(task_types.values())
    if total_score > 0:
        probabilities = {task: score / total_score for task, score in task_types.items()}
    else:
        probabilities = {task: 0 for task in task_types}
    
    # Get top 3 most likely task types
    top_tasks = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    result = {
        'probabilities': probabilities,
        'top_tasks': top_tasks,
        'explanations': explanations
    }
    
    return result

# --- Main Analysis Logic ---
def analyze_puzzle(json_path: str, output_dir: str, disable_viz=False):
    """
    Loads a puzzle JSON, performs analysis, and writes results to a text file.
    Returns analysis results dictionary.
    """
    puzzle_filename = os.path.basename(json_path)
    puzzle_name = os.path.splitext(puzzle_filename)[0]
    output_filename = f"{puzzle_name}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Create a directory for visualizations
    visualizations_dir = os.path.join(output_dir, f"{puzzle_name}_visualizations")
    
    print(f"Analyzing puzzle: {puzzle_filename}")

    try:
        with open(json_path, 'r') as f:
            puzzle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Puzzle file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {json_path}: {e}")
        return None

    analysis_results = {}

    # --- Perform Analysis on Training Examples ---
    analysis_results['train_examples'] = []
    all_train_input_colors = set()
    all_train_output_colors = set()
    
    # Create visualizations directory if we're going to save visualizations
    if not disable_viz and not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    for i, example in enumerate(puzzle_data.get('train', [])):
        example_results = {}

        # Analyze Input Grid
        input_grid = example.get('input', [])
        input_dims = analyze_grid_dimensions(input_grid)
        input_colors = find_unique_colors(input_grid)
        example_results['input'] = {
            'dimensions': input_dims,
            'unique_colors': sorted(list(input_colors)),
            'color_distribution': analyze_color_distribution(input_grid),
            'spatial_patterns': detect_spatial_patterns(input_grid)
        }
        all_train_input_colors.update(input_colors)
        
        # Save visualization of input grid
        if not disable_viz:
            vis_filename = os.path.join(visualizations_dir, f"train_{i+1}_input.png")
            visualize_grid(input_grid, vis_filename, f"Train {i+1} Input")

        # Analyze Output Grid
        output_grid = example.get('output', [])
        output_dims = analyze_grid_dimensions(output_grid)
        output_colors = find_unique_colors(output_grid)
        example_results['output'] = {
            'dimensions': output_dims,
            'unique_colors': sorted(list(output_colors)),
            'color_distribution': analyze_color_distribution(output_grid),
            'spatial_patterns': detect_spatial_patterns(output_grid)
        }
        all_train_output_colors.update(output_colors)
        
        # Save visualization of output grid
        if not disable_viz:
            vis_filename = os.path.join(visualizations_dir, f"train_{i+1}_output.png")
            visualize_grid(output_grid, vis_filename, f"Train {i+1} Output")

        # More detailed analysis of each grid
        example_results['input']['objects'] = identify_objects(input_grid)
        example_results['output']['objects'] = identify_objects(output_grid)
        
        example_results['input']['grid_structure'] = detect_grid_structure(input_grid)
        example_results['output']['grid_structure'] = detect_grid_structure(output_grid)
        
        example_results['input']['symmetry'] = detect_symmetry(input_grid)
        example_results['output']['symmetry'] = detect_symmetry(output_grid)
        
        example_results['input']['pattern_frequency_2x2'] = analyze_pattern_frequency(input_grid, (2, 2))
        example_results['output']['pattern_frequency_2x2'] = analyze_pattern_frequency(output_grid, (2, 2))
        
        # Analyze relationship between input and output
        example_results['input_output_relation'] = compare_input_output(input_grid, output_grid)

        analysis_results['train_examples'].append(example_results)

    analysis_results['all_train_input_colors'] = sorted(list(all_train_input_colors))
    analysis_results['all_train_output_colors'] = sorted(list(all_train_output_colors))
    analysis_results['all_train_colors'] = sorted(list(all_train_input_colors.union(all_train_output_colors)))


    # --- Perform Analysis on Test Examples ---
    analysis_results['test_examples'] = []
    all_test_input_colors = set()

    for i, example in enumerate(puzzle_data.get('test', [])):
         example_results = {}

         # Analyze Input Grid
         input_grid = example.get('input', [])
         input_dims = analyze_grid_dimensions(input_grid)
         input_colors = find_unique_colors(input_grid)
         example_results['input'] = {
             'dimensions': input_dims,
             'unique_colors': sorted(list(input_colors)),
             'color_distribution': analyze_color_distribution(input_grid),
             'objects': identify_objects(input_grid),
             'grid_structure': detect_grid_structure(input_grid),
             'symmetry': detect_symmetry(input_grid),
             'pattern_frequency_2x2': analyze_pattern_frequency(input_grid, (2, 2)),
             'spatial_patterns': detect_spatial_patterns(input_grid)
         }
         all_test_input_colors.update(input_colors)
         
         # Save visualization of test input grid
         if not disable_viz:
             vis_filename = os.path.join(visualizations_dir, f"test_{i+1}_input.png")
             visualize_grid(input_grid, vis_filename, f"Test {i+1} Input")

         analysis_results['test_examples'].append(example_results)

    analysis_results['all_test_input_colors'] = sorted(list(all_test_input_colors))
    analysis_results['all_colors'] = sorted(list(all_train_input_colors.union(all_train_output_colors).union(all_test_input_colors)))

    # --- Identify Task Type ---
    analysis_results['task_type'] = identify_task_type(analysis_results)

    # --- Write Results to File ---
    print(f"Writing analysis results to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            f.write(f"Puzzle Analysis: {puzzle_name}\n")
            f.write("=" * (len(puzzle_name) + 18) + "\n\n")

            # Task Type Section
            f.write("--- Task Type Analysis ---\n")
            top_tasks = analysis_results['task_type']['top_tasks']
            f.write("Top likely task types:\n")
            for task, prob in top_tasks:
                f.write(f"  - {task.replace('_', ' ').title()}: {prob:.1%}\n")
            
            f.write("\nEvidence for top task types:\n")
            for task, _ in top_tasks:
                if task in analysis_results['task_type']['explanations']:
                    f.write(f"  {task.replace('_', ' ').title()}:\n")
                    for explanation in analysis_results['task_type']['explanations'][task][:3]:  # Show top 3 pieces of evidence
                        f.write(f"    - {explanation}\n")
            f.write("\n")

            # Grid Dimensions Section
            f.write("--- Grid Dimensions ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                f.write(f"  Input: {example_results['input']['dimensions']}\n")
                f.write(f"  Output: {example_results['output']['dimensions']}\n")
            for i, example_results in enumerate(analysis_results['test_examples']):
                 f.write(f"Test Example {i+1}:\n")
                 f.write(f"  Input: {example_results['input']['dimensions']}\n")
            f.write("\n")

            # Unique Colors Section
            f.write("--- Unique Colors ---\n")
            f.write(f"Train Input Colors: {analysis_results['all_train_input_colors']}\n")
            f.write(f"Train Output Colors: {analysis_results['all_train_output_colors']}\n")
            f.write(f"All Train Colors: {analysis_results['all_train_colors']}\n")
            f.write(f"Test Input Colors: {analysis_results['all_test_input_colors']}\n")
            f.write(f"All Colors (Train & Test): {analysis_results['all_colors']}\n")
            f.write("\n")
            
            # Object Analysis Section
            f.write("--- Object Analysis ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                
                # Input objects
                input_objects = example_results['input']['objects']
                f.write(f"  Input Objects: {len(input_objects)}\n")
                for j, obj in enumerate(input_objects):
                    f.write(f"    Object {j+1}: color={obj['color']}, size={obj['size']}, position={obj['position']}, area={obj['area']}\n")
                
                # Output objects
                output_objects = example_results['output']['objects']
                f.write(f"  Output Objects: {len(output_objects)}\n")
                for j, obj in enumerate(output_objects):
                    f.write(f"    Object {j+1}: color={obj['color']}, size={obj['size']}, position={obj['position']}, area={obj['area']}\n")
            
            for i, example_results in enumerate(analysis_results['test_examples']):
                f.write(f"Test Example {i+1}:\n")
                input_objects = example_results['input']['objects']
                f.write(f"  Input Objects: {len(input_objects)}\n")
                for j, obj in enumerate(input_objects):
                    f.write(f"    Object {j+1}: color={obj['color']}, size={obj['size']}, position={obj['position']}, area={obj['area']}\n")
            f.write("\n")
            
            # Spatial Patterns Section
            f.write("--- Spatial Patterns Analysis ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                
                # Input spatial patterns
                input_patterns = example_results['input']['spatial_patterns']
                f.write(f"  Input Spatial Patterns:\n")
                f.write(f"    Lines: H={input_patterns['lines']['horizontal']}, V={input_patterns['lines']['vertical']}, D={input_patterns['lines']['diagonal']}\n")
                f.write(f"    Corners: {input_patterns['corners']}\n")
                f.write(f"    Enclosed Regions: {input_patterns['enclosed_regions']}\n")
                f.write(f"    Isolated Pixels: {input_patterns['isolated_pixels']}\n")
                f.write(f"    Border Pixels: {input_patterns['border_pixels']}\n")
                
                # Output spatial patterns
                output_patterns = example_results['output']['spatial_patterns']
                f.write(f"  Output Spatial Patterns:\n")
                f.write(f"    Lines: H={output_patterns['lines']['horizontal']}, V={output_patterns['lines']['vertical']}, D={output_patterns['lines']['diagonal']}\n")
                f.write(f"    Corners: {output_patterns['corners']}\n")
                f.write(f"    Enclosed Regions: {output_patterns['enclosed_regions']}\n")
                f.write(f"    Isolated Pixels: {output_patterns['isolated_pixels']}\n")
                f.write(f"    Border Pixels: {output_patterns['border_pixels']}\n")
            
            for i, example_results in enumerate(analysis_results['test_examples']):
                f.write(f"Test Example {i+1}:\n")
                input_patterns = example_results['input']['spatial_patterns']
                f.write(f"  Input Spatial Patterns:\n")
                f.write(f"    Lines: H={input_patterns['lines']['horizontal']}, V={input_patterns['lines']['vertical']}, D={input_patterns['lines']['diagonal']}\n")
                f.write(f"    Corners: {input_patterns['corners']}\n")
                f.write(f"    Enclosed Regions: {input_patterns['enclosed_regions']}\n")
                f.write(f"    Isolated Pixels: {input_patterns['isolated_pixels']}\n")
                f.write(f"    Border Pixels: {input_patterns['border_pixels']}\n")
            f.write("\n")
            
            # Symmetry Analysis Section
            f.write("--- Symmetry Analysis ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                
                # Input symmetry
                input_sym = example_results['input']['symmetry']
                f.write(f"  Input Symmetry:\n")
                f.write(f"    Horizontal: {input_sym['horizontal_symmetry']} (score: {input_sym['horizontal_symmetry_score']:.2f})\n")
                f.write(f"    Vertical: {input_sym['vertical_symmetry']} (score: {input_sym['vertical_symmetry_score']:.2f})\n")
                f.write(f"    Diagonal (main): {input_sym.get('diagonal_symmetry_main', False)}\n")
                f.write(f"    Diagonal (anti): {input_sym.get('diagonal_symmetry_anti', False)}\n")
                f.write(f"    Rotational (180°): {input_sym['rotational_symmetry_180']}\n")
                f.write(f"    Rotational (90°): {input_sym.get('rotational_symmetry_90', False)}\n")
                
                # Output symmetry
                output_sym = example_results['output']['symmetry']
                f.write(f"  Output Symmetry:\n")
                f.write(f"    Horizontal: {output_sym['horizontal_symmetry']} (score: {output_sym['horizontal_symmetry_score']:.2f})\n")
                f.write(f"    Vertical: {output_sym['vertical_symmetry']} (score: {output_sym['vertical_symmetry_score']:.2f})\n")
                f.write(f"    Diagonal (main): {output_sym.get('diagonal_symmetry_main', False)}\n")
                f.write(f"    Diagonal (anti): {output_sym.get('diagonal_symmetry_anti', False)}\n")
                f.write(f"    Rotational (180°): {output_sym['rotational_symmetry_180']}\n")
                f.write(f"    Rotational (90°): {output_sym.get('rotational_symmetry_90', False)}\n")
            
            for i, example_results in enumerate(analysis_results['test_examples']):
                f.write(f"Test Example {i+1}:\n")
                input_sym = example_results['input']['symmetry']
                f.write(f"  Input Symmetry:\n")
                f.write(f"    Horizontal: {input_sym['horizontal_symmetry']} (score: {input_sym['horizontal_symmetry_score']:.2f})\n")
                f.write(f"    Vertical: {input_sym['vertical_symmetry']} (score: {input_sym['vertical_symmetry_score']:.2f})\n")
                f.write(f"    Diagonal (main): {input_sym.get('diagonal_symmetry_main', False)}\n")
                f.write(f"    Diagonal (anti): {input_sym.get('diagonal_symmetry_anti', False)}\n")
                f.write(f"    Rotational (180°): {input_sym['rotational_symmetry_180']}\n")
                f.write(f"    Rotational (90°): {input_sym.get('rotational_symmetry_90', False)}\n")
            f.write("\n")
            
            # Grid Structure Analysis Section
            f.write("--- Grid Structure Analysis ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                
                # Input grid structure
                input_grid = example_results['input']['grid_structure']
                f.write(f"  Input Grid Structure: {input_grid['has_grid_structure']}\n")
                if input_grid['has_grid_structure']:
                    f.write(f"    Grid Color: {input_grid['grid_color']}\n")
                    f.write(f"    Cell Size: {input_grid['cell_height']}x{input_grid['cell_width']}\n")
                    f.write(f"    Grid Cells: {input_grid['grid_cells_y']}x{input_grid['grid_cells_x']}\n")
                
                # Output grid structure
                output_grid = example_results['output']['grid_structure']
                f.write(f"  Output Grid Structure: {output_grid['has_grid_structure']}\n")
                if output_grid['has_grid_structure']:
                    f.write(f"    Grid Color: {output_grid['grid_color']}\n")
                    f.write(f"    Cell Size: {output_grid['cell_height']}x{output_grid['cell_width']}\n")
                    f.write(f"    Grid Cells: {output_grid['grid_cells_y']}x{output_grid['grid_cells_x']}\n")
            
            for i, example_results in enumerate(analysis_results['test_examples']):
                f.write(f"Test Example {i+1}:\n")
                input_grid = example_results['input']['grid_structure']
                f.write(f"  Input Grid Structure: {input_grid['has_grid_structure']}\n")
                if input_grid['has_grid_structure']:
                    f.write(f"    Grid Color: {input_grid['grid_color']}\n")
                    f.write(f"    Cell Size: {input_grid['cell_height']}x{input_grid['cell_width']}\n")
                    f.write(f"    Grid Cells: {input_grid['grid_cells_y']}x{input_grid['grid_cells_x']}\n")
            f.write("\n")
            
            # Pattern Frequency Analysis Section
            f.write("--- Pattern Frequency Analysis (2x2) ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                
                # Input pattern frequency
                input_pattern = example_results['input']['pattern_frequency_2x2']
                if input_pattern['status'] == 'success':
                    f.write(f"  Input Unique Patterns: {input_pattern['unique_patterns_count']}\n")
                    f.write(f"  Top Patterns:\n")
                    for j, pattern_info in enumerate(input_pattern['most_common_patterns'][:3]):
                        pattern_str = '\n      '.join([str(row) for row in pattern_info['pattern']])
                        f.write(f"    Pattern {j+1}: count={pattern_info['count']}, frequency={pattern_info['frequency']:.2f}\n")
                        f.write(f"      {pattern_str}\n")
                else:
                    f.write(f"  Input Pattern Analysis: {input_pattern['status']}\n")
                
                # Output pattern frequency
                output_pattern = example_results['output']['pattern_frequency_2x2']
                if output_pattern['status'] == 'success':
                    f.write(f"  Output Unique Patterns: {output_pattern['unique_patterns_count']}\n")
                    f.write(f"  Top Patterns:\n")
                    for j, pattern_info in enumerate(output_pattern['most_common_patterns'][:3]):
                        pattern_str = '\n      '.join([str(row) for row in pattern_info['pattern']])
                        f.write(f"    Pattern {j+1}: count={pattern_info['count']}, frequency={pattern_info['frequency']:.2f}\n")
                        f.write(f"      {pattern_str}\n")
                else:
                    f.write(f"  Output Pattern Analysis: {output_pattern['status']}\n")
            
            for i, example_results in enumerate(analysis_results['test_examples']):
                f.write(f"Test Example {i+1}:\n")
                input_pattern = example_results['input']['pattern_frequency_2x2']
                if input_pattern['status'] == 'success':
                    f.write(f"  Input Unique Patterns: {input_pattern['unique_patterns_count']}\n")
                    f.write(f"  Top Patterns:\n")
                    for j, pattern_info in enumerate(input_pattern['most_common_patterns'][:3]):
                        pattern_str = '\n      '.join([str(row) for row in pattern_info['pattern']])
                        f.write(f"    Pattern {j+1}: count={pattern_info['count']}, frequency={pattern_info['frequency']:.2f}\n")
                        f.write(f"      {pattern_str}\n")
                else:
                    f.write(f"  Input Pattern Analysis: {input_pattern['status']}\n")
            f.write("\n")
            
            # Color Distribution Analysis Section
            f.write("--- Color Distribution Analysis ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                
                # Input color distribution
                input_color = example_results['input']['color_distribution']
                f.write(f"  Input Colors: {len(input_color['color_counts'])}\n")
                f.write(f"  Most Common: {input_color['most_common_color']} ({input_color['color_counts'][0]['percentage']:.1f}%)\n")
                f.write(f"  Color Entropy: {input_color['color_entropy']:.2f}\n")
                f.write(f"  Binary: {input_color['is_binary']}, Grayscale: {input_color['is_grayscale']}\n")
                f.write(f"  Color Counts:\n")
                for color_info in input_color['color_counts']:
                    f.write(f"    Color {color_info['color']}: {color_info['count']} ({color_info['percentage']:.1f}%)\n")
                
                # Output color distribution
                output_color = example_results['output']['color_distribution']
                f.write(f"  Output Colors: {len(output_color['color_counts'])}\n")
                f.write(f"  Most Common: {output_color['most_common_color']} ({output_color['color_counts'][0]['percentage']:.1f}%)\n")
                f.write(f"  Color Entropy: {output_color['color_entropy']:.2f}\n")
                f.write(f"  Binary: {output_color['is_binary']}, Grayscale: {output_color['is_grayscale']}\n")
                f.write(f"  Color Counts:\n")
                for color_info in output_color['color_counts']:
                    f.write(f"    Color {color_info['color']}: {color_info['count']} ({color_info['percentage']:.1f}%)\n")
            
            for i, example_results in enumerate(analysis_results['test_examples']):
                f.write(f"Test Example {i+1}:\n")
                input_color = example_results['input']['color_distribution']
                f.write(f"  Input Colors: {len(input_color['color_counts'])}\n")
                f.write(f"  Most Common: {input_color['most_common_color']} ({input_color['color_counts'][0]['percentage']:.1f}%)\n")
                f.write(f"  Color Entropy: {input_color['color_entropy']:.2f}\n")
                f.write(f"  Binary: {input_color['is_binary']}, Grayscale: {input_color['is_grayscale']}\n")
                f.write(f"  Color Counts:\n")
                for color_info in input_color['color_counts']:
                    f.write(f"    Color {color_info['color']}: {color_info['count']} ({color_info['percentage']:.1f}%)\n")
            f.write("\n")
            
            # Input-Output Transformation Analysis Section
            f.write("--- Input-Output Transformation Analysis ---\n")
            for i, example_results in enumerate(analysis_results['train_examples']):
                f.write(f"Train Example {i+1}:\n")
                rel = example_results['input_output_relation']
                
                f.write(f"  Size Transformation: {rel['size_transformation']}\n")
                f.write(f"  Size Ratios: rows={rel['size_relation']['rows_ratio']:.2f}, cols={rel['size_relation']['cols_ratio']:.2f}, area={rel['size_relation']['area_ratio']:.2f}\n")
                
                if 'transformation' in rel:
                    f.write(f"  Transformation: {rel['transformation']}\n")
                    if rel['transformation'] == 'shift':
                        f.write(f"    Shift Amount: {rel['shift_amount']}\n")
                
                if 'matching_cells_percentage' in rel:
                    f.write(f"  Matching Cells: {rel['matching_cells_percentage']:.1f}%\n")
                
                if 'color_transformation' in rel and rel['color_transformation']:
                    f.write(f"  Color Transformation: {rel['color_transformation']}\n")
                
                f.write(f"  Preserved Colors: {rel['preserved_colors']}\n")
                if 'new_colors' in rel:
                    f.write(f"  New Colors: {rel['new_colors']}\n")
                if 'disappeared_colors' in rel:
                    f.write(f"  Disappeared Colors: {rel['disappeared_colors']}\n")
            f.write("\n")
            
            # Summary and Pattern Hypotheses
            f.write("--- Summary and Pattern Hypotheses ---\n")
            
            # Analyze size relationships
            consistent_size_relation = True
            size_transformation = None
            for i, example_results in enumerate(analysis_results['train_examples']):
                if i == 0:
                    size_transformation = example_results['input_output_relation']['size_transformation']
                elif example_results['input_output_relation']['size_transformation'] != size_transformation:
                    consistent_size_relation = False
                    break
            
            if consistent_size_relation:
                f.write(f"Size Transformation: Consistent '{size_transformation}' pattern across all training examples\n")
            else:
                f.write("Size Transformation: Inconsistent patterns across training examples\n")
            
            # Analyze color transformations
            consistent_color_transform = True
            color_transformation = None
            for i, example_results in enumerate(analysis_results['train_examples']):
                relation = example_results['input_output_relation']
                if i == 0:
                    color_transformation = relation.get('color_transformation', None)
                elif relation.get('color_transformation', None) != color_transformation:
                    consistent_color_transform = False
                    break
            
            if consistent_color_transform and color_transformation:
                f.write(f"Color Transformation: Consistent '{color_transformation}' pattern across all training examples\n")
            else:
                f.write("Color Transformation: No consistent simple color transformation detected\n")
            
            # Check for consistent geometric transformations
            consistent_geo_transform = True
            geo_transformation = None
            for i, example_results in enumerate(analysis_results['train_examples']):
                relation = example_results['input_output_relation']
                if i == 0:
                    geo_transformation = relation.get('transformation', None)
                elif relation.get('transformation', None) != geo_transformation:
                    consistent_geo_transform = False
                    break
            
            if consistent_geo_transform and geo_transformation:
                f.write(f"Geometric Transformation: Consistent '{geo_transformation}' across all training examples\n")
            else:
                f.write("Geometric Transformation: No consistent simple geometric transformation detected\n")
            
            # Analyze object relationships
            f.write("\nObject Relationships:\n")
            consistent_object_count = True
            object_count_relation = None
            for i, example_results in enumerate(analysis_results['train_examples']):
                input_objs = len(example_results['input']['objects'])
                output_objs = len(example_results['output']['objects'])
                relation = None
                
                if input_objs == output_objs:
                    relation = "equal"
                elif input_objs < output_objs:
                    relation = "increased"
                else:
                    relation = "decreased"
                
                if i == 0:
                    object_count_relation = relation
                elif relation != object_count_relation:
                    consistent_object_count = False
                    break
            
            if consistent_object_count:
                f.write(f"  Object Count: Consistently '{object_count_relation}' across all training examples\n")
            else:
                f.write("  Object Count: Inconsistent pattern in object count changes\n")
            
            f.write("\nPossible Transformation Rules:\n")
            # Here we can add more sophisticated rule detection based on all the analyses above
            
            # Example rule detection
            rules_detected = []
            
            # Check for color inversion rule
            if consistent_color_transform and color_transformation == 'inverted':
                rules_detected.append("- Color inversion: All colors are inverted in the output")
            
            # Check for object counting rules
            if consistent_object_count:
                if object_count_relation == "equal":
                    rules_detected.append("- Object preservation: Same number of objects in input and output")
                elif object_count_relation == "increased":
                    rules_detected.append("- Object creation: More objects in output than input")
                else:
                    rules_detected.append("- Object reduction: Fewer objects in output than input")
            
            # Check for symmetry-related rules
            symmetry_consistent = True
            symmetry_type = None
            for i, example_results in enumerate(analysis_results['train_examples']):
                input_sym = example_results['input']['symmetry']
                output_sym = example_results['output']['symmetry']
                
                # Check if output has symmetry that input doesn't
                if (not input_sym['horizontal_symmetry'] and output_sym['horizontal_symmetry']) or \
                   (not input_sym['vertical_symmetry'] and output_sym['vertical_symmetry']) or \
                   (not input_sym['rotational_symmetry_180'] and output_sym['rotational_symmetry_180']):
                    
                    if output_sym['horizontal_symmetry'] and not input_sym['horizontal_symmetry']:
                        new_sym = "horizontal"
                    elif output_sym['vertical_symmetry'] and not input_sym['vertical_symmetry']:
                        new_sym = "vertical"
                    elif output_sym['rotational_symmetry_180'] and not input_sym['rotational_symmetry_180']:
                        new_sym = "rotational_180"
                    else:
                        new_sym = "other"
                    
                    if i == 0:
                        symmetry_type = new_sym
                    elif new_sym != symmetry_type:
                        symmetry_consistent = False
                        break
                else:
                    if i == 0:
                        symmetry_type = "none"
                    elif symmetry_type != "none":
                        symmetry_consistent = False
                        break
            
            if symmetry_consistent and symmetry_type and symmetry_type != "none":
                rules_detected.append(f"- Symmetry creation: Outputs have {symmetry_type} symmetry not present in inputs")
            
            # Write detected rules
            if rules_detected:
                for rule in rules_detected:
                    f.write(f"{rule}\n")
            else:
                f.write("No simple transformation rules detected. Consider analyzing more complex patterns.\n")
            
            print("Analysis complete.")
            
            return analysis_results

    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred writing results: {e}")
        
    return None

def parse_arguments():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ARC puzzles and generate feature reports.')
    parser.add_argument('--puzzle', type=str, default=PUZZLE_JSON_PATH,
                      help=f'Path to the puzzle JSON file (default: {PUZZLE_JSON_PATH})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                      help=f'Directory to save analysis results (default: {OUTPUT_DIR})')
    parser.add_argument('--batch', action='store_true',
                      help='Process all puzzle files in the directory')
    parser.add_argument('--no-viz', action='store_true',
                      help='Disable puzzle visualizations')
    parser.add_argument('--summary', action='store_true',
                      help='Generate a summary of all analyzed puzzles (only with --batch)')
    return parser.parse_args()

def find_all_puzzle_files(directory):
    """
    Find all JSON files in the specified directory and return their paths.
    """
    import glob
    return glob.glob(os.path.join(directory, '*.json'))

def analyze_batch(directory, output_dir, no_viz=False, generate_summary=False):
    """
    Analyze all puzzle files in the specified directory.
    """
    puzzle_files = find_all_puzzle_files(directory)
    print(f"Found {len(puzzle_files)} puzzle files in {directory}")
    
    results = []
    for puzzle_file in puzzle_files:
        try:
            print(f"\nProcessing: {os.path.basename(puzzle_file)}")
            result = analyze_puzzle(puzzle_file, output_dir, disable_viz=no_viz)
            results.append({
                'file': os.path.basename(puzzle_file),
                'result': result
            })
        except Exception as e:
            print(f"Error processing {puzzle_file}: {e}")
    
    if generate_summary and results:
        generate_batch_summary(results, output_dir)
    
    print(f"\nBatch processing complete. Analyzed {len(results)} puzzles.")

def generate_batch_summary(results, output_dir):
    """
    Generate a summary of all analyzed puzzles.
    """
    summary_path = os.path.join(output_dir, "batch_summary.txt")
    try:
        with open(summary_path, 'w') as f:
            f.write("ARC Puzzles Batch Analysis Summary\n")
            f.write("================================\n\n")
            
            f.write(f"Total puzzles analyzed: {len(results)}\n\n")
            
            # Add summary statistics here...
            
            f.write("Puzzle List:\n")
            for idx, res in enumerate(results):
                f.write(f"{idx+1}. {res['file']}\n")
            
        print(f"Batch summary written to: {summary_path}")
    except Exception as e:
        print(f"Error writing batch summary: {e}")

# --- Entry Point ---

if __name__ == "__main__":
    args = parse_arguments()
    
    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.batch:
        # Extract directory from the puzzle path
        puzzle_dir = os.path.dirname(args.puzzle)
        if not puzzle_dir:
            puzzle_dir = '.'
        analyze_batch(puzzle_dir, args.output_dir, args.no_viz, args.summary)
    else:
        analyze_puzzle(args.puzzle, args.output_dir, disable_viz=args.no_viz)

