#!/usr/bin/env python3
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def main(filename):
    # Read the file.
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Regular expression to match block lines, e.g.:
    # "Block: 254 Time: 16.3895 s"
    block_pattern = re.compile(r"Block:\s*(\d+)\s+([A-Za-z ]+):\s*([\d.]+)\s*s")
    
    # Dictionary to store block data.
    # Each key is a block id, and its value is a dict of metrics.
    blocks = {}
    # List to store summary information lines (those not matching the block pattern)
    summary_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = block_pattern.match(line)
        if m:
            bid = int(m.group(1))
            metric = m.group(2).strip()  # e.g. "Time", "Idle", "Memory Alloc", etc.
            value = float(m.group(3))
            if bid not in blocks:
                blocks[bid] = {}
            blocks[bid][metric] = value
        else:
            # Treat non-block lines as summary information.
            summary_lines.append(line)

    if not blocks:
        print("No block data found in the file!")
        return

    # Determine total number of blocks using the highest block id + 1.
    sorted_ids = sorted(blocks.keys())
    total_blocks = sorted_ids[-1] + 1

    # Calculate grid dimensions from the total number of blocks.
    # We assume total_blocks is a perfect square.
    dim = int(math.sqrt(total_blocks))
    if dim * dim != total_blocks:
        print(f"Warning: total blocks ({total_blocks}) is not a perfect square. Using dimension = {dim} (ignoring extra blocks).")
        total_blocks = dim * dim

    # Create lists for each metric (in order of block id from 0 to total_blocks-1).
    time_list = []
    idle_list = []
    mem_list = []
    scan_list = []
    merge_list = []

    for i in range(total_blocks):
        data = blocks.get(i, {})
        time_list.append(data.get("Time", np.nan))
        idle_list.append(data.get("Idle", np.nan))
        mem_list.append(data.get("Memory Alloc", np.nan))
        scan_list.append(data.get("Scan", np.nan))
        merge_list.append(data.get("Merge", np.nan))

    # Convert each list into a numpy array and reshape into (dim x dim).
    time_mat  = np.array(time_list).reshape(dim, dim)
    idle_mat  = np.array(idle_list).reshape(dim, dim)
    mem_mat   = np.array(mem_list).reshape(dim, dim)
    scan_mat  = np.array(scan_list).reshape(dim, dim)
    merge_mat = np.array(merge_list).reshape(dim, dim)

    # Prepare a list of (matrix, title) tuples for plotting.
    plots = [
        (time_mat, "Time (s)"),
        (idle_mat, "Idle (s)"),
        (mem_mat, "Memory Alloc (s)"),
        (scan_mat, "Scan (s)"),
        (merge_mat, "Merge (s)")
    ]

    # Determine subplot grid size for the 5 plots.
    n_plots = len(plots)
    ncols = int(math.ceil(math.sqrt(n_plots)))
    nrows = int(math.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axs = np.array(axs).flatten()  # Flatten for easy iteration.

    # Plot each metric as a heatmap.
    for i, (mat, title) in enumerate(plots):
        im = axs[i].imshow(mat, cmap='viridis')
        axs[i].set_title(title)
        fig.colorbar(im, ax=axs[i])
    
    # Hide any unused subplots.
    for j in range(n_plots, len(axs)):
        axs[j].axis('off')

    # Prepare the summary text (joining the summary lines with newlines).
    summary_text = "\n".join(summary_lines)
    # Place the summary text as a text box in the bottom right of the figure.
    plt.figtext(0.99, 0.01, summary_text, horizontalalignment='right', verticalalignment='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
    else:
        main(sys.argv[1])
