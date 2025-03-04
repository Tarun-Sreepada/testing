#!/usr/bin/env python3
import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

    

def find_closest_grid(total_blocks):
    """Finds the closest grid dimensions to a square for a given number of blocks."""
    # Start with the square root and go down to find a factor
    for rows in range(int(math.sqrt(total_blocks)), 0, -1):
        if total_blocks % rows == 0:
            cols = total_blocks // rows
            return rows, cols
    # Fallback, should not happen
    return total_blocks, 1


def format_summary_lines_3_columns(lines, ncols=3):
    """Format a list of text lines into n columns."""
    if not lines:
        return ""
    print(lines)
    total_char = sum(len(line) for line in lines)
    avg_char = total_char / len(lines)
    rows = ""
    temp = ""
    for line in lines:
        # if len(temp) < avg_char:
        #     temp += line + "\n"
        # else:
        #     rows += temp
        #     temp = ""
        temp += line + " || "
        if len(temp) > avg_char:
            rows += temp + "\n"
            temp = ""
        
    if temp:
        rows += temp
    return rows

def main(filename):
    # Read the file.
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Modified regex: trailing " s" is now optional.
    block_pattern = re.compile(r"Block:\s*(\d+)\s+([A-Za-z ]+):\s*([\d.]+)(?:\s*s)?")

    # Dictionary to store block data.
    blocks = {}
    # List to store summary information lines (those not matching the block pattern).
    summary_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = block_pattern.match(line)
        if m:
            bid = int(m.group(1))
            metric = m.group(2).strip()
            value = float(m.group(3))
            if bid not in blocks:
                blocks[bid] = {}
            blocks[bid][metric] = value
        else:
            summary_lines.append(line)

    if not blocks:
        print("No block data found in the file!")
        return

    # Determine total number of blocks using the highest block id + 1.
    sorted_ids = sorted(blocks.keys())
    total_blocks = sorted_ids[-1] + 1

    # Calculate grid dimensions from the total number of blocks.
    # Use the new function to get closest grid dimensions.
    rows, cols = find_closest_grid(total_blocks)

    print(f"Grid dimensions: {rows} x {cols} for {total_blocks} blocks.")

    # Create lists for each metric.
    time_list = []
    idle_list = []
    mem_list = []
    scan_list = []
    merge_list = []
    processed_list = []

    for i in range(total_blocks):
        data = blocks.get(i, {})
        time_list.append(data.get("Time", np.nan))
        idle_list.append(data.get("Idle", np.nan))
        mem_list.append(data.get("Memory Alloc", np.nan))
        scan_list.append(data.get("Scan", np.nan))
        merge_list.append(data.get("Merge", np.nan))
        processed_list.append(data.get("Processed", np.nan))

    # Convert each list into a numpy array and reshape into (rows x cols).
    time_mat  = np.array(time_list).reshape(rows, cols)
    idle_mat  = np.array(idle_list).reshape(rows, cols)
    mem_mat   = np.array(mem_list).reshape(rows, cols)
    scan_mat  = np.array(scan_list).reshape(rows, cols)
    merge_mat = np.array(merge_list).reshape(rows, cols)
    processed_mat = np.array(processed_list).reshape(rows, cols)

    # Prepare a list of (matrix, title) tuples for plotting, now including "Processed".
    plots = [
        (time_mat, "Time (s)"),
        (idle_mat, "Idle (s)"),
        (mem_mat, "Memory Alloc (s)"),
        (scan_mat, "Scan (s)"),
        (merge_mat, "Merge (s)"),
        (processed_mat, "Processed")
    ]

    # Determine subplot grid size for the plots.
    n_plots = len(plots)
    ncols_plots = int(math.ceil(math.sqrt(n_plots)))
    nrows_plots = int(math.ceil(n_plots / ncols_plots))

    # Increase figure height to make room for bottom summary text.
    fig, axs = plt.subplots(nrows_plots, ncols_plots, figsize=(ncols_plots * 4, nrows_plots * 4 + 2))
    axs = np.array(axs).flatten()  # Flatten for easy iteration.

    # Plot each metric as a heatmap.
    for i, (mat, title) in enumerate(plots):
        im = axs[i].imshow(mat, cmap='viridis', vmin=0)
        axs[i].set_title(title)
        # boundaries = 0 to max
        fig.colorbar(im, ax=axs[i])

    # Hide any unused subplots.
    for j in range(n_plots, len(axs)):
        axs[j].axis('off')

    # Format the summary text into 3 columns.
    summary_text = format_summary_lines_3_columns(summary_lines, ncols=3)

    # Adjust layout to leave extra space at the bottom and add the summary text.
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    plt.figtext(0.5, 0.02, summary_text, wrap=True,
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # plt.show()
    output_filename = filename.rsplit('.', 1)[0] + "_output.png"
    plt.savefig(output_filename)

def old(filename):
    # Read the file.
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Modified regex: trailing " s" is now optional.
    block_pattern = re.compile(r"Block:\s*(\d+)\s+([A-Za-z ]+):\s*([\d.]+)(?:\s*s)?")

    # Dictionary to store block data.
    blocks = {}
    # List to store summary information lines (those not matching the block pattern).
    summary_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = block_pattern.match(line)
        if m:
            bid = int(m.group(1))
            metric = m.group(2).strip()
            value = float(m.group(3))
            if bid not in blocks:
                blocks[bid] = {}
            blocks[bid][metric] = value
        else:
            summary_lines.append(line)

    if not blocks:
        print("No block data found in the file!")
        return

    # Determine total number of blocks using the highest block id + 1.
    sorted_ids = sorted(blocks.keys())
    total_blocks = sorted_ids[-1] + 1

    # Calculate grid dimensions from the total number of blocks.
    # Use the new function to get closest grid dimensions.
    rows, cols = find_closest_grid(total_blocks)

    print(f"Grid dimensions: {rows} x {cols} for {total_blocks} blocks.")

    # Create lists for each metric.
    time_list = []
    idle_list = []
    mem_list = []
    scan_list = []
    merge_list = []
    processed_list = []

    for i in range(total_blocks):
        data = blocks.get(i, {})
        time_list.append(data.get("Time", np.nan))
        idle_list.append(data.get("Idle", np.nan))
        mem_list.append(data.get("Memory Alloc", np.nan))
        scan_list.append(data.get("Scan", np.nan))
        merge_list.append(data.get("Merge", np.nan))
        processed_list.append(data.get("Processed", np.nan))

    # Convert each list into a numpy array and reshape into (rows x cols).
    time_mat  = np.array(time_list).reshape(rows, cols)
    idle_mat  = np.array(idle_list).reshape(rows, cols)
    mem_mat   = np.array(mem_list).reshape(rows, cols)
    scan_mat  = np.array(scan_list).reshape(rows, cols)
    merge_mat = np.array(merge_list).reshape(rows, cols)
    processed_mat = np.array(processed_list).reshape(rows, cols)

    # Prepare a list of (matrix, title) tuples for plotting, now including "Processed".
    plots = [
        (time_mat, "Time (s)"),
        (idle_mat, "Idle (s)"),
        (mem_mat, "Memory Alloc (s)"),
        (scan_mat, "Scan (s)"),
        (merge_mat, "Merge (s)"),
        (processed_mat, "Processed")
    ]

    # Determine subplot grid size for the plots.
    n_plots = len(plots)
    ncols_plots = int(math.ceil(math.sqrt(n_plots)))
    nrows_plots = int(math.ceil(n_plots / ncols_plots))

    # Increase figure height to make room for bottom summary text.
    fig, axs = plt.subplots(nrows_plots, ncols_plots, figsize=(ncols_plots * 4, nrows_plots * 4 + 2))
    axs = np.array(axs).flatten()  # Flatten for easy iteration.

    # Plot each metric as a heatmap.
    for i, (mat, title) in enumerate(plots):
        im = axs[i].imshow(mat, cmap='viridis')
        axs[i].set_title(title)
        # boundaries = 0 to max
        fig.colorbar(im, ax=axs[i])

    # Hide any unused subplots.
    for j in range(n_plots, len(axs)):
        axs[j].axis('off')

    # Format the summary text into 3 columns.
    summary_text = format_summary_lines_3_columns(summary_lines, ncols=3)

    # Adjust layout to leave extra space at the bottom and add the summary text.
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    plt.figtext(0.5, 0.02, summary_text, wrap=True,
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # plt.show()
    output_filename = filename.rsplit('.', 1)[0] + "_output_old.png"
    plt.savefig(output_filename)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
    else:
        main(sys.argv[1])
        old(sys.argv[1])
