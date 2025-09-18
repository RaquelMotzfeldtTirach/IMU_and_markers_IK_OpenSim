import sys
import numpy as np
import matplotlib.pyplot as plt

def read_mot_file(filepath):
    """Reads a .mot file and returns the header, column labels, and data as a numpy array."""
    header = []
    data = []
    columns = []
    with open(filepath, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                header.append(line)
                if line.strip().startswith('endheader'):
                    in_header = False
            elif not columns:
                columns = line.strip().split('\t')
            else:
                if line.strip() == '':
                    continue
                row = [float(x) for x in line.strip().split()]
                data.append(row)
    return header, columns, np.array(data)

def compare_mot_files(file1, file2, atol=1e-8):
    _, columns1, data1 = read_mot_file(file1)
    _, columns2, data2 = read_mot_file(file2)
    if columns1 != columns2:
        print("Column headers differ!")
        print("File 1 columns:", columns1)
        print("File 2 columns:", columns2)
        return
    if data1.shape != data2.shape:
        print(f"Shape mismatch: {data1.shape} vs {data2.shape}")
        return
    diff = np.abs(data1 - data2)
    max_diff = np.max(diff)
    if np.allclose(data1, data2, atol=atol):
        print("Files are identical within tolerance.")
        return
    print(f"Files differ! Maximum absolute difference: {max_diff}")
    rows, cols = np.where(diff > atol)
    for r, c in zip(rows, cols):
        print(f"Row {r+1}, Column '{columns1[c]}': File1={data1[r, c]}, File2={data2[r, c]}, Diff={diff[r, c]}")
    print(f"Largest difference: {max_diff} at row {rows[np.argmax(diff[rows, cols])]+1}, column '{columns1[cols[np.argmax(diff[rows, cols])]]}'")

    # Plot differences for main joints
    col_names = ['arm_flex_l', 'arm_add_l', 'arm_rot_l']
    plot_column_differences(col_names, columns1, data1, data2)
    col_names = ['arm_flex_r', 'arm_add_r', 'arm_rot_r']
    plot_column_differences(col_names, columns1, data1, data2)
    col_names = ['elbow_flex_l', 'pro_sup_l']
    plot_column_differences(col_names, columns1, data1, data2)
    col_names = ['elbow_flex_r', 'pro_sup_r']
    plot_column_differences(col_names, columns1, data1, data2)
    col_names = ['wrist_flex_l', 'wrist_dev_l']
    plot_column_differences(col_names, columns1, data1, data2)
    col_names = ['wrist_flex_r', 'wrist_dev_r']
    plot_column_differences(col_names, columns1, data1, data2)


def plot_column_differences(columns_to_plot, columns1, data1, data2, time=None, title=None):
    col_indices = []
    for name in columns_to_plot:
        if name in columns1:
            col_indices.append(columns1.index(name))
        else:
            print(f"Column '{name}' not found in file.")
            return
    if time is None:
        time_idx = columns1.index('time') if 'time' in columns1 else 0
        time = data1[:, time_idx]
    plt.figure(figsize=(10,6))
    for idx, label in zip(col_indices, columns_to_plot):
        plt.plot(time, data1[:, idx] - data2[:, idx], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Difference (File1 - File2)')
    if title is None:
        title = f"Differences for columns: {', '.join(columns_to_plot)}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diff_check_mot.py file1.mot file2.mot")
        sys.exit(1)
    compare_mot_files(sys.argv[1], sys.argv[2])
