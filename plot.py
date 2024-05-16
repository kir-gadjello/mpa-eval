import json
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate
import re

def extract_entries(jsonl_file, mode, tags=None):
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]

    if mode == 'auto':
        compatible_entries = []
        for entry in reversed(data):
            if not compatible_entries or (entry['seed'] == data[-1]['seed'] and entry['n_lines'] == data[-1]['n_lines'] and entry['n_digits'] == data[-1]['n_digits']):
                compatible_entries.append(entry)
            else:
                break
        compatible_entries.reverse()
    else:
        compatible_entries = [entry for entry in data if entry['tag'] in tags]

    return compatible_entries

def plot_table(entries, seed, n_lines, n_digits, n_trials, plot_style, tag_filter=None):
    entries = sorted(entries, key=lambda entry: entry['accuracy'], reverse=True)
    if plot_style == 'grid':
        table = []
        for entry in entries:
            table.append([entry['tag'], f"{entry['accuracy']*100.0:.2f}%"])
        print(f"MPA eval result, seed={seed}, n_digits={n_digits}, n_lines={n_lines}, n_trials={n_trials}]")
        print(tabulate(table, headers=['Tag', 'Accuracy'], tablefmt='grid'))
    
    elif plot_style == 'box':
        tags = [entry['tag'] for entry in entries]
        accuracies = [entry['accuracy'] for entry in entries]
        plt.bar(range(len(tags)), [acc*100 for acc in accuracies])
        plt.xticks(range(len(tags)), tags)
        plt.xlabel('Tag')
        plt.ylabel('Accuracy (%)')

        # Add percentage legends
        for i, accuracy in enumerate(accuracies):
            plt.text(i, max(5, accuracy*100-10), f"{accuracy*100:.1f}%", ha='center', va='top', fontsize=20)

        plt.title(f"MPA eval: seed={seed}, n_digits={n_digits}, n_lines={n_lines}, n_trials={n_trials}")
        plt.show()

    elif plot_style == 'asciibox':
        tags = [entry['tag'] for entry in entries]
        accuracies = [entry['accuracy'] for entry in entries]
        bar_width = 80
        # Create a table with ASCII characters
        table = []
        for i, tag in enumerate(tags):
            accuracy = accuracies[i]
            acc = f"{accuracy*100:.1f}%".ljust(5)
            filled = max(0, int(accuracy * bar_width)-5)
            table.append([tag, acc + ' ' + '▇' * filled + ' ' * int((1.0-accuracy) * bar_width) + '\ufeff'])

        # Print a header
        print(f"MPA eval result, seed={seed}, n_digits={n_digits}, n_lines={n_lines}, n_ trials={n_trials}")
        print(tabulate(table, tablefmt='grid'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('--mode', choices=['auto', 'default'], default='auto', help='Mode of operation')
    parser.add_argument('--tags', nargs='+', help='Tags to filter by (only applicable in default mode)')
    parser.add_argument('--plot-style', choices=['grid', 'box', 'asciibox'], default='grid', help='Plot style')
    parser.add_argument('--tag-filter', help='Filter tags using a string or a regexp')
    args = parser.parse_args()

    entries = extract_entries(args.input_file, args.mode, args.tags)

    if args.tag_filter:
        tag_filter = re.compile(args.tag_filter)
        entries = [entry for entry in entries if tag_filter.search(entry['tag'])]

    plot_table(entries, entries[0]['seed'], entries[0]['n_lines'], entries[0]['n_digits'], len(entries), args.plot_style, args.tag_filter)
