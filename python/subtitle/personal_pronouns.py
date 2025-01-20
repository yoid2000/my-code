import os
import re
import chardet

change_list = ["i'm", 'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', "you're", 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', "they're", 'they', 'theirs']

indir = 'infiles'
outdir = 'outfiles'

def do_swaps(file):
    # Create input and output file paths
    input_path = os.path.join(indir, file)
    output_path = os.path.join(outdir, 'mod_' + file)

    with open(input_path, 'rb') as f:
        result = chardet.detect(f.read())

    # Open the input file and output file
    with open(input_path, 'r', encoding=result['encoding']) as infile, open(os.path.join('outfiles', 'mod_' + file), 'w', encoding='utf8') as outfile:
        # Read each line from the input file
        for line in infile:
            # For each string in the change_list
            for change in change_list:
                # Create a pattern that matches the change word as a distinct word, ignoring case
                pattern = re.compile(r'\b' + re.escape(change) + r'\b', re.IGNORECASE)
                # Replace the matched word with '--'
                line = pattern.sub('--', line)
            # Write the modified line to the output file
            outfile.write(line)

for filename in os.listdir(indir):
    do_swaps(filename)