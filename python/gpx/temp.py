import re

def feet_to_meters(feet):
    return round(float(feet) / 3.28084, 2)

def convert_ele_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # Match <ele>value</ele>
            match = re.search(r'<ele>([\d.]+)</ele>', line)
            if match:
                feet = float(match.group(1))
                meters = feet_to_meters(feet)
                # Replace feet with meters in the line
                line = re.sub(r'<ele>[\d.]+</ele>', f'<ele>{meters}</ele>', line)
            outfile.write(line)

# Example usage
convert_ele_lines('temp1.gpx', 'temp1_converted.gpx')
convert_ele_lines('temp2.gpx', 'temp2_converted.gpx')