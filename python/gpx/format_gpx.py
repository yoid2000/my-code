import xml.dom.minidom

def meters_to_feet(meters):
    return round(float(meters) * 3.28084, 2)

def clean_and_format_gpx(input_file, output_file):
    # Read and parse the raw XML
    with open(input_file, 'r', encoding='utf-8') as file:
        raw_xml = file.read()
    dom = xml.dom.minidom.parseString(raw_xml)

    # Remove all <time> elements
    for time_node in dom.getElementsByTagName("time"):
        time_node.parentNode.removeChild(time_node)

    # Convert <ele> values from meters to feet
    for ele_node in dom.getElementsByTagName("ele"):
        try:
            meters = float(ele_node.firstChild.nodeValue.strip())
            feet = meters_to_feet(meters)
            ele_node.firstChild.nodeValue = str(feet)
        except (ValueError, AttributeError):
            continue  # Skip malformed <ele> entries

    # Normalize and strip whitespace inside <trkpt>
    for trkpt in dom.getElementsByTagName("trkpt"):
        trkpt.normalize()
        for node in trkpt.childNodes:
            if node.nodeType == node.TEXT_NODE:
                node.data = node.data.strip()

    # Generate pretty XML
    pretty_xml = dom.toprettyxml(indent="  ")

    # Collapse <trkpt> blocks into one line
    lines = pretty_xml.splitlines()
    collapsed = []
    buffer = []
    inside_trkpt = False

    for line in lines:
        if "<trkpt" in line:
            inside_trkpt = True
            buffer = [line.strip()]
        elif "</trkpt>" in line:
            buffer.append(line.strip())
            collapsed.append(" ".join(buffer))
            inside_trkpt = False
        elif inside_trkpt:
            buffer.append(line.strip())
        else:
            collapsed.append(line)

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(collapsed))

# Example usage
clean_and_format_gpx('alta-via-1-gs.gpx', 'alta-via-1-gs-pretty.gpx')