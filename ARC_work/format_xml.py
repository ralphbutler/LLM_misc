import json
import sys

def format_example_as_xml(example_features):
    """Format a single example's analysis as XML."""
    xml = ["<example_analysis>"]

    # Add basic example info
    xml.append(f"  <example_index>{example_features['example_index']}</example_index>")
    xml.append(f"  <input_dimensions>{example_features['input_dimensions']}</input_dimensions>")
    xml.append(f"  <output_dimensions>{example_features['output_dimensions']}</output_dimensions>")

    # Add symmetry information
    if "input_symmetry" in example_features:
        xml.append("  <input_symmetry>")
        for sym_type, is_sym in example_features["input_symmetry"].items():
            xml.append(f"    <{sym_type}>{str(is_sym).lower()}</{sym_type}>")
        xml.append("  </input_symmetry>")
    if "output_symmetry" in example_features:
        xml.append("  <output_symmetry>")
        for sym_type, is_sym in example_features["output_symmetry"].items():
            xml.append(f"    <{sym_type}>{str(is_sym).lower()}</{sym_type}>")
        xml.append("  </output_symmetry>")


    # Add component information
    xml.append("  <components>")
    for i, comp in enumerate(example_features["input_components"]):
        xml.append(f"    <input_component id='{i}'>")
        xml.append(f"      <color>{comp['color']}</color>")
        xml.append(f"      <size>{comp['area']}</size>")
        xml.append(f"      <position>{comp['centroid']}</position>")
        xml.append(f"      <width>{comp['width']}</width>")
        xml.append(f"      <height>{comp['height']}</height>")
        xml.append("    </input_component>")

    for j, comp in enumerate(example_features["output_components"]):
        xml.append(f"    <output_component id='{j}'>")
        xml.append(f"      <color>{comp['color']}</color>")
        xml.append(f"      <size>{comp['area']}</size>")
        xml.append(f"      <position>{comp['centroid']}</position>")
        xml.append(f"      <width>{comp['width']}</width>")
        xml.append(f"      <height>{comp['height']}</height>")
        xml.append("    </output_component>")
    xml.append("  </components>")

    # Add transformation information
    xml.append("  <transformations>")
    for match in example_features["transformations"]["potential_matches"]:
        xml.append("    <transformation>")
        xml.append(f"      <input_id>{match['input_component_idx']}</input_id>")
        xml.append(f"      <output_id>{match['output_component_idx']}</output_id>")

        # Translation information
        if "translation" in match:
            xml.append(f"      <translation>[{match['translation'][0]}, {match['translation'][1]}]</translation>")

        # Color change information
        if "color_change" in match:
            xml.append("      <color_change>")
            xml.append(f"        <from>{match['color_change']['from']}</from>")
            xml.append(f"        <to>{match['color_change']['to']}</to>")
            xml.append("      </color_change>")

        # Size change information
        if "size_change" in match:
            xml.append("      <size_change>")
            xml.append(f"        <from>{match['size_change']['from']}</from>")
            xml.append(f"        <to>{match['size_change']['to']}</to>")
            xml.append("      </size_change>")

        # Shape dimension change information
        if "shape_dimension_change" in match:
            xml.append("      <shape_dimension_change>")
            xml.append(f"        <from_width_height>{match['shape_dimension_change']['from']}</from_width_height>")
            xml.append(f"        <to_width_height>{match['shape_dimension_change']['to']}</to_width_height>")
            xml.append("      </shape_dimension_change>")

        # Potential rotation flag
        if "potential_rotation_90" in match and match["potential_rotation_90"]:
             xml.append("      <potential_rotation_90>true</potential_rotation_90>")


        xml.append("    </transformation>")
    xml.append("  </transformations>")

    # Add unmatched components
    if example_features["transformations"]["unmatched_inputs"] or example_features["transformations"]["unmatched_outputs"]:
        xml.append("  <unmatched_components>")
        if example_features["transformations"]["unmatched_inputs"]:
            xml.append("    <unmatched_inputs>")
            for idx in example_features["transformations"]["unmatched_inputs"]:
                xml.append(f"      <component_id>{idx}</component_id>")
            xml.append("    </unmatched_inputs>")

        if example_features["transformations"]["unmatched_outputs"]:
            xml.append("    <unmatched_outputs>")
            for idx in example_features["transformations"]["unmatched_outputs"]:
                xml.append(f"      <component_id>{idx}</component_id>")
            xml.append("    </unmatched_outputs>")

        xml.append("  </unmatched_components>")

    # Add common translation if available
    if "common_translation" in example_features["transformations"] and example_features["transformations"]["common_translation"]:
        # Renamed this section to be more general for example-level patterns
        xml.append("  <example_patterns>")
        common_trans = example_features["transformations"]["common_translation"]
        xml.append(f"    <common_translation>[{common_trans[0]}, {common_trans[1]}]</common_translation>")
        xml.append("  </example_patterns>")


    xml.append("</example_analysis>")
    return "\n".join(xml)

def format_test_input_as_xml(test_input):
    """Format test input analysis as XML."""
    if not test_input:
        return ""

    xml = ["<test_input>"]
    xml.append(f"  <dimensions>{test_input['dimensions']}</dimensions>")

    # Add symmetry information for test input
    if "symmetry" in test_input:
        xml.append("  <symmetry>")
        for sym_type, is_sym in test_input["symmetry"].items():
            xml.append(f"    <{sym_type}>{str(is_sym).lower()}</{sym_type}>")
        xml.append("  </symmetry>")

    # Add component information
    xml.append("  <components>")
    for i, comp in enumerate(test_input["components"]):
        xml.append(f"    <component id='{i}'>")
        xml.append(f"      <color>{comp['color']}</color>")
        xml.append(f"      <size>{comp['area']}</size>")
        xml.append(f"      <position>{comp['centroid']}</position>")
        xml.append(f"      <width>{comp['width']}</width>")
        xml.append(f"      <height>{comp['height']}</height>")
        xml.append("    </component>")
    xml.append("  </components>")

    xml.append("</test_input>")
    return "\n".join(xml)

def format_global_patterns_as_xml(global_patterns):
    """Format global patterns analysis as XML."""
    if not global_patterns:
        return ""

    xml = ["<global_patterns>"]

    # Add consistent transformations
    if "consistent_transformations" in global_patterns and global_patterns["consistent_transformations"]:
        xml.append("  <consistent_transformations>")
        for trans in global_patterns["consistent_transformations"]:
            xml.append(f"    <{trans['type']}>")
            for value in trans["values"]:
                # Format value based on type (assuming translation is list/tuple)
                if isinstance(value, (list, tuple)):
                     xml.append(f"      <value>[{value[0]}, {value[1]}]</value>")
                else:
                     xml.append(f"      <value>{value}</value>")
            xml.append(f"    </{trans['type']}>")
        xml.append("  </consistent_transformations>")

    # Add color patterns
    if "color_patterns" in global_patterns and global_patterns["color_patterns"]:
        xml.append("  <color_patterns>")
        for from_color, to_colors in global_patterns["color_patterns"].items():
            xml.append(f"    <from_color value='{from_color}'>")
            # Sort to_colors by count descending for readability
            sorted_to_colors = sorted(to_colors.items(), key=lambda item: item[1], reverse=True)
            for to_color, count in sorted_to_colors:
                xml.append(f"      <to_color value='{to_color}' count='{count}'/>")
            xml.append("    </from_color>")
        xml.append("  </color_patterns>")

    # Add size patterns (New)
    if "size_patterns" in global_patterns and global_patterns["size_patterns"]:
        xml.append("  <size_patterns>")
        for from_size, to_sizes in global_patterns["size_patterns"].items():
            xml.append(f"    <from_size value='{from_size}'>")
            # Sort to_sizes by count descending for readability
            sorted_to_sizes = sorted(to_sizes.items(), key=lambda item: item[1], reverse=True)
            for to_size, count in sorted_to_sizes:
                xml.append(f"      <to_size value='{to_size}' count='{count}'/>")
            xml.append("    </from_size>")
        xml.append("  </size_patterns>")

    # Add position patterns (Still placeholder)
    if "position_patterns" in global_patterns and global_patterns["position_patterns"]:
        xml.append("  <position_patterns>")
        # Format position patterns here
        xml.append("  </position_patterns>")

    # Add consistent symmetry (New)
    if "consistent_symmetry" in global_patterns and global_patterns["consistent_symmetry"]:
        xml.append("  <consistent_symmetry>")
        for sym_type, is_consistent in global_patterns["consistent_symmetry"].items():
             if is_consistent: # Only list symmetries that are consistent
                 xml.append(f"    <{sym_type}>true</{sym_type}>")
        xml.append("  </consistent_symmetry>")


    xml.append("</global_patterns>")
    return "\n".join(xml)

def format_as_xml(features):
    """Format entire puzzle analysis as XML for LLM input."""
    # Start with the root element
    xml = ["<puzzle_analysis>"]

    # Add each training example
    xml.append("  <training_examples>")
    for example in features["train_examples"]:
        example_xml = format_example_as_xml(example)
        # Indent the example XML
        example_xml_indented = "\n".join(f"    {line}" for line in example_xml.split("\n"))
        xml.append(example_xml_indented)
    xml.append("  </training_examples>")

    # Add test input
    if features["test_input"]:
        test_input_xml = format_test_input_as_xml(features["test_input"])
        # Indent the test input XML
        test_input_xml_indented = "\n".join(f"  {line}" for line in test_input_xml.split("\n"))
        xml.append(test_input_xml_indented)

    # Add global patterns
    if features["global_patterns"]:
        global_patterns_xml = format_global_patterns_as_xml(features["global_patterns"])
        # Indent the global patterns XML
        global_patterns_xml_indented = "\n".join(f"  {line}" for line in global_patterns_xml.split("\n"))
        xml.append(global_patterns_xml_indented)

    xml.append("</puzzle_analysis>")
    return "\n".join(xml)

def main():
    """Main function to read a JSON file and format it as XML."""
    if len(sys.argv) < 2:
        print("Usage: python format_xml.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            features = json.load(f)

        # Format the JSON as XML
        xml_output = format_as_xml(features)

        # Print the XML
        print(xml_output)

        # Optionally write to file
        # output_path = json_file_path.replace('.json', '.xml')
        # with open(output_path, 'w') as f:
        #     f.write(xml_output)
        # print(f"XML written to {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
