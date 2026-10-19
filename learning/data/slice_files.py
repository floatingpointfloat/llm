# Create a tiny custom dataset of your preferred size
input_file = "training_text_base.txt"
output_file = "training_text.txt"

# Change this number to however many lines you want to test with
max_lines = 157800
starting_from_line = 0

with open(input_file, "r", encoding="utf-8") as infile:
    with open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i >= max_lines + starting_from_line:
                break
            if (
                i >= starting_from_line
            ):  # only write the lines after the starting from line variable (to make a variable training text)
                outfile.write(line)

print(f"Sliced the text to {max_lines} lines at {output_file}!")
