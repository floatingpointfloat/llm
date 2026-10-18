# Create a tiny custom dataset of your preferred size
input_file = "training_text.txt"
output_file = "training_text_new.txt"

# Change this number to however many lines you want to test with
max_lines = 7000

with open(input_file, "r", encoding="utf-8") as infile:
    with open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i >= max_lines:
                break
            outfile.write(line)

print(f"Sliced the text to {max_lines} lines at {output_file}!")
