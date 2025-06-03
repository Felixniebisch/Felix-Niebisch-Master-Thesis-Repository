import os

def clean_csv_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path  # This will overwrite the original

    with open(input_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    cleaned_lines = []
    for line in lines:
        cleaned = line.replace('"', '').replace('\t', '').strip()
        cleaned_lines.append(cleaned + "\n")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(cleaned_lines)

    print(f" Cleaned CSV saved to: {output_path}")

def process_all_csvs(root_dir):
    for current_dir, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d.lower() != 'top']
        for file in files:
            if file.endswith(".csv") and not file.startswith("._"):
                input_path = os.path.join(current_dir, file)
                clean_csv_file(input_path)  # Overwrite


process_all_csvs("/Volumes/Expansion/Usable_data/M1794/")