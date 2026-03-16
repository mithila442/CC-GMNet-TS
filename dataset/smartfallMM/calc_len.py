import os
import csv

folder = "."   # current directory
output_file = "seq.csv"

results = []

for fname in sorted(os.listdir(folder)):
    if fname.lower().endswith(".csv") and fname != output_file:
        file_path = os.path.join(folder, fname)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()

                # detect delimiter
                if ";" in first_line and "," not in first_line:
                    delimiter = ";"
                elif "," in first_line:
                    delimiter = ","
                else:
                    delimiter = ","  # fallback

                f.seek(0)
                reader = csv.reader(f, delimiter=delimiter)
                row_count = sum(1 for row in reader if row)  # count non-empty rows

            results.append([fname, row_count])

        except Exception as e:
            print(f"Error reading {fname}: {e}")

# save result
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["csv_name", "length"])
    writer.writerows(results)

print(f"Saved {len(results)} entries to {output_file}")