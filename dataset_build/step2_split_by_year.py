"""Split a paper metadata file by publication year.
Outputs are written separately for 2024 and 2025 records."""

import json
import math
import os

def split_filtered_data():
    # Document-type filtering is already handled upstream, so this step only splits by year.
    input_file = "YOUR_INPUT_JSONL_PATH"
    output_2024 = "YOUR_OUTPUT_2024_JSON_PATH"
    output_2025 = "YOUR_OUTPUT_2025_JSON_PATH"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    count_2024 = 0
    count_2025 = 0
    count_other = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_2024, 'w', encoding='utf-8') as f24, \
         open(output_2025, 'w', encoding='utf-8') as f25:
        
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Publication Year processing
                pub_year = data.get('Publication Year')
                
                if pub_year is None:
                    count_other += 1
                    continue

                try:
                    year_val = int(float(pub_year))
                except ValueError:
                    count_other += 1
                    continue

                if year_val == 2024:
                    f24.write(line.strip() + "\n")
                    count_2024 += 1
                elif year_val == 2025:
                    f25.write(line.strip() + "\n")
                    count_2025 += 1
                else:
                    count_other += 1
            
            except json.JSONDecodeError as e:
                print(f"JSON error on line {i+1}: {e}")
            except Exception as e:
                print(f"Error on line {i+1}: {e}")

if __name__ == "__main__":
    split_filtered_data()
