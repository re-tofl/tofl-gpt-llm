import os
import json

def parse_files_in_directory(directory):
    data = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  
            print(f'reading : {filename}')
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                
                sections = content.split('================================================================')
                
                for section in sections:
                    if section.strip():  
                        parts = section.split('____________________________________________________________')
                        if len(parts) == 2:
                            question = parts[0].strip()
                            answer = parts[1].strip()
                            data.append({
                                "question": question,
                                "answer": answer
                            })
    return data

directory_path = './db'

parsed_data = parse_files_in_directory(directory_path)

with open('data.json', 'w', encoding='utf-8') as json_file:
    json.dump(parsed_data, json_file, ensure_ascii=False, indent=4)

print("data was parsed and written in data.json")
