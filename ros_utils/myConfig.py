#!/usr/bin/env python3
import json

class MyConfig:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load()  # Initialize the config_dict with loaded configurations
    
    def save(self, config_name, joint_positions):
        # Create a dictionary to store the joint positions with a name
        # Open the JSON file for writing
        self.config_dict[config_name]=list(joint_positions)
        print(self.config_dict)
        with open(self.file_path, 'w') as file:
            # Append the new joint configuration to the file
            json.dump(self.config_dict, file, indent=4)

    def load(self):
        try:
            # Open the JSON file for reading
            with open(self.file_path, 'r') as file:
                self.config_dict = json.load(file)
        except Exception as e:
            # Handle the case where the file does not exist
            print(f"Configuration file '{self.file_path}' not found.")
            self.config_dict ={}


    def get(self, name):
        return self.config_dict[name]

