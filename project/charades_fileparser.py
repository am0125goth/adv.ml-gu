import os
import torch

class charadesClassParser():
    def __init__(self, classes_file, object_classes_file, verb_classes_file, mapping_file):

        self.classes_file = classes_file
        self.object_classes_file = object_classes_file
        self.verb_classes_file = verb_classes_file
        self.mapping_file = mapping_file

        #load all classes
        self.action_classes = self._load_classes_file(classes_file)
        self.object_classes = self._load_classes_file(object_classes_file)
        self.verb_classes = self._load_classes_file(verb_classes_file)
        self.action_mappings = self._load_mapping_file(mapping_file)

        #create indicies
        self.verb_to_idx = {verb_id: idx for idx, verb_id in enumerate(self.verb_classes.keys())}
        self.object_to_idx = {obj_id: idx for idx, obj_id in enumerate(self.object_classes.keys())}
        self.action_to_idx = {action_id: idx for idx, action_id in enumerate(self.action_classes.keys())}

        #create reverse mappings with idx
        self.idx_to_verb = {idx: verb_id for verb_id, idx in self.verb_to_idx.items()}
        self.idx_to_object = {idx: obj_id for obj_id, idx in self.object_to_idx.items()}
        self.idx_to_action = {idx: action_id for action_id, idx in self.action_to_idx.items()}

    def _load_classes_file(self, file_path):
        #loades classes from file where each line in file consists of an id and an object/verb/action (action = verb + object)
        classes = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        #split only on first space (i.e. between id and its corresponding object/verb/action
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            class_id, description = parts
                            classes[class_id] = description
                        elif len(parts) == 1:
                            #handles what happends if no description is given
                            classes[parts[0]] = ""
        else:
            print(f"File {file_path} not found")
        return classes

    def _load_mapping_file(self, file_path):
        mappings = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 3:
                            action_id, obj_id, verb_id = parts
                            mappings[action_id] = (verb_id, obj_id)
        else:
            print(f"Mapping file {file_path} not found")

        return mappings

    def get_action_components(self, action_id):
        #get verb and object from an action
        return len(self.action_mappings.get(action_id, (None, None)))

    def get_num_verbs(self):
        return len(self.verb_classes)

    def get_num_objects(self):
        return len(self.object_classes)

    def get_num_actions(self):
        return len(self.action_classes)

    def action_to_component_tensors(self, action_id):
        #convert action id to multi-hot tensors for verb and object
        verb_tensor = torch.zeros(self.get_num_verbs())
        obj_tensor = torch.zeros(self.get_num_objects())

        if action_id in self.action_mappings:
            verb_id, obj_id = self.action_mappings[action_id]
            if verb_id in self.verb_to_idx:
                verb_tensor[self.verb_to_idx[verb_id]] = 1.0
            if obj_id in self.object_to_idx:
                obj_tensor[self.object_to_idx[obj_id]] = 1.0

        return verb_tensor, obj_tensor

    def get_verb_description(self, verb_id):
        return self.verb_classes.get(verb_id, "Unknown")

    def get_object_description(self, obj_id):
        return self.object_classes.get(obj_id, "Unknown")

    def get_action_description(self, action_id):
        return self.action_classes.get(action_id, "Unknown")

    def print_class_summary(self):
        #print summary of loaded classes
        print(f"Loaded {self.get_num_verbs()} verbs")
        print(f"Loaded {self.get_num_objects()} objects") 
        print(f"Loaded {self.get_num_actions()} actions")
        print(f"Loaded {len(self.action_mappings)} action mappings")

        #print some examples
        print("\nExample mappings:")
        for i, (action_id, (verb_id, obj_id)) in enumerate(list(self.action_mappings.items())[:5]):
            action_desc = self.get_action_description(action_id)
            verb_desc = self.get_verb_description(verb_id)
            obj_desc = self.get_object_description(obj_id)
            print(f"{action_id}: {action_desc}")
            print(f"Verb: {verb_id} ({verb_desc})")
            print(f"Object: {obj_id} ({obj_desc})")

    def get_all_verbs(self):
        return list(self.verb_classes.keys())

    def get_all_objects(self):
        return list(self.object_classes.keys())

    def get_all_actions(self):
        return list(self.action_classes.keys())