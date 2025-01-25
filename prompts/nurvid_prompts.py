cluster_main_prompt = '''You are a helpful medical assistant. You are now given a frame from a nursing education video. Your job is to determind if the frame is part of a predefined medical procedure. Here are all the procedures: {procedure}.
    Here are a few things for you to keep in mind:
    1. The frame belongs to an procedure ONLY IF it clearly shows the action in the frame. Here is an example for procedure 29 Draw bed curtains:
        Scene Description: A medical personalle standing next to a curtain.
        This is NOT an action of closing the curtain, as you are NOT ABSOLUTELY certain about the action of the subject.
        In this case ONLY IF the medical personalle is holding and dragging the curtain should be consider as the procedure 29.
    2. Don't be too interpretive on the scene or the action. Take the face value of the frame, do not be too imaginative on the action/scene. Here is an example:
        Scene Description: A nurse is standing and holding a antiseptic solution and sterile gauze.
        This should be taken as its face value: "A nurse holding some tools" and thats it. While it is possible that the nurse is going to perform skin disinfection, procedure 4, this is NOT. DO NOT consider the potential pre/post-processing as part of the procedure.{output_command}
'''

first_round = '''
    You need to output 2 things: 
    1. Potential matching procedures: If you consider the frame is one of the medical procedure, return the code of the top three procedures (1-177) that fits the scene, code 0 for non-procedure, as a list of int. Please follow EXACTLY the following format "[code1, code2, code3]".
    2. Explaination: Provide a one line detailed explaination for your answer.
'''

second_round = '''
    You need to output 2 things: 
    1. Matching procedures: If you consider the frame is one of the medical procedure, return the code of the procedure (1-3) that fits the scene, code 0 for non-procedure, as an int.
    2. Explaination: Provide a one line detailed explaination for your answer.
'''

import os
import json
import copy

class NurVidPromptGenerator:
    def __init__(self, annotation_dir):
        self.annotation_dir = annotation_dir

        self.code2action, self.actions = {}, ''
        for aidx, action in enumerate(open(f'{self.annotation_dir}/actions.txt').read().splitlines()):
            self.code2action[aidx + 1] = action
        self.action_pool = os.path.join(annotation_dir, 'action_pool')

    def generate_actions(self, round_id, codes = []):
        if round_id == 1:
            action_desc = ';'.join([f'{k} {v}' for k,v in self.code2action.items()])
        elif round_id == 2:
            action_desc = ''
            for cid, code in enumerate(codes):
                action = self.code2action[code]
                action_js = json.load(open(f'{self.action_pool}/{action.replace(" ", "_")}.json', 'r'))[action]
                description = ' '.join([a['sentence'] for a in action_js])
                action_desc += f'{cid + 1} {action} Description: {description}; '
        return action_desc
    
    def __call__(self, round_id = 1, codes = []):
        base_prompt = copy.deepcopy(cluster_main_prompt)
        procedures = self.generate_actions(round_id, codes)
        round_prompt = copy.deepcopy(first_round) if round_id == 1 else copy.deepcopy(second_round)
        base_prompt = base_prompt.replace('{procedure}', procedures).replace('{output_command}', round_prompt)
        return base_prompt




