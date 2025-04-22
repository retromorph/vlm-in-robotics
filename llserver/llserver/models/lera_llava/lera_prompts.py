# look_prompt = """
# You are a helpful embodied assistant. You are helping with resolving errors in robot execution.
# During execution of last action you have encountered a problem. Robot failed executing last action.
# You are given a goal and a current plan. 
# Look at the scene and describe what you see. Provide a detailed description of the scene. 
# Describe objects and their positions. Describe robot arm and its position. Describe gripper, its position and its contents.
# HINT: robot may accidently drop objects during gripper movement.

# Successfully executed actions:
# [success_actions]

# Current plan:
# [current_plan]

# Observation:
# <image>

# Current goal:
# [goal]
# """

look_prompt = """
You are a helpful embodied assistant. You are helping with resolving errors in robot execution.
During execution of last action you have encountered a problem. Robot failed executing last action.
You are given a goal and a current plan. 
Look at the scene and describe what you see. Provide a detailed description of the scene. 
Describe objects and their positions. Describe robot arm and its position. Describe gripper, its position and its contents.
HINT: robot may accidently drop objects during gripper movement.

Successfully executed actions:
[success_actions]

Current plan:
[current_plan]

Current goal:
[goal]
"""



explain_prompt = """
You are a helpful embodied assistant. You are helping with resolving errors in robot execution.
During execution of last action you have encountered a problem. Robot failed executing last action.
You are given a goal and a current plan and description of the current scene with ideas of what could be wrong.
Provde ideas how to fix the problem. HINT: robot may accidently drop objects during gripper movement but continue to move fripper with object.
Predict new plan what can fix the problem and reach the goal. Explain each choosen action briefly. You must use actions only from avaliable_actions in you predicted plan. Pay attention on action description. Do not use any other actions and formatting. Do not create combinations of actions.

Action description:
locate('object') - function that give you position of the object in the scene. Allows to move robot arm to the object in order to pick it up or place_on_top_of another object on it.
pick('object') - function that pick up an object. Can be used only if object location is known. Thus you must use locate('object') before pick('object').
place_on_top_of('object') - function that put something from gripper on top of the object, which name is provided to function. Example: place_on_top_of('blue block') - put anything from gripper on top of the blue block.
done() - function that indicate that the task is completed.
Each of locate, pick and place_on_top_of actions takes one argument - object name. You can not use any other arguments or multiple objects.

Strong advice: 
- always use locate('object') before pick('object')
- always use locate('object') before place_on_top_of('object')
- pick('object') can be used only if gripper is empty and object is located
- place_on_top_of('object') can be used only if gripper is NOT empty and object is located

Scene description and ideas of what could be wrong:
[look_response]

Avaliable actions:
[available_actions]

Successfully executed actions:
[success_actions]

Current plan:
[current_plan]

Current goal:
[goal]
"""


replan_prompt = """
You are a helpful embodied assistant. You are helping with resolving errors in robot execution.
During execution of last action you have encountered a problem. Robot failed executing last action.
You are given a goal and a current plan and error resolving ideas. 
Predict new plan to achieve the goal.
HINT: robot may accidently drop objects during gripper movement.

Action description:
locate('object') - function that give you position of the object in the scene. Allows to move robot arm to the object in order to pick it up or place_on_top_of another object on it.
pick('object') - function that pick up an object. Can be used only if object location is known. Thus you must use locate('object') before pick('object').
place_on_top_of('object') - function that put something from gripper on top of the object, which name is provided to function. Example: place_on_top_of('blue block') - put anything from gripper on top of the blue block.
done() - function that indicate that the task is completed.
Each of locate, pick and place_on_top_of actions takes one argument - object name. You can not use any other arguments or multiple objects.

Strong advice: 
- always use locate('object') before pick('object')
- always use locate('object') before place_on_top_of('object')
- pick('object') can be used only if gripper is empty and object is located
- place_on_top_of('object') can be used only if gripper is NOT empty and object is located

Scene description and ideas of what could be wrong:
[look_response]

Error resolving ideas:
[explain_response]

Answer format:
You must answer using actions from avaliable_actions. Pay attention on action description.
Do not use any other actions and formatting. Your output must be a list of actions.
Do not coment on your answer. Do not explain it. Return only new plan as a list of actions.
Separate actions by comma. Do not use any other formatting.

Answer examples:
["locate('cyan block')", "pick('cyan block')", "locate('blue block')", "place_on_top_of('blue block')", "done()"]
["locate('cyan block')", "pick('cyan block')", "locate('red bowl')", "place_on_top_of('cyan block')", "done()"]
["locate('blue block')", "place_on_top_of('blue block')", "done()"]

Successfully executed actions:
[success_actions]

Current plan:
[current_plan]

Avaliable actions:
[available_actions]

Current goal:
[goal]
"""


PROMPTS = {
    "look": look_prompt,
    "explain": explain_prompt,
    "replan": replan_prompt,
}