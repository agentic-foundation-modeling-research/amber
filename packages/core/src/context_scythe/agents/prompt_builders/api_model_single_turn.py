from textwrap import dedent
from .memory_single_turn import SingleTurnWithMemoryPromptBuilder


class APIModelPromptBuilder(SingleTurnWithMemoryPromptBuilder):

    def final_instruction_message(self) -> list[dict]:
        """Return the text representation of the final instructions."""
        

        final_inst_text = dedent(f"""# Next action
You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.

Perform only one action at a time. Chaining multiple actions is not allowed.

Explore the website thoroughly and answer when you're confident.

Put your reasoning in <think>...</think> tags and your action in <action>...</action> tags.
 
Once you have the final answer, return the final answer using the 'send_msg_to_user' action enclosed in <action>...</action> tags.

Your response should strictly follow <think>..</think>..<action>..</action> tags format.
""")
        
        final_instruction = [{
            "type": "text",
            "text": final_inst_text,
        }]
        return final_instruction
    
    def compression_seeking_instruction(self) -> list[dict]:
        """Return the text seeking compression."""
        instruction = """# Memory\n\n
You will now produce the key information from the observations, actions and previous memory (if any) 
and create a memory. The memory should contain any information that will help you achieve the goal
or navigate the website in future steps.

This includes (but is not limited to):
- Key facts, values, or partial answers gathered from the page so far
- URLs of useful pages (e.g. the current page, search results, product pages,
  forms) that you may need to revisit
- The structure or layout of the site (menus, sections, filters, navigation
  paths) so you don't have to rediscover them
- Identifiers such as product IDs, order numbers, usernames
- Progress so far: which subgoals are done, which are pending, and what to try
  next or avoid trying again

The memory block is the only thing carried over between steps, so write it as
self-contained notes that a fresh reader could use to continue the task without
seeing the prior pages.

The memory response can be free form with no requirement on the format.
"""
        final_instruction = [{
            "type": "text",
            "text": instruction,
        }]
        return final_instruction
