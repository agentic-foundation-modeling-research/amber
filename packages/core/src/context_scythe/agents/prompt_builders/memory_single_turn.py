from browsergym.core.action.highlevel import HighLevelActionSet
from .base import BasePromptBuilder
from ..trajectory_data import TrajectoryDataWithMemory, StepDataWithMemory
from . import SingleTurnPromptBuilder


class SingleTurnWithMemoryPromptBuilder(SingleTurnPromptBuilder):

    def __init__(self, use_privileged_information=False, memory_format="append"):
        """
        append: all past geenrated memories are kept in context
        overwrite: only the last memory block is kept in context
        """
        super().__init__(use_privileged_information)
        self.memory_format = memory_format

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
"""
        final_instruction = [{
            "type": "text",
            "text": instruction,
        }]
        return final_instruction
    
    def format_history(self, step_data: StepDataWithMemory):
        # Check if the step data has reasoning and action parsed
        step_num = step_data.step_num
        if step_data.response.action is not None:
            formatted_hist = f"## Step {step_num}\n<action>\n{step_data.response.action}\n</action>\n\n"
        else:
            # Just append the raw response
            formatted_hist = f"## Step {step_num}\n{step_data.response}"
        
        return formatted_hist
    
    def format_memory(self, step_data: StepDataWithMemory):
        step_num = step_data.step_num
        assert step_data.memory is not None
        formatted_mem = f"## Step {step_num}\n\n{step_data.memory}\n\n"
        return formatted_mem
    
    def build_history_messages(self, step_num: int, trajectory_data: TrajectoryDataWithMemory):
        hist_txt = ["# History of past actions\n"]
        for step in range(step_num):
            step_data = trajectory_data.steps[step]
            hist_txt.append(
                self.format_history(step_data)
            )
        hist_txt = "\n".join(hist_txt)

        if self.memory_format == "append":
            mem_txt = ["# History of past memories\n"]
            for step in range(step_num):
                step_data = trajectory_data.steps[step]
                mem_txt.append(
                    self.format_memory(step_data)
                )
            mem_txt = "\n".join(mem_txt)
        elif self.memory_format == "overwrite":
            prev_step_data = trajectory_data.steps[step_num-1]
            assert prev_step_data.memory is not None
            mem_txt = f"# Memory from previous step (Step {step_num-1})\n\n{prev_step_data.memory}\n\n"

        txt = "\n".join([hist_txt, mem_txt])
        
        return [{"type": "text", "text": txt}]

    def build_messages(
            self,
            step_num,
            mode: str,
            trajectory_data: TrajectoryDataWithMemory,
            action_set: HighLevelActionSet,
            current_step_data: StepDataWithMemory = None,
            privileged_information: str = None
        ):

        assert mode in ["action", "compression"]

        goal = trajectory_data.goal
        calculator_url = trajectory_data.calculator_url
        site_urls = trajectory_data.site_urls

        if current_step_data is None:
            assert step_num < len(trajectory_data)
        step_data: StepDataWithMemory = current_step_data or trajectory_data.steps[step_num]

        system_message = {
            "role": "system",
            "content": self.system_message_content()
        }

        user_message_content = self.goal_message_content(goal) + step_data.observation_message_content() + self.action_space_message(action_set)
        
        if step_num > 0:
            user_message_content.extend(self.build_history_messages(step_num, trajectory_data))
        
        user_message_content = user_message_content + self.allowed_website_message(
            calculator_url=calculator_url,
            site_urls=site_urls
        )

        if self.use_privileged_information:
            user_message_content = user_message_content + self.privileged_information_message(privileged_information)

        user_message_content = user_message_content + self.final_instruction_message()
        user_message = {
            "role": "user",
            "content": user_message_content
        }
        prompt_messages = [system_message, user_message]

        output = {}

        if step_data.response is not None:
            assistant_messages = {
                "role": "assistant",
                "content": step_data.response_message_content()
            }
            output["action_response"] = [assistant_messages]

        if mode == "compression":
            assert step_data.response is not None
            prompt_messages.extend(output["action_response"])

            compression_seeking_instruction = {
                "role": "user",
                "content": self.compression_seeking_instruction()
            }
            prompt_messages.append(compression_seeking_instruction)
        
        output = {"prompt": prompt_messages}

        if step_data.memory is not None:
            memory_messages = {
                "type": "text",
                "text": step_data.memory
            }
            assistant_response = {
                "role": "assistant",
                "content": [memory_messages]
            }
            output["compression_response"] = assistant_response

        return output
