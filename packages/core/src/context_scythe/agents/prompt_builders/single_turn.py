from textwrap import dedent

from browsergym.core.action.highlevel import HighLevelActionSet
from .base import BasePromptBuilder
from ..trajectory_data import TrajectoryData, StepData


class SingleTurnPromptBuilder(BasePromptBuilder):
    
    def system_message_content(self) -> list[dict]:
        """Return the text representation of the system message."""
        text = f"""
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. 
Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
"""
        return [{"type": "text", "text": text}]
    
    def goal_message_content(self, goal) -> list[dict]:
        return [{"type": "text", "text": f"# Goal\n\n{goal}\n\n"}]
    
    def cot_examples(self) -> list[dict]:
        return [
            {"reasoning": "I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.", "action": "click('12')"},
            {"reasoning": "I found the information requested by the user, I will send it to the chat.", "action": "send_msg_to_user('The price for a 15 inch laptop is 1499 USD.')"},
            {"reasoning": "I have finished navigating to the Products page. I will inform the user that I have completed the task.", "action": "send_msg_to_user('I have finished navigating to the Products page.')"},
        ]
    
    def format_cot_examples(self):
        examples = []
        for example in self.cot_examples():
            txt = f"<think>\n{example['reasoning']}\n</think>\n<action>\n{example['action']}\n</action>"
            examples.append(txt)
        return "\n\n".join(examples)
    
    def action_space_message(self, action_set: HighLevelActionSet) -> list[dict]:
        return  [{
            "type": "text",
            "text": ("# Action Space\n"
                f"{action_set.describe(with_long_description=False, with_examples=True)}\n\n"
                "Here are examples of actions with chain-of-thought reasoning:\n\n"
                f"{self.format_cot_examples()}\n\n"
            )
        }]
    
    def allowed_website_message(
        self,
        calculator_url: str,
        site_urls: dict[str, str],
    ):
        url_info = f"# Allowed list of websites\n\nUse the calculator provided at {calculator_url} for any mathematical calculations. "
        url_info += "You can access the following websites to solve the task:\n"
        url_info += "\n".join([f"- {url}" for name, url in site_urls.items()])
        url_info += f"\n\nDo not access any other website.\n\n"

        website_info = [{
            "type": "text",
            "text": url_info,
        }]
        return website_info
    
    def privileged_information_message(self, privileged_information) -> list[dict]:
        msg = "# Privileged Information\n\nYou can use the following information as a guide, which provides rough information about how to solve the task and key information that should be retained."
        msg += "The guide is a **rough** sequence of steps that solves the task, not a step by step map. Base your reasoning on the current website contents rather than soley relying on this information.\n"
        msg += privileged_information

        return [{
            "type": "text",
            "text": msg
        }]
    
    def final_instruction_message(self) -> list[dict]:
        """Return the text representation of the final instructions."""
        

        final_inst_text = dedent(f"""# Next action
You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.

Put your reasoning in <think>...</think> tags and your action in <action>...</action> tags.
 
Once you have the final answer, return the final answer using the 'send_msg_to_user' action enclosed in <action>...</action> tags.
""")
        
        final_instruction = [{
            "type": "text",
            "text": final_inst_text,
        }]
        return final_instruction
    
    def format_history(self, step_data: StepData):
        # Check if the step data has reasoning and action parsed
        step_num = step_data.step_num
        if step_data.response.reasoning is not None and step_data.response.action is not None:
            formatted_hist = f"## Step {step_num}\n<think>\n{step_data.response.reasoning}\n</think>\n<action>\n{step_data.response.action}\n</action>\n\n"
        else:
            # Just append the raw response
            formatted_hist = f"## Step {step_num}\n{step_data.response}"
        
        return formatted_hist
    
    def build_history_messages(self, step_num: int, trajectory_data: TrajectoryData):
        txt = ["# History of past actions\n"]
        for step in range(step_num):
            step_data = trajectory_data.steps[step]
            txt.append(
                self.format_history(step_data)
            )
        return [{"type": "text", "text": "\n".join(txt)}]

    def build_messages(
            self,
            step_num,
            trajectory_data: TrajectoryData,
            action_set: HighLevelActionSet,
            current_step_data: StepData = None,
            privileged_information: str = None
        ):

        goal = trajectory_data.goal
        calculator_url = trajectory_data.calculator_url
        site_urls = trajectory_data.site_urls

        if current_step_data is None:
            assert step_num < len(trajectory_data)
        
        step_data: StepData = current_step_data or trajectory_data.steps[step_num]

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
        output = {"prompt": [system_message, user_message]}

        if step_data.response is not None:
            assistant_messages = {
                "role": "assistant",
                "content": step_data.response_message_content()
            }
            output["response"] = [assistant_messages]

        return output
