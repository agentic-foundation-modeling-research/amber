from browsergym.core.action.highlevel import HighLevelActionSet
from .base import BasePromptBuilder
from ..trajectory_data import TrajectoryDataWithStateMemory, StepDataWithStateMemory
from . import SingleTurnPromptBuilder


class SingleTurnWithStateMemoryPromptBuilder(SingleTurnPromptBuilder):

    def compression_seeking_instruction(self) -> list[dict]:
        """Return the text seeking compression."""
        instruction = """# Update Task Memory and Current State

After reviewing the current observation, the current action response, the
action history, the previous cumulative memory, and the previous state,
produce:

1. a cumulative task memory, and
2. a compact snapshot of the current working state.

The purpose of task memory is to preserve both factual information and durable
execution progress needed to complete the task, so a future step does not have
to reconstruct them from old observations or actions. The purpose of current
state is to say where the task stands right now, what is pending, and what needs
attention next. Keep immediate next steps in current state rather than task
memory.

The goal is supplied separately on every step. Do not restate the goal or copy
its constraints into memory. Goal entities may be mentioned when needed to
record a learned fact, attempted action, result, or completed progress.

Only the latest task memory block is carried into the next step. Therefore
the new memory must be a complete cumulative replacement for the previous
memory, not merely a delta containing newly discovered information. Treat this
as continuous one-step summarization: keep the result substantially more
concise than an appended history while preserving all still-useful task
information. Write natural, unstructured prose; do not impose headings, fields,
or a fixed schema inside <memory>.

Begin with a reasoning section that decides which facts and progress remain
useful, what is only current working state, what is already recorded in previous
memory, and what must be updated or discarded.

## Part 1: Task Memory

Write a concise summary containing any information from the previous memory,
previous state, and current observation that may be useful again after the
immediate next step or needed for the final answer. This includes both factual
task knowledge and execution history. It may include:

- facts, values, partial answers, and identifiers learned during execution;
- useful URLs, navigation paths, site structure, and ways to find a page or
  control again;
- actions, searches, filters, form submissions, or strategies already tried,
  together with their outcomes;
- failures, dead ends, validation requirements, and approaches that should not
  be repeated;
- discoveries about how the site behaves or what must be done for an action to
  succeed;
- choices made, information entered, completed milestones, and other
  progress that later steps may need to remember; and
- unresolved leads or dependencies that may matter again later in the task.

Prefer concrete notes such as "Searching for X returned no results" or
"Continue requires a postal code" over vague notes such as "Made progress."
Keep information only when it could help a later decision, prevent repeated
work, recover after navigation, or support the final answer.

Preserve still-useful information from the previous memory, merge duplicates,
and incorporate relevant facts and progress from the previous state and current
observation. If new evidence refines or contradicts an old claim, rewrite or
remove that claim so memory contains only the best current version; do not keep
both the old claim and its correction. Omit prior information only when it is
stale, superseded, duplicated, available from the goal, or no longer useful.

The current observation was available before the current action. A current
action may be recorded as issued or attempted when that progress will matter,
but do not claim it succeeded or failed until an observation confirms the
outcome. On a later step, replace the attempted or unverified status with the
confirmed result.

Use this retention test: keep information when losing it could cause a future
step to lose a needed fact, repeat work, make a worse decision, or become unable
to produce the final answer. Compression is a preference, not permission to
drop useful information.

## Free-Form Update Examples

These examples demonstrate update decisions, not a required structure or style.
The text inside <memory> must remain natural, unstructured prose.

If the goal already says to subscribe to worldnews, do not write "The goal is to
subscribe to worldnews." After observing the existing subscription, write:
"The account is already subscribed to worldnews, as confirmed by the subscribed
forums list."

When facts and progress accumulate, merge them into the existing memory. If the
previous memory says "The product page is open; the item costs $19.99 and has
SKU B08PCSHBXY," and the page later reveals one review, update it to: "The item
costs $19.99, has SKU B08PCSHBXY, and has exactly one review, rated 1 out of 5."

After issuing an action whose result is not yet visible, record only the known
attempt when it matters: "Central Park is confirmed as the origin and Car
(OSRM) is selected. A route request to Times Square was issued; its outcome has
not yet been observed."

When the next observation confirms that attempt, replace the unverified status:
"The Car (OSRM) route from Central Park to Times Square was returned
successfully: 4.3 km and approximately 8 minutes." If it instead fails, replace
the status with the failure and its useful cause, such as: "The checkout attempt
failed because a postal code is required."

If new evidence corrects an old fact, keep only the correction. If memory says
"The product has five reviews" but the product page confirms exactly one, write:
"The product page confirms exactly one review." Do not preserve both counts or
append a narration of the correction.

## Part 2: Current State

Write a compact checkpoint for the next step. It should describe:

- the current page, view, dialog, or stage of the task;
- relevant current UI or form state, including selections and fields that are
  presently filled, empty, or blocked;
- the immediate subgoal and the next one or few actions; and
- remaining subgoals only when they are needed to guide the near-term plan.

Put the reasoning in <think>...</think>, the task memory in
<memory>...</memory>, and the current checkpoint in <state>...</state>.

Return exactly the following three tagged parts, in this order, with no text
outside them:
<think>
reasoning about which facts and progress to retain or update, what belongs in current state, and what to discard
</think>
<memory>
concise, unstructured cumulative replacement containing all still-useful facts and progress
</memory>
<state>
compact current working state and near-term plan
</state>
"""
        final_instruction = [{
            "type": "text",
            "text": instruction,
        }]
        return final_instruction
    
    def format_history(self, step_data: StepDataWithStateMemory):
        # Check if the step data has reasoning and action parsed
        step_num = step_data.step_num
        if step_data.response.action is not None:
            formatted_hist = f"## Step {step_num}\n<action>\n{step_data.response.action}\n</action>\n\n"
        else:
            # Just append the raw response
            formatted_hist = f"## Step {step_num}\n{step_data.response}"
        
        return formatted_hist
    
    def format_memory(self, step_num: int, memory: str):
        formatted_mem = f"## Step {step_num}\n\n{memory}\n\n"
        return formatted_mem

    def build_memory(self, step_num: int, trajectory_data: TrajectoryDataWithStateMemory):

        assert step_num > 0

        # Long-term memory from the previous step
        prev_step_data = trajectory_data.steps[step_num-1]
        memory = prev_step_data.memory.memory
        prev_state = prev_step_data.memory.state

        mem_txt = f"# Memory\n\n{memory}\n\n# State from previous step (Step {step_num-1})\n\n{prev_state}"

        return mem_txt
    
    def build_history_messages(self, step_num: int, trajectory_data: TrajectoryDataWithStateMemory):
        hist_txt = ["# History of past actions\n"]
        for step in range(step_num):
            step_data = trajectory_data.steps[step]
            hist_txt.append(
                self.format_history(step_data)
            )
        hist_txt = "\n".join(hist_txt)

        mem_txt = self.build_memory(step_num, trajectory_data)

        txt = "\n".join([hist_txt, mem_txt])
        
        return [{"type": "text", "text": txt}]

    def build_messages(
            self,
            step_num,
            mode: str,
            trajectory_data: TrajectoryDataWithStateMemory,
            action_set: HighLevelActionSet,
            current_step_data: StepDataWithStateMemory = None,
            privileged_information: str = None
        ):

        assert mode in ["action", "compression"]

        goal = trajectory_data.goal
        calculator_url = trajectory_data.calculator_url
        site_urls = trajectory_data.site_urls

        if current_step_data is None:
            assert step_num < len(trajectory_data)
        step_data: StepDataWithStateMemory = current_step_data or trajectory_data.steps[step_num]

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
            assistant_response = {
                "role": "assistant",
                "content": step_data.memory_message_content()
            }
            output["compression_response"] = assistant_response

        return output
