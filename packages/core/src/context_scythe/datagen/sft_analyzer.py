import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

evidence_prompt = """Analyze the trajectory in {trajectory_file}, which contains a step by step execution
of a task. The trajectory file is JSON with the following relevant shape:
{{
    "goal": <task given to the agent>,
    "steps": [
        {{
            "step_num": <step number in the trajectory>,
            "observation": {{
                "axtree": <full accessibility tree of the webpage at the step>,
                "last_action_error": <any error encountered in the previous action>
            }},
            "response": {{
                "model_full_response": <full response given by the model>,
                "reasoning": <reasoning trace extracted from the model response>,
                "action": <action extracted from the model response which is performed on the webpage>
            }},
            "memory": null
        }}
    ]
}}

The execution trace can contain multiple steps, and all steps share this schema.
The final answer provided by the agent is recorded in the action of the agent in the last step.

Your task is to create {save_file}. Do not edit {trajectory_file} or any other file.

For every step in the trajectory, create the memory that should be available immediately after that step.
The memory should be a compact running summary of the execution so far: what task is being pursued, what
has been accomplished, what useful page facts have been discovered, and what the next useful subgoal appears
to be from the current page state. A future agent should be able to continue after Step N without rereading
any previous accessibility tree, using only the saved memory for Step N and later observations.

You may inspect the entire trajectory, including future steps and the final answer, when deciding what memory
would have been useful after any earlier step. Use those future steps as oracle context for relevance: they can
tell you which page facts, filters, choices, errors, or intermediate results mattered later. However, do not leak
future-only information into an earlier step's memory. The memory for Step N may include only information
available from the goal, observations up to and including Step N, and high-level progress/subgoals justified by
that current state.

Preserve facts that would help a future agent solve the task, even if the recorded agent did not explicitly use
them. Include useful labels, values, table rows, option names, link destinations when semantically important,
identifiers visible as page content, item ordering, status text, selected filters, visible errors, or other page
facts needed to choose later actions or answer the task.

The memory at each step is cumulative, not a delta. If a fact or plan remains important after Step N, include it in Step N's
memory even if it first appeared in an earlier step. Update the running summary as the trajectory progresses.
Do not repeat the original goal verbatim in every memory entry. Mention the task goal only when it is needed
to clarify current progress or the next subgoal; otherwise focus on what has changed, what remains relevant,
and what should be done next.

Memory may include high-level progress and next-step intent, such as "the AskReddit forum is open and the
sort menu should be used to choose Most commented." Do not copy the agent's reasoning trace, do not include
raw action syntax, and do not mention low-level navigation mechanics unless they are necessary to describe the
current state or next subgoal. Do not include form values typed by the agent unless they are visible page content
or necessary task state.

Each memory value must be either:
- a plain string containing the useful running memory after that step, or
- null only when no useful page facts, progress, task state, error state, or next subgoal need to be retained.

{few_shot_prompt}

Use Python's json library to read {trajectory_file} and all the example trajectories. Write {save_file} in JSONL
format with exactly one object per trajectory step, in ascending step order. Each line must contain exactly the
fields `step` and `memory`:

```{save_file}
{{"step": 0, "memory": <memory for step 0>}}
{{"step": 1, "memory": <memory for step 1>}}
{{"step": 2, "memory": <memory for step 2>}}
...
{{"step": N, "memory": <memory for step N>}}
```

The `step` value must be the step's `step_num` from the trajectory. The `memory` value must be a JSON string or null.
The output file must contain one line for every step in the trajectory, including steps whose memory is null.

Create parent directories for {save_file} if needed. Do not edit any file other than {save_file}. Begin your analysis now.
"""

few_shot_prompt = """# Examples of memory

The following trajectory files contain examples of step-wise memory:

{examples}

The format is the same as the trajectory you are analyzing, with the memory field filled. Refer to these examples and
follow similar patterns when creating the memory for each of the steps for the trajectory you are analyzing. When in doubt,
treat these as the gold standard.

"""

class Analyzer:

    def __init__(
            self,
            exp_dir: str,
            few_shot_trajectories,
            save_dir: str,
        ):
        self.exp_dir = Path(exp_dir)
        self.few_shot_trajectories = few_shot_trajectories
        self.save_dir = Path(save_dir)

    def build_prompt(self, task_id):

        trajectory_file = self.exp_dir / f"{task_id}.json"
        save_file = self.save_dir / f"{task_id}.jsonl"

        few_shot_trajectories = [
            f"Trajectory {n}: {trajectory_file}" for n, trajectory_file in enumerate(self.few_shot_trajectories)
        ]
        task_few_shot_prompt = few_shot_prompt.format(
            examples="\n".join(few_shot_trajectories)
        )
        
        prompt = evidence_prompt.format(
            trajectory_file=trajectory_file,
            few_shot_prompt=task_few_shot_prompt,
            save_file=save_file,
        )
        return prompt
    
    def analyze(self, task_id, cwd, timeout=5*60):

        prompt = self.build_prompt(task_id)

        self.invoke_agent(prompt, cwd, timeout=timeout)

    def invoke_agent(self, prompt, cwd, timeout=5*60):
        cmd = ["pi", "-p", "--provider", "openai", "--model", "gpt-5.5", "--thinking", "medium", prompt]
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode:
            output = result.stderr.strip() or result.stdout.strip()
            message = f"Analyzer agent failed with exit code {result.returncode}"
            if output:
                message = f"{message}: {output}"
            raise RuntimeError(message)
        
        return result
