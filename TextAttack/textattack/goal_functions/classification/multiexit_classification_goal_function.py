"""
    Goal function for the multi-exit adversarial attacks
"""
import numpy as np
import torch

# customized classes
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results.multiexit_classification_goal_function_result import MultiexitClassificationGoalFunctionResult


# --------------------------------------------------------------------------------
#   MultiexitClassificationGoalFunction class
# --------------------------------------------------------------------------------
class MultiexitClassificationGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def _get_layer_score(self, layer_output):

        # Ensure the returned value is now a tensor.
        if not isinstance(layer_output, torch.Tensor):
            raise TypeError(
                "should be tensor "
                f"scores. Got type {type(layer_output)}"
            )
        #regression bug, doesn't check if is a regression or classification
        #adding here:
        elif (layer_output.numel() == 1) and isinstance(
                self.ground_truth_output, float):
            pass

        #otherwise here will squish all regressions guesses to <=1
        elif not ((layer_output.sum(dim=1) - 1).abs() < 1e-6).all():
            # Values in each row should sum up to 1. The model should return a
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            layer_output = torch.nn.functional.softmax(layer_output, dim=1)
            if not ((layer_output.sum(dim=1) - 1).abs() < 1e-6).all():
                raise ValueError("Model scores do not add up to 1.")
        return layer_output[0].cpu()


    def _process_model_outputs(self, attacked_text_list, multiexit_output, multiexit_outputs_data):
        if len(multiexit_outputs_data) != len(attacked_text_list):
            raise ValueError(
                f"Model return score of shape {len(multiexit_outputs_data)} for {len(attacked_text_list)} inputs.")

        result = []
        for example in multiexit_outputs_data:
            scores = []
            for x in example.exit_logits:
                scores.append(self._get_layer_score(x))
            result.append(scores)
        return result

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return MultiexitClassificationGoalFunctionResult

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        return [int(x.argmax()) for x in raw_output]