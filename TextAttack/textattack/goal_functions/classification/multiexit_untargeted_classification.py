"""
    Determine successful in untargeted classification
"""
import torch
from torch import linalg

# custom class
from .multiexit_classification_goal_function import MultiexitClassificationGoalFunction


# --------------------------------------------------------------------------------
#   MultiexitUntargetedClassification class
# --------------------------------------------------------------------------------
class MultiexitUntargetedClassification(MultiexitClassificationGoalFunction):
    """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """
    def __init__(self, *args, target_max_score=None, target_slowdown_score=-1., **kwargs):
        self.target_max_score = target_max_score
        self.target_slowdown_score = target_slowdown_score
        super().__init__(*args, **kwargs)

    # Note: model output here is a list of softmax (or regression score) at each exit

    def _is_goal_complete(self, model_output, data, attacked_text):
        # ----------------------------------------
        #   Depending on how we set the goals
        # ----------------------------------------
        if self.attack_goal == 'base':
            model_output = model_output[-1]
            if self.target_max_score:
                return model_output[self.ground_truth_output] < self.target_max_score
            elif (model_output.numel() == 1) \
                    and isinstance(self.ground_truth_output, float):
                return abs(self.ground_truth_output - model_output.item()) >= self.regression_threshold
            else:
                return model_output.argmax() != self.ground_truth_output

        elif self.attack_goal == 'oavg':
            if self.target_max_score:
                # : avg. output prob. < max_score
                model_output_avg = sum([ \
                    each_output[self.ground_truth_output] \
                    for each_output in model_output])
                return model_output_avg < self.target_max_score
            elif (model_output[-1].numel() == 1) \
                    and isinstance(self.ground_truth_output, float):
                # : avg. deviations >= threshold
                deviation_avg = sum([ \
                    abs(self.ground_truth_output - each_output.item()) \
                    for each_output in model_output])
                return deviation_avg >= self.regression_threshold
            else:
                # : all the classification outputs are hosed
                return all([each_output.argmax() != self.ground_truth_output \
                            for each_output in model_output])

        elif self.attack_goal == 'omax':
            raise NotImplementedError()

        elif self.attack_goal == 'ours':
            if self.target_slowdown_score:
                # : check if we achieve the slowdown
                return self._get_ours_score(model_output) > self.target_slowdown_score
            else:
                # : [Note] do not support regression, etc.
                raise NotImplementedError()

        else:
            assert False, ('Error" unsupported attack goal - {}'.format(self.attack_goal))


    def _get_score(self, model_output, model_data, attacked_text):
        # ----------------------------------------
        #   Depending on how we set the goals
        # ----------------------------------------
        if self.attack_goal == 'base':
            return self._get_base_score(model_output)
        elif self.attack_goal == 'oavg':
            return self._get_oavg_score(model_output)
        elif self.attack_goal == "omax":
            return self._get_omax_score(model_output)

        # our slowdown attacks
        elif self.attack_goal == "ours":
            return self._get_ours_score(model_output)
        else:
            assert False, ('Error: unsupported attack goal - {}'.format(self.attack_goal))


    # ------------------------------------------------
    # [Baseline] compute the score from the last layer
    # ------------------------------------------------
    def _get_base_score(self, model_output):
        output = model_output[-1]
        if (output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(output.item() - self.ground_truth_output)
        else:
            return 1 - output[self.ground_truth_output]


    # ------------------------------------------------
    # [Average] compute the averaged score over the exits
    # ------------------------------------------------
    def _get_oavg_score(self, model_output):
        tot_score  = 0.
        regression = ( 
            (model_output[0].numel() == 1) and 
            isinstance(self.ground_truth_output, float) )

        # loop over the layer outputs
        for layer_output in model_output:
            if regression:
                tot_score += abs(layer_output.item() - self.ground_truth_output)
            else:
                tot_score += (1 - layer_output[self.ground_truth_output])
        return tot_score / len(model_output)

    # ------------------------------------------------
    # [Max.] deprecated, average is the strongest one.
    # ------------------------------------------------
    def _get_omax_score(self, model_output):
        raise NotImplementedError()


    # ------------------------------------------------
    # [Ours] compute the averaged score over the exits
    # ------------------------------------------------
    def _get_ours_score(self, model_output):
        tot_score  = 0.
        regression = ( 
            (model_output[0].numel() == 1) and 
            isinstance(self.ground_truth_output, float) )

        # uniform tensor
        nclasses = len(model_output[0])
        eachprob = 1. / nclasses
        uni_prob = [eachprob for _ in range(nclasses)]
        uni_prob = torch.Tensor(uni_prob)

        # loop over the layer outputs
        for _, layer_output in enumerate(model_output):
            if regression:
                # : [deprecated] we use all the classification tasks, ignore
                tot_score += 0.
            else:
                # : this will bound the loss in [0, 1]
                tot_score += (1. - (1. / (nclasses - 1)) * sum(abs(layer_output - uni_prob))).item()

        # convert the tensor to a float value
        return tot_score / len(model_output)


    # ------------------------------------------------
    # [FIXME] for the later on...
    # ------------------------------------------------
    #get close to uniform distribution but with slight advantage to the correct answer
    def _get_uniform_plus(self, model_output):
        score=0
        reg = False
        uniform_ = 1/len(model_output[0])
        multiplier = 0.2
        plus_ = (1 + multiplier) * uniform_
        minus_ = (1 - (multiplier/(len(model_output[0])-1))) * uniform_
        uniform_plus = [minus_ for x in range(len(model_output[0]))]
        uniform_plus[self.ground_truth_output] = plus_
        uniform_plus = torch.Tensor(uniform_plus)
        if (model_output[0].numel() == 1) and isinstance(self.ground_truth_output, float):
            reg = True
        for layer_output in model_output:
            if reg:
                score = 0
            else:
                score += (1 - linalg.norm(layer_output - uniform_plus))
        return score/len(model_output)

    #get close to uniform distribution but with slight disadvantage to the correct answer
    def _get_uniform_minus(self, model_output):
        score=0
        reg = False
        uniform_ = 1/len(model_output[0])
        multiplier = 0.2
        minus_ = (1 - multiplier) * uniform_
        plus_ = (1 + (multiplier/(len(model_output[0])-1))) * uniform_
        uniform_minus = [plus_ for x in range(len(model_output[0]))]
        uniform_minus[self.ground_truth_output] = minus_
        uniform_minus = torch.Tensor(uniform_minus)
        if (model_output[0].numel() == 1) and isinstance(self.ground_truth_output, float):
            reg = True
        for layer_output in model_output:
            if reg:
                score = 0
            else:
                score += (1 - linalg.norm(layer_output - uniform_minus))
        return score/len(model_output)

    #weighted average of distance from last layer (if that prediction was correct)
    #assuming that models will usually be correct and will mostly predict correctly in most layers if so:
    #if correct, add value equal to layer, incentivizes being correct at later layer
    #if incorrect, add value equal to flipped layer, incentivizes being incorrect at earlier layer
    def _get_layer_dist(self, model_output):
        score=0
        reg = False
        if (model_output[0].numel() == 1) and isinstance(self.ground_truth_output, float):
            reg = True
        for i,layer_output in enumerate(model_output):
            if reg:
                if abs(self.ground_truth_output - layer_output.item()) <= self.regression_threshold:
                    score += i
                else:
                    score += (len(model_output)-i)
            else:
                if layer_output.argmax() == self.ground_truth_output:
                    score += i
                else:
                    score += (len(model_output)-i)
        return score/len(model_output)

    def _get_layer_dist2(self, model_output, model_data):
        return model_data.layer

    #try to make each layer output a different label than the previous
    def _get_alternate(self, model_output):
        score=0
        reg = False
        if (model_output[0].numel() == 1) and isinstance(self.ground_truth_output, float):
            reg = True
        for i in range(len(model_output)):
            if i == 0:
                continue
            if reg:
                cur = abs(self.ground_truth_output - model_output[i].item()) <= self.regression_threshold
                prev = abs(self.ground_truth_output - model_output[i-1].item()) <= self.regression_threshold
                if cur != prev:
                    score += 1
            else:
                cur = model_output[i].argmax() == self.ground_truth_output
                prev = model_output[i-1].argmax() == self.ground_truth_output
                if cur != prev:
                    score += 1
        return score/(len(model_output)-1)

    #score from exit layer
    def _get_exitl(self, model_output, model_data):
        output = model_output[model_data.layer-1]
        if (output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(output.item() - self.ground_truth_output)
        else:
            return 1 - output[self.ground_truth_output]
    
    def _layer(self, data):
        if data.layer == 12:
            return True
        else:
            return False
