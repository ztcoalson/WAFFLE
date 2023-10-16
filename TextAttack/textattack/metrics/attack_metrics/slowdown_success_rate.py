"""

Metrics on SlowdownSuccessRate
---------------------------------------------------------------------

"""

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class SlowdownSuccessRate(Metric):
    def __init__(self):
        self.failed_attacks = 0
        self.skipped_attacks = 0
        self.successful_attacks = 0
        
        # for slowdown
        self.orig_corrects = 0
        self.pert_corrects = 0

        self.all_metrics = {}

    def calculate(self, results):
        """Calculates all metrics related to number of succesful, failed and
        skipped results in an attack.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """
        self.results = results
        self.total_attacks = len(self.results)

        # loop over the results
        for _, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                self.failed_attacks += 1
            elif isinstance(result, SkippedAttackResult):
                self.skipped_attacks += 1
            else:
                self.successful_attacks += 1

            # : for slowdown (retrieve)
            each_oresult = result.original_result
            each_presult = result.perturbed_result

            # : for slowdown (computes)
            self.orig_corrects += ( \
                each_oresult.output[each_oresult.data.layer-1] == \
                each_oresult.ground_truth_output)
            self.pert_corrects += ( \
                each_presult.output[each_oresult.data.layer-1] == \
                each_presult.ground_truth_output)


        # Calculated numbers
        self.all_metrics["successful_attacks"] = self.successful_attacks
        self.all_metrics["failed_attacks"] = self.failed_attacks
        self.all_metrics["skipped_attacks"] = self.skipped_attacks

        # Should be computed from the total results
        self.all_metrics["original_accuracy"] = self.original_accuracy_perc()
        self.all_metrics["attack_accuracy_perc"] = self.attack_accuracy_perc()
        self.all_metrics["attack_success_rate"] = self.attack_success_rate_perc()

        return self.all_metrics

    def original_accuracy_perc(self):
        original_accuracy = self.orig_corrects * 100 / self.total_attacks
        original_accuracy = round(original_accuracy, 2)
        return original_accuracy

    def attack_accuracy_perc(self):
        accuracy_under_attack = self.pert_corrects * 100 / self.total_attacks
        accuracy_under_attack = round(accuracy_under_attack, 2)
        return accuracy_under_attack

    def attack_success_rate_perc(self):
        if self.successful_attacks + self.failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                self.successful_attacks
                * 100.0
                / (self.successful_attacks + self.failed_attacks)
            )
        attack_success_rate = round(attack_success_rate, 2)
        return attack_success_rate
