"""

attack_metrics package:
---------------------------------------------------------------------

TextAttack provide users common metrics on attacks' quality.

"""

from .attack_queries import AttackQueries
from .attack_success_rate import AttackSuccessRate
from .words_perturbed import WordsPerturbed

# for the slowdown attacks
from .slowdown_success_rate import SlowdownSuccessRate
from .slowdown_words_perturbed import SlowdownWordsPerturbed
