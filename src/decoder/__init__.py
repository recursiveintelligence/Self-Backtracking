from .decoders import (
    GreedyDecoder,
    SelfBackTrackingDecoder,
)
from .stopping import make_stop_criteria

DECODER_DICT={
    'greedy':GreedyDecoder,
    'self_backtrack':SelfBackTrackingDecoder
}
