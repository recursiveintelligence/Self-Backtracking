from .decoders import (
    GreedyDecoder,
    SelfBackTrackingDecoder,
)

DECODER_DICT={
    'greedy':GreedyDecoder,
    'self_backtrack':SelfBackTrackingDecoder
}