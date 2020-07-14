from .base import COCOAnalysis


class AveragePrecision(COCOAnalysis):

    def __init__(self, ap_mode=('map', 'ap50', 'ap75')):
        super().__init__()
        self.ap_mode = ap_mode

    def accumulate(self):
        super().accumulate()
        ap_dict = {
            'map': self.cocoEval.stats[0],
            'ap50': self.cocoEval.stats[1],
            'ap75': self.cocoEval.stats[2]
        }
        accumulate_state = {}
        for key in self.ap_mode:
            assert key in ('ap50', 'ap75', 'map'), \
                f'ap_mode {key} is not supported currently !'
            accumulate_state.update({
                key: ap_dict[key]
            })
        return accumulate_state
