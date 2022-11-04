from typing import List, Sequence

import torch.distributions as torch_dists


def check_batch_event_shape(dists: Sequence[torch_dists.Distribution]):
    batch_shape = dists[0].batch_shape
    event_shape_ = 0
    secs: List[int] = list()
    for dist in dists:
        # batch shape must be the same
        if batch_shape != dist.batch_shape:
            raise ValueError("batch_shape not the same")
        if len(dist.event_shape) == 0:
            event_shape_ += 1
            secs.append(1)
        elif len(dist.event_shape) == 1:
            event_shape_ += dist.event_shape[0]
            secs.append(dist.event_shape[0])
        else:
            raise ValueError("event_shape must be scaler or column vector")
    event_shape = (event_shape_, )
    return batch_shape, event_shape, secs
