import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks


class ForceModelCheckpoint(pl_callbacks.ModelCheckpoint):

    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        # remove self._last_global_step_saved == trainer.global_step  # already saved at the last step
        return (bool(
            trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
                or trainer.state.fn !=
                TrainerFn.FITTING  # don't save anything during non-fit
                or trainer.
                sanity_checking  # don't save anything during sanity check
                )
