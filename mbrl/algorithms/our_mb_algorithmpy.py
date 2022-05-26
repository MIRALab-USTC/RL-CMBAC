from mbrl.algorithms.mb_algorithm import MBRLBatchAlgorithm

class OurMBAlgorithm(MBRLBatchAlgorithm):
    def _train_batch(self):
        real_batch = self.pool.random_batch(self.real_batch_size, without_keys=['deltas'])
        imagined_batch= self.imagined_data_pool.random_batch(self.imagined_batch_size)
        params = self.trainer.train(real_batch, imagined_batch)
        return params


