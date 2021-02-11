import os
import tempfile
import unittest

from FilterbankModule import FilterbankModule
import Utils as util


class TestModel(unittest.TestCase):

    def setUp(self):
        debug = True
        use_cuda, device = util.get_device(debug)

        self.model_config = dict(
            # training
            epochs=20,
            batch_size=64,
            max_epochs_no_imprv=10,
            epochs_per_step=1,
            seed=1,

            # arch
            window_size=256,
            wavelet_kernels=500,
            arch=dict(
                conv_config=[
                    # format is [input chans (log 2), kernel size, pool stride], [...]
                    [5, 32, 1], [7, 32, 1], [5, 1, 10], [8, 1, 1]

                ],
                resid_cxns=[(0, 4)],
                lr=0.001,
                large_penalty=0.00005,
                small_penalty=0.0000142,
                sparse_reward=0.0000145,
            ),

            # flags
            restore_from_config=True,
            use_cuda=use_cuda,
            device=device,
            debug=debug
        )

        self.train_tensor, self.eval_tensor, _ = util.load_datasets(device, 1000, 1, 1000, 1)
        self.model = FilterbankModule(train_data=self.train_tensor,
                                      eval_data=self.eval_tensor,
                                      **self.model_config)

    def test_persist(self):
        model = self.model
        verbose = False

        import json
        config_hash_before = hash(json.dumps(model.config.__str__()))
        # _, _ = model.train_model()
        model_hash_before = hash(json.dumps(model.state_dict().__str__()))

        if verbose:
            print("Model conv dict:")
            for j, block in enumerate(model.conv_blocks):
                print("\t", j)
                conv_d = block.state_dict()

                for key in conv_d:
                    print("\t\t", key, " -> ", conv_d[key].size())

            print("Model resid dict:")
            for j, block in enumerate(model.resid_blocks):
                print("\t", j)
                resid_d = block.state_dict()

                for key in resid_d:
                    print("\t\t", key, " -> ", resid_d[key].size())

            print("\nOptimizer dict keys:")
            d1 = model.optimizer.state_dict()['state']
            for k2 in d1:
                print("\t", k2, " -> ", d1[k2]['exp_avg'].size())

            for k2 in model.optimizer.state_dict()['param_groups']:
                print(k2)

        optim_hash_before = hash(json.dumps(model.optimizer.state_dict().__str__()))
        with tempfile.TemporaryDirectory() as dirpath:
            file_name = model.save_checkpoint(dirpath)
            model.load_checkpoint(file_name)

        model_hash_after = hash(json.dumps(model.state_dict().__str__()))
        config_hash_after = hash(json.dumps(model.config.__str__()))
        optim_hash_after = hash(json.dumps(model.optimizer.state_dict().__str__()))

        if verbose:
            print("model diff: {}, optim diff: {}, config diff: {}".format(model_hash_before - model_hash_after,
                                                                           optim_hash_before - optim_hash_after,
                                                                           config_hash_before - config_hash_after))

        self.assertEqual(model_hash_before, model_hash_after)
        self.assertEqual(optim_hash_before, optim_hash_after)
        self.assertEqual(config_hash_before, config_hash_after)

