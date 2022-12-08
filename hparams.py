#import signatory
import tensorflow as tf
from text import symbols


class HyperParameters:
    def __init__(self, hparams_string=None, verbose=False):
        """Create model hyperparameters. Parse nondefault from given string."""
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 1000
        self.iters_per_checkpoint = 100
        self.seed = 1234
        self.dynamic_loss_scaling = True
        self.fp16_run = False
        self.distributed_run = False
        self.dist_backend = "nccl"
        self.dist_url = "tcp://localhost:54321"
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.ignore_layers = 0
        # ['embedding.weight']

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk = False
        self.training_files = 'filelists\\ljs_audio_text_train_filelist.txt'
        self.validation_files = 'filelists\\ljs_audio_text_val_filelist.txt'
        self.text_cleaners = ['english_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Model Parameters             #
        ################################
        self.param = 256

        self.n_symbols = len(symbols)
        self.symbols_embedding_dim = self.param

        # Encoder parameters
        self.e_augment_dim = 10
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = self.param
        self.e_transformer_num_heads = 8

        # Decoder parameters
        self.prenet_bias = True
        # self.augment_dim = 1
        self.n_frames_per_step = 1  # currently only 1 is supported
        self.d_transformer_num_heads = 4
        self.d_transformer_dim = self.param
        self.decoder_rnn_dim = self.param
        self.prenet_dim = self.param
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.3
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 256
        self.attention_dim = 256

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 256
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate = False
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.batch_size = 32
        self.mask_padding = True  # set model's padded outputs to padded values

        ################################
        # Neural CDE Hyperparameters #
        ################################
        self.ncde_augment_dim = 10
        self.ncde_input_dim = self.prenet_dim + self.ncde_augment_dim
        self.ncde_hidden_dim = self.prenet_dim + self.ncde_augment_dim
        self.ncde_hidden_hidden_dim = self.n_mel_channels
        self.ncde_output_dim = self.n_mel_channels
        self.ncde_static_dim = None
        self.ncde_solver = "reversible_heun"
        self.ncde_vector_field = "original"
        self.ncde_vector_field_type = "matmul"
        self.ncde_sparsity = None
        self.ncde_num_layers = 3
        self.ncde_use_initial = False
        self.ncde_adjoint = True
        self.ncde_interpolation = "rectilinear"
        self.ncde_interpolation_eps = None
        self.ncde_return_sequences = True
        self.ncde_backend = "torchsde"
        self.bm_size = 10
        self.ncde_t_size = 1

        if hparams_string:
            tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
            self.parse(hparams_string)

        if verbose:
            tf.compat.v1.logging.info('Final parsed hparams: %s', self.values())

    def parse(self, string):
        raise NotImplementedError

    def values(self):
        return dir(self)
