# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
import logging
import math

import numpy
import torch
import torch.nn as nn
import json

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.new_plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

def genUtt2Label(file):
    utt2h = {}
    f = open(file, 'r')
    line = f.readline()
    while line:
        row = line.split()
        utt = row[0]
        label = float(row[1])
        utt2h[utt] = label
        line = f.readline()
    f.close()
    return utt2h

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument('--pretrained-model', default="", type=str,
                           help='pretrained ASR model for initialization')
        group = add_arguments_transformer_common(group)

        return parser

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        ### Tung: classification code ####
        logging.warning('e2e_transformer_for_regression_general, using mean and std pooling')
        self.lossType=args.lossType
        self.linear2 = nn.Linear(in_features=2*args.adim,
                                 out_features=1)
        if self.lossType == 'MSE':
            self.loss_function = torch.nn.MSELoss()  # this is for regression mean squared loss
            logging.warning('Loss function = Root Mean Square Error (L2Loss)')
        elif self.lossType == 'MAE':
            self.loss_function = torch.nn.L1Loss()
            logging.warning('Loss function = Mean Absolute Error (L1Loss)')
        elif self.lossType == 'CE':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            logging.warning('Undefined loss')
            sys.exit(1)
        
        self.orgData = genUtt2Label(args.utt2labelTrain)
        self.orgDataDev = genUtt2Label(args.utt2labelDev)

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        if args.pretrained_model:
            path = args.pretrained_model
            logging.warning("load pretrained asr model from {}".format(path))
            if 'snapshot' in path:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
            else:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state_dict, strict=False)
            del model_state_dict
        else:
            initialize(self, args.transformer_init)
        # initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad, uttList):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        mean_f,std_f = torch.std_mean(hs_pad, 1)
        att_vec = torch.cat((mean_f, std_f), 1)
        #logging.warning('ilens = ' + str(ilens))
        x = self.linear2(att_vec).squeeze()
        labelList = []
        trainOrDev = "TRAIN"
        for utt in uttList:
            if utt not in self.orgData.keys(): ## This is for validation. This step does not have augmentation data
                trainOrDev = "DEV"
                label = self.orgDataDev[utt]
            else:
                label = self.orgData[utt]
            labelList.append(label)
        #target = torch.empty(hs_pad.size(0), dtype=torch.long).random_(2).to(hs_pad.device)
        target = torch.FloatTensor(labelList).to(hs_pad.device)
        loss = torch.sqrt(self.loss_function(x, target))
        
        acc = 100.0/loss
        #logging.warning('Acc = ' + str(acc))
        loss_data = float(loss)
        self.reporter.report(0, 0, float(acc), 0, 0, 0, loss_data)
        logging.warning("=====" + trainOrDev + ': Target = ' + str(target) + ', Pred = ' + str(x) + ', loss = ' + str(loss_data) )

        return loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        # logging.warning("x.shape: " + str(x.shape))
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize(self, feat):
        """Recognize input speech.
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(feat).unsqueeze(0)
        mean_f,std_f = torch.std_mean(enc_output, 1)
        att_vec = torch.cat((mean_f, std_f), 1)
        x = self.linear2(att_vec).squeeze()
        return x

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, utt_list):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, utt_list)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, utt_list):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, utt_list)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
