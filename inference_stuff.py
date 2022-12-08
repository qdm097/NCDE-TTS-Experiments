import os

import numpy as np
import torch
import torchaudio
from torch import nn
from tqdm import tqdm

# from BeamTransformer import BeamTransformerDecoder
import audio_processing
from data_utils import TextMelLoader
from model import Tacotron2
from hparams import HyperParameters as hparams
# from high_it_backups.it22300_loss_0_point_502.hparams_8_14_21 import HyperParameters as hparams
# from high_it_backups.it22300_loss_0_point_502.model_8_14_21 import Tacotron2
# from hparams import HyperParameters as hparams
# from model import Tacotron2
from text import text_to_sequence
from train import prepare_dataloaders
from waveglow_repo.denoiser import Denoiser
import IPython.display as ipd
from waveglow_repo import glow


def synthesis(text, hp):
    checkpoint_path = os.getcwd() + "\\high_it_backups\\it22300_loss_0_point_502\\checkpoint_22300"
    m = Tacotron2(hp).cuda()
    m.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model = m.cuda().eval()
    model.training = False
    model.decoder.training = False
    model.encoder.training = False
    # m_post = ModelPostNet()

    # m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    # m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, hp.text_cleaners))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1) + 1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m = m.cuda()
    # m_post = m_post.cuda()
    m.train(False)
    # m_post.train(False)

    pbar = tqdm(range(hp.max_decoder_steps))
    with torch.no_grad():
        for i in pbar:
            # pos_mel = torch.arange(1, mel_input.shape(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, stop_token, _ = \
                m.inference(text)
            mel_input = torch.cat([mel_input, mel_pred[:, -1:, :]], dim=1)
        mag_pred = postnet_pred
        # mag_pred = m_post.forward(postnet_pred)

    # wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    with torch.no_grad():
        audio = waveglow.infer(mag_pred.cuda(), sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    torchaudio.save('test_audio.wav', audio_denoised.cpu(), sample_rate=hparams.sampling_rate)
    # write(hp.sample_path + "/test.wav", hp.sr, wav)


def infer(hparams):
    checkpoint_path = os.getcwd() + "\\high_it_backups\\it22300_loss_0_point_502\\checkpoint_22300"
    model = Tacotron2(hparams).cuda()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model = model.cuda().eval()
    model.training = False
    model.decoder.training = False
    model.encoder.training = False

    # Generate a sample for review post-training
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    # model = load_model(hparams)
    # checkpoint_path = 'output_directory\\checkpoint_300_7-16'
    # model = warm_start_model(checkpoint_path, model, []).cuda()
    # waveglow_path = 'C:\\Users\\Garrett\\.cache\\torch\\hub\\NVIDIA_DeepLearningExamples_torchhub\\PyTorch\\SpeechSynthesis\\Tacotron2\\waveglow_repo'
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    text = "yes"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    _, mel_outputs, _ = model.inference(sequence)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs.cuda(), sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    torchaudio.save('test_audio.wav', audio_denoised.cpu(), sample_rate=hparams.sampling_rate)


def synth(text, hp):
    checkpoint_path = os.getcwd() + "\\high_it_backups\\it22300_loss_0_point_502\\checkpoint_22300"
    # checkpoint_path = os.getcwd() + '\\output_directory\\checkpoint_22400'
    m = Tacotron2(hp).cuda()
    m.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    m = m.cuda().eval()
    m.decoder.ncde.return_sequences = False
    m.training = False
    m.decoder.training = False
    m.encoder.training = False
    m.decoder.pad_id = 0
    m.decoder.sos_id = 1
    m.decoder.eos_id = 2
    # m.decoder = BeamTransformerDecoder(m.decoder, 5, 3)
    m.decoder.mel_dim = hp.n_mel_channels

    text = np.asarray(text_to_sequence(text, hp.text_cleaners))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    text = text.repeat(5, 1)
    mels = m.inference(text)

    # wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    with torch.no_grad():
        audio = waveglow.infer(mels.cuda(), sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    torchaudio.save('test_audio.wav', audio_denoised.cpu(), sample_rate=hparams.sampling_rate)
    # write(hp.sample_path + "/test.wav", hp.sr, wav)


from data_utils import TextMelLoader
import layers
from utils import load_wav_to_torch


def style_transfer(text, hp):
    checkpoint_path = os.getcwd() + '\\output_directory\\checkpoint_10200'
    # "\\high_it_backups\\it22300_loss_0_point_502\\checkpoint_22300"
    m = Tacotron2(hp).cuda()
    m.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    m = m.cuda().eval()
    m.training = False
    m.decoder.training = False
    m.encoder.training = False
    m.mel_dim = hp.n_mel_channels
    decoder_modules = [i for i in m.decoder.trans_decoder.children()]
    for sub_modules in decoder_modules:
        # This is not actually an error because all children here are iterable module lists
        for modules in sub_modules:
            sub_modules_modules = [p for p in modules.children()]
            for sub_modules_module in sub_modules_modules:
                if isinstance(sub_modules_module, nn.Dropout):
                    sub_modules_module.p = 0

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    # audio, sampling_rate = load_wav_to_torch("C:\\Users\\Garrett\\Documents\\041122142603_Normal.wav")
    # audio, sampling_rate = load_wav_to_torch("C:\\Users\\Garrett\\PycharmProjects\\tacotron2\\test_audio4.wav")
    audio, sampling_rate = load_wav_to_torch("C:\\Users\\Garrett\\Documents\\test_wav_export.wav")
    stft = layers.TacotronSTFT(
        hp.filter_length, hp.hop_length, hp.win_length,
        hp.n_mel_channels, hp.sampling_rate, hp.mel_fmin,
        hp.mel_fmax)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hp.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    # melspec = torch.squeeze(melspec, 0)

    # text_inputs, text_lengths, mels, max_len, output_lengths
    inputs = [sequence.cuda(), torch.IntTensor([sequence.size(1)]).cuda(), melspec.cuda(),
              torch.IntTensor([melspec.size(2)]).cuda(), torch.IntTensor([melspec.size(2)]).cuda()]
    mels = m.forward(inputs)

    # wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    with torch.no_grad():
        audio = waveglow.infer(mels[1][0, :, :].unsqueeze(0).cuda(), sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hp.sampling_rate)
    audio_denoised = denoiser(audio, strength=0.05)[:, 0]
    torchaudio.save('test_audio4.wav', audio.cpu(), sample_rate=hp.sampling_rate)



def s(text, hp):
    checkpoint_path = os.getcwd() + '\\output_directory\\checkpoint_39800'
    # checkpoint_path = os.getcwd() + "\\high_it_backups\\it22300_loss_0_point_502\\checkpoint_22300"
    m = Tacotron2(hp).cuda()
    m.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    m = m.cuda().eval()
    m.training = False
    m.decoder.training = False
    m.encoder.training = False
    m.mel_dim = hp.n_mel_channels

    # text = np.asarray(text_to_sequence(text, hp.text_cleaners))
    # text = torch.LongTensor(text).unsqueeze(0)
    # text = text.cuda()
    # mels = m.inference(text)

    '''
    train_loader, valset, collate_fn = prepare_dataloaders(hp)
    x = None
    for i, batch in enumerate(train_loader):
        if i > 0:
            break
        x, _ = m.parse_batch(batch)
    mels = m.forward(x)
    '''
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    # text_norm = torch.IntTensor(text_to_sequence(text, hp.text_cleaners)).cuda()
    mels = m.inference(sequence)[0]  # 0 is mel output, 1 is post_net mel output, 2 is gate output
    mel_in = mels.transpose(0, 2).cuda()

    # wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    with torch.no_grad():
        # todo: what sigma?
        audio = waveglow.infer(mel_in, sigma=1)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hp.sampling_rate)
    # audio_denoised = denoiser(audio, strength=0.05)[:, 0]
    torchaudio.save('test_audio4.wav', audio.cpu(), sample_rate=hp.sampling_rate)
    '''trainset = TextMelLoader(hp.training_files, hp)
    audio = audio_processing.griffin_lim(mels[1].cpu().transpose(1,2), trainset.stft.stft_fn.cpu())
    ipd.Audio(audio.data.cpu().numpy(), rate=hp.sampling_rate)
    torchaudio.save('test_audio4.wav', audio.cpu(), sample_rate=hp.sampling_rate)'''
    # write(hp.sample_path + "/test.wav", hp.sr, wav)

'''
import json
import requests

headers = {"Authorization": f"Bearer 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"}
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# output = query({"inputs": "This is a test"})'''

'''
from espnet2.bin.tts_inference import Text2Speech

model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

speech, *_ = model("text to generate speech from")'''

def rerun_speech(text, hp):
    checkpoint_path = os.getcwd() + '\\output_directory\\checkpoint_31900'
    # checkpoint_path = os.getcwd() + "\\high_it_backups\\it22300_loss_0_point_502\\checkpoint_22300"
    m = Tacotron2(hp).cuda()
    m.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    m = m.cuda().eval()
    m.training = False
    m.decoder.training = False
    m.encoder.training = False
    m.mel_dim = hp.n_mel_channels

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    # text_norm = torch.IntTensor(text_to_sequence(text, hp.text_cleaners)).cuda()
    tt2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
    tt2 = tt2.cuda().eval()

    txt_len = torch.tensor([sequence.shape[1]]).cuda()

    tt_out = tt2.infer(sequence, txt_len)[0]
    half_point = 150
    half_out = tt_out[:, :, :half_point]
    zeros = torch.zeros((tt_out.shape[0], tt_out.shape[1], tt_out.shape[2]-half_point)).cuda()
    zeroed = torch.cat((half_out, zeros), dim=2)

    input = (sequence, txt_len, zeroed, tt_out.shape[2], torch.tensor([tt_out.shape[2]]).cuda())
    decoder_modules = [i for i in m.decoder.trans_decoder.children()]
    for sub_modules in decoder_modules:
        # This is not actually an error because all children here are iterable module lists
        for modules in sub_modules:
            sub_modules_modules = [p for p in modules.children()]
            for sub_modules_module in sub_modules_modules:
                if isinstance(sub_modules_module, nn.Dropout):
                    sub_modules_module.p = 0
    m.decoder.pos_encoder.training = False
    m.encoder.pos_encoding.training = False
    mine_out = m.forward(input)[0]

    # mels = m.inference(sequence)[0]  # 0 is mel output, 1 is post_net mel output, 2 is gate output
    # mel_in = mels.transpose(0, 2).cuda()

    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    with torch.no_grad():
        # todo: what sigma?
        audio_tt2 = waveglow.infer(tt_out, sigma=.66)
        audio = waveglow.infer(mine_out, sigma=.66)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hp.sampling_rate)
    # audio_denoised = denoiser(audio, strength=0.05)[:, 0]
    torchaudio.save('tt2_test.wav', audio_tt2.cpu(), sample_rate=hp.sampling_rate)
    torchaudio.save('tt2_test1.wav', audio.cpu(), sample_rate=hp.sampling_rate)

if __name__ == '__main__':
    hp = hparams()
    # output = query({"inputs": "This is a test"})
    # torchaudio.save('test_audio4.wav', speech.cpu(), sample_rate=hp.sampling_rate)
    # torchaudio.save('test_audio4.wav', output.cpu(), sample_rate=hp.sampling_rate)
    torch.backends.cudnn.enabled = hp.cudnn_enabled
    torch.backends.cudnn.benchmark = hp.cudnn_benchmark
    # infer(hparams)
    # "Hello, my name is Volacrum, I am an alpha-stage speech synthesizer."
    """s("this is a test of a high-iteration model, performed after running a voice style transfer test. It is "
      "currently unclear whether the model is capable of performing this process independently, or whether"
      "some form of bug has been introduced in the inference process. If not, hi, I am Volacrum.", hp)"""
    # s("Would try", hp)
    # rerun_speech("Printing, in the only sense with which we are at present concerned", hp)
    s("Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", hp)
    # s("Hello, this is a test of fire.", hp)
    # s("Hello, my name is john", hp)
    """style_transfer("this is a voice transcription test, I am unsure of the outcome of this test or whether it will"
                   " be of any value yet it is important to make anyway because different waveforms hvae different"
                   " patterns and the model may or may not be any good at generating them from scratch", hp)"""
