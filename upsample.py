import jukebox
import torch as t
import librosa
import os
import sys
import pickle
import random
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache

port = random.randint(10000, 20000)
rank, local_rank, device = setup_dist_from_mpi(port=port)

model = "5b_lyrics" # or "1b_lyrics"
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model=='5b_lyrics' else 16
# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.

if len(sys.argv) > 1:
  this_run_slug = sys.argv[1]
else:
  this_run_slug = "co_compose_synth2"

hps.name = '/home/robin/google-drive/samples/' + this_run_slug + '/'

meta = pickle.load( open( f'{hps.name}meta.p', "rb" ) )

hps.sample_length = 1048576 if model=="5b_lyrics" else 786432
chunk_size = 16 if model=="5b_lyrics" else 32
max_batch_size = 3 if model=="5b_lyrics" else 16
hps.hop_fraction = [.5, .5, .125]
hps.levels = 3

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)

metas_1 = meta[0]
metas_2 = meta[1]

print(metas_1)
print(metas_2)

zs = t.load(f'{hps.name}zs-top-level-final.t')

top_prior_raw_to_tokens = 128

hps.sample_length = zs[2].shape[1] * top_prior_raw_to_tokens

upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]

labels_1 = [prior.labeller.get_batch_labels(metas_1, 'cuda') for prior in upsamplers]
labels_2 = [prior.labeller.get_batch_labels(metas_2, 'cuda') for prior in upsamplers]

sampling_kwargs = [dict(temp=0.985, fp16=True, max_batch_size=16, chunk_size=32),
                   dict(temp=0.985, fp16=True, max_batch_size=16, chunk_size=32),
                   None]

#if type(labels_1[2])==dict:
#  labels_1[2] = [prior.labeller.get_batch_labels(metas_1, 'cuda') for prior in upsamplers] + [labels_1[2]]
#  labels_2[2] = [prior.labeller.get_batch_labels(metas_2, 'cuda') for prior in upsamplers] + [labels_2[2]]

top_prior = None

print(labels_1)
print(labels_2)

zs = upsample(zs, labels_1, labels_2, sampling_kwargs, [*upsamplers, top_prior], hps)

print(f'DONE with ${this_run_slug}')
