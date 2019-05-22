import numpy as np
import matplotlib.pyplot as plt
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from PIL import Image
import utils
import torchvision.transforms as transforms
import torch.utils.data as data
import time
from tqdm import tqdm
import sys
import os
import nltk
import spacy
sys.path.append("pycocotools")
from coco import COCO
from build_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
SIZE_CONSTANT = 32

print("loading word vectors...")
nlp = spacy.load("en_vectors_web_lg")

def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
    q: queries. tensor of shape [B,m,d_k].
    v: values. tensor of shape [B,n,d_v].

    Returns:
    tensor of shape [B,m,d_v].
    """
    total_points = q.shape[1]
    rep = v.mean(dim=1, keepdim=True)
    rep = tile(rep, [1, total_points, 1])
    return rep

def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
    q: queries. tensor of shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    scale: float that scales the L1 distance.
    normalise: Boolean that determines whether weights sum to 1.

    Returns:
    tensor of shape [B,m,d_v].
    """
    k = k.unsqueeze(1)
    q = q.unsqueeze(2)
    unnorm_weights = -torch.abs((k - q) / scale)
    unnorm_weights = unnorm_weights.sum(dim=-1) #[B,m,n]
    if normalise:
        weight_fn = lambda x: F.softmax(x, dim=-1)
    else:
        weight_fn = lambda x: 1 + F.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = torch.einsum('bik,bkj->bij', weights, v)
    return rep

def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    normalise: Boolean that determines whether weights sum to 1.

    Returns:
    tensor of shape [B,m,d_v].
    """
    d_k = q.shape[-1]
    scale = np.sqrt(d_k)
    unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale # [B, m, n]
    if normalise:
        weight_fn = lambda x: F.softmax(x, dim=-1)
    else:
        weight_fn = F.sigmoid
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = torch.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def multihead_attention(q, k, v, Wqs, Wks, Wvs, Wo, num_heads=8):
    """Computes multi-head attention.

    Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    Wqs: list of linear query transformation. [Linear(?, d_k)]
    Wks: list of linear key transformations. [Linear(?, d_k), ...]
    Wvs: list of linear value transformations. [Linear(?, d_v), ...]
    Wo: linear transformation for output of dot-product attention
    num_heads: number of heads. Should divide d_v.

    Returns:
    tensor of shape [B,m,d_v].
    """

    d_k = q.shape[-1]
    d_v = v.shape[-1]
    head_size = d_v / num_heads
    rep = 0

    for h in range(num_heads):
        q_h = Wqs[h](q)
        k_h = Wks[h](k)
        v_h = Wvs[h](v)
        o = dot_product_attention(q_h, k_h, v_h, normalise=True)
        rep += Wo(o)

    return rep


class Attention(nn.Module):
    """The Attention module."""

    def __init__(self, rep, x_size, r_size, output_sizes, att_type, scale=1., normalise=True,
               num_heads=8):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
          rep: transformation to apply to contexts before computing attention.
              One of: ['identity','mlp'].
          output_sizes: list of number of hidden units per layer of mlp.
              Used only if rep == 'mlp'.
          att_type: type of attention. One of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: Boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        super().__init__()
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise

        d_v = r_size
        if self._rep =='mlp':
            self._mlp = BatchMLP(xy_size=x_size, output_sizes=output_sizes)
            d_k = output_sizes[-1] # dimension of keys and queries
        else:
            d_k = x_size

        if self._type == 'multihead':
            head_size = d_v // num_heads
            self._num_heads = num_heads
            self._wqs = nn.ModuleList([
              BatchMLP(d_k, output_sizes=[head_size])
              for h in range(num_heads)
            ])
            self._wks = nn.ModuleList([
              BatchMLP(d_k, output_sizes=[head_size])
              for h in range(num_heads)
            ])
            self._wvs = nn.ModuleList([
              BatchMLP(d_v, output_sizes=[head_size])
              for h in range(num_heads)
            ])
            self._wo = BatchMLP(head_size, [d_v])

    def forward(self, context_x, target_x, r):
        """Apply attention to create aggregated representation of r.

        Args:
          context_x: tensor of shape [B,n1,d_x] (keys)
          target_x: tensor of shape [B,n2,d_x] (queries)
          r: tensor of shape [B,n1,d] (values)

        Returns:
          tensor of shape [B,n2,d]

        Raises:
          NameError: The argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            target_x *= self._coef # This has grad
            k, q = context_x, target_x
        elif self._rep == 'mlp':
          # Pass through MLP
            k = self._mlp(context_x)
            q = self._mlp(target_x)
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == 'multihead':
            rep = multihead_attention(q, k, r, self._wqs, self._wks, self._wvs, self._wo, self._num_heads)
        else:
            raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                           ",'multihead']"))

        return rep


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        image = torch.Tensor(np.array(image))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_coco_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

def get_sentence(target, vocab):
    return [vocab.idx2word[x.item()] for x in target]

#I'm not sure this class is ever going to get used
class TextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(300,128,1)
        self.conv2 = nn.Conv1d(128,64,1)
        self.conv3 = nn.Conv1d(64,32,1)
        self.conv4 = nn.Conv1d(32,3,1)

    def forward(self,x): 
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        out = x.transpose(1,2)
        return out

class ImagePath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,5,stride=2)
        self.conv2 = nn.Conv2d(3,1,3)
        self.fc1 = nn.Linear(144, 300)

    def forward(self, x):
        x = x.transpose(3,2).transpose(2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        out = self.fc1(x)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_path = ImagePath()
        self.fc1 = nn.Linear(300,128)
        self.fc2 = nn.Linear(128,128)
        
    def forward(self, context_x, context_y):
        # text = self.text_path(context_x)
        img = self.image_path(context_y)
        x = torch.cat((img, context_x), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = torch.mean(x, dim=1, keepdim=True)
        return out
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(428,128,1)
        self.conv2 = nn.Conv1d(128,64,1)
        self.conv3 = nn.Conv1d(64,3,1)
        self.linear_transition = nn.Linear(3, 128)
        self.fc1 = nn.Linear(128, 32*32)
        self.fc2 = nn.Linear(32*32, 48*48)
        self.fc3 = nn.Linear(48*48, 64*64)

    def forward(self,x):
        # it is likely that I will need to use some concatenation since this probably won't train
        # due to vanishing gradients
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1,2)
        x = F.relu(self.linear_transition(x))
        out = self.fc1(x)
        # x = F.relu(self.fc2(x))
        # out = self.fc3(x)
        return torch.sigmoid(out[:,-3:,:]) # since this output will be used for image data, it needs to be constrained to be positive

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32, 28*28) 
        self.fc2 = nn.Linear(28*28, 20*20) 
        self.fc3 =nn.Linear(20*20, 16*16) 
        self.fc4 =nn.Linear(16*16, 10*10) 
        self.output = nn.Linear(10*10, 1)

    def forward(self, x):
        x = x.contiguous().view(-1, 32*32)
        out = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        out = self.output(out)
        return out.mean()

# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(33,16)
#         self.fc2 = nn.Linear(16,2)
#         self.dist = torch.distributions.Uniform(-1,1)
        
#     def forward(self, r):
#         x = torch.Tensor(np.linspace(0,5,data_points)).view(data_points,1).float()
#         out = torch.cat((x, r.repeat(1,data_points).view(data_points,32)), 1)

#         return self.fc2(F.relu(self.fc1(out))) + self.dist.sample()

def train():
    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
    PLOT_AFTER = 100 #10000 #@param {type:"number"}
    HIDDEN_SIZE = 300 #@param {type:"number"}
    MODEL_TYPE = 'ANP' #@param ['NP','ANP']
    ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']
    batch_size=64
    X_SIZE = 1
    Y_SIZE = 1
    vocab = pickle.load(open("vocab.pkl", "rb"))
    test_sentences = ["Two men seated at an open air restaurant", "flowers in a pot sitting on a cement wall", "a vase and lids are sitting on a table", "a teddy bear that is sitting next to some item on a table", "a plant in a vase by the window", "a young girl is similing and she has food around her on a table"]

    dataset_train = get_coco_loader(
        "./resized_small_train2014/",
         "./annotations/captions_train2014.json",
         vocab=vocab,
          transform=None,
           batch_size=batch_size,
            shuffle=True,
             num_workers=4
             )

    dataset_test = dataset_train # we will need to build out the test dataset soon
    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE 
    deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
    decoder_output_sizes = [32]*2 + [2]
    use_deterministic_path = True
    xy_size = X_SIZE + Y_SIZE

    # # ANP with multihead attention
    # if MODEL_TYPE == 'ANP':
    #     attention = Attention(rep='mlp', x_size=X_SIZE, r_size=deterministic_encoder_output_sizes[-1], output_sizes=[HIDDEN_SIZE]*2,
    #                         att_type=ATTENTION_TYPE).to(device) # CHANGE: rep was originally 'mlp'
    # # NP - equivalent to uniform attention
    # elif MODEL_TYPE == 'NP':
    #     attention = Attention(rep='identity', x_size=None, output_sizes=None, att_type='uniform').to(device)
    # else:
    #     raise NameError("MODEL_TYPE not among ['ANP,'NP']")

    # # Define the model
    # print("num_latents: {}, latent_encoder_output_sizes: {}, deterministic_encoder_output_sizes: {}, decoder_output_sizes: {}".format(
    #     num_latents, latent_encoder_output_sizes, deterministic_encoder_output_sizes, decoder_output_sizes))
    # decoder_input_size = 2 * HIDDEN_SIZE + X_SIZE
    # model_wass = LatentModel(X_SIZE, Y_SIZE, latent_encoder_output_sizes, num_latents,
    #                     decoder_output_sizes, use_deterministic_path,
    #                     deterministic_encoder_output_sizes, attention, loss_type="wass").to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    critic = Critic().to(device)

    optimizer_critic = torch.optim.Adam(critic.parameters())

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    for epoch in range(10):
        progress = tqdm(enumerate(dataset_train))
        total_loss = 0
        count = 0

        for i, (images, targets, lengths) in progress:
            try:
                optimizer.zero_grad()
                gen_loss = 0
                for instance in range(batch_size):
                    image, target, length = images[instance].to(device), targets[instance], lengths[instance]
                    sentence = get_sentence(target, vocab)
                    vectors = torch.Tensor([nlp(word).vector for word in sentence if "<" not in word]).to(device)
                    vectors = vectors.unsqueeze(0)
                    image = image.unsqueeze(0)
                    r = encoder(vectors, image)
        
                    decoder_input = torch.cat((r.repeat(1,vectors.shape[1], 1), vectors.float()), -1)
                    out = decoder(decoder_input)
                    fake_image = out.view(32,32,3)

                    disc_fake = critic(fake_image)
                    disc_fake.backward()
                    gen_loss = - disc_fake
                optimizer.step()

                for t in range(5):
                    optimizer_critic.zero_grad()
                    loss = 0
                    for instance in range(batch_size):
                        image, target, length = images[instance].to(device), targets[instance], lengths[instance]
                        sentence = get_sentence(target, vocab)
                        vectors = torch.Tensor([nlp(word).vector for word in sentence if "<" not in word]).to(device)
                        vectors = vectors.unsqueeze(0)
                        image = image.unsqueeze(0)
                        r = encoder(vectors, image)
                        decoder_input = torch.cat((r.repeat(1,vectors.shape[1], 1), vectors.float()), -1)
                        out = decoder(decoder_input)
                        fake_image = out.view(32,32,3)
                        # fake_image = fake_image.transpose(1,0).transpose(2,1)

                        disc_real = critic(image)
                        disc_fake = critic(fake_image)
                        gradient_penalty = utils.calc_gradient_penalty(critic, image, fake_image)
                        loss = disc_fake - disc_real + gradient_penalty
                        loss.backward()
                        w_dist = disc_real - disc_fake
                    optimizer_critic.step()
                progress.set_description("E{} - L{:.4f}".format(epoch, w_dist.item()))

                with open("encoder.pkl", "wb") as of:
                    pickle.dump(encoder, of)

                with open("decoder.pkl", "wb") as of:
                    pickle.dump(decoder, of)

                with open("critic.pkl", "wb") as of:
                    pickle.dump(critic, of)
            except Exception as e:
                print(e)
                continue
            try:
                if i % 100 ==0:
                    with torch.no_grad():
                        decoder_input = torch.cat((r.repeat(1,vectors.shape[1], 1), vectors.float()), -1)
                        out = decoder(decoder_input)
                        fake_image = out.view(32,32,3)
                        plt.imshow(fake_image.detach().cpu())
                        plt.xlabel(" ".join([x for x in sentence if x not in {"<end>", "<pad>", "<start>", "<unk>"}]), wrap=True)
                        plt.tight_layout()
                        plt.savefig("{}generated{}.png".format(i+1, sentence[1]))
                        plt.close()
            except:
                continue
        print("done")

if __name__ == "__main__":
    train()
