import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy

class Beam(object):
    """
    Implements beam search.

    When using this class with a model, the model will be passed not one input sentence, but a batch of
    k = beam_width input sentences. This will produce k probability distributions over the vocabulary. This
    class will select the top k tokens by "score"" from this k x vocabulary_size matrix. This means that
    some (hypothetically, maybe all) of those top k tokens may correspond to the same input sentence. These k
    tokens are appended to their parent input sentences to form the next batch of k input sentences for the
    model.

    The "score" of an output token at a given timestep is the score of its parent input sentence plus the
    model's estimated probability of that token. In other words, the score of a sentence (a branch of the
    search) is the sum of the probabilities of the generated tokens in it.
    """


    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        self._eos = eos
        self.eosTop = False

        # A 1d tensor of length beam_width.
        #   Contains the scores (probabilities) of the k selected tokens at the last time-step.
        #   Initially all zero - these values are meaningless, the last scores are simply not
        #   defined when the beam is initialized.
        self.scores = self.tt.FloatTensor(size).zero_()
        
        # A list of tensors, each tensor corresponding to one timestep.
        # Each tensor is a 1d array of length beam_width, containing integers
        # from 0 to beam_width, which should be thought of as referring to branches
        # of the beam search.
        #   
        self.prevKs = []
        
        # A list of tensors, each tensor corresponding to one timestep.
        # Each tensor is a 1d array of length beam_width, containing token IDs.
        #   These are the IDs of the selected token at each timestep, so nextYs[1]
        #   contains the top k tokens from the model's distribution when it was fed
        #   just the base input sentence. nextYs[2] contains the top k tokens from
        #   the k different inputs formed by appending each of nextYs[0] to the base
        #   input, and so on.
        #
        #   nextYs[0] contains just the start-of-sentence token.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][:] = sos

        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        # Basically just returns the last element of self.nextYs, with some trivial re-shaping of the tensor
        # The reshaping (into a tensor of shape (beam_width, 1)) is so that it can be passed to a huggingface model
        # and be interpreted as a batch of beam_width sentences, each of length 1 token.
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
            beam_res: the output of getFinal()
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
