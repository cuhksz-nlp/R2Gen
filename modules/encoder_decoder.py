from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import modules.utils as utils
from modules.layers import *


class BaseLanguageModel(nn.Module):
    def __init__(self):
        super(BaseLanguageModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]  # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1),
                                            change.new_ones(batch_size, 1))

                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda

            return logprobs, unaug_logprobs

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # beam_logprobs_sum Nxb logprobs is NxbxV
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size  # Nxb which beam
            selected_ix = ix % vocab_size  # Nxb # which world
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1)

            if t > 0:
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
                        beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))

                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size))
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([beam_seq_logprobs, beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        # Start diverse_beam_search
        temperature = getattr(self.args, 'temperature', 1)
        beam_size = getattr(self.args, 'beam_size', 10)
        group_size = getattr(self.args, 'group_size', 1)
        diversity_lambda = getattr(self.args, 'diversity_lambda', 0.5)
        decoding_constraint = getattr(self.args, 'decoding_constraint', 0)
        suppress_UNK = getattr(self.args, 'suppress_UNK', 0)
        length_penalty = utils.penalty_builder(getattr(self.args, 'length_penalty', ''))
        bdash = beam_size // group_size  # beam per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1).to(device) for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]

        # Chunk elements in the args
        args = list(args)
        args = utils.split_tensors(group_size, args)  # For each arg, turn (Bbg)x... to (Bb)x(g)x...
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in
                    range(group_size)]  # group_name, arg_name, model_name
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.max_seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.max_seq_length + divm - 1:
                    # add diversity
                    logprobs = logprobs_table[divm]
                    # suppress previous word
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1).to(device),
                                          float('-inf'))
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)

                    # infer new beams
                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm] = beam_step(logprobs,
                                                  unaug_logprobs,
                                                  bdash,
                                                  t - divm,
                                                  beam_seq_table[divm],
                                                  beam_seq_logprobs_table[divm],
                                                  beam_logprobs_sum_table[divm],
                                                  state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t - divm] == self.eos_idx
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1
                        if t == self.max_seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(),
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    # move the current group one step forward in time

                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(
                            args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
                            for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel':
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).cuda()
                return -torch.log(-torch.log(U + eps) + eps)

            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return F.log_softmax(y / temperature, dim=-1)

            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        else:
            logprobs = logprobs / temperature
            if sample_method.startswith('top'):
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    probs = F.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
                    sorted_probs = sorted_probs * mask.float()
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        return it, sampleLogprobs


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class EncoderDecoder(BaseLanguageModel):
    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.vocab_size = tokenizer.get_vocab_size()
        self.d_model = self.args.d_model
        self.d_ff = self.args.d_ff
        self.num_layers = self.args.num_layers
        self.drop_prob_lm = self.args.drop_prob_lm
        self.max_seq_length = self.args.max_seq_length
        self.d_vf = self.args.d_vf

        self.bos_idx = self.args.bos_idx
        self.eos_idx = self.args.eos_idx
        self.pad_idx = self.args.pad_idx

        self.use_bn = getattr(self.args, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = lambda x: x
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.d_vf),) if self.use_bn else ()) +
                (nn.Linear(self.d_vf, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        self.logit = nn.Linear(self.d_model, self.vocab_size + 1)

        self.tokenizer = tokenizer
        self.model = make_model(self.args, self.vocab_size + 1)

    def init_hidden(self, bsz):
        return []

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        hidden_states = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], hidden_states, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=1)

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, hidden_states, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(hidden_states, mask, ys, subsequent_mask(ys.size(1)).to(hidden_states.device))
        return out[:, -1], [ys.unsqueeze(0)]

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None):
        beam_size = getattr(self.args, 'beam_size', 10)
        group_size = getattr(self.args, 'group_size', 1)
        sample_n = getattr(self.args, 'sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None):
        sample_method = getattr(self.args, 'sample_method', 'greedy')
        beam_size = getattr(self.args, 'beam_size', 1)
        temperature = getattr(self.args, 'temperature', 1.0)
        sample_n = int(getattr(self.args, 'sample_n', 1))
        group_size = getattr(self.args, 'group_size', 1)
        output_logsoftmax = getattr(self.args, 'output_logsoftmax', 1)
        decoding_constraint = getattr(self.args, 'decoding_constraint', 0)
        block_trigrams = getattr(self.args, 'block_trigrams', 0)

        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        for t in range(self.max_seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.max_seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None):
        sample_method = getattr(self.args, 'sample_method', 'greedy')
        beam_size = getattr(self.args, 'beam_size', 1)
        temperature = getattr(self.args, 'temperature', 1.0)
        group_size = getattr(self.args, 'group_size', 1)
        diversity_lambda = getattr(self.args, 'diversity_lambda', 0.5)
        decoding_constraint = getattr(self.args, 'decoding_constraint', 0)
        block_trigrams = getattr(self.args, 'block_trigrams', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]  # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.max_seq_length), self.pad_idx, dtype=torch.long) for _ in
                     range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.max_seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.max_seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.max_seq_length - 1:
                    if t == 0:  # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t - 1]  # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state_table[divm])
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda

                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:, t - 3:t - 1]
                        for i in range(batch_size):  # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current = seq[i][t - 1]
                            if t == 3:  # initialize
                                trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]:  # add to list
                                    trigrams[i][prev_two].append(current)
                                else:  # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:, t - 2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i, j] += 1
                        # Apply mask to log probs
                        # logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0  # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = seq[:, t - 1] != self.pad_idx & seq[:, t - 1] != self.eos_idx
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx)  # changed
                    seq[:, t] = it
                    seqLogprobs[:, t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table, 1).reshape(batch_size * group_size, -1)
