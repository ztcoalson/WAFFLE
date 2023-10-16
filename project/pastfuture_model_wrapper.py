"""
    Code adapted from the TextAttack's repository.
"""
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# transformers
import transformers

# textattack
from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper

# clear-out the cache
torch.cuda.empty_cache()


# ----------------------------------------
#   ResultHook: unused function for now
# ----------------------------------------
class ResultHook():
    def __init__(self, max_layers):
        pass
    #calls near the end of _attack() in attacker.py
    def call(self, result):
        pass

#modified textattack to accept this additional object from model and add it to results
class OutputData():
    def __init__(self, layer, exit_logits=None):
        self.layer = layer
        self.exit_logits = exit_logits
    #called during logging/outputting attack result on terminal
    def print_out(self):
        return "exit layer: " + str(self.layer)


# ----------------------------------------
#   PastfutureModelWrapper class
# ----------------------------------------
class PastfutureModelWrapper(PyTorchModelWrapper):

    def __init__(self, model, tokenizer, max_seq_length, model_type, multiexit, attgoal):
        assert isinstance(
            model, transformers.PreTrainedModel
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."
        self.max_length = max_seq_length
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = max_seq_length        # set the max length
        self.multiexit = multiexit
        self.model_type = model_type
        self.attgoal = attgoal

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs, all_logits = self.model(**inputs_dict)
        
        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            res = outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            res = outputs[0]

        exit_logits = None
        if (self.multiexit):
            exit_logits = all_logits
            
        if self.model_type == "albert":
            exitlayer = self.model.albert.exit_layer
        elif self.model_type == "bert":
            exitlayer = self.model.bert.exit_layer
        elif self.model_type == "roberta":
            exitlayer = self.model.roberta.exit_layer
        else:
            raise NotImplementedError()


        data = [OutputData(int(exitlayer), exit_logits) for x in range(len(outputs))]

        assert len(data) == len(res), f"Wrapper: {len(data)} data for {len(res)} results"

        return res, data


    def get_grad(self, text_input, loss_fn=CrossEntropyLoss(), adv_train=False):
        # sanity checks
        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )
        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        # set the model to 'train' mode, only when we compute gradients for adversarial training
        if adv_train: self.model.train()

        # compose inputs
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device

        # Change ids from dictionary to list of encodings
        inputs_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs_dict.to(model_device)

        # make multi-exit predictions
        # : a single logit and 12 logits
        final_logit, total_logits = self.model(**inputs_dict)

        # compute the loss w.r.t the attack goals
        if self.attgoal == 'base':
            final_logit = final_logit[0]
            output = final_logit.argmax(dim=1)
            loss = loss_fn(final_logit, output)

        elif self.attgoal == 'oavg':
            loss = 0.
            for each_logit in total_logits:
                each_output = each_logit.argmax(dim=1)
                loss += loss_fn(each_logit, each_output)

        elif self.attgoal == 'ours':
            loss = 0.

            # : compose uniform probability
            nclasses = len(final_logit[0][0])
            eachprob = 1. / nclasses
            uni_prob = [eachprob for _ in range(nclasses)]
            uni_prob = torch.Tensor(uni_prob)
            uni_prob = uni_prob.to(model_device)

            # : compute the loss (between 0 and 1)
            for each_logit in total_logits:
                loss += (1. - (1. / (nclasses - 1)) * sum(abs(F.softmax(each_logit, dim=1)[0] - uni_prob)))
            loss /= len(total_logits)

        else:
            assert False, ('Error: unsupported attack goal - {}'.format(self.attgoal))

        # compute the input gradients
        loss.backward()

        # grad w.r.t to word embeddings

        # Fix for Issue #601

        # Check if gradient has shape [max_sequence,1,_] ( when model input in transpose of input sequence)

        if emb_grads[0].shape[1] == 1:
            grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()
        else:
            # gradient has shape [1,max_sequence,_]
            grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": inputs_dict['input_ids'].tolist(), "gradient": grad}
        return output


    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
