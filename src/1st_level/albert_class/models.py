import torch
import transformers

import config


class ColeridgeModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(ColeridgeModel, self).__init__(conf)
        self.albert = transformers.AlbertModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.HIGH_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE, 1),
        )
        
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, ids, mask):
        # sequence_output of N_LAST_HIDDEN + Embedding states
        # (N_LAST_HIDDEN + 1, batch_size, num_tokens, 768)
        out = self.albert(ids, attention_mask=mask)
        out = out.last_hidden_state 

        return self.classifier(out[:, 0, :].squeeze(1))
