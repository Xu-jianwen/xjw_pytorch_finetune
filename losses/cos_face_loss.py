import torch.nn.functional as F
import torch.nn as nn
from util import proxies_reducer
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import CosFaceLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class CosFace(CosFaceLoss):
    def __init__(self, margin=0.4, scale=64, **kwargs):
        super(CosFace, self).__init__(margin=margin, scale=scale, **kwargs)
        
    def forward(self, W, embeddings, labels, indices_tuple=None):
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        loss_dict = self.compute_loss(W, embeddings, labels, indices_tuple)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)
    
    def compute_loss(self, W, embeddings, labels, indices_tuple=None):

        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = embeddings @ W.t()
        cosine = proxies_reducer(self.num_classes, int(W.size(0)/self.num_classes), cosine)
        # cosine = F.softmax(cosine, dim=1)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)
        unweighted_loss = self.cross_entropy(logits, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, W)

        return loss_dict