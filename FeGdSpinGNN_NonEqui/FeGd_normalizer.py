import torch


class FeGdNormalizer:
    """
    Normalizes:
      - data.x[:, 2:5]  (moments)
      - data.edge_attr[:, 3:4] (edge distance rij)
      - data.y          (targets)

    Usage:
      normalizer = FeGdNormalizer().fit(dataset_train)
      data = normalizer.normalize(data)
      y_denorm = normalizer.unnormalize_target(y_norm)
    """

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.stats = None

    def fit(self, dataset):
        """
        Compute mean/std from a dataset.
        """
        moments = torch.cat([data.x[:, 2:5] for data in dataset], dim=0)
        edge_dist = torch.cat([data.edge_attr[:, 3:4] for data in dataset], dim=0)
        targets = torch.cat([data.y for data in dataset], dim=0)

        self.stats = {
            "moment_mean": moments.mean(dim=0),
            "moment_std": moments.std(dim=0).clamp(min=self.eps),
            "edge_dist_mean": edge_dist.mean(dim=0),
            "edge_dist_std": edge_dist.std(dim=0).clamp(min=self.eps),
            "target_mean": targets.mean(dim=0),
            "target_std": targets.std(dim=0).clamp(min=self.eps),
        }
        return self

    def normalize(self, data, inplace=True):
        """
        Normalize a single PyG Data object.
        If inplace=False, returns a (shallow) clone before modifying.
        """
        if self.stats is None:
            raise ValueError("Normalizer has no stats. Call .fit(dataset) first.")

        if not inplace:
            data = data.clone()

        s = self.stats
        data.x[:, 2:5] = (data.x[:, 2:5] - s["moment_mean"]) / s["moment_std"]
        data.edge_attr[:, 3:4] = (data.edge_attr[:, 3:4] - s["edge_dist_mean"]) / s["edge_dist_std"]
        data.y = (data.y - s["target_mean"]) / s["target_std"]
        return data

    def normalize_dataset(self, dataset, inplace=True):
        """
        Normalize every item in a dataset. Returns a list.
        (For PyG Dataset, you often create a new list: [normalize(ds[i]) for i in ...])
        """
        return [self.normalize(data, inplace=inplace) for data in dataset]

    def unnormalize_target(self, y_norm):
        """
        Denormalize predicted or true y back to original scale.
        y_norm: tensor [..., 3]
        """
        if self.stats is None:
            raise ValueError("Normalizer has no stats. Call .fit(dataset) first.")

        s = self.stats
        return y_norm * s["target_std"] + s["target_mean"]
