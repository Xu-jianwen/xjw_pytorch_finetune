import numpy as np
from sklearn.manifold import TSNE
from PIL import Image


class VisTSNE:
    def __init__(self, feat, path_list):
        self.feats = feat
        self.path_list = path_list

    def tsne_transform(self, feats):
        tsne_feats = TSNE(n_components=2, perplexity=40, verbose=1).fit_transform(feats)
        return tsne_feats

    @staticmethod
    def _norm(feat):
        """
        To normalize feature to [0, 1] x [0,1]
        """

        min_x, max_x, min_y, max_y = (
            feat[:, 0].min(),
            feat[:, 0].max(),
            feat[:, 1].min(),
            feat[:, 1].max(),
        )
        w, h = max_x - min_x, max_y - min_y

        feat = (feat - np.array([min_x, min_y])) / np.array([w, h])
        return feat

    @staticmethod
    def _img_loader(img_path, img_size):
        img = Image.open(img_path)
        w, h = img.size
        # _img_size = (int(img_size * w / max(w, h)), int(img_size * h / max(w, h)))
        _img_size = (img_size, img_size)
        img = img.resize(_img_size, Image.ANTIALIAS)
        img_array = np.array(img)
        if not len(img_array.shape) == 3:
            return
        else:
            return img_array

    @staticmethod
    def _feat2cor(feat, grid):
        return feat * np.array(grid)

    def vis_tsne(self, feats, img_list, img_size=64, grid=[32, 32], save_path=None):
        assert (
            feats.shape[1] == 2
        ), f"Expect to visualize 2D feature, but get {feats.shape[1]}D feature!"
        feats = self.tsne_transform(feats)
        feats = VisTSNE._norm(feats)
        w, h = (grid[0] + 1) * img_size, (grid[1] + 1) * img_size
        template = 255 * np.ones((w, h, 3), dtype=np.uint8)

        for img, feat in zip(img_list, feats):
            img_array = VisTSNE._img_loader(img, img_size)
            cor = np.round(VisTSNE._feat2cor(feat, grid))
            x, y = int(cor[0]), int(cor[1])
            template[
                x * img_size : (x + 1) * img_size, y * img_size : (y + 1) * img_size, :
            ] = img_array

        tsne_img = Image.fromarray(template).convert("RGB")
        if save_path is not None:
            tsne_img.save(save_path)
        else:
            return tsne_img
