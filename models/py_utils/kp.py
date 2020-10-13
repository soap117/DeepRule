import numpy as np
import torch
import torch.nn as nn

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr, cls, offset, line_cls
import time
from .kp_utils import _tranpose_and_gather_feat, _decode, _decode_pure, _decode_gt, _decode_line_cls, _decode_pure_cls, _decode_pure_line
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss, _ae_line_loss, _offset_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_center_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()
        print("use kp")
        self.nstack    = nstack
        self._decode   = _decode

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-6:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_cls_pure(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp_cls_pure, self).__init__()
        print("use kp")
        self.nstack    = nstack
        self._decode   = _decode_pure_cls

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])


        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

        self.cls = cls(2, cnv_dim, cnv_dim, 3, stride=2)
        self.offset = offset(2, cnv_dim, cnv_dim, 1, stride=2)

    def _train(self, *xs):
            image   = xs[0]
            tl_inds = xs[1]
            br_inds = xs[2]

            inter = self.pre(image)
            outs  = []

            layers = zip(
                self.kps, self.cnvs,
                self.tl_cnvs, self.br_cnvs,
                self.tl_heats, self.br_heats,
                self.tl_regrs, self.br_regrs,
            )
            for ind, layer in enumerate(layers):
                kp_, cnv_          = layer[0:2]
                tl_cnv_, br_cnv_   = layer[2:4]
                tl_heat_, br_heat_ = layer[4:6]
                tl_regr_, br_regr_ = layer[6:8]

                kp  = kp_(inter)
                cnv = cnv_(kp)
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
                br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

                outs += [tl_heat, br_heat, tl_regr, br_regr]
                if ind == self.nstack - 1:
                    cls_p = self.cls(cnv)
                    offset_p = self.offset(cnv)
                    outs += [cls_p, offset_p]
                if ind < self.nstack - 1:
                    inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                    inter = self.relu(inter)
                    inter = self.inters[ind](inter)
            return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs,
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            tl_cnv_, br_cnv_ = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                cls_p = self.cls(cnv)
                offset_p = self.offset(cnv)
                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_regr, br_regr]
                outs += [cls_p, offset_p]
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-6:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_gt(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp_gt, self).__init__()
        print("use kp")
        self.nstack    = nstack
        self._decode   = _decode_gt

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_   = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag,  br_tag  = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-6:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class kp_pure(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual, if_dcn=False
    ):
        super(kp_pure, self).__init__()
        print("use kp pure")
        self.nstack    = nstack
        self._decode   = _decode_pure
        self.if_dcn = if_dcn
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)


    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            if self.if_dcn:
                cnv = self.dcn(cnv)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-4:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_pure_line_cls(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual, if_dcn=False
    ):
        super(kp_pure_line_cls, self).__init__()
        print("use kp pure")
        self.nstack    = nstack
        self._decode   = _decode_line_cls
        self.if_dcn = if_dcn
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])


        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cls = nn.ModuleList([
            line_cls(cnv_dim*8, 2) for _ in range(nstack)
        ])
        self.relu = nn.ReLU(inplace=True)

    def _group_features(self, features, weight):
        features = features.view(features.size(0), -1, 4, features.size(2))
        weight = weight.view(weight.size(0), -1, 4)
        weight = weight.unsqueeze(3)
        weighted_features = features * weight
        weighted_features = torch.sum(weighted_features, 2)
        weighted_features = weighted_features.view(weighted_features.size(0), -1, 8*weighted_features.size(2))
        return weighted_features

    def _train(self, *xs):
        image   = xs[0]
        ps_inds = xs[1]
        ng_inds = xs[2]
        ps_weight = xs[3]
        ng_weight = xs[4]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs, self.cls
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            cls_ = layer[2]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            ps_features = _tranpose_and_gather_feat(cnv, ps_inds)
            ng_features = _tranpose_and_gather_feat(cnv, ng_inds)
            ps_features_group = self._group_features(ps_features, ps_weight)
            ng_features_group = self._group_features(ng_features, ng_weight)
            ps_prediction = cls_(ps_features_group)
            ng_prediction = cls_(ng_features_group)
            outs += [ps_prediction, ng_prediction]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]
        ps_inds = xs[1]
        ng_inds = xs[2]
        ps_weight = xs[3]
        ng_weight = xs[4]
        ps_mask = xs[5]
        ng_mask = xs[6]
        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs, self.cls
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            cls_ = layer[2]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            if self.if_dcn:
                cnv = self.dcn(cnv)

            if ind == self.nstack - 1:
                ps_features = _tranpose_and_gather_feat(cnv, ps_inds)
                ng_features = _tranpose_and_gather_feat(cnv, ng_inds)
                ps_features_group = self._group_features(ps_features, ps_weight)
                ng_features_group = self._group_features(ng_features, ng_weight)
                ps_prediction = cls_(ps_features_group)
                ng_prediction = cls_(ng_features_group)
                outs += [ps_prediction, ng_prediction]
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-2:]), 0, 0

    def _test_real(self, *xs, **kwargs):
        image = xs[0]
        inds = xs[1]
        weight = xs[2]
        mask = xs[3]
        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs, self.cls
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            cls_ = layer[2]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            if self.if_dcn:
                cnv = self.dcn(cnv)

            if ind == self.nstack - 1:
                features = _tranpose_and_gather_feat(cnv, inds)
                features_group = self._group_features(features, weight)
                prediction = cls_(features_group)
                outs += [prediction]
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        final_ans = torch.argmax(outs[-1], dim=1)
        return final_ans, 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) == 5:
            return self._train(*xs, **kwargs)
        if len(xs) == 7:
            return self._test(*xs, **kwargs)
        if len(xs) == 4:
            return self._test_real(*xs, **kwargs)



class kp_pure_bar(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual, if_dcn=False
    ):
        super(kp_pure_bar, self).__init__()
        print("use kp pure")
        self.nstack    = nstack
        self._decode   = _decode_pure
        self.if_dcn = if_dcn
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)


    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            if self.if_dcn:
                cnv = self.dcn(cnv)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-4:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_pure_pie(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual, if_dcn=False
    ):
        super(kp_pure_pie, self).__init__()
        print("use kp pure pie")
        self.nstack    = nstack
        self._decode   = _decode_pure
        self.if_dcn = if_dcn
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)


    def _train(self, *xs):
        image   = xs[0]
        center_inds = xs[1]
        key_inds_tl = xs[2]
        key_inds_br = xs[3]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            center_cnv_, key_cnv_   = layer[2:4]
            center_heat_, key_heat_ = layer[4:6]
            center_regr_, key_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            center_cnv = center_cnv_(cnv)
            key_cnv = key_cnv_(cnv)

            center_heat, key_heat = center_heat_(center_cnv), key_heat_(key_cnv)
            center_regr, key_regr = center_regr_(center_cnv), key_regr_(key_cnv)
            center_regr = _tranpose_and_gather_feat(center_regr, center_inds)
            key_regr_tl = _tranpose_and_gather_feat(key_regr, key_inds_tl)
            key_regr_br = _tranpose_and_gather_feat(key_regr, key_inds_br)

            outs += [center_heat, key_heat, center_regr, key_regr_tl, key_regr_br]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            center_cnv_, key_cnv_ = layer[2:4]
            center_heat_, key_heat_ = layer[4:6]
            center_regr_, key_regr_ = layer[6:8]

            kp = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                center_cnv = center_cnv_(cnv)
                key_cnv = key_cnv_(cnv)

                center_heat, key_heat = center_heat_(center_cnv), key_heat_(key_cnv)
                center_regr, key_regr = center_regr_(center_cnv), key_regr_(key_cnv)
                outs += [center_heat, key_heat, center_regr, key_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return self._decode(*outs[-4:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_pure_pie_s(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual, if_dcn=False
    ):
        super(kp_pure_pie_s, self).__init__()
        print("use kp pure pie")
        self.nstack    = nstack
        self._decode   = _decode_pure
        self.if_dcn = if_dcn
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)


    def _train(self, *xs):
        image   = xs[0]
        center_inds = xs[1]
        key_inds_tl = xs[2]
        key_inds_br = xs[3]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            center_heat_, key_heat_ = layer[4:6]
            center_regr_, key_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            #tl_cnv = tl_cnv_(cnv)
            #br_cnv = br_cnv_(cnv)

            center_heat, key_heat = center_heat_(cnv), key_heat_(cnv)
            center_regr, key_regr = center_regr_(cnv), key_regr_(cnv)
            center_regr = _tranpose_and_gather_feat(center_regr, center_inds)
            key_regr_tl = _tranpose_and_gather_feat(key_regr, key_inds_tl)
            key_regr_br = _tranpose_and_gather_feat(key_regr, key_inds_br)

            outs += [center_heat, key_heat, center_regr, key_regr_tl, key_regr_br]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            center_cnv_, key_cnv_ = layer[2:4]
            center_heat_, key_heat_ = layer[4:6]
            center_regr_, key_regr_ = layer[6:8]

            kp = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                #center_cnv = center_cnv_(cnv)
                #key_cnv = key_cnv_(cnv)

                center_heat, key_heat = center_heat_(cnv), key_heat_(cnv)
                center_regr, key_regr = center_regr_(cnv), key_regr_(cnv)
                outs += [center_heat, key_heat, center_regr, key_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return self._decode(*outs[-4:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_line(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_center_layer=make_center_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp_line, self).__init__()
        print("use kp")
        self.nstack    = nstack
        self._decode   = _decode_pure_line

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.key_cnvs = nn.ModuleList([
            make_center_layer(cnv_dim) for _ in range(nstack)
        ])
        self.hybrid_cnvs = nn.ModuleList([
            make_center_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.key_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.hybrid_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.key_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for key_heat, hybrid_heat in zip(self.key_heats, self.hybrid_heats):
            key_heat[-1].bias.data.fill_(-2.19)
            hybrid_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.key_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        key_inds = xs[1]
        key_inds_grouped = xs[2]
        tag_group_lens = xs[3]

        inter = self.pre(image)
        outs  = []
        layers = zip(
            self.kps, self.cnvs,
            self.key_cnvs, self.hybrid_cnvs,
            self.key_heats, self.hybrid_heats,
            self.key_tags,
            self.key_regrs,
        )
        for ind, layer in enumerate(layers):
            key_tag_grouped = []
            kp_, cnv_          = layer[0:2]
            key_cnv_, hybrid_cnv_   = layer[2:4]
            key_heat_, hybrid_heat_ = layer[4:6]
            key_tag_ = layer[6]
            key_regr_= layer[7]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            key_cnv = key_cnv_(cnv)
            hybrid_cnv = hybrid_cnv_(cnv)

            key_heat, hybrid_heat = key_heat_(key_cnv), hybrid_heat_(hybrid_cnv)
            key_tag_ori  = key_tag_(cnv)
            key_regr_ori = key_regr_(key_cnv)

            key_tag  = _tranpose_and_gather_feat(key_tag_ori, key_inds)
            key_regr = _tranpose_and_gather_feat(key_regr_ori, key_inds)
            for g_id in range(16):
                key_tag_grouped.append(torch.unsqueeze(_tranpose_and_gather_feat(key_tag_ori, key_inds_grouped[:, g_id,:]), 1))
            key_tag_grouped = torch.cat(key_tag_grouped, 1)
            outs += [key_heat, hybrid_heat, key_tag, key_tag_grouped, key_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs = []
        layers = zip(
            self.kps, self.cnvs,
            self.key_cnvs, self.hybrid_cnvs,
            self.key_heats, self.hybrid_heats,
            self.key_tags,
            self.key_regrs,
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            key_cnv_, hybrid_cnv_ = layer[2:4]
            key_heat_, hybrid_heat_ = layer[4:6]
            key_tag_ = layer[6]
            key_regr_ = layer[7]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                key_cnv = key_cnv_(cnv)
                hybrid_cnv = hybrid_cnv_(cnv)

                key_heat, hybrid_heat = key_heat_(key_cnv), hybrid_heat_(hybrid_cnv)
                key_tag_ori = key_tag_(cnv)
                key_regr_ori = key_regr_(key_cnv)

                outs += [key_heat, hybrid_heat, key_tag_ori, key_regr_ori]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-4:], **kwargs), 0, 0

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class kp_pure_dcn(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual, if_dcn=False
    ):
        super(kp_pure_dcn, self).__init__()
        print("use kp pure")
        self.nstack    = nstack
        self._decode   = _decode_pure
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])
        self.dcns = nn.ModuleList([
            dcn(4, cnv_dim, cnv_dim, 3, 3) for _ in range(nstack)
        ])
        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs, self.dcns,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ , dcn_   = layer[0:3]
            tl_cnv_, br_cnv_   = layer[3:5]
            tl_heat_, br_heat_ = layer[5:7]
            tl_regr_, br_regr_ = layer[7:9]

            ts = time.time()
            kp  = kp_(inter)
            cnv = cnv_(kp)
            dcn = dcn_(cnv)
            te = time.time()
            tl_cnv = tl_cnv_(dcn)
            br_cnv = br_cnv_(dcn)

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](dcn)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        tp = time.time()
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs, self.dcns,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_, dcn_    = layer[0:3]
            tl_cnv_, br_cnv_   = layer[3:5]
            tl_heat_, br_heat_ = layer[5:7]
            tl_regr_, br_regr_ = layer[7:9]
            kp = kp_(inter)
            cnv = cnv_(kp)
            dcn = dcn_(cnv)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(dcn)
                br_cnv = br_cnv_(dcn)
                ts = time.time()
                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
                te = time.time()
                outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](dcn)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-4:], **kwargs), ts-tp, te-ts

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class kp_pure_mix(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256,
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
        kp_layer=residual
    ):
        super(kp_pure_mix, self).__init__()
        print("use kp mix")
        self.nstack    = nstack
        self._decode   = _decode_pure

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(2*cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(2*cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(2*cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(2*cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            tl_cnv_mixed = torch.cat((tl_cnv, cnv), dim=1)
            br_cnv_mixed = torch.cat((br_cnv, cnv), dim=1)

            tl_heat, br_heat = tl_heat_(tl_cnv_mixed), br_heat_(br_cnv_mixed)
            tl_regr, br_regr = tl_regr_(tl_cnv_mixed), br_regr_(br_cnv_mixed)

            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.tl_heats, self.br_heats,
            self.tl_regrs, self.br_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_   = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_regr_, br_regr_ = layer[6:8]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                tl_cnv_mixed = torch.cat((tl_cnv, cnv), dim=1)
                br_cnv_mixed = torch.cat((br_cnv, cnv), dim=1)

                tl_heat, br_heat = tl_heat_(tl_cnv_mixed), br_heat_(br_cnv_mixed)
                tl_regr, br_regr = tl_regr_(tl_cnv_mixed), br_regr_(br_cnv_mixed)

                outs += [tl_heat, br_heat, tl_regr, br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-4:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb

    def forward(self, outs, targets):
        stride = 6

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_tags  = outs[2::stride]
        br_tags  = outs[3::stride]
        tl_regrs = outs[4::stride]
        br_regrs = outs[5::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, self.lamda, self.lamdb)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, self.lamda, self.lamdb)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0)


class AELossPureCls(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELossPureCls, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb
        self.cls_loss = nn.CrossEntropyLoss(size_average=True)
        self.offset_loss = _offset_loss

    def forward(self, outs, targets):
        stride = 4

        tl_heats = outs[0:-2:stride]
        br_heats = outs[1:-2:stride]
        tl_regrs = outs[2:-2:stride]
        br_regrs = outs[3:-2:stride]
        cls = outs[-2]
        offset = outs[-1]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]
        gt_cls     = targets[5]
        gt_offset  = targets[6]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, self.lamda, self.lamdb)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, self.lamda, self.lamdb)

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        cls_loss = self.cls_loss(cls, gt_cls)
        cls_loss = self.regr_weight * cls_loss

        offset_loss = self.offset_loss(offset, gt_offset)
        offset_loss = self.regr_weight * offset_loss

        loss = (focal_loss + regr_loss) / len(tl_heats) + cls_loss + offset_loss
        return loss.unsqueeze(0)

class AELossLineCls(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELossLineCls, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb
        self.cls_loss = nn.CrossEntropyLoss(size_average=True)
        self.offset_loss = _offset_loss

    def forward(self, outs, targets):
        stride = 2

        ps_predictions = outs[0::stride]
        ng_predictions = outs[1::stride]
        ps_ind = targets[0].view(-1)
        ng_ind = targets[1].view(-1)
        ps_mask = targets[2].view(-1)
        ng_mask = targets[2].view(-1)
        # focal loss
        cls_loss = 0
        for ps_pre, ng_pre in zip(ps_predictions, ng_predictions):
            ps_pre = ps_pre.view(-1, 2)
            ng_pre = ng_pre.view(-1, 2)
            if ps_mask.sum() > 0:
                cls_loss += (self.cls_loss(ps_pre[ps_mask], ps_ind[ps_mask])/2)
            if ng_mask.sum() > 0:
                cls_loss += (self.cls_loss(ng_pre[ng_mask], ng_ind[ng_mask])/2)

        loss = cls_loss
        return loss.unsqueeze(0)

class AELossLineClsFocal(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELossLineClsFocal, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb
        self.cls_loss = nn.CrossEntropyLoss(size_average=False, reduce=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, outs, targets):
        stride = 2

        ps_predictions = outs[0::stride]
        ng_predictions = outs[1::stride]
        ps_ind = targets[0].view(-1)
        ng_ind = targets[1].view(-1)
        ps_mask = targets[2].view(-1)
        ng_mask = targets[2].view(-1)
        # focal loss
        cls_loss = 0
        for ps_pre, ng_pre in zip(ps_predictions, ng_predictions):
            ps_pre = ps_pre.view(-1, 2)
            ng_pre = ng_pre.view(-1, 2)
            ps_pre_n = self.softmax(ps_pre)
            ng_pre_n = self.softmax(ng_pre)

            if ps_mask.sum() > 0:
                cls_loss += (torch.pow(1 - ps_pre_n[ps_mask][:, 0], self.lamdb) * self.cls_loss(ps_pre[ps_mask],
                                                                                                ps_ind[
                                                                                                    ps_mask]) / 2).mean()
            if ng_mask.sum() > 0:
                cls_loss += (torch.pow(1 - ng_pre_n[ng_mask][:, 1], self.lamdb) * self.cls_loss(ng_pre[ng_mask],
                                                                                                ng_ind[
                                                                                                    ng_mask]) / 2).mean()

        loss = cls_loss
        return loss.unsqueeze(0)

class AELossLine(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELossLine, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_line_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb

    def forward(self, outs, targets):
        stride = 5
        key_heats = outs[0::stride]
        hybrid_heats = outs[1::stride]
        key_tags  = outs[2::stride]
        key_tags_grouped  = outs[3::stride]
        key_regrs = outs[4::stride]


        gt_key_heat = targets[0]
        gt_hybrid_heat = targets[1]
        gt_mask    = targets[2]
        gt_mask_grouped = targets[3]
        gt_key_regr = targets[4]

        # focal loss
        focal_loss = 0

        key_heats = [_sigmoid(t) for t in key_heats]
        hybrid_heats = [_sigmoid(b) for b in hybrid_heats]

        focal_loss += self.focal_loss(key_heats, gt_key_heat, self.lamda, self.lamdb)
        focal_loss += self.focal_loss(hybrid_heats, gt_hybrid_heat, self.lamda, self.lamdb)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for key_tag_grouped in key_tags_grouped:
            pull, push = self.ae_loss(key_tag_grouped, gt_mask_grouped)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for key_regr in key_regrs:
            regr_loss += self.regr_loss(key_regr, gt_key_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(key_heats)
        return loss.unsqueeze(0)


class AELossPurePie(nn.Module):
    def __init__(self, lamda, lamdb, regr_weight=1, focal_loss=_neg_loss):
        super(AELossPurePie, self).__init__()

        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb

    def forward(self, outs, targets):
        stride = 5

        center_heats = outs[0::stride]
        key_heats = outs[1::stride]
        center_regrs = outs[2::stride]
        key_regrs_tl = outs[3::stride]
        key_regrs_br = outs[4::stride]

        gt_center_heat = targets[0]
        gt_key_heat = targets[1]
        gt_mask    = targets[2]
        gt_center_regr = targets[3]
        gt_key_regr_tl = targets[4]
        gt_key_regr_br = targets[5]

        # focal loss
        focal_loss = 0
        center_heats = [_sigmoid(t) for t in center_heats]
        key_heats = [_sigmoid(b) for b in key_heats]

        #print(center_heats[0].shape)
        #print(gt_center_heat.shape)
        focal_loss += self.focal_loss(center_heats, gt_center_heat, self.lamda, self.lamdb)
        focal_loss += self.focal_loss(key_heats, gt_key_heat, self.lamda, self.lamdb)


        regr_loss = 0
        for center_regr, key_regr_tl, key_regr_br in zip(center_regrs, key_regrs_tl, key_regrs_br):
            regr_loss += self.regr_loss(center_regr, gt_center_regr, gt_mask)
            regr_loss += self.regr_loss(key_regr_tl, gt_key_regr_tl, gt_mask)/2
            regr_loss += self.regr_loss(key_regr_br, gt_key_regr_br, gt_mask)/2
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + regr_loss) / len(center_heats)
        return loss.unsqueeze(0)


class AELossPure(nn.Module):
    def __init__(self, lamda, lamdb, regr_weight=1, focal_loss=_neg_loss):
        super(AELossPure, self).__init__()

        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss
        self.lamda = lamda
        self.lamdb = lamdb

    def forward(self, outs, targets):
        stride = 4

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_regrs = outs[2::stride]
        br_regrs = outs[3::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]

        # focal loss
        focal_loss = 0
        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, self.lamda, self.lamdb)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, self.lamda, self.lamdb)


        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0)