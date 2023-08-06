from typing import OrderedDict
import warnings
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle, Arrow

from mmdet.core.visualization import color_val_matplotlib


def init_detector(config, checkpoint=None, device="cuda:0", cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if "CLASSES" in checkpoint["meta"]:
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "Class names are not saved in the checkpoint's "
                "meta data, use COCO classes by default."
            )
            model.CLASSES = get_classes("coco")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = mmcv.imread(results["img"])
        results["img"] = img
        results["img_fields"] = ["img"]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def inference_detector(model, img, rescale=True):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # just get the actual data from DataContainer
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), "CPU inference with RoIPool is not supported currently."

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=rescale, **data)[0]
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(
    model,
    img,
    result,
    score_thr=0.3,
    fig_size=(15, 10),
    title="result",
    block=True,
    wait_time=0,
):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI. Default: True
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    warnings.warn('"block" will be deprecated in v2.9.0,' 'Please use "wait_time"')
    if hasattr(model, "module"):
        model = model.module

    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        fig_size=fig_size,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
    )


def show_pts_pyplot(
    model,
    img,
    result,
    result_form=dict(box=5, center=1, pts_i=9, pts_r=9),
    result_show=dict(box=True, center=True, pts_i=True, pts_r=True, arrow=True),
    score_thr=0.3,
    det_color=(72, 101, 241),
    text_color=(72, 101, 241),
    thickness=2,
    font_size=13,
    show=True,
    out_file=None,
    fig_size=(15, 10),
    win_name="result",
    wait_time=0,
):

    if hasattr(model, "module"):
        model = model.module
    class_names = model.CLASSES

    img = mmcv.imread(img).copy()
    dets = np.vstack(result)
    labels = [np.full(det.shape[0], i, dtype=np.int32) for i, det in enumerate(result)]

    labels = np.concatenate(labels)
    if out_file is not None:
        show = False

    if score_thr > 0:
        scores = dets[:, 4]
        inds = scores > score_thr
        dets = dets[inds, :]
        labels = labels[inds]

    det_color = color_val_matplotlib(det_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    img = np.ascontiguousarray(img)

    plt.figure(win_name, figsize=fig_size)
    # plt.title(win_name)
    plt.axis("off")
    ax = plt.gca()

    polygons = []
    pts_center = []
    pts_init = []
    pts_refine = []
    arrows_init = []
    arrows_refine = []
    color = []

    for i, (det, label) in enumerate(zip(dets, labels)):
        index = 0

        color.append(det_color)
        label_text = class_names[label] if class_names is not None else f"class {label}"
        if len(det) > 4:
            label_text += f"|{det[4]:.02f}"

        if "box" in result_form:
            det_int = det.astype(np.int32)
            poly = [
                [det_int[0], det_int[1]],
                [det_int[0], det_int[3]],
                [det_int[2], det_int[3]],
                [det_int[2], det_int[1]],
            ]
            np_poly = np.array(poly).reshape((4, 2))
            index = index + result_form["box"]

        if "center" in result_form:
            center = np.array([det_int[index], det_int[index + 1]])
            index = index + result_form["center"] * 2

        nums = 0
        if "pts_i" in result_form:
            nums = result_form["pts_i"]
            pts_i = det_int[index : index + nums * 2]
            if "pts_r" in result_form:
                assert result_form["pts_i"] == result_form["pts_r"]
                pts_r = det_int[index + nums * 2 : index + nums * 4]

        if result_show["box"]:
            polygons.append(Polygon(np_poly))
            ax.text(
                det_int[0],
                det_int[1],
                f"{label_text}",
                bbox={
                    "facecolor": "black",
                    "alpha": 1.0,
                    "pad": 0.7,
                    "edgecolor": "none",
                },
                color="white",
                fontsize=font_size,
                verticalalignment="top",
                horizontalalignment="left",
            )

        if result_show["center"]:
            pts_center.append(Circle(center, 8.0))

        for i in range(nums):
            if result_show["pts_i"]:
                pts_init.append(Circle(np.array([pts_i[i * 2], pts_i[i * 2 + 1]]), 8.0))
            if result_show["pts_r"]:
                pts_refine.append(
                    Circle(np.array([pts_r[i * 2], pts_r[i * 2 + 1]]), 4.0)
                )
            if result_show["pts_i"] and result_show["arrow"]:
                arrows_init.append(
                    Arrow(
                        np.array(center[0]),
                        np.array(center[1]),
                        np.array(pts_i[i * 2] - center[0]),
                        np.array(pts_i[i * 2 + 1] - center[1]),
                        width=1.0,
                    )
                )
            if result_show["pts_i"] and result_show["pts_r"] and result_show["arrow"]:
                arrows_refine.append(
                    Arrow(
                        np.array(pts_i[i * 2]),
                        np.array(pts_i[i * 2 + 1]),
                        np.array(pts_r[i * 2] - pts_i[i * 2]),
                        np.array(pts_r[i * 2 + 1] - pts_i[i * 2 + 1]),
                        width=1.0,
                    )
                )

        # color.append(det_color)
        # label_text = class_names[label] if class_names is not None else f"class {label}"
        # if len(det) > 4:
        #     label_text += f"|{det[4]:.02f}"
        # ax.text(
        #     det_int[0],
        #     det_int[1],
        #     f"{label_text}",
        #     bbox={"facecolor": "black", "alpha": 1.0, "pad": 0.7, "edgecolor": "none"},
        #     color="white",
        #     fontsize=font_size,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        # )

    plt.imshow(img)

    p = PatchCollection(
        polygons,
        facecolor="none",
        # facecolor="#36b050",
        alpha=1.0,
        edgecolors="#f9c001",
        # edgecolors="black",
        # edgecolors="none",
        # linestyles="dashed",
        linewidths=4,  # 6
    )
    ax.add_collection(p)

    ai = PatchCollection(
        arrows_init, facecolor="#f40104", edgecolors="#f40104", linewidths=2
    )
    ax.add_collection(ai)

    ar = PatchCollection(
        arrows_refine, facecolor="#f9c001", edgecolors="#f9c001", linewidths=2
    )
    ax.add_collection(ar)

    c = PatchCollection(
        pts_center, facecolor="#36b050", edgecolors="#36b050", linewidths=1
    )
    ax.add_collection(c)

    i = PatchCollection(
        pts_init, facecolor="#f40104", edgecolors="#f40104", linewidths=1
    )
    ax.add_collection(i)

    r = PatchCollection(
        pts_refine, facecolor="#f9c001", edgecolors="#f9c001", linewidths=1
    )
    ax.add_collection(r)

    if out_file is not None:
        dir_name = osp.abspath(osp.dirname(out_file))
        mmcv.mkdir_or_exist(dir_name)
        plt.savefig(out_file)
        if not show:
            plt.close()
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
            plt.close()

    if not (show or out_file):
        return mmcv.rgb2bgr(img)


# def show_pts_pyplot(
#                     model,
#                     img,
#                     result,
#                     items=['bbox', 'center', 'pts_init', 'arrow'],
#                     score_thr=0.3,
#                     det_color=(72, 101, 241),
#                     text_color=(72, 101, 241),
#                     thickness=2,
#                     font_size=13,
#                     show=True,
#                     out_file=None,
#                     fig_size=(15, 10),
#                     win_name='result',
#                     wait_time=0):

#     if hasattr(model, 'module'):
#         model = model.module
#     class_names = model.CLASSES

#     img = mmcv.imread(img).copy()
#     dets = np.vstack(result)
#     labels = [
#         np.full(det.shape[0], i, dtype=np.int32)
#         for i, det in enumerate(result)
#     ]
#     labels = np.concatenate(labels)
#     if out_file is not None:
#         show = False

#     if score_thr > 0:
#         scores = dets[:, 4]
#         inds = scores > score_thr
#         dets = dets[inds, :]
#         labels = labels[inds]

#     det_color = color_val_matplotlib(det_color)
#     text_color = color_val_matplotlib(text_color)

#     img = mmcv.bgr2rgb(img)
#     img = np.ascontiguousarray(img)

#     plt.figure(win_name, figsize=fig_size)
#     plt.title(win_name)
#     plt.axis('off')
#     ax = plt.gca()

#     polygons = []
#     pts_center = []
#     pts_init = []
#     arrows = []
#     color = []
#     for i, (det, label) in enumerate(zip(dets, labels)):
#         det_int = det.astype(np.int32)

#         if 'center' in items:
#             center = np.array([det_int[5], det_int[6]])
#             pts_center.append(Circle(center, 2.0))

#         # nums = (det_int.shape[0] - 7) // 2
#         nums = 9

#         pts_i = det_int[7:7+nums*2]

#         for i in range(nums):
#             if 'pts_init' in items:
#                 pts_init.append(Circle(
#                     np.array([pts_i[i*2], pts_i[i*2+1]]),
#                     2.0))
#             if ('pts_init' in items) and ('arrow' in items):
#                 arrows.append(Arrow(
#                     np.array(det_int[5]), np.array(det_int[6]),
#                     np.array(pts_i[i*2]-det_int[5]), np.array(pts_i[i*2+1]-det_int[6]),
#                     width=1.0))

#         color.append(det_color)
#         label_text = class_names[
#             label] if class_names is not None else f'class {label}'
#         if len(det) > 4:
#             label_text += f'|{det[4]:.02f}'
#         if 'bbox' in items:
#             poly = [[det_int[0], det_int[1]], [det_int[0], det_int[3]],
#                     [det_int[2], det_int[3]], [det_int[2], det_int[1]]]
#             np_poly = np.array(poly).reshape((4, 2))
#             polygons.append(Polygon(np_poly))
#             ax.text(
#                 det_int[0],
#                 det_int[1],
#                 f'{label_text}',
#                 bbox={
#                     'facecolor': 'black',
#                     'alpha': 0.8,
#                     'pad': 0.7,
#                     'edgecolor': 'none'
#                 },
#                 color=text_color,
#                 fontsize=font_size,
#                 verticalalignment='top',
#                 horizontalalignment='left')

#     plt.imshow(img)

#     p = PatchCollection(
#         polygons, facecolor='none', edgecolors=color, linewidths=thickness)
#     ax.add_collection(p)

#     c = PatchCollection(
#         pts_center, facecolor='black', edgecolors='black', linewidths=thickness)
#     ax.add_collection(c)

#     a = PatchCollection(
#         arrows, facecolor='black', edgecolors='black', linewidths=thickness)
#     ax.add_collection(a)

#     i = PatchCollection(
#         pts_init, facecolor='blue', edgecolors='blue', linewidths=thickness)
#     ax.add_collection(i)


#     if out_file is not None:
#         dir_name = osp.abspath(osp.dirname(out_file))
#         mmcv.mkdir_or_exist(dir_name)
#         plt.savefig(out_file)
#         if not show:
#             plt.close()
#     if show:
#         if wait_time == 0:
#             plt.show()
#         else:
#             plt.show(block=False)
#             plt.pause(wait_time)
#             plt.close()

#     if not (show or out_file):
#         return mmcv.rgb2bgr(img)
