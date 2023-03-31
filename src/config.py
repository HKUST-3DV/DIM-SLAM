from email.policy import default
from secrets import choice
from typing import overload
from xml.etree.ElementInclude import default_loader
import yaml
import argparse
import os
import json

from easydict import EasyDict as edict


def load_yaml(path, default_path=None):
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)
    
    inherit_from = cfg_special.get('inherit_from')
    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_yaml(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dict2parser(d, pre=""):
    ret_dict = edict()
    for key,val in d.items():
        if type(val) == edict or type(val) == dict:
            ret_dict.update(dict2parser(val, pre+key+"."))
        else:
            ret_dict[pre+key] = val
    return ret_dict

def build_keys(d, keys=[],val=None):
    if len(keys) == 1:
        d[keys[0]] = val
        return
    if keys[0] not in d:
        d[keys[0]] = {}
    build_keys(d[keys[0]], keys[1:], val)

def parser2dict(d):
    ret_dict = {}
    
    for key, val in d.items():
        keys = key.split(".")
        build_keys(ret_dict, keys,val)
        # ret_dict[keys[-1]] = val
    
    return ret_dict
    

class yamlparser():
    def __init__(self, args=None, verbose=True) -> None:
        
        config_file_parser = argparse.ArgumentParser()
        config_file_parser.add_argument("config", type=str, help="the configure file")

        config_file_args, remaining_args = config_file_parser.parse_known_args(args=args)

        default_config_in_file = None
        if config_file_args.config is not None and os.path.exists(config_file_args.config):
            if verbose:
                print("Loading Config from %s" % config_file_args.config)
            default_config_in_file = load_yaml(config_file_args.config)
        
        override_config_parser = argparse.ArgumentParser()

        ##### OVERRIDE CONFIG IN HERE, REPLACE THE CONFIG IN FILE

        # data 
        override_config_parser.add_argument("--exp_name", default="base", type=str, help="exp name")
        override_config_parser.add_argument("--exp_tag", default="base", type=str, help="exp tag")
        override_config_parser.add_argument("--output", default="output", type=str, help="exp dir")
        override_config_parser.add_argument("--scale", default=1.0)
        override_config_parser.add_argument("--device", default="cuda:0", type=str, help="default using cuda:0")

        # camera
        override_config_parser.add_argument("--color_cam.H", default=None,type=float, help="input image H")
        override_config_parser.add_argument("--color_cam.W", default=None, type=float,help="input image W")
        override_config_parser.add_argument("--color_cam.fx", default=None, type=float,help="camera fx")
        override_config_parser.add_argument("--color_cam.fy", default=None, type=float,help="camera fy")
        override_config_parser.add_argument("--color_cam.cx", default=None, type=float,help="camera cx")
        override_config_parser.add_argument("--color_cam.cy", default=None, type=float,help="camera cy")
        override_config_parser.add_argument("--color_cam.crop_edge", default=None,type=float, help="crop edge before resize")
        override_config_parser.add_argument("--color_cam.crop_size", nargs="+",default=None,help="reize image")
        override_config_parser.add_argument('--color_cam.distortion', nargs='+', default=None, help='cam.distortion Set flag')
        override_config_parser.add_argument("--color_cam.transform", default=None)

        override_config_parser.add_argument("--depth_cam.H", default=None,type=float, help="input image H")
        override_config_parser.add_argument("--depth_cam.W", default=None, type=float,help="input image W")
        override_config_parser.add_argument("--depth_cam.fx", default=None, type=float,help="camera fx")
        override_config_parser.add_argument("--depth_cam.fy", default=None, type=float,help="camera fy")
        override_config_parser.add_argument("--depth_cam.cx", default=None, type=float,help="camera cx")
        override_config_parser.add_argument("--depth_cam.cy", default=None, type=float,help="camera cy")
        override_config_parser.add_argument("--depth_cam.crop_edge", default=None,type=float, help="crop edge before resize")
        override_config_parser.add_argument("--depth_cam.crop_size", nargs="+",default=None,help="reize image")
        override_config_parser.add_argument("--depth_cam.depth_scale", default=None, type=float,help="store_true")
        override_config_parser.add_argument("--depth_cam.transform", default=None)
            

        # dataset
        override_config_parser.add_argument("--dataset.name", default=None, type=str, help="dataset name")
        override_config_parser.add_argument("--dataset.scene", default=None, type=str, help="dataset scene")
        override_config_parser.add_argument("--dataset.input_folder", default=None, type=str, help="dataset root path")
        override_config_parser.add_argument("--dataset.ds_start", default=None, type=int, help="dataset start")
        override_config_parser.add_argument("--dataset.ds_interval", default=None, type=int, help="dataset end")

        # nerf
        override_config_parser.add_argument("--nerf.bound", default=None, type=float, help="nerf bound")
        override_config_parser.add_argument("--nerf.grids_lens", default=None, type=int, help="nerf grids_lens")
        override_config_parser.add_argument("--nerf.grids_dim", default=None, type=int, help="nerf grids_dim")
        override_config_parser.add_argument("--nerf.decoder", default=None, type=str, help="nerf decoder type")
        override_config_parser.add_argument("--nerf.hidden_dim", default=None, type=int, help="nerf hidden_dim")
        override_config_parser.add_argument("--nerf.n_blocks", default=None, type=int, help="nerf n_blocks")
        override_config_parser.add_argument("--nerf.skips", default=None, help="nerf skips")

        # renderer

        # sfm
        override_config_parser.add_argument("--sfm.pixels", default=None, type=int, help="sfm pixels")
        override_config_parser.add_argument("--sfm.n_views", default=None, type=int, help="sfm n_views")

        # rerun
        override_config_parser.add_argument("--rerun", action="store_true")

        if default_config_in_file is not None:
            # override_config_parser.set_defaults(**default_config_in_file)
            override_config_parser.set_defaults(**dict2parser(default_config_in_file))
        
        args = override_config_parser.parse_args(remaining_args)

        args = edict(vars(args))
        args = parser2dict(args)
        args = edict(args)
        if verbose:
            print(json.dumps(args, indent=2, ensure_ascii=False))
        self.args = args
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        print("delete inhert from")
        if "inherit_from" in self.args:
            del self.args.inherit_from
        
    def print(self):
        print(json.dumps(self.args, indent=2, ensure_ascii=False))