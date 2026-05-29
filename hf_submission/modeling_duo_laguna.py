import os
import json

import torch
from huggingface_hub import hf_hub_download
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

from .duo_laguna_remote import enable_duo_laguna_eval


def _load_repo_json(pretrained_model_name_or_path, filename, revision=None):
    if os.path.isdir(pretrained_model_name_or_path):
        path = os.path.join(pretrained_model_name_or_path, filename)
    else:
        path = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename=filename,
            revision=revision,
        )
    with open(path) as f:
        return json.load(f)


def _load_duo_tensor(pretrained_model_name_or_path, filename, revision=None):
    if os.path.isdir(pretrained_model_name_or_path):
        path = os.path.join(pretrained_model_name_or_path, filename)
    else:
        path = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename=filename,
            revision=revision,
        )
    return torch.load(path, map_location="cpu", weights_only=True)


class DuoLagunaForCausalLM(LagunaForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        duo_attention = kwargs.pop("duo_attention", True)
        duo_sink_size = kwargs.pop("duo_sink_size", None)
        duo_recent_size = kwargs.pop("duo_recent_size", None)
        duo_heads_file = kwargs.pop("duo_heads_file", None)
        revision = kwargs.get("revision")
        config_dict = _load_repo_json(
            pretrained_model_name_or_path,
            "config.json",
            revision=revision,
        )
        duo_config = config_dict.get("duo_attention")
        base_model_name_or_path = kwargs.pop("duo_base_model_name_or_path", None)
        if duo_config is not None:
            base_model_name_or_path = (
                base_model_name_or_path
                or duo_config.get("base_model_name_or_path")
                or duo_config.get("base_model")
            )
        base_revision = kwargs.pop("duo_base_revision", None)
        if duo_config is not None:
            base_revision = base_revision or duo_config.get("base_model_revision")

        load_path = base_model_name_or_path or pretrained_model_name_or_path
        if base_revision is not None:
            kwargs["revision"] = base_revision

        model = super().from_pretrained(
            load_path,
            *model_args,
            **kwargs,
        )
        if base_revision is not None and revision is not None:
            kwargs["revision"] = revision

        if not duo_attention:
            return model

        if duo_config is None:
            duo_config = getattr(model.config, "duo_attention", None)
        if duo_config is None:
            raise ValueError(
                "This repository does not define config.duo_attention; "
                "reload with duo_attention=False to use the unpatched Laguna model."
            )

        full_attention_heads = _load_duo_tensor(
            pretrained_model_name_or_path,
            duo_heads_file or duo_config["full_attention_heads_file"],
            revision=revision,
        )
        sink_size = duo_sink_size or duo_config["sink_size"]
        recent_size = duo_recent_size or duo_config["recent_size"]

        enable_duo_laguna_eval(
            model,
            full_attention_heads,
            sink_size=sink_size,
            recent_size=recent_size,
        )
        model.duo_attention_config = {
            **duo_config,
            "sink_size": sink_size,
            "recent_size": recent_size,
        }
        return model
